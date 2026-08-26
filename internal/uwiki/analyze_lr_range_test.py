"""
Read the LR range test and report, per method, the usable learning-rate window.

Consumes what `launch_lr_range_test.sh` leaves behind:

    <OUTPUT_ROOT>/<run-tag-base>/lr<LR>/<method>/<knob>-<value>/
        <method>_config.json    learning_rate, gradient_accumulation_steps, method
        metrics.jsonl           per-micro-batch losses

and classifies each probe as flat / moving / diverged, then brackets the moving
range and proposes a four-point log-spaced grid inside it.

No evaluation pass is involved. The question here is only "does this LR move the
forget signal, and does it wreck the retain signal", which the training metrics
answer on their own.

Reading the verdicts:

  flat      the forget signal barely moved. Below the useful window -- these
            cells would be indistinguishable from the starting checkpoint.
  moving    the forget signal moved without the retain signal blowing up. This
            is the band the sweep grid should live in.
  diverged  a non-finite loss, a retain signal that climbed past --retain-max,
            or a forget signal that moved more than --forget-max in a handful of
            steps. Above the useful window.

The forget signal differs by method, so the first available field is used, and
which one is reported:

    ce_forget        gradient-ascent, wga, satimp, grad-diff, ce-u
    nll_avg_mean     simnpo
    nll_theta_mean   npo
    loss_forget      rmu   (an MSE toward the steering vector -- it DECREASES
                            as the method works, so expect a negative delta)

Stdlib only -- runs on a login node, no GPU and no torch.

Usage:
    python internal/uwiki/analyze_lr_range_test.py
    python internal/uwiki/analyze_lr_range_test.py --output-root /project/.../unlearning-pareto
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

FORGET_FIELDS = ("ce_forget", "nll_avg_mean", "nll_theta_mean", "loss_forget")
RETAIN_FIELD = "loss_retain"


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def window_mean(rows, field, n, from_end=False):
    """Mean of `field` over the first (or last) n rows that carry it."""
    seq = rows[-n:] if from_end else rows[:n]
    vals = [r[field] for r in seq if field in r and finite(r[field])]
    if not vals:
        vals = [r[field] for r in rows if field in r and finite(r[field])]
        if not vals:
            return None
    return sum(vals) / len(vals)


def load_probe(run_dir):
    """Return a dict describing one probe, or None if it produced nothing."""
    cfgs = glob.glob(os.path.join(run_dir, "*_config.json"))
    metrics = os.path.join(run_dir, "metrics.jsonl")
    if not cfgs or not os.path.isfile(metrics):
        return None
    with open(cfgs[0]) as f:
        cfg = json.load(f)

    rows = []
    any_nonfinite = False
    with open(metrics) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k, v in r.items():
                if isinstance(v, float) and not math.isfinite(v):
                    any_nonfinite = True
            rows.append(r)
    if not rows:
        return None

    field = next((f for f in FORGET_FIELDS if any(f in r for r in rows)), None)
    accum = int(cfg.get("gradient_accumulation_steps") or 1)
    win = max(1, min(accum, len(rows) // 4 or 1))

    probe = {
        "method": cfg.get("method", "?"),
        "lr": float(cfg.get("learning_rate", float("nan"))),
        "run_dir": run_dir,
        "steps": max((r.get("optimizer_step", 0) for r in rows), default=0),
        "n_rows": len(rows),
        "forget_field": field,
        "nonfinite": any_nonfinite,
    }

    for label, fld in (("forget", field), ("retain", RETAIN_FIELD)):
        if fld is None:
            probe[f"{label}_first"] = probe[f"{label}_last"] = probe[f"{label}_delta"] = None
            continue
        a = window_mean(rows, fld, win, from_end=False)
        b = window_mean(rows, fld, win, from_end=True)
        probe[f"{label}_first"] = a
        probe[f"{label}_last"] = b
        probe[f"{label}_delta"] = (b - a) if (a is not None and b is not None) else None

    return probe


def verdict(p, move_eps, retain_max, forget_max):
    """Classify a probe by the RELATIVE change in its forget signal.

    Relative, not absolute, because the forget signal is not on one scale across
    methods. ce_forget and nll_avg_mean are per-token nats starting near 1.8, but
    npo's nll_theta_mean is a SUMMED per-sequence NLL starting near 476. Absolute
    nat thresholds flagged every single npo probe as "diverged" on units alone,
    hiding a window identical to simnpo's.

    Dividing by the starting value reproduces every verdict the absolute test got
    right on the per-token methods, and fixes npo.
    """
    if p["nonfinite"]:
        return "diverged", "non-finite loss"
    rd = p["retain_delta"]
    if rd is not None and rd > retain_max:
        return "diverged", f"retain +{rd:.2f}"
    fd = p["forget_delta"]
    if fd is None:
        return "no data", "no forget signal"
    first = p.get("forget_first")
    if not first:
        return "no data", "no baseline to normalize against"
    rel = abs(fd) / abs(first)
    if rel > forget_max:
        return "diverged", f"forget {rel * 100:.0f}% in {p['steps']} steps"
    if rel < move_eps:
        return "flat", ""
    return "moving", ""


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def log_grid(lo, hi, n=4):
    if lo <= 0 or hi <= 0 or hi < lo:
        return []
    if n == 1 or math.isclose(lo, hi):
        return [lo]
    step = (math.log10(hi) - math.log10(lo)) / (n - 1)
    return [10 ** (math.log10(lo) + i * step) for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=str,
                        default=os.path.expanduser("~/pretrain-experiments/unlearning-pareto"))
    parser.add_argument("--run-tag-base", type=str, default="lr-range")
    parser.add_argument("--move-eps", type=float, default=0.03,
                        help="RELATIVE forget-signal change below which a probe counts as "
                             "flat (0.03 = 3%% of its starting value).")
    parser.add_argument("--retain-max", type=float, default=0.5,
                        help="Retain-signal rise above which a probe counts as diverged (nats).")
    parser.add_argument("--forget-max", type=float, default=2.8,
                        help="RELATIVE forget-signal change above which a probe counts as "
                             "diverged (2.8 = 280%% of its starting value).")
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--output-json", type=str, default="lr_range_test.json")
    args = parser.parse_args()

    pattern = os.path.join(args.output_root, args.run_tag_base, "lr*", "*", "*")
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(
            f"No probe directories under {os.path.join(args.output_root, args.run_tag_base)}.\n"
            f"Check --output-root / --run-tag-base, or that the jobs have finished."
        )

    by_method = defaultdict(list)
    skipped = 0
    for d in dirs:
        p = load_probe(d)
        if p is None:
            skipped += 1
            continue
        by_method[p["method"]].append(p)

    print(f"Scanned {len(dirs)} probe dirs under {args.run_tag_base}/"
          f"{f' ({skipped} with no usable metrics)' if skipped else ''}\n")

    summary = {}
    for method in sorted(by_method):
        probes = sorted(by_method[method], key=lambda p: p["lr"])
        field = next((p["forget_field"] for p in probes if p["forget_field"]), "?")
        print("=" * 86)
        print(f"  {method}    forget signal: {field}")
        print("=" * 86)
        print(f"  {'lr':>10s} {'steps':>6s} {'forget first->last':>26s} "
              f"{'retain delta':>13s}  verdict")

        moving = []
        probe_rows = []
        for p in probes:
            v, why = verdict(p, args.move_eps, args.retain_max, args.forget_max)
            if v == "moving":
                moving.append(p["lr"])
            probe_rows.append({
                "lr": p["lr"], "verdict": v, "why": why,
                "steps": p["steps"], "forget_field": p["forget_field"],
                "forget_first": p["forget_first"], "forget_last": p["forget_last"],
                "forget_delta": p["forget_delta"], "retain_delta": p["retain_delta"],
            })
            fwd = (f"{fmt(p['forget_first'])} -> {fmt(p['forget_last'])} "
                   f"({p['forget_delta']:+.3f})") if p["forget_delta"] is not None else "-"
            rd = f"{p['retain_delta']:+.3f}" if p["retain_delta"] is not None else "-"
            tail = f"  {v}" + (f"  [{why}]" if why else "")
            print(f"  {p['lr']:>10.1e} {p['steps']:>6d} {fwd:>26s} {rd:>13s}{tail}")

        if moving:
            lo, hi = min(moving), max(moving)
            grid = log_grid(lo, hi, args.grid_points)
            print(f"\n  usable window: {lo:.1e} .. {hi:.1e}")
            print("  suggested grid: " + "  ".join(f"{g:.2e}" for g in grid))
            summary[method] = {"window": [lo, hi], "suggested_grid": grid,
                               "forget_field": field, "probes": probe_rows}
        else:
            hint = ("every probe was flat -- extend LRS upward"
                    if not any(verdict(p, args.move_eps, args.retain_max,
                                       args.forget_max)[0] == "diverged" for p in probes)
                    else "every probe was flat or diverged -- the window is between "
                         "two rungs; refine LRS around the transition")
            print(f"\n  no usable window found: {hint}")
            summary[method] = {"window": None, "suggested_grid": [],
                               "forget_field": field, "probes": probe_rows}
        print()

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.output_json}")
    print("\nThese windows are training-side only. They say where an LR does "
          "something without destabilising the run;\nthey do not say where it "
          "sits on the forgetting/utility trade-off -- that still needs the eval pass.")


if __name__ == "__main__":
    main()
