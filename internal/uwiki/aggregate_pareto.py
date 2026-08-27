"""
Collect the Pareto sweep into one tidy table: one row per metric per point.

Walks what internal/uwiki/eval_pareto_cell.sh leaves behind:

    <OUTPUT_ROOT>/<RUN_TAG>/<method>/<knob>-<value>/
        <method>_config.json          learning_rate, method, ...
        metrics.jsonl                 the training signal (ce_forget etc.)
        evals/<name>/results.yaml            any eval on the standard interface:
                                            c4_perplexity, fictional_knowledge,
                                            verbatim_memorization,
                                            insertion_likelihood,
                                            benchmark_contamination,
                                            prompt_extraction, denial_of_service
        evals/gaussian_watermark/gaussian_privacy_scores_{in,out}_*.pt
        evals/mia/*.json                     one JSON per MIA condition
    <OUTPUT_ROOT>/anchors/<name>/...  the same, for the reference points

NOTHING IS COLLAPSED. Every numeric key found in every eval becomes its own row,
so a figure can pick whichever unlearning axis it wants without a re-run. The
output is long format:

    point_type,point,method,knob,value,eval,metric,score

Pivot it however the figure needs, e.g. for the classic plot:
    x = eval=verbatim_memorization, metric=num_memorized_sequences
    y = eval=c4_perplexity,         metric=<whatever perplexity.py reports>

The Gaussian-watermark eval writes raw dot products rather than a scalar, so it
is reduced here the same way internal/uwiki/gw_summary_mid.py does it:

    signal = mean_in / sem_in

i.e. how many standard errors the in-distribution mean sits from zero. A model
that still carries the watermark scores |signal| >> 3; one that never saw it (or
has genuinely unlearned it) scores ~0. torch is imported lazily, so the rest of
the table still builds on a login node without it.

Usage:
    python internal/uwiki/aggregate_pareto.py
    python internal/uwiki/aggregate_pareto.py --output-root /data/.../unlearning-pareto
    python internal/uwiki/aggregate_pareto.py --csv pareto.csv --json pareto.json
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required (pip install pyyaml)", file=sys.stderr)
    raise


def as_float(x):
    """Return x as a float, or None if it is not a finite number."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(float(x)) else None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def flatten(obj, prefix=""):
    """Yield (dotted_key, float) for every numeric leaf in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        # Lists of numbers are summarised rather than exploded -- a 2500-element
        # per-sample list is detail, not a metric.
        vals = [as_float(v) for v in obj]
        vals = [v for v in vals if v is not None]
        if vals and len(vals) == len(obj):
            yield (f"{prefix}.mean" if prefix else "mean", sum(vals) / len(vals))
            yield (f"{prefix}.n" if prefix else "n", float(len(vals)))
    else:
        v = as_float(obj)
        if v is not None:
            yield (prefix or "value", v)


def read_yaml_metrics(path):
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"
    if data is None:
        return {}, "empty file"
    return dict(flatten(data)), None


def read_gw_signal(gw_dir):
    """Reduce the watermark dot-product tensors to mean/sem/signal."""
    hits_in = sorted(glob.glob(os.path.join(gw_dir, "gaussian_privacy_scores_in_*.pt")))
    hits_out = sorted(glob.glob(os.path.join(gw_dir, "gaussian_privacy_scores_out_*.pt")))
    if not hits_in:
        return {}, "no gaussian_privacy_scores_in_*.pt"
    try:
        import torch
    except ImportError:
        return {}, "torch not importable (skipping GW reduction)"

    def stats(path, tag):
        t = torch.load(path, map_location="cpu").float().flatten()
        n = t.numel()
        if n == 0:
            return {}
        m = t.mean().item()
        s = t.std().item()
        sem = s / math.sqrt(n) if n > 1 else float("nan")
        out = {f"mean_{tag}": m, f"sem_{tag}": sem, f"n_{tag}": float(n)}
        return {k: v for k, v in out.items() if math.isfinite(v)}

    metrics = {}
    try:
        metrics.update(stats(hits_in[0], "in"))
        if hits_out:
            metrics.update(stats(hits_out[0], "out"))
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"

    if "mean_in" in metrics and metrics.get("sem_in", 0):
        metrics["signal"] = metrics["mean_in"] / metrics["sem_in"]
    return metrics, None


def read_training_signal(cell_dir):
    """First/last ce_forget from metrics.jsonl, AVERAGED over an accumulation cycle.

    One row of metrics.jsonl is a single micro-batch -- 4 sequences. At batch 512
    that is 1/128th of an optimizer step, and its CE varies by more than a nat
    between rows. Taking vals[0] and vals[-1] therefore measured noise, not
    learning: it reported ce_first=2.723 on a forget set whose true mean is 1.793,
    and turned gradient ascent's real (positive) forgetting into a negative delta.

    Average over `gradient_accumulation_steps` rows at each end instead, matching
    what analyze_lr_range_test.py does.
    """
    path = os.path.join(cell_dir, "metrics.jsonl")
    if not os.path.isfile(path):
        return {}
    fields = ("ce_forget", "nll_avg_mean", "nll_theta_mean", "loss_forget")

    accum = 1
    for cfg_path in glob.glob(os.path.join(cell_dir, "*_config.json")):
        try:
            with open(cfg_path) as f:
                accum = int(json.load(f).get("gradient_accumulation_steps") or 1)
        except Exception:
            pass
        break

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return {}
    field = next((x for x in fields if any(x in r for r in rows)), None)
    if field is None:
        return {}
    vals = [as_float(r[field]) for r in rows if field in r]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}

    # Never let the two windows overlap, or first and last become the same mean.
    win = max(1, min(accum, len(vals) // 4 or 1))
    first = sum(vals[:win]) / win
    last = sum(vals[-win:]) / win
    out = {
        f"{field}_first": first,
        f"{field}_last": last,
        f"{field}_delta": last - first,
        f"{field}_window": float(win),
        "optimizer_steps": float(max((r.get("optimizer_step", 0) for r in rows), default=0)),
        "n_micro_batches": float(len(vals)),
    }
    return out


def collect_point(rows, point_type, point, method, knob, value, base_dir, eval_dir):
    """Append every metric found for one point of the plot."""
    def add(eval_name, metrics, note=None):
        for metric, score in sorted(metrics.items()):
            rows.append({
                "point_type": point_type, "point": point, "method": method,
                "knob": knob, "value": value, "eval": eval_name,
                "metric": metric, "score": score,
            })
        if note:
            print(f"    [{eval_name}] {note}")

    if base_dir:
        train = read_training_signal(base_dir)
        if train:
            add("training", train)

    if not os.path.isdir(eval_dir):
        print(f"    no evals/ yet")
        return

    for name in sorted(os.listdir(eval_dir)):
        sub = os.path.join(eval_dir, name)
        if not os.path.isdir(sub):
            continue
        if name == "gaussian_watermark":
            metrics, err = read_gw_signal(sub)
            add(name, metrics, err)
            continue
        # MIA writes one JSON per condition, not a results.yaml. "mia" is the
        # current dir name; the older one is accepted so existing trees still read.
        if name in ("mia", "memorization_patterns_mia"):
            for jf in sorted(glob.glob(os.path.join(sub, "*.json"))):
                try:
                    with open(jf) as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"    [mia] {os.path.basename(jf)}: {e}")
                    continue
                # The filename embeds model and step, so it is unusable as a key.
                # The JSON is {condition: {...}}, and the condition is the same
                # across every point -- that is what makes rows comparable.
                for cond, entry in data.items():
                    if isinstance(entry, dict):
                        add(f"mia/{cond}", dict(flatten(entry)))
                    else:
                        add(f"mia/{os.path.splitext(os.path.basename(jf))[0]}",
                            dict(flatten(data)))
                        break
            continue
        y = os.path.join(sub, "results.yaml")
        if os.path.isfile(y):
            metrics, err = read_yaml_metrics(y)
            add(name, metrics, err)



# The headline view: the plot as a table. Every metric stays in the CSV; this
# just picks the handful you actually stare at, anchors first so the cells have
# something to be read against.
PIVOT_AXES = [
    ("fictional_knowledge", "probability",     "fk_prob",    11, "e", 2),
    ("c4_perplexity",       "perplexity",      "c4_ppl",      9, "f", 2),
    ("gaussian_watermark",  "mean_in",         "gw_mean_in", 12, "f", 2),
    ("gaussian_watermark",  "sem_in",          "gw_sem",      8, "f", 2),
    ("prompt_extraction",   "leakage_at_1",    "pe_leak",     9, "f", 4),
    ("mia/" + os.environ.get("MIA_COND", "rare_1tok_16x"),
                            "calibrated_auc",  "mia_auc",    10, "f", 4),
    ("training",            "ce_forget_delta", "ce_delta",   10, "f", 3),
]


def print_pivot(rows):
    from collections import defaultdict
    d = defaultdict(dict)
    for r in rows:
        for ev, me, tag, _w, _p, _n in PIVOT_AXES:
            if r["eval"] == ev and r["metric"] == me:
                d[(r["point_type"], r["method"], r["knob"], r["value"])][tag] = r["score"]
    if not d:
        return

    def cell(v, w, p, n):
        return " " * (w - 1) + "-" if v is None else f"{v:>{w}.{n}{p}}"

    hdr = f"  {'method':<17}{'knob':<7}{'value':<9}"
    for _e, _m, tag, w, _p, _n in PIVOT_AXES:
        hdr += f"{tag:>{w}}"
    print("")
    print("=" * len(hdr))
    print("  THE PLOT  (x = fk_prob, lower = more forgotten; y = c4_ppl)")
    print("=" * len(hdr))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for kind in ("anchor", "cell"):
        if kind == "cell":
            print("")
        for (t, m, k, v), x in sorted(d.items(), key=lambda kv: (kv[0][1], kv[0][3])):
            if t != kind:
                continue
            name = m if t == "cell" else f"[{m}]"
            line = f"  {name:<17}{k:<7}{v:<9}"
            for _e, _m2, tag, w, pp, n in PIVOT_AXES:
                line += cell(x.get(tag), w, pp, n)
            print(line)


def write_wide_csv(rows, path):
    """One row per point, one column per eval/metric -- ready for pandas.

    The long CSV is the source of truth; this is the same data pivoted so a
    notebook can do df.plot(x=..., y=...) without reshaping. Column names are
    "<eval>.<metric>", e.g. "fictional_knowledge.probability".
    """
    from collections import defaultdict
    d = defaultdict(dict)
    cols = set()
    for r in rows:
        key = (r["point_type"], r["point"], r["method"], r["knob"], r["value"])
        col = f'{r["eval"]}.{r["metric"]}'
        d[key][col] = r["score"]
        cols.add(col)

    head = ["point_type", "point", "method", "knob", "value"] + sorted(cols)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for key in sorted(d, key=lambda k: (k[0] != "anchor", k[2], k[4])):
            rec = d[key]
            w.writerow(list(key) + [rec.get(c, "") for c in sorted(cols)])
    return len(d), len(head)

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=str,
                        default=os.path.expanduser("~/pretrain-experiments/unlearning-pareto"))
    parser.add_argument("--run-tag", type=str, default="1B-pareto")
    parser.add_argument("--anchor-dir", type=str, default=None,
                        help="default: <output-root>/anchors")
    parser.add_argument("--csv", type=str, default="pareto_results.csv")
    parser.add_argument("--json", type=str, default="pareto_results.json")
    parser.add_argument("--wide-csv", type=str, default="pareto_wide.csv",
                        help="Pivoted one-row-per-point CSV for plotting.")
    args = parser.parse_args()

    sweep_dir = os.path.join(args.output_root, args.run_tag)
    anchor_dir = args.anchor_dir or os.path.join(args.output_root, "anchors")
    rows = []

    if os.path.isdir(sweep_dir):
        for method in sorted(os.listdir(sweep_dir)):
            mdir = os.path.join(sweep_dir, method)
            if not os.path.isdir(mdir):
                continue
            print(f"\n--- {method} ---")
            for cell in sorted(os.listdir(mdir)):
                cdir = os.path.join(mdir, cell)
                if not os.path.isdir(cdir):
                    continue
                # Split on the FIRST hyphen: values like "1e-5" contain one,
                # so rpartition would yield knob="lr-1e", value="5".
                knob, _, value = cell.partition("-")
                print(f"  {cell}")
                collect_point(rows, "cell", cell, method, knob, value,
                              cdir, os.path.join(cdir, "evals"))
    else:
        print(f"(no sweep at {sweep_dir})")

    if os.path.isdir(anchor_dir):
        print(f"\n--- anchors ---")
        for name in sorted(os.listdir(anchor_dir)):
            adir = os.path.join(anchor_dir, name)
            if not os.path.isdir(adir):
                continue
            print(f"  {name}")
            collect_point(rows, "anchor", name, name, "", "", None, adir)
    else:
        print(f"\n(no anchors at {anchor_dir} -- the curves will have no reference points)")

    if not rows:
        raise SystemExit(
            "\nNothing collected. Either the eval jobs have not run yet, or "
            "--output-root/--run-tag point somewhere else."
        )

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["point_type", "point", "method", "knob",
                                          "value", "eval", "metric", "score"])
        w.writeheader()
        w.writerows(rows)
    with open(args.json, "w") as f:
        json.dump(rows, f, indent=2)
    n_pts, n_cols = write_wide_csv(rows, args.wide_csv)

    evals = sorted({r["eval"] for r in rows})
    points = sorted({(r["point_type"], r["point"]) for r in rows})
    print(f"\n{len(rows)} rows, {len(points)} points, {len(evals)} evals")
    print(f"  evals: {', '.join(evals)}")
    print(f"wrote {args.csv}, {args.json}, and {args.wide_csv} "
          f"({n_pts} points x {n_cols} columns)")

    print_pivot(rows)

    print("\nAvailable axes (eval / metric):")
    seen = sorted({(r["eval"], r["metric"]) for r in rows})
    for e, m in seen:
        print(f"    {e:<34} {m}")


if __name__ == "__main__":
    main()
