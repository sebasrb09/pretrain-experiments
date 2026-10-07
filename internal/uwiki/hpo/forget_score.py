"""The search objective: forgetting on three tasks, each measured against its own anchors.

For each task t, with measurement x_t at a checkpoint:

    p_t = clip( (s_t(x_t) - s_t(B_t)) / (s_t(C_t) - s_t(B_t)), 0, 1 )
    F   = (p_fk + p_il + p_wm) / 3

    B_t   the BASELINE anchor: the model every method starts from
    C_t   the COUNTERFACTUAL anchor (deep-ignorance): same corpus, targets removed
    s_t   the scale the task is compared on
            fk  log10 of the fictional-knowledge probability  (spans ~5 decades)
            il  insertion-likelihood perplexity, linear       (spans ~3x)
            wm  signed Gaussian-watermark Q4 score, linear    (a test statistic)

p_t = 0 means nothing forgotten and 1 means the counterfactual was reached, so
every task is in the same unit: the fraction of the way a perfect unlearner
travels. These are export_results.py's fk_forgot, il_forgot and wm_removed,
clipped to [0, 1].

Why each choice, because each one decides how much a task counts:

  * Anchor-gap scaling, not z-scores across trials. Standardising by the spread
    of the trials would hand the watermark, which barely moves, the same weight
    as the tasks that do, and would amplify its noise. The anchor gap is fixed
    before the search starts, so a task's weight cannot depend on the trials.
  * log10 for fk. Linear would score a 10x drop in probability as 90% forgotten
    while the model still sits 10,000x above the counterfactual, and that one
    task would carry ~4x the weight of insertion. In log space the two carry
    comparable weight (sd of their terms across the 1B sweep: 0.016 vs 0.021).
  * Clipping. Past the counterfactual (a broken model drives fk to ~1e-78) earns
    nothing extra, so no task can be gamed by destroying the model, and moving
    away from the counterfactual cannot cancel progress on another task.
  * Equal weights. Each task counts once, as in the paper's seven-task figure.

The other tasks are held out and never seen by the search: verbatim,
contamination, prompt extraction and DoS have anchor gaps too small (or of the
wrong sign) to normalise against at 1B, and nothing moves MIA within budget.

ANCHORS ARE MEASURED WITH THE TRIALS' EXACT EVALUATION SETTINGS. An anchor from
another site or another batch size is a different measurement: the 1B LUMI
export carried an ASC baseline for insertion (3.603) while every untouched LUMI
checkpoint reads ~12.9, which would credit every trial with 29% insertion
forgetting before a single step. settings.json pins what the anchors were
measured with, and check_settings() refuses a trial that differs.

    python forget_score.py show  --anchor-root DIR   print the anchors and gaps
    python forget_score.py check --anchor-root DIR   exit 1 unless the current
                                                     environment matches them
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORTER = os.path.join(HERE, os.pardir, "export_results.py")

TASKS = ("fk", "il", "wm")
POINTS = ("baseline", "deep-ignorance")

# The variables that decide what an evaluation measures. The anchors and every
# trial must agree on all of them, plus the content of the C4 file.
EVAL_KEYS = ("NOISE_DIR", "NOISE_STD", "EVAL_MAX_NUM_SEQS", "INFERENCE_MAX_NUM_SEQS",
             "C4_TASK_FILE", "IL_EXPERIMENT", "IL_MAX_TOKENS")

# Minimum anchor separation per task, in the task's own scale. Below these the
# anchors do not separate and p_t would be noise divided by noise.
MIN_GAP = {"fk": 1.0, "il": 1.0, "wm": 0.5}


class AnchorError(RuntimeError):
    pass


def load_exporter():
    spec = importlib.util.spec_from_file_location("_export_results", EXPORTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _num(v):
    """A float, or None for anything missing or NaN. +inf is kept: it is how a
    destroyed model's perplexity overflows, and it is still a measurement."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def measure(eval_dir, il_experiment, ex=None):
    """The four raw numbers a rung contributes, read with the exporter's readers."""
    ex = ex or load_exporter()
    _, wm_q4 = ex.watermark(eval_dir)
    return {
        "c4": _num(ex.scalar(eval_dir, "c4_perplexity", "perplexity")),
        "fk": _num(ex.scalar(eval_dir, "fictional_knowledge", "probability")),
        "il": _num(ex.insertion_likelihood(eval_dir, il_experiment)),
        "wm": _num(wm_q4),
    }


def _scale(task, x):
    if task == "fk":
        return math.log10(x) if x > 0 else -math.inf
    return x


def progress(task, x, anchors):
    """p_t for one measurement, clipped to [0, 1]."""
    b = _scale(task, anchors["baseline"][task])
    c = _scale(task, anchors["deep-ignorance"][task])
    s = _scale(task, x)
    if math.isinf(s):
        # fk probability of exactly 0, or an overflowed perplexity: further than
        # the counterfactual in the direction of forgetting.
        v = 1.0 if (s - b) * (c - b) > 0 else 0.0
    else:
        v = (s - b) / (c - b)
    return min(1.0, max(0.0, v))


def score(m, anchors):
    """{'p_fk', 'p_il', 'p_wm', 'F'}, or None if any task is missing."""
    if any(m.get(t) is None for t in TASKS):
        return None
    out = {f"p_{t}": progress(t, m[t], anchors) for t in TASKS}
    out["F"] = sum(out[f"p_{t}"] for t in TASKS) / len(TASKS)
    return out


def fingerprint(env):
    """What the anchors were, or a trial is, measured with."""
    missing = [k for k in EVAL_KEYS if not env.get(k)]
    if missing:
        raise AnchorError("evaluation settings not set: " + " ".join(missing))
    fp = {k: env[k] for k in EVAL_KEYS}
    with open(env["C4_TASK_FILE"], "rb") as fh:
        fp["c4_task_file_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    return fp


def write_settings(root, env, models):
    os.makedirs(root, exist_ok=True)
    blob = {"eval": fingerprint(env), "models": models}
    path = os.path.join(root, "settings.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2, sort_keys=True)
    return path


def read_settings(root):
    path = os.path.join(root, "settings.json")
    if not os.path.exists(path):
        raise AnchorError(f"no {path}: the anchors were never submitted (optuna_hpo.py --anchors)")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_settings(root, env):
    """Raise unless env evaluates exactly the way the anchors were evaluated."""
    want = read_settings(root)["eval"]
    have = fingerprint(env)
    diff = {k: (want.get(k), have.get(k)) for k in sorted(set(want) | set(have))
            if want.get(k) != have.get(k)}
    if diff:
        lines = [f"  {k}: anchors {a!r}, this run {b!r}" for k, (a, b) in diff.items()]
        raise AnchorError("evaluation settings differ from the anchors':\n" + "\n".join(lines))


def load_anchors(root, il_experiment, ex=None):
    """Both anchors' measurements, validated. Raises AnchorError with the reason."""
    try:
        import torch   # noqa: F401  the exporter reads the watermark .pt files with it
    except ImportError:
        # export_results.watermark() returns None without torch, which would
        # surface as "no finite wm" and hide the real cause.
        raise AnchorError("torch is not importable in this Python, so the watermark scores "
                          "cannot be read. Run inside the container: source internal/lumi/env.sh")
    ex = ex or load_exporter()
    out = {}
    for pt in POINTS:
        d = os.path.join(root, pt, "step-0")
        m = measure(d, il_experiment, ex)
        bad = [k for k, v in m.items() if v is None or math.isinf(v)]
        if bad:
            raise AnchorError(f"anchor {pt} has no finite {', '.join(bad)} under {d}")
        out[pt] = m
    b, c = out["baseline"], out["deep-ignorance"]
    if b["fk"] <= 0 or c["fk"] <= 0:
        raise AnchorError(f"fk anchors must be positive (baseline {b['fk']}, counterfactual {c['fk']})")
    gaps = {
        "fk": math.log10(b["fk"]) - math.log10(c["fk"]),   # baseline knows more
        "il": c["il"] - b["il"],                           # counterfactual is more surprised
        "wm": c["wm"] - b["wm"],                           # baseline is more negative
    }
    for t, g in gaps.items():
        if g < MIN_GAP[t]:
            raise AnchorError(f"anchors do not separate on {t}: gap {g:.4g} in its own scale, "
                              f"need at least {MIN_GAP[t]}")
    out["gaps"] = gaps
    return out


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["show", "check"])
    ap.add_argument("--anchor-root", required=True)
    ap.add_argument("--il-experiment", default=None,
                    help="default: IL_EXPERIMENT from settings.json")
    args = ap.parse_args()
    try:
        st = read_settings(args.anchor_root)
        il_exp = args.il_experiment or st["eval"]["IL_EXPERIMENT"]
        if args.cmd == "check":
            check_settings(args.anchor_root, os.environ)
        a = load_anchors(args.anchor_root, il_exp)
    except (AnchorError, OSError, KeyError) as e:
        print(f"ANCHORS NOT USABLE: {e}", file=sys.stderr)
        return 1
    print(f"anchors under {args.anchor_root} (il experiment {il_exp})")
    for k, v in sorted(st["eval"].items()):
        print(f"  {k} = {v}")
    print(f"  {'':16s} {'c4':>10s} {'fk':>12s} {'il':>10s} {'wm_q4':>10s}")
    for pt in POINTS:
        m = a[pt]
        print(f"  {pt:16s} {m['c4']:10.4f} {m['fk']:12.4e} {m['il']:10.4f} {m['wm']:+10.4f}")
    g = a["gaps"]
    print(f"  gaps: fk {g['fk']:.3f} decades, il {g['il']:.3f}, wm {g['wm']:.4f}")
    # The batch-invariance diagnostic that optuna_hpo.py --anchors also submits.
    bc = os.path.join(args.anchor_root.rstrip("/") + "-batchcheck", "baseline-mns8", "step-0")
    if os.path.isdir(bc):
        ex = load_exporter()
        fk8 = _num(ex.scalar(bc, "fictional_knowledge", "probability"))
        il8 = _num(ex.insertion_likelihood(bc, il_exp))
        b = a["baseline"]
        rel = lambda x, y: "-" if x is None else f"{100.0 * (x - y) / y:+.3f}%"
        print(f"  batch check, baseline at INFERENCE_MAX_NUM_SEQS=8 vs the anchor's setting:")
        print(f"    fk {fk8 if fk8 is not None else 'missing'} vs {b['fk']:.6g}  ({rel(fk8, b['fk'])})")
        print(f"    il {il8 if il8 is not None else 'missing'} vs {b['il']:.6g}  ({rel(il8, b['il'])})")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
