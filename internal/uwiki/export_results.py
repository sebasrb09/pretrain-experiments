"""Export every evaluated checkpoint to tidy CSVs for plotting.

WHY THIS EXISTS
---------------
`aggregate_pareto.py` produces a wide CSV keyed to one RUN_TAG at a time and a
terminal pivot for eyeballing. That is the wrong shape for paper figures, which
need every cell and anchor in one long table with the identifying columns
(method, variant, learning rate, step) split out so a plotting script can group
and filter without parsing directory names.

This walks the whole output root once and writes:

    results_cells.csv     one row per evaluated checkpoint
    results_anchors.csv   one row per evaluated anchor checkpoint
    results_meta.json     the normalisation endpoints, so the CSVs are
                          reproducible and the paper can cite the numbers

USAGE
-----
    python internal/uwiki/export_results.py                     # -> ./exports
    python internal/uwiki/export_results.py --out paper/data
    python internal/uwiki/export_results.py --tags '1B-p3-*'    # subset

NORMALISATION
-------------
Two derived columns per forget metric, both fractions of the distance from the
un-unlearned baseline to the deep-ignorance floor:

    fk_forgot = log10(fk_base / fk_cell) / log10(fk_base / fk_di)
    il_forgot = (il_cell - il_base) / (il_di - il_base)

fk_prob is normalised in LOG space because it spans four orders of magnitude; a
linear fraction would compress everything below 1e-3 into one point. Insertion
likelihood spans a factor of ~4.7 and is normalised linearly. The watermark uses
its Q4 partition only -- the first three quartiles carry no measurable signal --
normalised linearly between the same two anchors.

Values above 1.0 mean the checkpoint went past the deep-ignorance floor, which
in practice means a broken model rather than deeper forgetting. They are NOT
clipped here: clipping is a plotting decision and belongs downstream.

The endpoints are read from the anchors on disk, not hardcoded, so re-running
after adding anchor evaluations updates them. Anything missing falls back to the
values recorded in results_meta.json's `fallback` block, which are the ones used
in the analysis so far.
"""

import argparse
import csv
import glob
import json
import math
import os
import re

FALLBACK = {
    "fk_baseline": 3.493e-02,
    "fk_deep_ignorance": 2.872e-06,
    "il_baseline": 3.60,
    "il_deep_ignorance": 16.77,
    "wm_baseline": -1.169,
    "wm_deep_ignorance": 0.077,
    "c4_baseline": 18.77,
}

# The paper's measures added 2026-10-08 (see extra_columns): the backdoors WITH
# their trigger (pe_leak / dos_garbage are the untriggered control), iGSM
# accuracy per number of operations, and MUSE's news metrics on MUSE's own
# items. Per-condition values are in results_conditions.csv.
EXTRA_FIELDS = ["pe_leak_trig", "dos_garbage_trig", "dos_ppl_trig",
                "igsm_ops1", "igsm_ops3", "igsm_ops5",
                "news_verbmem", "news_knowmem_f", "news_knowmem_r", "news_privleak_auc"]

FIELDS = [
    "run_tag", "method", "variant", "lr", "knob", "knob_value", "step",
    "fk_prob", "fk_forgot",
    "il_ppl", "il_forgot",
    "c4_ppl", "c4_delta_pct",
    "wm_full", "wm_q4", "wm_removed",
    # the rest of the seven TOAA categories. These ran on only part of the
    # sweep, so a blank here means "not measured on that cell", never zero.
    "vm_memorized", "bm_acc", "pe_leak", "dos_garbage", "dos_ppl", "mia_auc",
    "mia_tpr1",
    *EXTRA_FIELDS,
    "eval_dir",
]


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}/"))
    else:
        out[prefix.rstrip("/")] = d
    return out


def read_yaml(path):
    # ImportError is NOT caught with OSError. A missing file means "this cell
    # was not evaluated" and None is right; a missing PyYAML means every
    # YAML-backed column silently exports blank, which reads downstream as a
    # flat zero bar rather than as a broken run. Fail once, loudly, instead.
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "PyYAML is not installed, so no results.yaml can be read and every "
            "metric except the watermark would export blank. pip install pyyaml")
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except OSError:
        return None


def scalar(eval_dir, evaluation, key):
    d = read_yaml(os.path.join(eval_dir, evaluation, "results.yaml"))
    return None if d is None else d.get(key)


def benchmark_acc(eval_dir):
    """Contamination accuracy.

    benchmark.py names the key after --norm: plain `acc`, or `acc_char` /
    `acc_mixed` when a normalisation was requested. Which one was used is not
    recorded anywhere else, so accept whichever is present.
    """
    # Split 0, the HELD-OUT questions (never inserted). The contamination effect
    # is accuracy on the inserted splits minus this; those are per split in
    # results_conditions.csv. Pre-2026-10-07 runs wrote split 0 without suffix.
    d = read_yaml(os.path.join(eval_dir, "benchmark_contamination_s0", "results.yaml"))
    if d is None:
        d = read_yaml(os.path.join(eval_dir, "benchmark_contamination", "results.yaml"))
    if d is None:
        return None
    for k in ("acc", "acc_char", "acc_mixed"):
        if k in d:
            return d[k]
    return None


def prompt_leakage(eval_dir, name="prompt_extraction"):
    """Fraction of prompts reproduced at RougeL recall > 0.9.

    The key is leakage_at_<k+1>, so it depends on --num-generations. Take the
    first-generation number when it is there, else the lowest k present.
    """
    d = read_yaml(os.path.join(eval_dir, name, "results.yaml"))
    if d is None:
        return None
    if "leakage_at_1" in d:
        return d["leakage_at_1"]
    ks = sorted(k for k in d if k.startswith("leakage_at_"))
    return d[ks[0]] if ks else None


def mia_auc(eval_dir):
    """Membership-inference AUC, calibrated against the reference model.

    newtoken_mia.py writes JSON (not YAML) named per model and condition, so
    glob rather than index. Prefer the calibrated number: the uncalibrated one
    moves with the target's overall fluency and would track perplexity.
    """
    # "mia*" not "mia": eval_cell_body.sh writes one directory per condition
    # (mia_rare_1tok_16x), while older runs wrote a single mia/. Both are read.
    hits = sorted(glob.glob(os.path.join(
        eval_dir, "mia*", "results_mia_samples_*.json")))
    if not hits:
        return None
    try:
        with open(hits[0], encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return None
    flat = flatten(blob)
    for want in ("calibrated_auc", "auc"):
        hit = [k for k in flat if k.split("/")[-1] == want
               and isinstance(flat[k], (int, float))]
        if hit:
            return flat[hit[0]]
    return None


def mia_tpr1(eval_dir):
    """TPR at 1% FPR of the loss-based membership-inference attack.

    The metric the paper reports for the canary dialogues. Read from the raw
    (uncalibrated) primary-region ROC that newtoken_mia.py keeps under the
    legacy `fpr` / `tpr` keys: the largest TPR whose FPR does not exceed 1%.
    A model that never saw the canaries scores about 0.01.
    """
    hits = sorted(glob.glob(os.path.join(eval_dir, "mia*", "results_mia_samples_*.json")))
    if not hits:
        return None
    try:
        with open(hits[0], encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return None
    entries = [blob] if "fpr" in blob else [v for v in blob.values() if isinstance(v, dict)]
    for e in entries:
        fpr, tpr = e.get("fpr"), e.get("tpr")
        if isinstance(fpr, list) and isinstance(tpr, list) and len(fpr) == len(tpr):
            ok = [t for f_, t in zip(fpr, tpr) if f_ <= 0.01]
            return max(ok) if ok else None
    return None


# ---------------------------------------------------------------- conditions
# results_conditions.csv: one row per (checkpoint or anchor) x task x condition
# x metric, with the condition's design spelled out in its own columns, so any
# difficulty level can be plotted on its own.
COND_FIELDS = ["kind", "run_tag", "method", "lr", "knob_value", "step", "point",
               "task", "condition", "copies", "group", "format", "canary_type", "canary_tokens",
               "quarter", "metric", "value", "eval_dir"]

# sbordt/toaa_benchmark_contamination splits (checked 2026-10-07): 0 is the
# held-out set; 1-4 inserted uniformly, 5-8 the second group ("forgetting-curves"),
# same copies. The copies are 4/12/32/144 (the insertion dataset's experiments
# and row counts: 2k questions x 32 = 62,272 rows), not the 36 the paper's text
# says. Splits 3 and 4 both hold 2k questions; 3 = 32x and 4 = 144x is assumed
# from the ordering, to be confirmed by the baseline's accuracy (144x > 32x).
BM_SPLIT_COPIES = {0: 0, 1: 4, 2: 12, 3: 32, 4: 144, 5: 4, 6: 12, 7: 32, 8: 144}
BM_SPLIT_GROUP = {0: "held-out", 1: "uniform", 2: "uniform", 3: "uniform", 4: "uniform",
                  5: "window", 6: "window", 7: "window", 8: "window"}


# verbatim_memorization.py --task news; see check_news_memorization there.
# Per insertion condition (whole articles), and for MUSE's own items ("muse").
NEWS_METRICS = ("nll", "min40", "privleak_auc_min40", "privleak_auc_ppl", "verbmem_rougeL",
                "verbmem_rougeL_recall", "n_scored", "n_verbmem")
MUSE_METRICS = ("verbmem_rougeL", "verbmem_rougeL_recall", "knowmem_f_rougeL", "knowmem_r_rougeL",
                "privleak_auc_min40", "privleak_auc_ppl", "n_verbmem", "n_knowmem_f", "n_knowmem_r",
                "n_privleak_forget", "n_privleak_holdout")


def extra_columns(eval_dir):
    """EXTRA_FIELDS for one eval directory."""
    news = (read_yaml(os.path.join(eval_dir, "news_memorization", "results.yaml")) or {}).get("muse") or {}
    return {
        "pe_leak_trig": prompt_leakage(eval_dir, "prompt_extraction_triggered"),
        "dos_garbage_trig": scalar(eval_dir, "denial_of_service_triggered", "is_garbage"),
        "dos_ppl_trig": scalar(eval_dir, "denial_of_service_triggered", "mean_ppl"),
        **{f"igsm_ops{k}": scalar(eval_dir, f"mathematical_reasoning_ops{k}", "acc") for k in (1, 3, 5)},
        "news_verbmem": news.get("verbmem_rougeL"),
        "news_knowmem_f": news.get("knowmem_f_rougeL"),
        "news_knowmem_r": news.get("knowmem_r_rougeL"),
        "news_privleak_auc": news.get("privleak_auc_min40"),
    }


def _tpr_at(fpr, tpr, at=0.01):
    ok = [t for f_, t in zip(fpr, tpr) if f_ <= at]
    return max(ok) if ok else None


def condition_rows(eval_dir):
    """Every per-condition measurement under one eval directory."""
    out = []
    # canary dialogues: one JSON per condition, labels from the result itself
    for path in sorted(glob.glob(os.path.join(eval_dir, "mia*", "results_mia_samples_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError):
            continue
        entries = [blob] if "fpr" in blob else [v for v in blob.values() if isinstance(v, dict)]
        for e in entries:
            lab = dict(task="mia", condition=e.get("condition"), copies=e.get("duplication"),
                       canary_type=e.get("suffix_type"), canary_tokens=e.get("n_suffix_tokens"))
            for metric, val in (("auc", e.get("auc")), ("calibrated_auc", e.get("calibrated_auc")),
                                ("tpr_at_1pct_fpr", _tpr_at(e.get("fpr") or [], e.get("tpr") or []))):
                if val is not None:
                    out.append(dict(lab, metric=metric, value=val))
    # contamination: one directory per split; the old single directory was split 0
    split_dirs = {int(m.group(1)): d for d in glob.glob(os.path.join(eval_dir, "benchmark_contamination_s*"))
                  for m in [re.search(r"_s(\d+)$", d)] if m}
    if 0 not in split_dirs and os.path.isdir(os.path.join(eval_dir, "benchmark_contamination")):
        split_dirs[0] = os.path.join(eval_dir, "benchmark_contamination")
    for k, d in sorted(split_dirs.items()):
        y = read_yaml(os.path.join(d, "results.yaml"))
        acc = None if y is None else next((y[a] for a in ("acc", "acc_char", "acc_mixed") if a in y), None)
        if acc is not None:
            out.append(dict(task="contamination", condition=f"split{k}", copies=BM_SPLIT_COPIES.get(k),
                            group=BM_SPLIT_GROUP.get(k), metric="acc", value=acc))
    # watermark by quarter of training (1 = earliest insertions, 4 = latest)
    paths = sorted(glob.glob(os.path.join(eval_dir, "gaussian_watermark", "gaussian_privacy_scores_in_*.pt")))
    if paths:
        try:
            import torch
            a = torch.cat([torch.load(p, map_location="cpu").float().flatten() for p in paths])
            n = len(a)
            for q in range(4):
                out.append(dict(task="watermark", condition=f"Q{q + 1}", quarter=q + 1,
                                metric="mean_score", value=a[q * n // 4:(q + 1) * n // 4].mean().item()))
        except Exception:
            pass
    # news articles: one row per insertion condition (copies x format) and for
    # the never-inserted holdout control (copies 0, group held-out)
    y = read_yaml(os.path.join(eval_dir, "news_memorization", "results.yaml"))
    for cond, d in sorted(((y or {}).get("conditions") or {}).items()):
        lab = dict(task="news", condition=cond, copies=d.get("copies"), format=d.get("format"),
                   group="held-out" if d.get("copies") == 0 else "inserted")
        for metric in NEWS_METRICS:
            if d.get(metric) is not None:
                out.append(dict(lab, metric=metric, value=d[metric]))
    for metric in MUSE_METRICS:
        v = ((y or {}).get("muse") or {}).get(metric)
        if v is not None:
            out.append(dict(task="news", condition="muse", group="muse-items", metric=metric, value=v))
    # backdoors: attack success with the trigger (the paper's measure) and on
    # the same prompts without it (the control)
    for name, cond in (("prompt_extraction_triggered", "triggered"), ("prompt_extraction", "untriggered")):
        v = prompt_leakage(eval_dir, name)
        if v is not None:
            out.append(dict(task="prompt_extraction", condition=cond, metric="leakage_at_1", value=v))
    for name, cond in (("denial_of_service_triggered", "triggered"), ("denial_of_service", "untriggered")):
        d = read_yaml(os.path.join(eval_dir, name, "results.yaml")) or {}
        for metric in ("is_garbage", "mean_ppl"):
            if d.get(metric) is not None:
                out.append(dict(task="denial_of_service", condition=cond, metric=metric, value=d[metric]))
    # iGSM: accuracy and answer NLL per number of operations
    for path in sorted(glob.glob(os.path.join(eval_dir, "mathematical_reasoning_ops*", "results.yaml"))):
        d = read_yaml(path) or {}
        cond = os.path.basename(os.path.dirname(path))[len("mathematical_reasoning_"):]
        for metric in ("acc", "mean_nll"):
            if d.get(metric) is not None:
                out.append(dict(task="igsm", condition=cond, metric=metric, value=d[metric]))
    # insertion likelihood per inserted experiment (only where all were run)
    y = read_yaml(os.path.join(eval_dir, "insertion_likelihood", "results.yaml"))
    if y:
        for k, v in flatten(y).items():
            parts = k.split("/")
            if len(parts) >= 3 and parts[-1] == "perplexity" and isinstance(v, (int, float)):
                out.append(dict(task="insertion", condition=parts[-2], metric="perplexity", value=v))
    return out


def insertion_likelihood(eval_dir, experiment):
    """Perplexity on one insertion experiment.

    The YAML nests as <eval>/<experiment>/<metric>, and which experiments are
    present depends on how the eval was invoked, so match rather than index.
    """
    d = read_yaml(os.path.join(eval_dir, "insertion_likelihood", "results.yaml"))
    if d is None:
        return None
    flat = flatten(d)
    keys = [k for k in flat if "perplexity" in k.lower()]
    hit = [k for k in keys if experiment in k] or keys
    return flat[hit[0]] if hit else None


_LR_CACHE = {}


def cell_lr(cell_dir):
    """learning_rate from the driver's own config dump, or None.

    Authoritative, unlike parsing it out of the RUN_TAG: every driver writes
    <driver>_config.json into the cell directory with the rate it actually
    used. The tag only carries it when the tag happens to end in lr<value>,
    which is false for 20 of the sweeps -- `1B-lr1e-5-base` names the rate in
    the middle, `1B-p2-satimp-rt` not at all -- and a cell launched without
    LR= silently takes the driver default, which no tag records.

    tokenizer_config.json and generation_config.json sit in the same tree and
    match the same glob, so they are excluded by name rather than by position.
    """
    if cell_dir in _LR_CACHE:
        return _LR_CACHE[cell_dir]
    lr = None
    for p in sorted(glob.glob(os.path.join(cell_dir, "*_config.json"))):
        if os.path.basename(p) in ("tokenizer_config.json", "generation_config.json"):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                v = json.load(f).get("learning_rate")
            if v is not None:
                lr = v
                break
        except Exception:
            continue
    _LR_CACHE[cell_dir] = lr
    return lr


def watermark(eval_dir):
    """(full-set mean, Q4 mean) from the saved per-sequence scores.

    Scores are concatenated in noise-file order, which is sorted by training
    step, so the last quarter is the most recently inserted watermarks -- the
    only partition that separates baseline from deep-ignorance.
    """
    paths = sorted(glob.glob(os.path.join(
        eval_dir, "gaussian_watermark", "gaussian_privacy_scores_in_*.pt")))
    if not paths:
        return None, None
    try:
        import torch
    except ImportError:
        return None, None
    vals = []
    for p in paths:
        try:
            vals.append(torch.load(p, map_location="cpu").float().flatten())
        except Exception:
            continue
    if not vals:
        return None, None
    a = torch.cat(vals)
    return a.mean().item(), a[3 * len(a) // 4:].mean().item()


def parse_tag(tag):
    """(variant, lr) from a RUN_TAG. Both may be None."""
    variant = ("forget-only" if "-fo-" in tag or tag.endswith("-fo")
               else "retain" if "-rt-" in tag or tag.endswith("-rt")
               else None)
    m = re.search(r"lr([0-9][0-9.eE+-]*)$", tag)
    return variant, (m.group(1) if m else None)


def collect_anchors(root):
    rows, ends = [], {}
    for results in glob.glob(os.path.join(root, "anchors", "**", "results.yaml"),
                             recursive=True):
        eval_dir = os.path.dirname(os.path.dirname(results))
        parts = eval_dir.split(os.sep)
        name = parts[-2] if parts[-1].startswith("step-") else parts[-1]
        step = parts[-1][len("step-"):] if parts[-1].startswith("step-") else ""
        key = (name, step)
        if key in ends:
            continue
        wm_full, wm_q4 = watermark(eval_dir)
        ends[key] = {
            "point": name, "step": step,
            "fk_prob": scalar(eval_dir, "fictional_knowledge", "probability"),
            "il_ppl": insertion_likelihood(eval_dir, "knowledge-acquisition"),
            "c4_ppl": scalar(eval_dir, "c4_perplexity", "perplexity"),
            "wm_full": wm_full, "wm_q4": wm_q4,
            "vm_memorized": scalar(eval_dir, "verbatim_memorization",
                                    "num_memorized_sequences"),
            "bm_acc": benchmark_acc(eval_dir),
            "pe_leak": prompt_leakage(eval_dir),
            "dos_garbage": scalar(eval_dir, "denial_of_service", "is_garbage"),
            "dos_ppl": scalar(eval_dir, "denial_of_service", "mean_ppl"),
            "mia_auc": mia_auc(eval_dir),
            "mia_tpr1": mia_tpr1(eval_dir),
            **extra_columns(eval_dir),
            "eval_dir": eval_dir,
        }
    return list(ends.values())


def endpoints(anchors):
    """Normalisation endpoints, measured from the anchors where available.

    deep-ignorance is taken as the MEDIAN across its own checkpoints rather
    than any single one: its scatter is the metric's noise floor, and one
    checkpoint would make the normalisation depend on which.
    """
    out = dict(FALLBACK)
    base = [a for a in anchors if a["point"] == "baseline"]
    di = [a for a in anchors if a["point"] == "deep-ignorance"]

    def med(rows, field):
        vals = sorted(r[field] for r in rows if r.get(field) is not None)
        return vals[len(vals) // 2] if vals else None

    for field, base_key, di_key in (
            ("fk_prob", "fk_baseline", "fk_deep_ignorance"),
            ("il_ppl", "il_baseline", "il_deep_ignorance"),
            ("wm_q4", "wm_baseline", "wm_deep_ignorance")):
        b, d = med(base, field), med(di, field)
        if b is not None:
            out[base_key] = b
        if d is not None:
            out[di_key] = d
    b = med(base, "c4_ppl")
    if b is not None:
        out["c4_baseline"] = b
    return out


def derive(row, e):
    fk, il, c4, wm = row.get("fk_prob"), row.get("il_ppl"), row.get("c4_ppl"), row.get("wm_q4")
    if fk and fk > 0:
        span = math.log10(e["fk_baseline"] / e["fk_deep_ignorance"])
        row["fk_forgot"] = round(math.log10(e["fk_baseline"] / fk) / span, 5)
    if il is not None:
        row["il_forgot"] = round((il - e["il_baseline"])
                                 / (e["il_deep_ignorance"] - e["il_baseline"]), 5)
    if c4 is not None:
        row["c4_delta_pct"] = round((c4 - e["c4_baseline"]) / e["c4_baseline"] * 100, 4)
    if wm is not None:
        row["wm_removed"] = round((wm - e["wm_baseline"])
                                  / (e["wm_deep_ignorance"] - e["wm_baseline"]), 5)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    data = os.environ.get("PE_DATA") or os.environ.get("DATA") or os.path.expanduser("~")
    ap.add_argument("--output-root", default=os.path.join(data, "unlearning-pareto"))
    ap.add_argument("--out", default="exports")
    ap.add_argument("--tags", default="*", help="glob over RUN_TAG directories")
    ap.add_argument("--il-experiment", default="knowledge-acquisition")
    args = ap.parse_args()

    root = args.output_root
    os.makedirs(args.out, exist_ok=True)

    anchors = collect_anchors(root)
    ends = endpoints(anchors)
    print(f"anchors: {len(anchors)} rows")
    for k, v in ends.items():
        print(f"  {k:22s} {v:.6g}" + ("" if k in FALLBACK and v != FALLBACK[k]
                                      else "   (fallback)" if v == FALLBACK.get(k) else ""))

    cells, skipped = [], 0
    pattern = os.path.join(root, args.tags, "*", "*", "step-*", "evals")
    for eval_dir in sorted(glob.glob(pattern)):
        parts = eval_dir.split(os.sep)
        tag, method, cell, step_dir = parts[-5], parts[-4], parts[-3], parts[-2]
        if tag == "anchors":
            continue
        variant, tag_lr = parse_tag(tag)
        knob, _, knob_value = cell.partition("-")
        # The driver's config dump is the only authoritative record -- see
        # cell_lr(). Fall back to the tag, then to the swept value for ce-u and
        # gradient-ascent, which take the learning rate AS their knob.
        cell_dir = os.path.dirname(os.path.dirname(eval_dir))
        lr = cell_lr(cell_dir) or tag_lr or (knob_value if knob == "lr" else None)
        wm_full, wm_q4 = watermark(eval_dir)
        row = {
            "run_tag": tag, "method": method, "variant": variant or "",
            "lr": lr or "", "knob": knob, "knob_value": knob_value,
            "step": int(step_dir[len("step-"):]),
            "fk_prob": scalar(eval_dir, "fictional_knowledge", "probability"),
            "il_ppl": insertion_likelihood(eval_dir, args.il_experiment),
            "c4_ppl": scalar(eval_dir, "c4_perplexity", "perplexity"),
            "wm_full": wm_full, "wm_q4": wm_q4,
            "vm_memorized": scalar(eval_dir, "verbatim_memorization",
                                    "num_memorized_sequences"),
            "bm_acc": benchmark_acc(eval_dir),
            "pe_leak": prompt_leakage(eval_dir),
            "dos_garbage": scalar(eval_dir, "denial_of_service", "is_garbage"),
            "dos_ppl": scalar(eval_dir, "denial_of_service", "mean_ppl"),
            "mia_auc": mia_auc(eval_dir),
            "mia_tpr1": mia_tpr1(eval_dir),
            **extra_columns(eval_dir),
            "eval_dir": eval_dir,
        }
        if row["c4_ppl"] is None and row["fk_prob"] is None:
            skipped += 1
            continue
        cells.append(derive(row, ends))

    def write(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {path}  ({len(rows)} rows)")

    cells.sort(key=lambda r: (r["method"], r["variant"], str(r["lr"]), r["step"]))
    write(os.path.join(args.out, "results_cells.csv"), cells, FIELDS)

    afields = ["point", "step", "fk_prob", "fk_forgot", "il_ppl", "il_forgot",
               "c4_ppl", "c4_delta_pct", "wm_full", "wm_q4", "wm_removed",
               "vm_memorized", "bm_acc", "pe_leak", "dos_garbage", "dos_ppl",
               "mia_auc", "mia_tpr1", *EXTRA_FIELDS, "eval_dir"]
    for a in anchors:
        derive(a, ends)
    anchors.sort(key=lambda r: (r["point"], int(r["step"] or -1)))
    write(os.path.join(args.out, "results_anchors.csv"), anchors, afields)

    conds = []
    for r in cells:
        for c in condition_rows(r["eval_dir"]):
            conds.append(dict(c, kind="cell", run_tag=r["run_tag"], method=r["method"], lr=r["lr"],
                              knob_value=r["knob_value"], step=r["step"], eval_dir=r["eval_dir"]))
    for a in anchors:
        for c in condition_rows(a["eval_dir"]):
            conds.append(dict(c, kind="anchor", point=a["point"], step=a["step"], eval_dir=a["eval_dir"]))
    write(os.path.join(args.out, "results_conditions.csv"), conds, COND_FIELDS)

    meta = {
        "output_root": root,
        "il_experiment": args.il_experiment,
        "endpoints": ends,
        "fallback": FALLBACK,
        "n_cells": len(cells),
        "n_anchors": len(anchors),
        "methods": sorted({r["method"] for r in cells}),
        "run_tags": sorted({r["run_tag"] for r in cells}),
        "notes": {
            "fk_forgot": "log-space fraction of baseline -> deep-ignorance",
            "il_forgot": "linear fraction of baseline -> deep-ignorance",
            "wm_removed": "linear fraction, Q4 partition only",
            "above_one": "values > 1 are past the deep-ignorance floor; not clipped",
        },
    }
    with open(os.path.join(args.out, "results_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # Say which suite tasks are thin. They were enabled partway through, so a
    # column can be 17% full; silence there would read as "all zero" in a plot.
    for col in ("vm_memorized", "bm_acc", "pe_leak", "dos_garbage", "mia_auc", "mia_tpr1", *EXTRA_FIELDS):
        n = sum(1 for r in cells if r.get(col) is not None)
        if n < len(cells):
            print(f"  {col}: {n}/{len(cells)} cells measured"
                  + ("  -- NOT RUN" if n == 0 else ""))
    print(f"wrote {os.path.join(args.out, 'results_meta.json')}")
    if skipped:
        print(f"{skipped} checkpoints had no readable results and were skipped")


if __name__ == "__main__":
    main()
