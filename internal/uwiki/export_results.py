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

FIELDS = [
    "run_tag", "method", "variant", "lr", "knob", "knob_value", "step",
    "fk_prob", "fk_forgot",
    "il_ppl", "il_forgot",
    "c4_ppl", "c4_delta_pct",
    "wm_full", "wm_q4", "wm_removed",
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
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, ImportError):
        return None


def scalar(eval_dir, evaluation, key):
    d = read_yaml(os.path.join(eval_dir, evaluation, "results.yaml"))
    return None if d is None else d.get(key)


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
            "wm_full": wm_full, "wm_q4": wm_q4, "eval_dir": eval_dir,
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
        # ce-u and gradient-ascent take the learning rate AS their swept value,
        # so the cell name carries it and the tag may not.
        lr = tag_lr or (knob_value if knob == "lr" else None)
        wm_full, wm_q4 = watermark(eval_dir)
        row = {
            "run_tag": tag, "method": method, "variant": variant or "",
            "lr": lr or "", "knob": knob, "knob_value": knob_value,
            "step": int(step_dir[len("step-"):]),
            "fk_prob": scalar(eval_dir, "fictional_knowledge", "probability"),
            "il_ppl": insertion_likelihood(eval_dir, args.il_experiment),
            "c4_ppl": scalar(eval_dir, "c4_perplexity", "perplexity"),
            "wm_full": wm_full, "wm_q4": wm_q4,
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
               "c4_ppl", "c4_delta_pct", "wm_full", "wm_q4", "wm_removed", "eval_dir"]
    for a in anchors:
        derive(a, ends)
    anchors.sort(key=lambda r: (r["point"], int(r["step"] or -1)))
    write(os.path.join(args.out, "results_anchors.csv"), anchors, afields)

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
    print(f"wrote {os.path.join(args.out, 'results_meta.json')}")
    if skipped:
        print(f"{skipped} checkpoints had no readable results and were skipped")


if __name__ == "__main__":
    main()
