"""Read every cell's own config record and emit the run-level facts as a CSV.

WHY THIS EXISTS
---------------
`results_cells.csv` carries what was MEASURED. It does not carry what was
CONFIGURED -- retain weight, model, revision -- and those are exactly the
fields that decide whether two cells may be compared at all.

That gap has now caused the same error twice. A hyperparameter panel pooled
one retain run with three forget-only runs at neighbouring knob values, and the
retain term's ~4x effect read as a knob optimum. Both times the status was
recovered by an ad hoc `grep` over a hand-written list of paths, which is how
`1B-lr1e-5-base` -- a tag carrying DIFFERENT retain weights for different
methods -- went unnoticed.

Every driver already writes a `<method>_config.json` next to its checkpoints
with `retain_loss_weight`, `model` and `revision` in it. This walks all of
them, so the answer comes from the run itself rather than from a naming
convention. Tags are not evidence: `1B-p3-satimp-b10.0` has no `-fo` suffix and
is forget-only; `1B-p2-satimp-rt` and `1B-p2-satimp-rt-b10` differ only past
the point a glob usually stops.

USAGE
-----
    # on the cluster, where the output root lives
    python internal/uwiki/audit_configs.py --output-root $DATA/unlearning-pareto
    python internal/uwiki/audit_configs.py --output-root $DATA/decayed-root \
        --out exports-decayed/results_configs.csv

    # just look, do not write
    python internal/uwiki/audit_configs.py --output-root $DATA/unlearning-pareto \
        --method satimp --print

Writes `results_configs.csv` (default: alongside the output root's exports) with
one row per cell, and prints a summary that flags:

  * tags whose cells disagree on retain weight   (a tag is then NOT a config)
  * tags whose cells disagree on model/revision  (two models in one tag)
  * cells with no config file at all             (trained by an older driver,
                                                  or never actually started)

Join it to results_cells.csv on (run_tag, method, knob, knob_value).
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

# Fields worth carrying. Drivers differ, so every one is optional; a driver
# that does not record a field leaves it blank rather than failing the row.
FIELDS = [
    "run_tag", "method", "knob", "knob_value", "cell_dir", "config_file",
    "retain_loss_weight", "model", "revision", "learning_rate",
    "beta1", "beta2", "beta", "gamma", "steering_coefficient", "alpha",
    "max_steps", "epochs", "seed", "dtype", "lr_schedule",
]


def find_cells(root):
    """<root>/<run_tag>/<method>/<knob>-<value>/ -- the layout the launchers use."""
    for tag_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(tag_dir):
            continue
        tag = os.path.basename(tag_dir)
        if tag == "anchors":
            continue
        for meth_dir in sorted(glob.glob(os.path.join(tag_dir, "*"))):
            if not os.path.isdir(meth_dir):
                continue
            method = os.path.basename(meth_dir)
            for cell in sorted(glob.glob(os.path.join(meth_dir, "*"))):
                if not os.path.isdir(cell):
                    continue
                knob_value = os.path.basename(cell)
                knob, _, value = knob_value.partition("-")
                yield tag, method, knob, value, cell


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-root", required=True,
                    help="sweep root, e.g. $DATA/unlearning-pareto")
    ap.add_argument("--out", help="CSV to write (default: ./results_configs.csv)")
    ap.add_argument("--method", action="append", dest="methods",
                    help="restrict to a method; repeatable")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="print every row instead of only the summary")
    a = ap.parse_args()

    if not os.path.isdir(a.output_root):
        sys.exit(f"ERROR: no such output root: {a.output_root}")

    rows, missing = [], []
    for tag, method, knob, value, cell in find_cells(a.output_root):
        if a.methods and method not in a.methods:
            continue
        cfgs = sorted(glob.glob(os.path.join(cell, "*_config.json")))
        if not cfgs:
            missing.append((tag, method, knob, value, cell))
            continue
        # One cell should hold one config. If a cell was re-run under a
        # different driver both survive; take the newest and say so.
        cfg = max(cfgs, key=os.path.getmtime)
        try:
            with open(cfg, encoding="utf-8") as f:
                d = json.load(f) or {}
        except (OSError, json.JSONDecodeError) as e:
            print(f"  !! unreadable {cfg}: {e}", file=sys.stderr)
            continue
        row = {k: "" for k in FIELDS}
        row.update(run_tag=tag, method=method, knob=knob, knob_value=value,
                   cell_dir=cell, config_file=os.path.basename(cfg))
        for k in FIELDS:
            if k in d and d[k] is not None:
                row[k] = d[k]
        # grad-diff's lambda IS its retain weight, recorded under the generic
        # name; nothing to remap, but note it so the summary reads correctly.
        rows.append(row)
        if len(cfgs) > 1:
            print(f"  note: {len(cfgs)} config files in {cell}; used {os.path.basename(cfg)}",
                  file=sys.stderr)

    if not rows:
        sys.exit("ERROR: no cells with config files found under "
                 f"{a.output_root}. Check the path.")

    out = a.out or "results_configs.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out}  ({len(rows)} cells)")

    if a.do_print:
        print()
        print(f"  {'run_tag':<34}{'method':<17}{'knob':<8}{'value':<8}"
              f"{'retain':>8}  model")
        for r in sorted(rows, key=lambda r: (r["method"], r["run_tag"])):
            print(f"  {r['run_tag']:<34}{r['method']:<17}{r['knob']:<8}"
                  f"{str(r['knob_value']):<8}{str(r['retain_loss_weight']):>8}  "
                  f"{r['model']}")

    # ---- summary: the three things that invalidate a comparison -----------
    print("\n" + "=" * 92)
    print("RETAIN STATUS BY (run_tag, method) -- a tag is not a configuration")
    print("=" * 92)
    by = defaultdict(set)
    models = defaultdict(set)
    for r in rows:
        by[(r["run_tag"], r["method"])].add(str(r["retain_loss_weight"]))
        models[(r["run_tag"], r["method"])].add(f"{r['model']}@{r['revision'] or 'main'}")
    for k in sorted(by):
        vals = sorted(by[k])
        flag = "  <<< INCONSISTENT" if len(vals) > 1 else ""
        print(f"  {k[0]:<34}{k[1]:<17}retain={','.join(vals)}{flag}")

    incons = [k for k in by if len(by[k]) > 1]
    multi_model = [k for k in models if len(models[k]) > 1]
    print("\n" + "=" * 92)
    print("FLAGS")
    print("=" * 92)
    print(f"  cells audited                  : {len(rows)}")
    print(f"  tags mixing retain weights     : {len(incons)}")
    for k in incons:
        print(f"      {k[0]} / {k[1]}: {sorted(by[k])}")
    print(f"  tags mixing model/revision     : {len(multi_model)}")
    for k in multi_model:
        print(f"      {k[0]} / {k[1]}: {sorted(models[k])}")
    print(f"  cells with NO config file      : {len(missing)}")
    for t, m, k, v, c in missing[:20]:
        print(f"      {t} / {m} / {k}-{v}")
    if len(missing) > 20:
        print(f"      ... and {len(missing) - 20} more")

    print("\n  Join to results_cells.csv on (run_tag, method, knob, knob_value).")
    print("  Any tag flagged INCONSISTENT must be split before it is plotted or")
    print("  aggregated -- that is the defect this script exists to surface.")


if __name__ == "__main__":
    main()
