"""Summarize an Optuna HPO study: every trial, its outcome, and the best one.

    python internal/uwiki/hpo/report.py --method ce-u --out "$PE_WORK/exports-hpo-ceu"

Reads the study from the journal the driver wrote, and each trial's
hpo_result.json for the per-rung measurements behind its objective. Writes
trials.csv (one row per trial) and rungs.csv (one row per evaluated rung) so the
search can be inspected and plotted off the cluster. With --sacct (a dump of the
user's jobs taken on the host) also compute.csv, the cost of every submitted job;
internal/lumi/pe_tools.sh pe_export runs it that way for all the searches.

The objective is F, the three-task forgetting score in [0, 1] (see
forget_score.py): 0 is the baseline, 1 is the counterfactual on all three tasks.

Things it flags, because each has bitten this project before:
  - NaN or missing measurements on a rung
  - in-budget objectives within the scatter of untouched checkpoints, which
    would mean the eval measured a model that did not move
  - trials that never finished (still RUNNING in the study)
  - trials that failed after their job was submitted
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PE = os.environ.get("PE_WORK", "/scratch/project_465003383/unlearning_baselines")

# F on checkpoints that barely moved (C4 within 0.5% of the baseline) in the 1B
# sweep, scored against LUMI-consistent anchors: median 0.001, sd 0.009, 95th
# percentile 0.018. An F below this cannot be told apart from an untouched model.
NOISE = 0.02


def natural(trial):
    if "params" in trial.user_attrs:
        return dict(trial.user_attrs["params"])
    p = {}
    for k, v in trial.params.items():
        if k.endswith("_logit"):
            p[k[:-len("_logit")]] = 1.0 / (1.0 + math.exp(-v))
        else:
            p[k] = v
    return p


def result_json(out_root, tag, method):
    hits = glob.glob(os.path.join(out_root, tag, method, "*", "hpo_result.json"))
    if not hits:
        return None
    with open(hits[0], encoding="utf-8") as fh:
        return json.load(fh)


def fmt(v, spec=".4g"):
    if v is None:
        return "-"
    if isinstance(v, float) and not math.isfinite(v):
        return "NaN" if math.isnan(v) else "inf"
    return format(v, spec) if isinstance(v, (int, float)) else str(v)


RUNG_FIELDS = ["trial", "step", "c4_ppl", "c4_delta_pct", "fk_prob", "il_ppl", "wm_q4",
               "p_fk", "p_il", "p_wm", "F"]
COMPUTE_FIELDS = ["category", "trial", "job", "name", "slurm_state", "hours", "gpus", "gpu_hours", "billing",
                  "billing_hours"]


SACCT_COLUMNS = "JobID,JobName,State,ElapsedRaw,AllocTRES"


def sacct(args_list):
    """Rows of `sacct -X -n -P -o JobID,JobName,State,ElapsedRaw,AllocTRES <args>`, or None
    without SLURM accounting (off the cluster, and inside the container)."""
    import subprocess
    try:
        out = subprocess.run(["sacct", "-X", "-n", "-P", "-o", SACCT_COLUMNS] + args_list,
                             capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return parse_sacct(out.stdout)


def parse_sacct(text):
    rows = []
    for line in text.splitlines():
        f = line.split("|")
        if len(f) < 5 or not f[0]:
            continue
        tres = dict(kv.split("=", 1) for kv in f[4].split(",") if "=" in kv)
        gpus = next((float(v) for k, v in tres.items() if k.startswith("gres/gpu")), 0.0)
        hours = (int(f[3]) if f[3].isdigit() else 0) / 3600.0
        billing = float(tres["billing"]) if tres.get("billing", "").replace(".", "", 1).isdigit() else None
        rows.append({"job": f[0], "name": f[1], "slurm_state": f[2].split()[0], "hours": round(hours, 4),
                     "gpus": gpus, "gpu_hours": round(hours * gpus, 4), "billing": billing,
                     "billing_hours": None if billing is None else round(hours * billing, 4)})
    return rows


def final_row(study, r):
    """A finalize job of this study (its retraining or one of its evaluations), or None."""
    if study not in r["name"] or "-final-" not in r["name"]:
        return None
    m = re.search(r"-final-t(\d+)", r["name"])
    return {"category": "finalize evaluation" if r["name"].startswith("pe-") else "finalize training",
            "trial": int(m.group(1)) if m else None, **r}


def compute_ledger(study, trials, dump=None):
    """Every SLURM job the study cost: one per trial that was submitted (finished,
    failed, pruned or still running alike), and the finalize retraining and its
    evaluations (names containing <study>...-final-). From `dump` (parse_sacct rows
    of the user's jobs) when given, else from sacct itself; None without either."""
    jobs = {str(t.user_attrs["job"]): t.number for t in trials if t.user_attrs.get("job")}
    if dump is not None:
        rows = []
        for r in dump:
            if r["job"] in jobs:
                rows.append({"category": "trial", "trial": jobs[r["job"]], **r})
            elif final_row(study, r):
                rows.append(final_row(study, r))
        return rows
    rows = []
    ids = sorted(jobs)
    for i in range(0, len(ids), 200):
        got = sacct(["-j", ",".join(ids[i:i + 200])])
        if got is None:
            return None
        for r in got:
            rows.append({"category": "trial", "trial": jobs.get(r["job"]), **r})
    starts = [t.datetime_start for t in trials if t.datetime_start]
    if starts:
        got = sacct(["-u", os.environ.get("USER", ""), "-S", min(starts).strftime("%Y-%m-%dT%H:%M:%S")])
        for r in got or []:
            if r["job"] not in jobs and final_row(study, r):
                rows.append(final_row(study, r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--storage", default=os.path.join(PE, "hpo", "optuna_journal.log"))
    ap.add_argument("--study", default=None, help="default hpo-<method>-wm")
    ap.add_argument("--output-root", default=os.path.join(PE, "hpo"),
                    help="where the trials' cell directories live")
    ap.add_argument("--out", required=True, help="directory for trials.csv and rungs.csv")
    ap.add_argument("--sacct", default=None,
                    help=f"the output of `sacct -u $USER -X -n -P -o {SACCT_COLUMNS} -S <before the search>` "
                         "taken on the host: this script runs in the container, which has no sacct "
                         "(internal/lumi/pe_tools.sh pe_export writes it)")
    args = ap.parse_args()

    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if not os.path.exists(args.storage):
        sys.exit(f"no study journal at {args.storage}")
    study = optuna.load_study(study_name=args.study or f"hpo-{args.method}-wm",
                              storage=JournalStorage(JournalFileBackend(args.storage)))

    rows, rungs, warn = [], [], []
    # what the search maximised and its reference points, for the search figure
    # (notebooks/pareto_tasks.py --search); studies made before objectives had
    # names are f3
    meta = {"study": study.study_name, "method": args.method,
            "objective": study.user_attrs.get("objective", "f3")}
    for t in study.trials:
        p = natural(t)
        tag = t.user_attrs.get("tag", f"{study.study_name}-t{t.number:04d}")
        res = result_json(args.output_root, tag, args.method)
        row = {"trial": t.number, "state": t.state.name, "F": t.value,
               "p_fk": t.user_attrs.get("p_fk"), "p_il": t.user_attrs.get("p_il"),
               "p_wm": t.user_attrs.get("p_wm"),
               "best_step": t.user_attrs.get("best_step"),
               "c4_delta_pct": t.user_attrs.get("c4_delta_pct"),
               "feasible": t.user_attrs.get("feasible"),
               "tokens": t.user_attrs.get("tokens"), "job": t.user_attrs.get("job"), "tag": tag,
               "kept_steps": " ".join(map(str, (res or {}).get("kept_steps", [])))}
        row.update({f"hp_{k}": v for k, v in sorted(p.items())})
        rows.append(row)
        if res:
            if "anchors" not in meta and res.get("anchors"):
                meta.update(anchors=res["anchors"], base_c4_ppl=res.get("base_c4_ppl"),
                            util_cap_pct=res.get("util_cap_pct"))
            for pt in res.get("points", []):
                rungs.append({"trial": t.number, **{k: pt.get(k) for k in RUNG_FIELDS[1:]}})
                # only what the objective measures: a wm search never runs fk or il
                import forget_score as fs
                col = {"fk": "fk_prob", "il": "il_ppl", "wm": "wm_q4"}
                for k in ("c4_ppl",) + tuple(col[x] for x in fs.OBJECTIVES[meta["objective"]]):
                    v = pt.get(k)
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        warn.append(f"trial {t.number} step {pt.get('step')}: {k} is {v}")
        if t.state.name == "RUNNING":
            warn.append(f"trial {t.number}: still RUNNING in the study (job {row['job']}), never told")
        if t.state.name == "FAIL" and t.user_attrs.get("job"):
            warn.append(f"trial {t.number}: FAILED after submission (job {row['job']}): read its .out/.err")
        if t.value is not None and row["feasible"] and t.value < NOISE:
            warn.append(f"trial {t.number}: in-budget F {t.value:.4f} is within the scatter "
                        f"of untouched checkpoints (< {NOISE})")

    # COMPUTE, every submitted trial whatever its outcome (2026-10-14): the
    # allocation is spent by dominated, failed and pruned trials as much as by
    # the winners. compute.csv is the ledger, one row per SLURM job.
    dump = None
    if args.sacct:
        with open(args.sacct, encoding="utf-8") as fh:
            dump = parse_sacct(fh.read())
    ledger = compute_ledger(study.study_name, study.trials, dump)
    if dump is not None:
        seen = {r["job"] for r in dump}
        lost = [t.number for t in study.trials if t.user_attrs.get("job") and str(t.user_attrs["job"]) not in seen]
        if lost:
            warn.append(f"compute: {len(lost)} submitted trial(s) not in the sacct dump (older than its start?): "
                        + " ".join(map(str, lost[:20])))
    if ledger is not None:
        per_trial = {}
        for r in ledger:
            if r["category"] == "trial":
                per_trial[r["trial"]] = r
        for row in rows:
            j = per_trial.get(row["trial"], {})
            row.update({k: j.get(k) for k in ("slurm_state", "hours", "gpu_hours", "billing_hours")})

    os.makedirs(args.out, exist_ok=True)
    if ledger is not None:
        with open(os.path.join(args.out, "compute.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COMPUTE_FIELDS); w.writeheader(); w.writerows(ledger)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("hp_"), k))
    with open(os.path.join(args.out, "trials.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, "rungs.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RUNG_FIELDS); w.writeheader(); w.writerows(rungs)
    with open(os.path.join(args.out, "study.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    states = {}
    for r in rows:
        states[r["state"]] = states.get(r["state"], 0) + 1
    print(f"=== study {study.study_name}: {len(rows)} trials {states} ===")
    print(f"    F in [0, 1]: 0 = baseline, 1 = counterfactual on fk, il and wm; "
          f"untouched checkpoints scatter below {NOISE}")
    print(f"{'trial':>5} {'state':8} {'F':>6} {'p_fk':>5} {'p_il':>5} {'p_wm':>5} {'step':>4} "
          f"{'c4%':>7} {'feas':>5}  hyperparameters")
    for r in rows:
        hp = " ".join(f"{k[3:]}={fmt(v)}" for k, v in r.items() if k.startswith("hp_"))
        print(f"{r['trial']:>5} {r['state']:8} {fmt(r['F'], '.4f'):>6} {fmt(r['p_fk'], '.3f'):>5} "
              f"{fmt(r['p_il'], '.3f'):>5} {fmt(r['p_wm'], '.3f'):>5} {fmt(r['best_step']):>4} "
              f"{fmt(r['c4_delta_pct'], '+.2f'):>7} {fmt(r['feasible']):>5}  {hp}")
    feas = [r for r in rows if r["feasible"] and r["F"] is not None and math.isfinite(r["F"])]
    if feas:
        b = max(feas, key=lambda r: r["F"])
        print(f"\nbest in-budget: trial {b['trial']}, F {b['F']:.4f} at step {b['best_step']} "
              f"(fk {fmt(b['p_fk'], '.3f')}, il {fmt(b['p_il'], '.3f')}, wm {fmt(b['p_wm'], '.3f')}), "
              f"c4 {b['c4_delta_pct']:+.2f}%"
              + ("" if b["F"] >= NOISE else "  (within the scatter of untouched checkpoints)"))
    else:
        print("\nno in-budget trial")
    kept = [(r["trial"], r["kept_steps"]) for r in rows if r.get("kept_steps")]
    if kept:
        print(f"\ncheckpoints kept on the frontier (full suite: optuna_hpo.py --eval-cells): "
              + ", ".join(f"t{t} step {s}" for t, s in kept))
    if ledger is None:
        print("\ncompute: no SLURM accounting here (the container has no sacct: pass --sacct, as pe_export does)")
    else:
        print("\ncompute (SLURM accounting; GPU-hours = hours x GCDs; billing = sacct's billing TRES):")
        cats = {}
        for r in ledger:
            key = (r["category"], r["slurm_state"])
            c = cats.setdefault(key, [0, 0.0, 0.0])
            c[0] += 1; c[1] += r["gpu_hours"]; c[2] += r["billing_hours"] or 0.0
        for (cat, st), (n, g, b) in sorted(cats.items()):
            print(f"  {cat:20s} {st:12s} {n:4d} job(s) {g:9.1f} GPU-h {b:11.1f} billing-h")
        tot_g = sum(r["gpu_hours"] for r in ledger); tot_b = sum(r["billing_hours"] or 0.0 for r in ledger)
        tok = sum(r["tokens"] or 0 for r in rows)
        print(f"  {'total':33s} {len(ledger):4d} job(s) {tot_g:9.1f} GPU-h {tot_b:11.1f} billing-h; "
              f"{tok / 1e9:.2f}B training tokens planned over {sum(1 for r in rows if r['job'])} submitted trials")
    print("\nWARNINGS:" if warn else "\nno warnings")
    for x in warn:
        print("  " + x)
    print(f"\nwrote {os.path.join(args.out, 'trials.csv')}, rungs.csv" + (", compute.csv" if ledger is not None else ""))


if __name__ == "__main__":
    main()
