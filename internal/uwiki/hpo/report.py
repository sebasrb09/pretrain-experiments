"""Summarize an Optuna HPO study: every trial, its outcome, and the best one.

    python internal/uwiki/hpo/report.py --method ce-u --out "$PE_WORK/exports-hpo-ceu"

Reads the study from the journal the driver wrote, and each trial's
hpo_result.json for the per-rung measurements behind its objective. Writes
trials.csv (one row per trial) and rungs.csv (one row per evaluated rung) so the
search can be inspected and plotted off the cluster.

Things it flags, because each has bitten this project before:
  - NaN or missing objectives (a trial that trained into NaN, or whose eval
    never produced the watermark)
  - objectives at the baseline |wm_q4| (~1.17), which would mean the eval
    measured an untouched model
  - trials that never finished (still RUNNING in the study)
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PE = os.environ.get("PE_WORK", "/scratch/project_465003383/unlearning_baselines")

BASELINE = 1.170          # |wm_q4| of the baseline anchor (wm-q4-sign-convention)
FLOOR = 0.077             # |wm_q4| of the counterfactual
NOISE = 0.10              # spread of the continued-pretraining control over steps


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
        return "NaN"
    return format(v, spec) if isinstance(v, (int, float)) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--storage", default=os.path.join(PE, "hpo", "optuna_journal.log"))
    ap.add_argument("--study", default=None)
    ap.add_argument("--output-root", default=os.path.join(PE, "hpo"),
                    help="where the trials' cell directories live")
    ap.add_argument("--out", required=True, help="directory for trials.csv and rungs.csv")
    args = ap.parse_args()

    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if not os.path.exists(args.storage):
        sys.exit(f"no study journal at {args.storage}")
    study = optuna.load_study(study_name=args.study or f"hpo-{args.method}",
                              storage=JournalStorage(JournalFileBackend(args.storage)))

    rows, rungs, warn = [], [], []
    for t in study.trials:
        p = natural(t)
        tag = t.user_attrs.get("tag", f"hpo-{args.method}-t{t.number:04d}")
        res = result_json(args.output_root, tag, args.method)
        row = {"trial": t.number, "state": t.state.name, "objective_abs_wm_q4": t.value,
               "best_step": t.user_attrs.get("best_step"),
               "c4_delta_pct": t.user_attrs.get("c4_delta_pct"),
               "feasible": t.user_attrs.get("feasible"),
               "tokens": t.user_attrs.get("tokens"), "job": t.user_attrs.get("job"), "tag": tag}
        row.update({f"hp_{k}": v for k, v in sorted(p.items())})
        rows.append(row)
        if res:
            for pt in res.get("points", []):
                rungs.append({"trial": t.number, "step": pt.get("step"), "c4_ppl": pt.get("c4_ppl"),
                              "c4_delta_pct": pt.get("c4_delta_pct"), "wm_q4": pt.get("wm_q4"),
                              "abs_wm_q4": pt.get("abs_wm_q4")})
                for k in ("c4_ppl", "wm_q4"):
                    v = pt.get(k)
                    if v is None or (isinstance(v, float) and not math.isfinite(v)):
                        warn.append(f"trial {t.number} step {pt.get('step')}: {k} is {v}")
        if t.state.name == "RUNNING":
            warn.append(f"trial {t.number}: still RUNNING in the study (job {row['job']}), never told")
        if t.state.name == "FAIL" and t.user_attrs.get("job"):
            warn.append(f"trial {t.number}: FAILED after submission (job {row['job']}): read its .out/.err")
        if t.value is not None and abs(t.value - BASELINE) < 0.02:
            warn.append(f"trial {t.number}: objective {t.value:.4f} sits at the baseline")

    os.makedirs(args.out, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("hp_"), k))
    with open(os.path.join(args.out, "trials.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, "rungs.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial", "step", "c4_ppl", "c4_delta_pct", "wm_q4", "abs_wm_q4"])
        w.writeheader(); w.writerows(rungs)

    states = {}
    for r in rows:
        states[r["state"]] = states.get(r["state"], 0) + 1
    print(f"=== study {study.study_name}: {len(rows)} trials {states} ===")
    print(f"    baseline |wm_q4| {BASELINE}, counterfactual floor {FLOOR}, "
          f"step-to-step noise of the control ~{NOISE}")
    print(f"{'trial':>5} {'state':8} {'|wm_q4|':>8} {'step':>4} {'c4%':>7} {'feas':>5}  hyperparameters")
    for r in rows:
        hp = " ".join(f"{k[3:]}={fmt(v)}" for k, v in r.items() if k.startswith("hp_"))
        print(f"{r['trial']:>5} {r['state']:8} {fmt(r['objective_abs_wm_q4']):>8} "
              f"{fmt(r['best_step']):>4} {fmt(r['c4_delta_pct'], '+.2f'):>7} "
              f"{fmt(r['feasible']):>5}  {hp}")
    feas = [r for r in rows if r["feasible"] and r["objective_abs_wm_q4"] is not None
            and math.isfinite(r["objective_abs_wm_q4"])]
    if feas:
        b = min(feas, key=lambda r: r["objective_abs_wm_q4"])
        gain = BASELINE - b["objective_abs_wm_q4"]
        print(f"\nbest in-budget: trial {b['trial']}, |wm_q4| {b['objective_abs_wm_q4']:.4f} at step "
              f"{b['best_step']}, c4 {b['c4_delta_pct']:+.2f}%: {100 * gain / (BASELINE - FLOOR):.1f}% "
              f"of the way to the floor" + ("" if gain > NOISE else "  (within the control's noise)"))
    else:
        print("\nno in-budget trial")
    print("\nWARNINGS:" if warn else "\nno warnings")
    for x in warn:
        print("  " + x)
    print(f"\nwrote {os.path.join(args.out, 'trials.csv')} and rungs.csv")


if __name__ == "__main__":
    main()
