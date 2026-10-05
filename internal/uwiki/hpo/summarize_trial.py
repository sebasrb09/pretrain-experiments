"""Reduce one HPO trial's evaluations to the single JSON the driver reads.

Runs inside the trial's own SLURM job, where torch is already available. That
is deliberate: the Gaussian watermark scores are .pt files, so reading them
needs torch, and keeping that on the compute node lets the Optuna driver live
in a tiny login-node venv with nothing but optuna in it. The training venv is
never touched.

The reader functions come from export_results.py rather than being
reimplemented here, so the objective the search optimizes is by construction
the same number the paper's tables report.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORTER = os.path.join(HERE, os.pardir, "export_results.py")


def _load_exporter():
    spec = importlib.util.spec_from_file_location("_export_results", EXPORTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cell_dir = os.environ["CELL_DIR"]
    rungs = [int(r) for r in os.environ["RUNGS"].split()]
    base = float(os.environ.get("BASE_C4_PPL", "19.71"))
    cap = float(os.environ.get("UTIL_CAP_PCT", "5.0"))

    ex = _load_exporter()

    points = []
    for r in rungs:
        eval_dir = os.path.join(cell_dir, "evals", f"step-{r}")
        if not os.path.isdir(eval_dir):
            continue
        c4 = ex.scalar(eval_dir, "c4_perplexity", "perplexity")
        wm_full, wm_q4 = ex.watermark(eval_dir)
        if c4 is None or wm_q4 is None:
            # A rung whose evaluation did not land is absent, not zero. Scoring
            # it as zero would read as perfect forgetting at no utility cost.
            continue
        points.append({
            "step": r,
            "c4_ppl": float(c4),
            "c4_delta_pct": 100.0 * (float(c4) - base) / base,
            "wm_full": None if wm_full is None else float(wm_full),
            "wm_q4": float(wm_q4),
            # THE OBJECTIVE IS THE ABSOLUTE VALUE. The score is signed and the
            # baseline sits at wm_q4 = -1.170 while a model that never saw the
            # poisons sits at +0.077, so detectability is |wm_q4| and forgetting
            # means driving it toward zero. TPR at 1% FPR is
            # 1 - Phi(z_0.99 - |wm_q4|), which is monotone in |wm_q4|, so the two
            # give the same ranking. Minimizing the SIGNED score instead would
            # optimize toward -1.17, which is the baseline: no forgetting at all.
            "abs_wm_q4": abs(float(wm_q4)),
        })

    feasible = [p for p in points if p["c4_delta_pct"] <= cap]

    # The objective is the best IN-BUDGET point along the trajectory, which is
    # exactly how the paper selects a method's operating point. One trial
    # therefore yields several candidate points for the price of one training
    # run, which is where most of the sample efficiency comes from.
    if feasible:
        best = min(feasible, key=lambda p: p["abs_wm_q4"])
    elif points:
        # Infeasible: report the least damaging point so the sampler still
        # learns the shape of the constraint boundary, and let the driver mark
        # it violated through constraints_func.
        best = min(points, key=lambda p: p["c4_delta_pct"])
    else:
        best = None

    result = {
        "trial": int(os.environ["TRIAL"]),
        "method": os.environ["METHOD"],
        "cell_dir": cell_dir,
        "base_c4_ppl": base,
        "util_cap_pct": cap,
        "points": points,
        "best": best,
        "feasible": bool(feasible),
        # Positive means the cap was exceeded, which is the sign convention
        # Optuna's constraints_func expects.
        "constraint": None if best is None else best["c4_delta_pct"] - cap,
        # What the driver passes to study.tell. Named separately so the sign
        # convention cannot be mistaken at the other end.
        "objective": None if best is None else best["abs_wm_q4"],
        # For context in the log: the two fixed points of the axis.
        "baseline_abs_wm_q4": 1.170,
        "counterfactual_abs_wm_q4": 0.077,
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    if best is None:
        print("WARNING: no rung produced both a perplexity and a watermark score",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
