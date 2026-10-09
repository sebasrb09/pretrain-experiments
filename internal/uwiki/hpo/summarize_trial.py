"""Reduce one HPO trial's evaluations to the single JSON the driver reads.

Runs inside the trial's own SLURM job, where torch is available: the Gaussian
watermark scores are .pt files.

The objective is forget_score.F, the three-task forgetting score (see that
file for the equation and why it is scaled the way it is), at the best rung
inside the utility budget. Since 2026-10-08 hpo_trial.sh evaluates C4 at every
rung but F only at the last rung inside the budget (see its comment for the
evidence that this is the best one), so there is one complete point; the C4 at
every rung is kept as c4_trajectory. The utility baseline is the BASELINE ANCHOR's C4
perplexity, read from disk rather than passed in: the anchors were measured on
the same C4 file as the trials, which is not the file the paper reports on, so
no hard-coded number can be right for both.

Every input is required. A default here is how a trial silently measures the
wrong thing.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import forget_score as fs   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = os.environ
    cell_dir = env["CELL_DIR"]
    rungs = [int(r) for r in env["RUNGS"].split()]
    anchor_root = env["ANCHOR_ROOT"]
    il_exp = env["IL_EXPERIMENT"]
    cap = float(env["UTIL_CAP_PCT"])
    # Trials submitted before objectives had names carry no HPO_OBJECTIVE: f3.
    objective = env.get("HPO_OBJECTIVE", "f3")
    tasks = fs.OBJECTIVES[objective]

    ex = fs.load_exporter()
    anchors = fs.load_anchors(anchor_root, il_exp, ex)
    base = anchors["baseline"]["c4"]

    points = []
    c4_trajectory = []
    for r in rungs:
        eval_dir = os.path.join(cell_dir, "evals", f"step-{r}")
        if not os.path.isdir(eval_dir):
            continue
        m = fs.measure(eval_dir, il_exp, ex)
        if m["c4"] is not None:
            c4_trajectory.append({"step": r, "c4_ppl": m["c4"],
                                  "c4_delta_pct": 100.0 * (m["c4"] - base) / base})
        s = fs.score(m, anchors, objective)
        if m["c4"] is not None and s is None and all(m[t] is None for t in tasks):
            continue    # a C4-only rung of the first pass, by design
        if m["c4"] is None or s is None:
            # A rung whose evaluation did not land is absent, not zero.
            print(f"  step-{r}: incomplete evaluation {m}, left out", file=sys.stderr)
            continue
        points.append({
            "step": r,
            "c4_ppl": m["c4"],
            "c4_delta_pct": 100.0 * (m["c4"] - base) / base,
            "fk_prob": m["fk"], "il_ppl": m["il"], "wm_q4": m["wm"],
            **s,
        })

    feasible = [p for p in points if p["c4_delta_pct"] <= cap]
    # The best IN-BUDGET point along the trajectory, which is how the paper
    # picks a method's operating point.
    if feasible:
        best = max(feasible, key=lambda p: p["F"])
    elif points:
        # Infeasible: report the least damaging point, so the sampler still
        # learns where the wall is; constraints_func marks it violated.
        best = min(points, key=lambda p: p["c4_delta_pct"])
    else:
        best = None

    result = {
        "trial": int(env["TRIAL"]),
        "method": env["METHOD"],
        "cell_dir": cell_dir,
        "objective_name": f"F = mean(p_{', p_'.join(tasks)}), maximized ({objective})",
        "objective_tasks": list(tasks),
        "anchor_root": anchor_root,
        "anchors": {pt: anchors[pt] for pt in fs.POINTS},
        "base_c4_ppl": base,
        "util_cap_pct": cap,
        "points": points,
        "c4_trajectory": c4_trajectory,
        "best": best,
        "feasible": bool(feasible),
        # Positive means the cap was exceeded, Optuna's constraints convention.
        "constraint": None if best is None else best["c4_delta_pct"] - cap,
        # What the driver passes to study.tell (direction=maximize).
        "objective": None if best is None else best["F"],
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    if best is None:
        print("WARNING: no rung produced all four measurements", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
