"""Optuna ask/tell driver for the unlearning hyperparameter search.

Runs on a login node as a submit-and-poll loop, the same shape as the existing
sweep orchestrators: it sleeps, calls squeue and sbatch, and does no compute
itself.

Why ask/tell rather than study.optimize: an objective here takes hours and runs
on another machine, so the sampling step and the evaluation step have to be
decoupled. ask() hands out a configuration, the trial becomes a SLURM job, and
tell() reports back whenever that job lands. Several trials are therefore in
flight at once, and GPSampler conditions each new batch on everything that has
finished.

The budget is tokens, not trials. See spaces.py for why.

Install: optuna goes into the container venv you already use, which after
`source internal/lumi/env.sh` is what plain `python -m pip install optuna`
writes to. GPSampler needs scipy and torch and the container supplies both.

Usage:
    # 1. budget arithmetic only, no optuna and no submission
    python internal/uwiki/hpo/optuna_hpo.py --method ce-u --plan

    # 2. the real search
    python internal/uwiki/hpo/optuna_hpo.py \
        --method ce-u --budget-tokens 300e6 --max-parallel 8
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spaces                                                  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
PE = os.environ.get("PE_WORK", "/scratch/project_465003383/unlearning_baselines")

# The rung ladder. These are the steps a trial checkpoints and evaluates at, and
# they are a subset of the sweep's 1,2,3,5,8,13,21,34,55 schedule. Four rather
# than nine because each rung costs a separate evaluation, and the objective
# only needs enough of the trajectory to find the in-budget optimum. Observed
# best steps on the working sweep were 5, 8, 13, 14, 21, 25, 28 and 51, so the
# ladder brackets all of them.
DEFAULT_RUNGS = [3, 8, 21, 55]


# --------------------------------------------------------------- stub sampler
class _StubTrial:
    """Uniform random sampling with no optuna, for --plan.

    Mirrors the three suggest_* methods spaces.py uses so the search space and
    the cost model can be exercised anywhere, including a laptop.
    """

    def __init__(self, rng):
        self.rng = rng
        self.params = {}

    def suggest_categorical(self, name, choices):
        v = self.rng.choice(choices)
        self.params[name] = v
        return v

    def suggest_float(self, name, low, high, log=False):
        if log:
            import math
            v = math.exp(self.rng.uniform(math.log(low), math.log(high)))
        else:
            v = self.rng.uniform(low, high)
        self.params[name] = v
        return v


def plan(method, budget, steps, rungs, n_show, max_batch, tph, emin):
    rng = random.Random(0)
    print(f"=== budget plan: {method} ===")
    print(f"  pre-training run     {spaces.PRETRAIN_TOKENS/1e9:8.1f}B tokens")
    print(f"  10% allowance        {spaces.METHOD_BUDGET_TOKENS/1e9:8.1f}B tokens  (PER METHOD)")
    print(f"  this method's budget {budget/1e9:8.3f}B tokens")
    print(f"  steps per trial      {steps}")
    print(f"  rungs                {rungs}")
    print(f"  retain-carrying      {spaces.USES_RETAIN[method]} "
          f"({'2x' if spaces.USES_RETAIN[method] else '1x'} tokens per step)")
    print()
    costs = []
    for b in [x for x in spaces.BATCH_CHOICES if x <= max_batch]:
        c = spaces.trial_tokens(method, b, steps)
        costs.append(c)
        wt = trial_walltime(method, {"batch_size": b}, steps, rungs, tph, emin)
        risk = "  <-- HITS THE 24h CAP, would be truncated" if wt.startswith("24:") else ""
        print(f"  batch {b:4d}: {c/1e6:8.1f}M tokens/trial -> "
              f"{int(budget // c):4d} trials, walltime {wt}{risk}")
    avg = sum(costs) / len(costs)
    print(f"\n  uniform over batch sizes: ~{avg/1e6:.1f}M per trial -> "
          f"~{int(budget // avg)} trials expected")
    print(f"  evaluations            ~{int(budget // avg) * len(rungs)} "
          f"(={len(rungs)} rungs per trial)")
    print(f"\n=== {n_show} sample configurations ===")
    for i in range(n_show):
        t = _StubTrial(rng)
        p = spaces.suggest(t, method, steps, max_batch)
        cost = spaces.trial_tokens(method, p["batch_size"], steps)
        bits = " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(p.items()))
        print(f"  [{i:2d}] {cost/1e6:7.1f}M  {bits}")


# ------------------------------------------------------------------ the search
def trial_walltime(method, params, steps, rungs, tok_per_hour, eval_min, cap_h=24):
    """SLURM walltime for one trial, scaled by what the trial actually costs.

    A FIXED walltime is a trap here. Centring the batch space on 512 means
    trials span 128 to 2048, a 16x spread in accumulation work, so one limit
    either times out the large-batch trials or wildly over-books the small
    ones. Timeouts have already cost this project two full sweeps.

    Both rates are rough and configurable. The first pilot measures them, and
    the 1.6x safety factor is there because being wrong in this direction
    costs a queue slot while being wrong in the other costs the trial.
    """
    toks = spaces.trial_tokens(method, params["batch_size"], steps)
    # Throughput is NOT independent of MICRO_BATCH. At MICRO_BATCH=1 the GPU
    # runs one 4,096-token sequence per forward pass, so utilisation is far
    # below MICRO_BATCH=4 for the same tokens. --tokens-per-hour is quoted at
    # MICRO_BATCH=4 and scaled down from there. Crude, and the pilot replaces
    # it with a measurement, but ignoring it would under-book every rmu and npo
    # trial, which is the direction that loses the work.
    eff = float(tok_per_hour) * (spaces.MICRO_BATCH[method] / 4.0)
    hours = 1.6 * (toks / eff + len(rungs) * eval_min / 60.0)
    hours = max(2.0, min(float(cap_h), hours))
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h:02d}:{m:02d}:00"


def _submit(method, params, trial_no, steps, rungs, out_root, time_limit, dry):
    tag = f"hpo-{method}-t{trial_no:04d}"
    env = dict(os.environ)
    trial_env = spaces.env_for(method, params, steps, rungs)
    # Refuse to submit a trial carrying a dimension the cell cannot forward.
    # Two such dimensions existed before this check: grad-diff's retain_weight
    # and rmu's, both sampled and both inert.
    spaces.validate(method, params, trial_env)
    env.update(trial_env)
    env.update({
        "REPO": REPO,
        "TRIAL": str(trial_no),
        "RUN_TAG": tag,
        "OUTPUT_ROOT": out_root,
        "RUNGS": " ".join(str(r) for r in rungs),
        # The 1.5B identity, the same overrides sweep_1B_v2.sh uses. Without
        # these the LUMI wrapper supplies its 2.7B defaults.
        "MODEL": "sbordt/OLMo-2-1B-Exp-Unlearning",
        "REVISION": "stage1-step100000-tokens210B",
        "OPTIM_REPO": "sbordt/OLMo-2-1B-Exp-Unlearning",
        "OPTIM_REVISION": "step100000-unsharded",
        "OLMO_CONFIG": "",
    })
    cmd = ["sbatch", "-J", tag, f"--time={time_limit}", "--export=ALL",
           os.path.join(HERE, "hpo_trial.sh")]
    if dry:
        print(f"  [dry] {tag}  " + " ".join(
            f"{k}={env[k]}" for k in sorted(spaces.env_for(method, params, steps, rungs))))
        return None, tag
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        print(f"  SUBMIT FAILED for {tag}: {out.stderr.strip()}", file=sys.stderr)
        return None, tag
    job = out.stdout.strip().split()[-1]
    print(f"  submitted {tag} as job {job}")
    return job, tag


def _running(job):
    r = subprocess.run(["squeue", "-j", str(job), "-h"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def _harvest(out_root, tag, method):
    """Read a finished trial's result file, or None if it is not there."""
    cell = os.path.join(out_root, tag, method)
    if not os.path.isdir(cell):
        return None
    for d in sorted(os.listdir(cell)):
        f = os.path.join(cell, d, "hpo_result.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                return json.load(fh)
    return None


def search(args):
    import optuna
    from optuna.samplers import GPSampler

    os.makedirs(os.path.dirname(args.storage.replace("sqlite:///", "")) or ".",
                exist_ok=True)

    def constraints_func(trial):
        # Positive is a violation. A trial whose every rung blew the utility cap
        # is still informative, since it tells the GP where the wall is.
        c = trial.user_attrs.get("constraint")
        return [1.0 if c is None else float(c)]

    study = optuna.create_study(
        study_name=args.study or f"hpo-{args.method}",
        storage=args.storage,
        load_if_exists=True,
        # Minimize |wm_q4|. The score is signed: the baseline sits at -1.170 and
        # a model that never saw the poisons at +0.077, so detectability is the
        # magnitude and forgetting drives it toward zero. summarize_trial.py
        # computes it and reports it as "objective".
        direction="minimize",
        sampler=GPSampler(
            n_startup_trials=args.startup,    # uniform random until this many land
            constraints_func=constraints_func,
            seed=args.seed,
        ),
    )

    spent = sum(t.user_attrs.get("tokens", 0) for t in study.trials
                if t.user_attrs.get("tokens"))
    print(f"=== {study.study_name} ===")
    print(f"  budget {args.budget_tokens/1e9:.3f}B tokens, already spent "
          f"{spent/1e9:.3f}B over {len(study.trials)} trial(s)")
    print(f"  uniform random for the first {args.startup} trials, then GP")

    inflight = {}        # trial_no -> (optuna trial, job, tag, tokens)

    while True:
        # Fill the queue while there is budget and room.
        while len(inflight) < args.max_parallel:
            if spent >= args.budget_tokens:
                break
            trial = study.ask()
            params = spaces.suggest(trial, args.method, args.steps, args.max_batch)
            cost = spaces.trial_tokens(args.method, params["batch_size"], args.steps)
            if spent + cost > args.budget_tokens:
                # Do not overspend. Returning the trial as failed keeps the
                # study's trial numbering honest about what was never run.
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                print(f"  trial {trial.number} would overspend "
                      f"({cost/1e6:.1f}M, {(args.budget_tokens-spent)/1e6:.1f}M left), stopping")
                spent = args.budget_tokens
                break
            wt = args.time or trial_walltime(
                args.method, params, args.steps, args.rungs,
                args.tokens_per_hour, args.eval_minutes)
            job, tag = _submit(args.method, params, trial.number, args.steps,
                               args.rungs, args.output_root, wt, args.dry_run)
            if args.dry_run:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                spent += cost
                continue
            if job is None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                continue
            trial.set_user_attr("tokens", cost)
            trial.set_user_attr("job", job)
            trial.set_user_attr("tag", tag)
            inflight[trial.number] = (trial, job, tag, cost)
            spent += cost

        if not inflight:
            break

        time.sleep(args.poll)

        for no in list(inflight):
            trial, job, tag, cost = inflight[no]
            if _running(job):
                continue
            res = _harvest(args.output_root, tag, args.method)
            if res is None or res.get("best") is None:
                print(f"  trial {no} ({tag}) produced no result, marked failed")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                best = res["best"]
                trial.set_user_attr("constraint", res["constraint"])
                trial.set_user_attr("best_step", best["step"])
                trial.set_user_attr("c4_delta_pct", best["c4_delta_pct"])
                trial.set_user_attr("feasible", res["feasible"])
                study.tell(trial, res["objective"])
                flag = "" if res["feasible"] else "  INFEASIBLE"
                print(f"  trial {no}: |wm_q4|={res['objective']:.4f} at step "
                      f"{best['step']}, c4 +{best['c4_delta_pct']:.2f}%{flag}"
                      f"   (baseline 1.170, floor 0.077)")
            del inflight[no]

    print("\n=== done ===")
    feas = [t for t in study.trials
            if t.value is not None and t.user_attrs.get("feasible")]
    if feas:
        b = min(feas, key=lambda t: t.value)
        print(f"  best in-budget trial {b.number}: |wm_q4|={b.value:.4f} "
              f"(baseline 1.170, counterfactual floor 0.077, so "
              f"{100*(1.170-b.value)/(1.170-0.077):.1f}% of the way to the floor)")
        for k, v in sorted(b.params.items()):
            print(f"    {k} = {v}")
    else:
        print("  no in-budget trial yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(spaces.USES_RETAIN))
    ap.add_argument("--plan", action="store_true",
                    help="print the budget arithmetic and sample configurations, "
                         "then exit. Needs no optuna and submits nothing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the real sampler but print sbatch instead of calling it")
    ap.add_argument("--budget-tokens", type=float, default=None,
                    help="default: the full 10%% allowance, which is PER METHOD")
    ap.add_argument("--steps", type=int, default=55)
    ap.add_argument("--max-batch", type=int, default=10**9,
                    help="trim batch choices above this. Use it when a large "
                         "batch cannot finish in the walltime, which the plan "
                         "output flags.")
    ap.add_argument("--rungs", type=int, nargs="+", default=DEFAULT_RUNGS)
    ap.add_argument("--startup", type=int, default=None,
                    help="uniform random trials before the GP takes over. "
                         "Default 3x the number of dimensions, minimum 16, since "
                         "a GP given fewer points than dimensions is guessing.")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", default=None,
                    help="fixed walltime per trial job. Default is to scale it "
                         "with the sampled batch size, which is what you want: "
                         "trials span a 16x range in cost.")
    ap.add_argument("--tokens-per-hour", type=float, default=120e6,
                    help="training throughput, for sizing walltime. Rough. The "
                         "pilot measures the real number.")
    ap.add_argument("--eval-minutes", type=float, default=20.0,
                    help="minutes per rung evaluation, for sizing walltime")
    ap.add_argument("--output-root", default=os.path.join(PE, "hpo"))
    ap.add_argument("--storage", default=f"sqlite:///{os.path.join(PE, 'hpo', 'optuna.db')}")
    ap.add_argument("--study", default=None)
    args = ap.parse_args()

    if args.startup is None:
        # Dimension aware. The space is 7 to 10 dimensions depending on method,
        # and seeding a GP with fewer random points than it has dimensions makes
        # its first suggestions arbitrary.
        import random as _r
        n_dims = len(spaces.suggest(_StubTrial(_r.Random(0)), args.method,
                                   args.steps, args.max_batch))
        args.startup = max(16, 3 * n_dims)
        print(f"  {n_dims} dimensions -> {args.startup} uniform random startup trials")

    if args.budget_tokens is None:
        # Per method, not shared. Each method gets its own 10% of the continual
        # pre-training compute.
        args.budget_tokens = spaces.METHOD_BUDGET_TOKENS

    if args.plan:
        plan(args.method, args.budget_tokens, args.steps, args.rungs, 10,
             args.max_batch, args.tokens_per_hour, args.eval_minutes)
        return 0
    search(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
