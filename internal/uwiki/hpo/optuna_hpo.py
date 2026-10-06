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
import glob
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import types

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

# The sweep's own checkpoint schedule. A finalized winner is retrained on it so
# its trajectory lines up rung for rung with every sweep cell.
FINAL_RUNGS = [1, 2, 3, 5, 8, 13, 21, 34, 55]

# The 1.5B identity, the same overrides sweep_1B_v2.sh uses. Without these the
# LUMI wrapper supplies its 2.7B defaults.
IDENTITY = {
    "MODEL": "sbordt/OLMo-2-1B-Exp-Unlearning",
    "REVISION": "stage1-step100000-tokens210B",
    "OPTIM_REPO": "sbordt/OLMo-2-1B-Exp-Unlearning",
    "OPTIM_REVISION": "step100000-unsharded",
    "OLMO_CONFIG": "",
}


def _scrub_vars():
    """SCRUB_VARS from internal/uwiki/scrub_env.sh, so there is one list."""
    path = os.path.join(REPO, "internal", "uwiki", "scrub_env.sh")
    src = open(path, encoding="utf-8").read()
    body = src[src.index("SCRUB_VARS=(") + len("SCRUB_VARS=("):]
    body = body[:body.index(chr(10) + ")")]
    names = []
    for line in body.splitlines():
        names += line.split("#", 1)[0].split()
    return names


SCRUB_VARS = _scrub_vars()


def _clean_env():
    """The login environment minus everything that defines the experiment.

    sbatch --export=ALL hands the job this whole environment, and the cell
    scripts read dozens of experiment variables with ${VAR:-default}. So a
    value left in the shell, and env.sh exports a 2.7B OUTPUT_ROOT on every
    source, would silently change what a trial runs. Each such variable is
    either set explicitly by this driver or absent.
    """
    return {k: v for k, v in os.environ.items() if k not in SCRUB_VARS}


BRIDGE_DIR = os.path.join(PE, "hpo", "bridge")

# Where build_noise_dir.sh puts the 1.5B watermark vectors on LUMI. Passed to
# every trial and eval EXPLICITLY: the eval body's own default resolves to a
# $HOME path that does not exist here, and the first pilot died on exactly that.
NOISE_DIR_1B = os.path.join(PE, "noise-vectors", "OLMo-2-1B-Exp")

# 18.7734 is the C4 perplexity of the 1.5B BASELINE anchor (step 0) on LUMI,
# from exports-1B-lumi/results_anchors.csv, identical on ASC. 19.71 is the 5%
# CAP (18.7734 x 1.05), and an earlier version used it here as the baseline,
# which put the feasibility line at 20.70, a 10.3% budget.
C4_BASELINE_1B = "18.7734"
UTIL_CAP_PCT = "5.0"

# Eval batch for perplexity. Every LUMI eval launcher before the HPO set 1:
# perplexity.py documents batch 8 asking for ~19 GB in one allocation, an OOM
# on a 64 GB MI250X GCD. Memory only: LUMI and ASC baselines agree to 4 d.p.
EVAL_MAX_NUM_SEQS = "1"
# Batch for every other eval task (inference_engine.py, default 8). No LUMI
# launcher ever set it, so the decayed arm's full suite ran at 8 on these
# GCDs: a proven value, set explicitly so a stale shell value cannot leak in.
INFERENCE_MAX_NUM_SEQS = "8"


def _require_noise():
    if not glob.glob(os.path.join(NOISE_DIR_1B, "gaussian_poisoning_*.pkl")):
        sys.exit(f"no gaussian_poisoning_*.pkl in {NOISE_DIR_1B}: the watermark, which is "
                 "the objective, cannot be scored. Not launching.")


def _host_run(cmd, explicit=None, timeout=900):
    """Run a SLURM command on the host, even from inside the container.

    On LUMI this driver runs inside the PyTorch Singularity container, because
    that is where optuna and torch live, and the container has no sbatch or
    squeue: SLURM exists only on the host. When the command is on PATH (a
    host Python, or a test) it runs directly. Otherwise it is handed to
    hpo_bridge.sh, a host-side loop that executes request scripts dropped in
    BRIDGE_DIR, and the reply is read back.

    `explicit` holds the variables the job needs, and in bridge mode nothing
    else from this process crosses over: inside the container os.environ
    carries container paths (PATH, LD_LIBRARY_PATH, ...) that must never reach
    a host job. The job inherits the bridge's host environment instead, with
    every SCRUB_VARS entry removed and `explicit` exported on top.

    Returns an object with returncode, stdout and stderr, like subprocess.run.
    """
    explicit = dict(explicit or {})
    if shutil.which(cmd[0]):
        env = _clean_env()
        env.update(explicit)
        return subprocess.run(cmd, env=env, capture_output=True, text=True)

    os.makedirs(BRIDGE_DIR, mode=0o700, exist_ok=True)
    rid = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    lines = ["set -u", f"cd {shlex.quote(os.getcwd())} || exit 97"]
    drop = [v for v in SCRUB_VARS if v not in explicit]
    if drop:
        lines.append("unset " + " ".join(drop))
    for k, v in explicit.items():
        lines.append(f"export {k}={shlex.quote(str(v))}")
    lines.append("exec " + " ".join(shlex.quote(c) for c in cmd))
    base = os.path.join(BRIDGE_DIR, rid)
    with open(base + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(chr(10).join(lines) + chr(10))
    # Rename into place so the bridge never picks up half a request.
    os.replace(base + ".tmp", base + ".req")

    deadline = time.time() + timeout
    while not os.path.exists(base + ".rc"):
        if time.time() > deadline:
            return types.SimpleNamespace(
                returncode=124, stdout="",
                stderr=f"no reply from hpo_bridge.sh within {timeout}s ({BRIDGE_DIR})")
        time.sleep(1)

    def _read(ext):
        try:
            with open(base + ext, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""
    rc = int((_read(".rc").strip() or "1"))
    res = types.SimpleNamespace(returncode=rc, stdout=_read(".out"), stderr=_read(".err"))
    for ext in (".rc", ".out", ".err"):
        try:
            os.remove(base + ext)
        except OSError:
            pass
    return res


def _require_host_slurm():
    """Fail fast when SLURM is unreachable, instead of hanging on every call."""
    if shutil.which("sbatch"):
        return
    hb = os.path.join(BRIDGE_DIR, ".heartbeat")
    age = time.time() - os.path.getmtime(hb) if os.path.exists(hb) else None
    if age is None or age > 60:
        sys.exit(
            "sbatch is not available here (this is the container) and no live\n"
            f"hpo_bridge.sh is serving {BRIDGE_DIR}. Start it on the HOST first:\n"
            "  setsid nohup bash internal/uwiki/hpo/hpo_bridge.sh \\\n"
            "      > \"$PE_WORK/hpo/bridge.log\" 2>&1 < /dev/null &\n"
            "  disown")


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
    # Per-method measured anchor, not one number scaled by micro-batch. See
    # spaces.THROUGHPUT. --tokens-per-hour overrides it for all methods.
    eff = float(tok_per_hour) if tok_per_hour else spaces.THROUGHPUT[method]
    hours = 1.6 * (toks / eff + len(rungs) * eval_min / 60.0)
    hours = max(2.0, min(float(cap_h), hours))
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h:02d}:{m:02d}:00"


def _submit(method, params, trial_no, steps, rungs, out_root, time_limit, dry):
    tag = f"hpo-{method}-t{trial_no:04d}"
    env = {}            # only what the job needs; see _host_run
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
        "NOISE_DIR": NOISE_DIR_1B,
        "BASE_C4_PPL": C4_BASELINE_1B,
        "UTIL_CAP_PCT": UTIL_CAP_PCT,
        "EVAL_MAX_NUM_SEQS": EVAL_MAX_NUM_SEQS,
        "INFERENCE_MAX_NUM_SEQS": INFERENCE_MAX_NUM_SEQS,
    })
    env.update(IDENTITY)
    cmd = ["sbatch", "-J", tag, f"--time={time_limit}", "--export=ALL",
           os.path.join(HERE, "hpo_trial.sh")]
    if dry:
        print(f"  [dry] {tag}  " + " ".join(
            f"{k}={env[k]}" for k in sorted(spaces.env_for(method, params, steps, rungs))))
        return None, tag
    out = _host_run(cmd, env)
    if out.returncode != 0:
        print(f"  SUBMIT FAILED for {tag}: {out.stderr.strip()}", file=sys.stderr)
        return None, tag
    job = out.stdout.strip().split()[-1]
    print(f"  submitted {tag} as job {job}")
    return job, tag


def _running(job):
    """True while the job is queued or running, and also when we cannot tell.

    squeue on LUMI occasionally fails with "Socket timed out". Reading that
    empty reply as "finished" would harvest a trial that is still training,
    find no result, and mark it failed, losing its tokens and its result. A
    finished job makes squeue -j exit non-zero with "Invalid job id", which is
    the only failure taken to mean done.
    """
    r = _host_run(["squeue", "-j", str(job), "-h"])
    if r.returncode != 0:
        return "Invalid job id" not in (r.stderr or "")
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


def _storage(args):
    # Journal storage by default, not SQLite. The study lives on Lustre, and
    # SQLite relies on POSIX byte-range locks that parallel filesystems do not
    # reliably honour, which shows up as "database is locked" or "disk I/O
    # error" at create_study. JournalFileOpenLock takes its lock by atomic file
    # creation instead, which Lustre does support. An explicit sqlite:/// or
    # other URL is still accepted.
    if "://" in args.storage:
        if args.storage.startswith("sqlite:///"):
            os.makedirs(os.path.dirname(args.storage[len("sqlite:///"):]) or ".",
                        exist_ok=True)
        return args.storage
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    os.makedirs(os.path.dirname(args.storage) or ".", exist_ok=True)
    return JournalStorage(JournalFileBackend(
        args.storage, lock_obj=JournalFileOpenLock(args.storage)))


def search(args):
    import optuna
    from optuna.samplers import GPSampler

    if not args.dry_run:
        _require_host_slurm()
        _require_noise()
    storage = _storage(args)

    def constraints_func(trial):
        # Positive is a violation. A trial whose every rung blew the utility cap
        # is still informative, since it tells the GP where the wall is.
        c = trial.user_attrs.get("constraint")
        return [1.0 if c is None else float(c)]

    study = optuna.create_study(
        study_name=args.study or f"hpo-{args.method}",
        storage=storage,
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
    gone = sorted(k for k in os.environ if k in SCRUB_VARS)
    if gone:
        print(f"  ignoring {len(gone)} inherited experiment variable(s): {' '.join(gone)}")

    inflight = {}        # trial_no -> (optuna trial, job, tag, tokens)
    submit_failures = 0
    # Trials that ran but returned nothing. Three in a row means something
    # systematic (a missing input, a broken path), and continuing would spend
    # the whole budget on trials that cannot produce an objective.
    empty_results = 0

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
                # Budget only grows on success, so a persistently failing
                # sbatch (bad account, partition, missing script) would
                # otherwise spin here forever, filling the study with failed
                # trials and hammering the scheduler. Stop and say why.
                submit_failures += 1
                if submit_failures >= 3:
                    sys.exit("ABORT: 3 consecutive sbatch failures, see the "
                             "SUBMIT FAILED lines above")
                continue
            submit_failures = 0
            trial.set_user_attr("tokens", cost)
            # The natural values (betas, not their logits) exactly as sent to
            # the job, so a winner can be retrained without re-deriving them.
            trial.set_user_attr("params", params)
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
                empty_results += 1
                if empty_results >= 3:
                    sys.exit("ABORT: 3 consecutive trials produced no result. Read one "
                             f"hpo-{args.method}-t*.out before relaunching. Trials still "
                             "in the queue are not cancelled.")
            else:
                best = res["best"]
                trial.set_user_attr("constraint", res["constraint"])
                trial.set_user_attr("best_step", best["step"])
                trial.set_user_attr("c4_delta_pct", best["c4_delta_pct"])
                trial.set_user_attr("feasible", res["feasible"])
                study.tell(trial, res["objective"])
                empty_results = 0
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
            if k.endswith("_logit"):
                # registered with Optuna in logit space, see spaces.logit_uniform
                k, v = k[:-len("_logit")], 1.0 / (1.0 + math.exp(-v))
            print(f"    {k} = {v}")
    else:
        print("  no in-budget trial yet")


def _natural_params(trial):
    """A trial's hyperparameters as values, not as the logits Optuna stores.

    Trials submitted by this version carry the exact values in user_attrs.
    Older ones are reconstructed by squashing each *_logit parameter back.
    """
    if "params" in trial.user_attrs:
        return dict(trial.user_attrs["params"])
    p = {}
    for k, v in trial.params.items():
        if k.endswith("_logit"):
            p[k[:-len("_logit")]] = 1.0 / (1.0 + math.exp(-v))
        else:
            p[k] = v
    return p


def _queued_names():
    """Job names in the queue, or None when squeue could not be read."""
    r = _host_run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"])
    if r.returncode != 0:
        return None
    return set(r.stdout.split())


def _cell_dir(root, tag, method):
    parent = os.path.join(root, tag, method)
    if not os.path.isdir(parent):
        return None
    subs = sorted(d for d in os.listdir(parent)
                  if os.path.isdir(os.path.join(parent, d)))
    return os.path.join(parent, subs[0]) if subs else None


def finalize(args):
    """Retrain the best trial(s) on the sweep schedule and run the FULL suite.

    The search measures only its objective and constraint. The winner then gets
    everything a sweep cell gets: every task including MIA and denial of
    service, on every rung of the sweep's own checkpoint schedule, so it sits in
    the same tables and on the same axes as the sweep.

    Retraining uses the trial's exact settings and seed, so its trajectory
    reproduces the trial's up to GPU nondeterminism, and the |wm_q4| it reports
    at the trial's best step is a direct check on the search.

    Idempotent. Training is skipped for a winner that already has its last
    checkpoint or is queued, and an eval for a checkpoint that already has an
    evals/ directory or a queued job, so a rerun does only what is missing.
    """
    import optuna
    if not args.dry_run:
        _require_host_slurm()
        _require_noise()
    study = optuna.load_study(study_name=args.study or f"hpo-{args.method}",
                              storage=_storage(args))
    ok = [t for t in study.trials
          if t.state.name == "COMPLETE" and t.value is not None
          and t.user_attrs.get("feasible")]
    if not ok:
        sys.exit("no feasible completed trial to finalize")
    winners = sorted(ok, key=lambda t: t.value)[:args.top_k]
    root = args.final_root
    last = FINAL_RUNGS[-1]
    print(f"=== finalize {args.method}: top {len(winners)} of {len(ok)} feasible trial(s) ===")
    print(f"  root {root}, schedule {FINAL_RUNGS}, full suite incl. MIA and DoS")

    queued = set() if args.dry_run else _queued_names()
    if queued is None:
        sys.exit("squeue could not be read, so duplicates cannot be ruled out. Rerun.")

    tags = []
    for t in winners:
        params = _natural_params(t)
        tag = f"hpo-final-{args.method}-t{t.number:04d}"
        tags.append(tag)
        print(f"  trial {t.number}: |wm_q4|={t.value:.4f} at step "
              f"{t.user_attrs.get('best_step')}, c4 "
              f"+{t.user_attrs.get('c4_delta_pct', float('nan')):.2f}% in the search")
        for k, v in sorted(params.items()):
            print(f"      {k} = {v}")
        cell = _cell_dir(root, tag, args.method)
        if cell and os.path.isdir(os.path.join(cell, f"step-{last}")):
            print(f"    already trained: {cell}")
            continue
        if tag in queued:
            print("    already queued")
            continue
        tenv = spaces.env_for(args.method, params, args.steps, FINAL_RUNGS)
        spaces.validate(args.method, params, tenv)
        env = {}        # only what the job needs; see _host_run
        env.update(tenv)
        env.update(IDENTITY)
        env.update({"RUN_TAG": tag, "OUTPUT_ROOT": root})
        wt = args.time or trial_walltime(args.method, params, args.steps, [],
                                         args.tokens_per_hour, 0.0)
        cmd = ["sbatch", "-J", tag, f"--time={wt}", "--export=ALL",
               os.path.join(REPO, "internal", "lumi", "unlearn_cell.sh")]
        if args.dry_run:
            print(f"    [dry] {' '.join(cmd)}")
            print(f"          CKPT_STEPS={tenv['CKPT_STEPS']} MICRO_BATCH={tenv['MICRO_BATCH']} "
                  f"NO_TRAINER_STATE={tenv['NO_TRAINER_STATE']}")
            continue
        out = _host_run(cmd, env)
        if out.returncode != 0:
            print(f"    TRAIN SUBMIT FAILED: {out.stderr.strip()}", file=sys.stderr)
            continue
        print(f"    training submitted as job {out.stdout.strip().split()[-1]}, walltime {wt}")

    if args.dry_run:
        print("[dry] nothing submitted")
        return

    # Wait by NAME, so a restarted finalize also waits for jobs it did not
    # submit itself. An unreadable queue counts as still busy.
    while True:
        q = _queued_names()
        if q is not None and not [tag for tag in tags if tag in q]:
            break
        print("  winner(s) still training" if q is not None else "  squeue unreadable, waiting")
        time.sleep(args.poll)

    mia = os.path.join(PE, "hf", "mia-cache")
    off = "1" if (os.path.isdir(mia) and os.listdir(mia)) else "0"
    hub = os.path.join(os.environ.get("HF_HOME", os.path.join(PE, "hf")), "hub")
    judge = os.path.join(hub, "models--meta-llama--Meta-Llama-3-8B-Instruct", "snapshots")
    if off == "1" and not glob.glob(os.path.join(judge, "*", "*.safetensors")):
        sys.exit(f"no cached DoS judge under {judge} and evals run offline. Not submitting.")

    queued = _queued_names()
    if queued is None:
        sys.exit("squeue could not be read before submitting evals. Rerun to resume.")
    n = 0
    for tag in tags:
        cell = _cell_dir(root, tag, args.method)
        if cell is None:
            print(f"  {tag}: no cell directory, training did not run")
            continue
        cname = os.path.basename(cell)
        have = sorted((d for d in os.listdir(cell) if d.startswith("step-")),
                      key=lambda d: int(d[len("step-"):]))
        missing = [r for r in FINAL_RUNGS if f"step-{r}" not in have]
        if missing:
            print(f"  WARNING {tag}: no checkpoint at steps {missing}")
        for st in have:
            ck = os.path.join(cell, st)
            jn = f"pe-{tag}-{args.method}-{cname}-{st}"
            if os.path.isdir(os.path.join(ck, "evals")) or jn in queued:
                continue
            eenv = {}   # explicit only; MODEL in particular must never reach an eval
            eenv.update({"SKIP_MIA": "0", "SKIP_DOS": "0", "NOISE_DIR": NOISE_DIR_1B,
                         "EVAL_MAX_NUM_SEQS": EVAL_MAX_NUM_SEQS,
                         "INFERENCE_MAX_NUM_SEQS": INFERENCE_MAX_NUM_SEQS,
                         "MIA_CACHE_DIR": mia,
                         "MIA_REF_CACHE_DIR": os.path.join(mia, "ref"),
                         "HF_HUB_OFFLINE": off, "HF_DATASETS_OFFLINE": off})
            cmd = ["sbatch", "-J", jn, "-t", args.eval_time,
                   f"--export=ALL,CELL_DIR={cell},CKPT={ck},EVAL_OUT={os.path.join(ck, 'evals')}",
                   os.path.join(REPO, "internal", "lumi", "eval_pareto_cell.sh")]
            out = _host_run(cmd, eenv)
            if out.returncode != 0:
                print(f"  EVAL SUBMIT FAILED {jn}: {out.stderr.strip()}", file=sys.stderr)
                continue
            n += 1
    print(f"=== {n} full-suite eval job(s) submitted ===")
    print("  once they finish, export both together:")
    print(f"    python internal/uwiki/audit_configs.py  --output-root {root} "
          f"--out \"$PE_WORK/exports-hpo-final/results_configs.csv\"")
    print(f"    python internal/uwiki/export_results.py --output-root {root} "
          f"--out \"$PE_WORK/exports-hpo-final\" --tags 'hpo-final-*'")


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
    ap.add_argument("--tokens-per-hour", type=float, default=None,
                    help="override the per-method measured throughput in "
                         "spaces.THROUGHPUT, which is what sizes walltime")
    ap.add_argument("--eval-minutes", type=float, default=20.0,
                    help="minutes per rung evaluation, for sizing walltime")
    ap.add_argument("--output-root", default=os.path.join(PE, "hpo"))
    ap.add_argument("--storage", default=os.path.join(PE, "hpo", "optuna_journal.log"),
                    help="a file path selects Lustre-safe journal storage (default); "
                         "a URL such as sqlite:///... is passed to Optuna as is")
    ap.add_argument("--study", default=None)
    ap.add_argument("--finalize", action="store_true",
                    help="retrain the best feasible trial(s) on the sweep schedule "
                         "and run the FULL eval suite on every checkpoint")
    ap.add_argument("--top-k", type=int, default=1,
                    help="how many of the best trials --finalize retrains")
    ap.add_argument("--final-root", default=os.path.join(PE, "hpo-final"))
    ap.add_argument("--eval-time", default="12:00:00",
                    help="walltime per full-suite eval job in --finalize")
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
    if args.finalize:
        finalize(args)
        return 0
    search(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
