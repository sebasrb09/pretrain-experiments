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

    # 2. the anchors, once per model: the baseline and the counterfactual,
    #    evaluated with exactly the trials' settings (see forget_score.py)
    python internal/uwiki/hpo/optuna_hpo.py --method ce-u --anchors

    # 3. the search, once both anchor jobs have finished
    python internal/uwiki/hpo/optuna_hpo.py --method ce-u --max-parallel 8
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

# The rung ladder: the steps a trial checkpoints and evaluates at. It is the
# sweep's own 1,2,3,5,8,13,21,34,55 schedule. The pilot ran a coarse 3,8,21,55
# and the budget boundary fell BETWEEN rungs every time (trial 0: step 3 at
# +0.1%, step 8 at +10.3%), so no trial's in-budget optimum was measured. The
# early stop keeps the finer ladder affordable: a trial stops evaluating at its
# first rung over the cap. It also lines every search point up with a sweep cell.
DEFAULT_RUNGS = [1, 2, 3, 5, 8, 13, 21, 34, 55]

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

UTIL_CAP_PCT = "5.0"

# The objective is forget_score.F: forgetting on fictional knowledge, insertion
# likelihood and the watermark, each scaled by its own two anchors. The tag is
# part of every study and trial name, so a search under this objective can
# never resume the |wm_q4| pilot's study or reuse its trial directories.
OBJECTIVE_TAG = "f3"

# BASELINE is the model every method starts from (IDENTITY); the counterfactual
# is the same corpus with the targets removed, at the same pretraining step.
ANCHOR_MODELS = {
    "baseline": (IDENTITY["MODEL"], IDENTITY["REVISION"]),
    "deep-ignorance": ("sbordt/OLMo-2-1B-Unlearning", "stage1-step100000-tokens210B"),
}
DEFAULT_ANCHOR_ROOT = os.path.join(PE, "hpo", "anchors-" + OBJECTIVE_TAG)

# Two disjoint C4 files. The search's utility constraint uses a SECOND slice of
# C4's validation split (documents 2500-4999 of the stream), so the documents
# the paper reports utility on are never used to pick the winner. The slice was
# built from the first validation shard after checking that its documents
# 0-2499 are exactly the reporting file, and it shares no document with it.
HPO_C4_FILE = os.path.join(REPO, "resources", "validation-set", "c4_en_validation_hpo.jsonl")
REPORT_C4_FILE = os.path.join(REPO, "resources", "validation-set", "c4_en_validation.jsonl")

# What every evaluation in the SEARCH runs with, anchors and trials alike.
# forget_score.check_settings refuses a trial that differs in any of these or
# in the content of the C4 file.
#   EVAL_MAX_NUM_SEQS = INFERENCE_MAX_NUM_SEQS = 1: no padding anywhere.
#     Batched log-likelihoods were batch-dependent before the position_ids fix
#     (2026-09-20), and LUMI and ASC disagree ~3.6x on insertion likelihood for
#     the same model. Batch 1 takes padding out of the objective entirely.
#   IL_EXPERIMENT: the one insertion experiment the exporter reports. Each
#     experiment is sampled with its own fixed seed, so this equals that entry
#     of an all-experiments run at 1/57 of the cost.
#   HF_*_OFFLINE = 0: resolve online, as the pilot did, at <= 8 jobs at a time.
EVAL_ENV = {
    "NOISE_DIR": NOISE_DIR_1B,
    "NOISE_STD": "0.075",
    "EVAL_MAX_NUM_SEQS": "1",
    "INFERENCE_MAX_NUM_SEQS": "1",
    "C4_TASK_FILE": HPO_C4_FILE,
    "IL_EXPERIMENT": "knowledge-acquisition",
    "IL_MAX_TOKENS": "1000000",
    "HF_HUB_OFFLINE": "0",
    "HF_DATASETS_OFFLINE": "0",
}

# What the WINNERS' full suite runs with in --finalize, on the reporting C4
# file. Batch 1: the --anchors batch check showed batch 8 is wrong on LUMI
# (insertion 12.84 vs 3.60 on the same model), so every LUMI eval runs at 1.
FINAL_EVAL_ENV = {
    "NOISE_DIR": NOISE_DIR_1B,
    "NOISE_STD": "0.075",
    "EVAL_MAX_NUM_SEQS": "1",
    "INFERENCE_MAX_NUM_SEQS": "1",
    "C4_TASK_FILE": REPORT_C4_FILE,
    "IL_EXPERIMENT": "all",
    "IL_MAX_TOKENS": "1000000",
}


def _require_noise():
    if not glob.glob(os.path.join(NOISE_DIR_1B, "gaussian_poisoning_*.pkl")):
        sys.exit(f"no gaussian_poisoning_*.pkl in {NOISE_DIR_1B}: the watermark, which is "
                 "the objective, cannot be scored. Not launching.")


def _require_anchors(anchor_root):
    """The objective's scale must exist, and match EVAL_ENV, before any trial runs."""
    import forget_score as fs
    try:
        fs.check_settings(anchor_root, EVAL_ENV)
        a = fs.load_anchors(anchor_root, EVAL_ENV["IL_EXPERIMENT"])
    except (fs.AnchorError, OSError) as e:
        sys.exit(f"anchors not usable: {e}" + chr(10) +
                 "  Run --anchors first and wait for both anchor jobs to finish.")
    print(f"  anchors {anchor_root}")
    for pt in fs.POINTS:
        m = a[pt]
        print(f"    {pt:15s} c4 {m['c4']:.4f}  fk {m['fk']:.4e}  il {m['il']:.4f}  "
              f"wm_q4 {m['wm']:+.4f}")
    return a


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
# Measured on the pilot (four trials, fitted to within a minute): ~4 min of
# startup per trial, 10.7 min per rung for C4 + watermark, and 89M tokens/h of
# CE-U training. 0.15 h covers the startup with margin. The rung cost is
# --eval-minutes, which now also pays for fictional knowledge and one insertion
# experiment at batch 1, so it stays an estimate until the next pilot.
STARTUP_HOURS = 0.15


def _space_distributions(method, steps, max_batch):
    """The Optuna distributions of this method's space, at these settings."""
    import optuna
    tmp = optuna.create_study()          # in memory, never stored
    t = tmp.ask()
    spaces.suggest(t, method, steps, max_batch)
    return dict(t.distributions)


def _seed_study(study, args):
    """Copy another study's COMPLETE trials that fit this space into this (empty) one.

    For a new study over a NARROWER space (--max-batch 512 after a pilot that
    allowed 1024 and 2048). Changing the space inside the old study would not
    work: Optuna models a parameter only where every finished trial has the
    identical distribution, so the GP would silently stop learning batch size.
    Copied trials keep their value, constraint and user attributes (their
    "tokens" count toward this study's budget, since they were trained) and are
    re-registered with this space's distributions; a trial whose value lies
    outside the new space is not copied.
    """
    import optuna
    dists = _space_distributions(args.method, args.steps, args.max_batch)
    n = 0
    for name in args.seed_from.split(","):
      src = optuna.load_study(study_name=name, storage=_storage(args))
      src_obj = src.user_attrs.get("objective", "f3")
      if src_obj != args.objective:
          print(f"  seed: {name} maximises {src_obj}, not {args.objective}; not copied")
          continue
      for t in src.trials:
        if t.state.name != "COMPLETE":
            continue
        try:
            fits = set(t.params) == set(dists) and all(
                dists[k]._contains(dists[k].to_internal_repr(v)) for k, v in t.params.items())
        except (ValueError, TypeError):
            fits = False
        if not fits:
            print(f"  seed: {name} trial {t.number} lies outside this space, not copied")
            continue
        study.add_trial(optuna.trial.create_trial(
            params=t.params, distributions=dists, value=t.value,
            user_attrs=dict(t.user_attrs, seeded_from=f"{name}:{t.number}"),
            system_attrs=t.system_attrs))
        n += 1
    print(f"  seeded {n} completed trial(s) from {args.seed_from}")


def _enqueue(study, args):
    """Queue one configuration: a trial of another study, with --set overrides.

    For controls, e.g. --enqueue-from hpo-ce-u-f3:4 --set adam_beta1=0.9 reruns
    trial 4 with the pre-training beta1. Names are the natural ones; the Adam
    betas are registered as logits and converted here. Idempotent: a relaunch
    does not queue the same configuration twice.
    """
    import optuna
    name, _, no = args.enqueue_from.rpartition(":")
    src = optuna.load_study(study_name=name, storage=_storage(args)).trials[int(no)]
    params = dict(src.params)
    for kv in args.set:
        k, _, v = kv.partition("=")
        v = float(v)
        if k + "_logit" in params:
            k, v = k + "_logit", math.log(v / (1.0 - v))
        if k not in params:
            sys.exit(f"--set {kv}: {k} is not a parameter of {args.enqueue_from}")
        params[k] = int(v) if k == "batch_size" else v
    tag = " ".join([args.enqueue_from] + list(args.set))
    if any(t.user_attrs.get("enqueued_from") == tag for t in study.trials):
        print(f"  already enqueued: {tag}")
        return
    study.enqueue_trial(params, user_attrs={"enqueued_from": tag})
    print(f"  enqueued: {tag}")


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
    # STARTUP_HOURS: model load, forget-set tokenization and optimizer state,
    # none of which is in the measured per-step rate.
    hours = 1.6 * (STARTUP_HOURS + toks / eff + len(rungs) * eval_min / 60.0)
    hours = max(2.0, min(float(cap_h), hours))
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h:02d}:{m:02d}:00"


class StaleTrialDir(RuntimeError):
    pass


def _submit(method, params, trial_no, steps, rungs, out_root, time_limit, dry,
            study_name, anchor_root, objective="f3"):
    # The study name carries the objective tag, so no two studies share a tag.
    tag = f"{study_name}-t{trial_no:04d}"
    if not dry and os.path.exists(os.path.join(out_root, tag)):
        # hpo_trial.sh refuses this too, but by then a GCD has been queued for it.
        raise StaleTrialDir(f"{os.path.join(out_root, tag)} already exists: this study name "
                            "was used before. Pass a new --study rather than reuse its "
                            "trial directories.")
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
        "ANCHOR_ROOT": anchor_root,
        "UTIL_CAP_PCT": UTIL_CAP_PCT,
        "HPO_OBJECTIVE": objective,
    })
    env.update(EVAL_ENV)
    env.update(IDENTITY)
    cmd = ["sbatch", "-J", tag, f"--time={time_limit}", "--export=ALL",
           os.path.join(HERE, "hpo_trial.sh")]
    if dry:
        print(f"  [dry] {tag}  " + " ".join(
            f"{k}={env[k]}" for k in sorted(spaces.env_for(method, params, steps, rungs))))
        return None, tag
    global _LAST_SUBMIT_ERROR
    out = _host_run(cmd, env)
    if out.returncode != 0:
        _LAST_SUBMIT_ERROR = out.stderr
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
    import forget_score as fs
    from optuna.samplers import GPSampler

    if not args.dry_run:
        _require_host_slurm()
        _require_noise()
        _require_anchors(args.anchor_root)
    storage = _storage(args)

    def constraints_func(trial):
        # Positive is a violation. A trial whose every rung blew the utility cap
        # is still informative, since it tells the GP where the wall is.
        c = trial.user_attrs.get("constraint")
        return [1.0 if c is None else float(c)]

    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        load_if_exists=True,
        # MAXIMIZE F, the three-task forgetting score in [0, 1]: 0 is the
        # baseline, 1 is the counterfactual on all three tasks. See
        # forget_score.py; summarize_trial.py reports it as "objective".
        direction="maximize",
        sampler=GPSampler(
            n_startup_trials=args.startup,    # uniform random until this many land
            constraints_func=constraints_func,
            seed=args.seed,
        ),
    )

    # Every study records what it maximises. One made before objectives had
    # names and holding trials is f3.
    have = study.user_attrs.get("objective")
    if have is None:
        have = "f3" if study.trials else args.objective
        study.set_user_attr("objective", have)
    if have != args.objective:
        sys.exit(f"study {study.study_name} maximises {have!r}, not {args.objective!r}: "
                 "pass --objective " + have + " to continue it, or a new --study")
    print(f"  objective {args.objective}: mean progress on {', '.join(fs.OBJECTIVES[args.objective])}")
    if args.seed_from and not study.trials:
        _seed_study(study, args)
    if args.enqueue_from:
        _enqueue(study, args)

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

    # RESUMING. A fixed sampler seed makes every launch redraw the same random
    # configurations: the CE-U pilot's trials 4-6 repeated trials 0-2 exactly.
    # Seed by the number of trials already in the study instead.
    if study.trials:
        study.sampler = GPSampler(n_startup_trials=args.startup,
                                  constraints_func=constraints_func,
                                  seed=args.seed + len(study.trials))
    # Trials a previous driver left RUNNING (it was stopped while they were in
    # flight) are adopted: still queued -> polled as usual; finished -> their
    # hpo_result.json is harvested on the first pass, or they are closed FAIL.
    # Without this a finished trial's result was never recorded.
    if not args.dry_run:
        for t in study.trials:
            if t.state.name == "RUNNING" and t.user_attrs.get("job"):
                handle = optuna.trial.Trial(study, t._trial_id)
                inflight[t.number] = (handle, t.user_attrs["job"],
                                      t.user_attrs.get("tag", f"{study.study_name}-t{t.number:04d}"),
                                      t.user_attrs.get("tokens", 0))
                print(f"  adopting trial {t.number} (job {t.user_attrs['job']}) from an earlier launch")
    submit_failures = 0
    waiting = False      # the user's queue is at --submit-cap: wait for room, do not ask
    # Trials that ran but returned nothing. Three in a row means something
    # systematic (a missing input, a broken path), and continuing would spend
    # the whole budget on trials that cannot produce an objective.
    empty_results = 0

    while True:
        # Fill the queue while there is budget and room.
        while len(inflight) < args.max_parallel:
            if spent >= args.budget_tokens:
                break
            if not args.dry_run:
                n_queued = _queue_size()
                if n_queued is not None and n_queued >= args.submit_cap:
                    if not waiting:
                        print(f"  queue holds {n_queued} of your jobs (cap {args.submit_cap}); "
                              "waiting for room")
                    waiting = True
                    break
            waiting = False
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
            try:
                job, tag = _submit(args.method, params, trial.number, args.steps,
                                   args.rungs, args.output_root, wt, args.dry_run,
                                   study.study_name, args.anchor_root, args.objective)
            except StaleTrialDir as e:
                # Close the asked trial first, so the study is not left with a
                # trial that is RUNNING forever.
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                sys.exit(f"ABORT: {e}")
            if args.dry_run:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                spent += cost
                continue
            if job is None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                if not args.dry_run and _queue_full_error(_LAST_SUBMIT_ERROR):
                    # A full queue, not a broken submission: the trial was never
                    # run (no compute, no budget), so wait for room rather than
                    # count it toward the abort below.
                    print("  the scheduler's per-user job limit was reached; waiting for room")
                    waiting = True
                    break
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

        if not inflight and not waiting:
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
                             f"{study.study_name}-t*.out before relaunching. Trials still "
                             "in the queue are not cancelled.")
            else:
                best = res["best"]
                trial.set_user_attr("constraint", res["constraint"])
                trial.set_user_attr("best_step", best["step"])
                trial.set_user_attr("c4_delta_pct", best["c4_delta_pct"])
                trial.set_user_attr("feasible", res["feasible"])
                for k in ("p_fk", "p_il", "p_wm"):
                    if best.get(k) is not None:
                        trial.set_user_attr(k, best[k])
                study.tell(trial, res["objective"])
                empty_results = 0
                flag = "" if res["feasible"] else "  INFEASIBLE"
                parts = ", ".join(f"{t} {best[f'p_{t}']:.3f}" for t in ("fk", "il", "wm")
                                  if best.get(f"p_{t}") is not None)
                print(f"  trial {no}: F={res['objective']:.4f} at step {best['step']} "
                      f"({parts}), c4 {best['c4_delta_pct']:+.2f}%{flag}")
            del inflight[no]

    print("\n=== done ===")
    feas = [t for t in study.trials
            if t.value is not None and t.user_attrs.get("feasible")]
    if feas:
        b = max(feas, key=lambda t: t.value)
        print(f"  best in-budget trial {b.number}: F={b.value:.4f} at step "
              f"{b.user_attrs.get('best_step')} (fk {b.user_attrs.get('p_fk', float('nan')):.3f}, "
              f"il {b.user_attrs.get('p_il', float('nan')):.3f}, "
              f"wm {b.user_attrs.get('p_wm', float('nan')):.3f}; 0 = baseline, 1 = counterfactual)")
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


# LUMI's per-user cap on submitted jobs (small-g: MaxSubmit 210) is shared by
# everything the user has queued. launch_pareto_evals.sh fills the queue up to
# its SUBMIT_CAP of 195, so a driver capped a little higher always finds room
# between the two, and a big re-evaluation can no longer starve the search.
_LAST_SUBMIT_ERROR = ""


def _queue_size():
    """Number of this user's jobs in the queue, or None when squeue could not be read."""
    r = _host_run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"])
    if r.returncode != 0:
        return None
    return len([line for line in r.stdout.splitlines() if line.strip()])


def _queue_full_error(err):
    """The scheduler refused a job because the user's queue is at its limit."""
    return "MaxSubmitJob" in err or "job submit limit" in err


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


def anchors(args):
    """Submit the two anchor evaluations with EVAL_ENV, plus one diagnostic.

    settings.json is written first, so every trial can verify it is measured
    exactly like the anchors. Idempotent: the eval body's .done markers skip
    finished tasks, and a settings.json that disagrees with EVAL_ENV is refused
    rather than overwritten, because anchors measured two ways cannot be mixed.

    The diagnostic re-measures the baseline's fk and il at batch 8, the sweep's
    INFERENCE_MAX_NUM_SEQS, into <anchor-root>-batchcheck. If it matches the
    batch-1 anchor, batched log-likelihoods are batch-invariant on LUMI and the
    sweep's fk and il columns stand. If not, they depend on padding.
    """
    import forget_score as fs
    root = args.anchor_root
    models = {pt: {"model": m, "revision": r} for pt, (m, r) in ANCHOR_MODELS.items()}
    sp = os.path.join(root, "settings.json")
    if os.path.exists(sp):
        try:
            fs.check_settings(root, EVAL_ENV)
        except fs.AnchorError as e:
            sys.exit(f"{e}" + chr(10) + f"  {root} holds anchors measured differently; "
                     "use a new --anchor-root.")
        if fs.read_settings(root)["models"] != models:
            sys.exit(f"{sp} names different anchor models; use a new --anchor-root.")
    if not args.dry_run:
        _require_host_slurm()
        _require_noise()
        if not os.path.exists(sp):
            fs.write_settings(root, EVAL_ENV, models)
            print(f"  wrote {sp}")

    base = {"SKIP_PPL": "0", "SKIP_FK": "0", "SKIP_IL": "0", "SKIP_GW": "0",
            "SKIP_VM": "1", "SKIP_BM": "1", "SKIP_PE": "1", "SKIP_MIA": "1",
            "SKIP_DOS": "1", "SKIP_NEWS": "1", "SKIP_MATH": "1", "FORCE_EVAL": "0"}
    jobs = []
    for pt, (model, rev) in ANCHOR_MODELS.items():
        env = dict(base)
        env.update(EVAL_ENV)
        env.update({"MODEL": model, "REVISION": rev,
                    "EVAL_OUT": os.path.join(root, pt, "step-0")})
        jobs.append((f"hpo-anchor-{OBJECTIVE_TAG}-{pt}", env))
    diag = dict(base)
    diag.update(EVAL_ENV)
    diag.update({"SKIP_PPL": "1", "SKIP_GW": "1", "INFERENCE_MAX_NUM_SEQS": "8",
                 "MODEL": ANCHOR_MODELS["baseline"][0], "REVISION": ANCHOR_MODELS["baseline"][1],
                 "EVAL_OUT": os.path.join(root + "-batchcheck", "baseline-mns8", "step-0")})
    jobs.append((f"hpo-anchor-{OBJECTIVE_TAG}-batchcheck", diag))

    script = os.path.join(REPO, "internal", "lumi", "eval_pareto_cell.sh")
    for name, env in jobs:
        cmd = ["sbatch", "-J", name, "--time=02:00:00", "--export=ALL", script]
        if args.dry_run:
            print(f"  [dry] {name}: " + " ".join(f"{k}={env[k]}" for k in sorted(env)))
            continue
        out = _host_run(cmd, env)
        if out.returncode != 0:
            sys.exit(f"ANCHOR SUBMIT FAILED for {name}: {out.stderr.strip()}")
        print(f"  submitted {name} as job {out.stdout.strip().split()[-1]}")
    print("  when all three have finished:")
    print(f"    python internal/uwiki/hpo/forget_score.py show --anchor-root {root}")


def finalize(args):
    """Retrain the best trial(s) on the sweep schedule and run the FULL suite.

    The search measures only its objective and constraint. The winner then gets
    everything a sweep cell gets: every task including MIA and denial of
    service, on every rung of the sweep's own checkpoint schedule, so it sits in
    the same tables and on the same axes as the sweep.

    Retraining uses the trial's exact settings and seed, so its trajectory
    reproduces the trial's up to GPU nondeterminism. Its evaluations use
    FINAL_EVAL_ENV (the sweep's settings and the REPORTING C4 file), not the
    search's, so the winner is reported on documents the search never saw.

    Idempotent. Training is skipped for a winner that already has its last
    checkpoint or is queued, and an eval for a checkpoint that already has an
    evals/ directory or a queued job, so a rerun does only what is missing.
    """
    import optuna
    if not args.dry_run:
        _require_host_slurm()
        _require_noise()
    study = optuna.load_study(study_name=args.study, storage=_storage(args))
    ok = [t for t in study.trials
          if t.state.name == "COMPLETE" and t.value is not None
          and t.user_attrs.get("feasible")]
    if not ok:
        sys.exit("no feasible completed trial to finalize")
    # direction=maximize: the largest F first.
    winners = sorted(ok, key=lambda t: t.value, reverse=True)[:args.top_k]
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
        tag = f"{args.study}-final-t{t.number:04d}"
        tags.append(tag)
        print(f"  trial {t.number}: F={t.value:.4f} at step "
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

    mia = os.path.join(PE, "hf", "mia-cache-b1")     # batch-1 reference scores, as the cells
    off = "1" if (os.path.isdir(mia) and os.listdir(mia)) else "0"
    hub = os.path.join(os.environ.get("HF_HOME", os.path.join(PE, "hf")), "hub")
    judge = os.path.join(hub, "models--meta-llama--Meta-Llama-3-8B-Instruct", "snapshots")
    if off == "1" and not glob.glob(os.path.join(judge, "*", "*.safetensors")):
        sys.exit(f"no cached DoS judge under {judge} and evals run offline. Not submitting.")

    queued = _queued_names()
    if queued is None:
        sys.exit("squeue could not be read before submitting evals. Rerun to resume.")
    # The full suite in THREE jobs per checkpoint, each with the settings its
    # tasks were measured with on the sweep cells and the anchors (2026-10-08/09):
    #   b1     default attention, batch 1: C4, knowledge, insertion, watermark,
    #          verbatim and MIA (12 paper conditions; batch 32 reads lower on LUMI)
    #   e8     eager attention, batch 8: contamination, news, prompt extraction, DoS
    #   e8math eager attention, batch 8: iGSM ops 1, 3, 5 (the longest, on its own)
    # Within a task, the winner, the cells and the anchors must share one setting.
    skip_all = {f"SKIP_{k}": "1" for k in ("PPL", "FK", "IL", "GW", "VM", "BM", "PE", "MIA", "DOS", "NEWS", "MATH")}
    eager = {"INFERENCE_MAX_NUM_SEQS": "8", "INFERENCE_ATTN_IMPL": "eager", "ALLOW_ROCM_PADDED_BATCHES": "1"}
    groups = [
        ("b1", "12:00:00", {"SKIP_PPL": "0", "SKIP_FK": "0", "SKIP_IL": "0", "SKIP_GW": "0", "SKIP_VM": "0",
                            "SKIP_MIA": "0", "MIA_CONDITIONS": "paper", "MIA_BATCH": "1",
                            "MIA_CACHE_DIR": mia, "MIA_REF_CACHE_DIR": os.path.join(mia, "ref")}),
        ("e8", "08:00:00", dict(eager, SKIP_BM="0", SKIP_NEWS="0", SKIP_PE="0", SKIP_DOS="0",
                                BM_SPLITS="0 1 2 3 4 5 6 7 8", NEWS_N="0", NEWS_N_GENERATE="0",
                                PE_QUERIES="1000", PE_GENERATIONS="1", DOS_QUERIES="1000")),
        ("e8math", "16:00:00", dict(eager, SKIP_MATH="0", MATH_OPS="1 3 5")),
    ]
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
            for gname, gtime, genv in groups:
                jn = f"pe-{tag}-{args.method}-{cname}-{st}-{gname}"
                if jn in queued:
                    continue      # a rerun resubmits the rest; finished tasks skip on their .done markers
                eenv = {}   # explicit only; MODEL in particular must never reach an eval
                eenv.update(FINAL_EVAL_ENV)
                eenv.update(skip_all)
                eenv.update(genv)
                eenv.update({"FORCE_EVAL": "0", "HF_HUB_OFFLINE": off, "HF_DATASETS_OFFLINE": off})
                cmd = ["sbatch", "-J", jn, "-t", args.eval_time or gtime,
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
          f"--out \"$PE_WORK/exports-hpo-final\" --tags '{args.study}-final-*'")


def main():
    import forget_score as fs
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
    ap.add_argument("--submit-cap", type=int, default=205,
                    help="wait while you have this many jobs queued (LUMI small-g MaxSubmit is "
                         "210; launch_pareto_evals.sh stops at 195, so the search keeps room)")
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
                    help="minutes per rung evaluation (C4, fictional knowledge, one "
                         "insertion experiment, watermark), for sizing walltime")
    ap.add_argument("--output-root", default=os.path.join(PE, "hpo"))
    ap.add_argument("--storage", default=os.path.join(PE, "hpo", "optuna_journal.log"),
                    help="a file path selects Lustre-safe journal storage (default); "
                         "a URL such as sqlite:///... is passed to Optuna as is")
    ap.add_argument("--objective", choices=sorted(fs.OBJECTIVES), default="wm",
                    help="what the search maximises: wm (the Gaussian poison alone, the "
                         "default since 2026-10-09) or f3 (mean of knowledge, insertion "
                         "and the poison). Recorded in the study; a mismatch is refused.")
    ap.add_argument("--study", default=None,
                    help="default hpo-<method>-<objective>")
    ap.add_argument("--seed-from", default=None,
                    help="copy the COMPLETE trials of these studies (comma separated) that "
                         "fit the current space into a new, empty --study (e.g. after "
                         "narrowing --max-batch)")
    ap.add_argument("--enqueue-from", default=None,
                    help="STUDY:TRIAL to run again first, with --set overrides (a control)")
    ap.add_argument("--set", action="append", default=[],
                    help="NAME=VALUE override for --enqueue-from, natural names "
                         "(adam_beta1=0.9); repeatable")
    ap.add_argument("--anchor-root", default=DEFAULT_ANCHOR_ROOT,
                    help="baseline and counterfactual evaluated with EVAL_ENV")
    ap.add_argument("--anchors", action="store_true",
                    help="submit the anchor evaluations (and the batch check), then exit")
    ap.add_argument("--finalize", action="store_true",
                    help="retrain the best feasible trial(s) on the sweep schedule "
                         "and run the FULL eval suite on every checkpoint")
    ap.add_argument("--top-k", type=int, default=1,
                    help="how many of the best trials --finalize retrains")
    ap.add_argument("--final-root", default=os.path.join(PE, "hpo-final"))
    ap.add_argument("--eval-time", default=None,
                    help="walltime per full-suite eval job in --finalize")
    args = ap.parse_args()

    if args.study is None:
        args.study = f"hpo-{args.method}-{args.objective}"
    if args.anchors:
        anchors(args)
        return 0

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
