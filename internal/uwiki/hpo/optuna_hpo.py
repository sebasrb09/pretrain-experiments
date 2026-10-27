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

    # 3b. the full-range search (2026-10-15): scored up to C4 33, valued by the
    #     area under the trajectory, wider space, starting from the first search
    python internal/uwiki/hpo/optuna_hpo.py --method npo --objective wmarea --space v2 \
        --warm-start hpo-npo-wm --startup 16 --max-trials 100 --budget-tokens 4e9 --max-batch 512

    # 3c. after an outage: the same command plus --rescore scores the checkpoints of
    #     trials that trained and then died, instead of training them again

    # 4. the full suite on the searches' frontier checkpoints in a plan
    #    (forget_score.py frontier); retrains only what is not on disk
    python internal/uwiki/hpo/optuna_hpo.py --frontier PLAN.csv --dry-run
"""
import argparse
import csv
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

# Everything that differs between model sizes, in ONE place (2026-10-13, the
# 2.7B transfer). --size picks a row and nothing size-dependent is read from
# anywhere else, so a 1B value cannot reach a 2.7B job by default:
#   identity        the model a method starts from and its Adam moments, which
#                   must be the same size (the wrong moments fail silently)
#   counterfactual  the Retraining model, same corpus without the insertions
#   noise           the Gaussian-poison vectors, one set per embedding width
#   mia             the canary test's cache of reference-model scores
#   hidden          the model's hidden size, checked on every checkpoint before
#                   an evaluation is submitted
#   slow            how much longer a job takes than at 1B (time limits only)
SIZES = {
    "1B": {
        "identity": IDENTITY, "counterfactual": "sbordt/OLMo-2-1B-Unlearning",
        "noise": NOISE_DIR_1B, "mia": "mia-cache-b1", "hidden": 2048, "slow": 1.0,
        "tag": "", "final_root": os.path.join(PE, "hpo-final"),
    },
    "2.7B": {
        "identity": {
            "MODEL": "sbordt/OLMo-2-2.7B-Exp-Unlearning",
            "REVISION": "stage1-step100000-tokens210B",
            "OPTIM_REPO": "sbordt/OLMo-2-2.7B-Exp-Unlearning",
            "OPTIM_REVISION": "step100000-unsharded",
            "OLMO_CONFIG": "",       # the wrapper takes it from OPTIM_REPO@OPTIM_REVISION
        },
        "counterfactual": "sbordt/OLMo-2-2.7B-Unlearning",
        "noise": os.path.join(PE, "noise-vectors", "OLMo-2-2.7B-Exp"), "mia": "mia-cache-b1-2.7B",
        "hidden": 2880, "slow": 3.0,
        "tag": "-2.7B", "final_root": os.path.join(PE, "hpo-final-2.7B"),
    },
}


def _scaled(hms, k):
    """A HH:MM:SS limit times k, at most small-g's three days."""
    h, m, sec = (int(x) for x in hms.split(":"))
    t = min(72 * 3600, int((h * 3600 + m * 60 + sec) * k))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:00"


def _other_sizes(text, size):
    return [o for o in SIZES if o != size and f"OLMo-2-{o}" in text]


def _check_training_env(env, size):
    """A training job of `size` must name that size's model and moments and nothing of another size."""
    bad = [f"{k}={v}" for k, v in sorted(env.items()) if _other_sizes(str(v), size)]
    for k in ("MODEL", "OPTIM_REPO"):
        if f"OLMo-2-{size}" not in env.get(k, ""):
            bad.append(f"{k}={env.get(k)!r} is not an OLMo-2-{size} repository")
    if bad:
        sys.exit(f"REFUSING to submit a {size} training job with: " + "; ".join(bad))


def _check_targets(targets, size):
    """Every evaluation target must be a model of `size`.

    A checkpoint or model folder is checked by the hidden size in its
    config.json, a Hub repository by its name. The poison vectors and the
    canary reference scores are per size, so a model of another size would be
    scored against the wrong ones, and some of those evaluations skip quietly.
    """
    want, bad = SIZES[size]["hidden"], []
    for t in targets:
        if _other_sizes(t["exports"], size):
            bad.append(f"{t['what']}: {t['exports'].split(',')[0]} names another size")
        if t.get("repo") and f"OLMo-2-{size}" not in t["repo"]:
            bad.append(f"{t['what']}: {t['repo']} is not an OLMo-2-{size} repository")
        cfg = os.path.join(t["ckpt"], "config.json") if t.get("ckpt") else None
        if cfg and os.path.exists(cfg):
            try:
                with open(cfg, encoding="utf-8") as fh:
                    got = json.load(fh).get("hidden_size")
            except (OSError, ValueError):
                continue        # unreadable: the eval job itself fails loudly on it
            if got != want:
                bad.append(f"{t['what']}: hidden size {got} in {cfg}, a {size} model has {want}")
    if bad:
        sys.exit(f"REFUSING to evaluate as {size}:\n  " + "\n  ".join(bad[:12]))


UTIL_CAP_PCT = "5.0"     # the first searches' cap; per objective: forget_score.CAP_PCT

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
# file. Batch 1 here; finalize() raises it to eager batch 8 for the generation
# tasks, as on the cells. Insertion on the one experiment the export reads, as
# on the cells, the anchors and the SGD control: the cap is per experiment, so
# "all" gives the same number at ~70 min more per checkpoint.
FINAL_EVAL_ENV = {
    "NOISE_DIR": NOISE_DIR_1B,
    "NOISE_STD": "0.075",
    "EVAL_MAX_NUM_SEQS": "1",
    "INFERENCE_MAX_NUM_SEQS": "1",
    "C4_TASK_FILE": REPORT_C4_FILE,
    "IL_EXPERIMENT": "knowledge-acquisition",
    "IL_MAX_TOKENS": "1000000",
}


def _require_noise(size="1B"):
    noise = SIZES[size]["noise"]
    if not glob.glob(os.path.join(noise, "gaussian_poisoning_*.pkl")):
        sys.exit(f"no gaussian_poisoning_*.pkl in {noise}: the watermark, which is "
                 "the objective, cannot be scored. Not launching."
                 + ("" if size == "1B" else f" Build the {size} vectors with internal/lumi/build_noise_dir.sh."))


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

    def suggest_int(self, name, low, high):
        v = self.rng.randint(low, high)
        self.params[name] = v
        return v


def plan(method, budget, steps, rungs, n_show, max_batch, tph, emin, space="v1"):
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
    for b in [x for x in spaces.batch_choices(method, space) if x <= max_batch]:
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
        p = spaces.suggest(t, method, steps, max_batch, space)
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


def _space_distributions(method, steps, max_batch, space="v1"):
    """The Optuna distributions of this method's space, at these settings."""
    import optuna
    tmp = optuna.create_study()          # in memory, never stored
    t = tmp.ask()
    spaces.suggest(t, method, steps, max_batch, space)
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
    dists = _space_distributions(args.method, args.steps, args.max_batch, args.space)
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


def _warm_start(study, args, busy=()):
    """Queue the configurations on an earlier study's frontier as this study's first trials.

    For the full-range searches (2026-10-15): the first searches spent most or
    all of their budget on random startup trials (SatImp and SimNPO never reached
    the GP), so a new study starts from what they found instead of from
    scratch. In space v2, the best configurations the space was derived from
    (spaces.v2_spec, by the stage-2 objective, so inside the space by
    construction); otherwise trials with a point on the source's own frontier
    (forget_score.frontier_trials, 0.25 TPR points), spread along it. Up to
    --warm-max of them. Idempotent: a relaunch queues nothing twice. They count
    toward --startup.
    """
    import optuna
    import forget_score as fs
    dists = _space_distributions(args.method, args.steps, args.max_batch, args.space)
    # queued once, unless that attempt FAILED (the 2026-10-18 shutdown killed every warm-start
    # trial of NPO, SatImp and SimNPO): those are queued again, except the ones --rescore is
    # scoring (busy), which come back without training; search() re-queues one that does not
    have = {t.user_attrs.get("enqueued_from") for t in study.trials if t.state.name != "FAIL"} | set(busy)
    for name in args.warm_start.split(","):
        src = optuna.load_study(study_name=name, storage=_storage(args))
        trials = {t.number: t for t in src.trials}
        spec = spaces.v2_spec(args.method) if args.space == "v2" else {}
        if spec.get("study") == name:
            picked = [int(n) for n in spec["best_trials"]][:args.warm_max]
        else:
            picked = fs.frontier_trials(args.output_root, name, args.method, n=args.warm_max)
        n = 0
        for no in picked:
            t, tag = trials.get(no), f"warm {name}:{no}"
            if t is None or tag in have:
                continue
            try:
                fits = set(t.params) == set(dists) and all(
                    dists[k]._contains(dists[k].to_internal_repr(v)) for k, v in t.params.items())
            except (ValueError, TypeError):
                fits = False
            if not fits:
                print(f"  warm start: {name} trial {no} lies outside this space, not queued")
                continue
            study.enqueue_trial(dict(t.params), user_attrs={"enqueued_from": tag})
            n += 1
        print(f"  warm start: {n} configuration(s) of {name} queued "
              f"(trials {' '.join(map(str, picked)) or 'none'})")


def _unscored_steps(cell):
    """The rung checkpoints in a trial's cell directory without a watermark score."""
    out = []
    for d in glob.glob(os.path.join(cell, "step-*")):
        s = os.path.basename(d)[len("step-"):]
        if s.isdigit() and os.path.isdir(d) and not os.path.exists(
                os.path.join(cell, "evals", f"step-{s}", "gaussian_watermark.done")):
            out.append(int(s))
    return sorted(out)


def _trained_steps(cell):
    """The last optimizer step a trial logged (its metrics.jsonl), or None."""
    try:
        with open(os.path.join(cell, "metrics.jsonl"), encoding="utf-8") as fh:
            last = [line for line in fh if line.strip()][-1]
        return int(json.loads(last)["optimizer_step"])
    except (OSError, IndexError, KeyError, ValueError, TypeError):
        return None


def _complete(res, trained, steps):
    """Whether a dead trial's result measures what its objective needs: it trained to the
    end, or it crossed the C4 cap before it stopped (the objective ends at the cap). A
    run the network cut short below the cap has a truncated trajectory, whose value is
    only a lower bound and would steer the GP away from a configuration unfairly."""
    if trained is not None and trained >= steps:
        return True
    cap = res.get("util_cap_pct")
    return cap is not None and any(p.get("c4_delta_pct") is not None and p["c4_delta_pct"] > cap
                                   for p in res.get("points", []))


def _tell_result(study, trial, no, res):
    """Close a running trial with its hpo_result.json; returns the line to print."""
    best = res["best"]
    trial.set_user_attr("constraint", res["constraint"])
    trial.set_user_attr("best_step", best["step"])
    trial.set_user_attr("c4_delta_pct", best["c4_delta_pct"])
    trial.set_user_attr("feasible", res["feasible"])
    for k in ("p_fk", "p_il", "p_wm"):
        if best.get(k) is not None:
            trial.set_user_attr(k, best[k])
    study.tell(trial, res["objective"])
    flag = "" if res["feasible"] else "  INFEASIBLE"
    parts = ", ".join(f"{t} {best[f'p_{t}']:.3f}" for t in ("fk", "il", "wm")
                      if best.get(f"p_{t}") is not None)
    return (f"  trial {no}: F={res['objective']:.4f} at step {best['step']} "
            f"({parts}), c4 {best['c4_delta_pct']:+.2f}%{flag}")


def _record_rescored(study, t, res, job=None):
    """Add dead trial t back to the study as a new COMPLETE trial carrying result res."""
    import optuna
    best = res["best"]
    c = res.get("constraint")
    attrs = dict(t.user_attrs, rescored_from=t.number, constraint=c, best_step=best["step"],
                 c4_delta_pct=best["c4_delta_pct"], feasible=res["feasible"])
    for k in ("p_fk", "p_il", "p_wm"):
        if best.get(k) is not None:
            attrs[k] = best[k]
    if job:
        attrs["rescore_job"] = job
    # add_trial does not call the sampler, so the constraint goes where GPSampler reads it
    # (the key and format of Optuna's _process_constraints_after_trial, read by 4.x and 5.x);
    # without it the GP refuses the study ("number of constraints must be the same")
    study.add_trial(optuna.trial.create_trial(
        params=t.params, distributions=t.distributions, value=res["objective"], user_attrs=attrs,
        system_attrs={"constraints": (1.0 if c is None else float(c),)}))


def _rescore(study, args):
    """Bring back the trials an outage killed after they trained (--rescore, 2026-10-19).

    The 2026-10-18 shutdown ended 8-11 trials of each search in a timeout after
    training, their checkpoints on disk and unscored. Optuna cannot reopen a FAILED
    trial, so each comes back as a new COMPLETE trial with the same parameters and
    attributes (rescored_from: the dead trial's number; report.py lists the copy
    only), and is charged its tokens, which a FAILED trial is not. One whose result
    is already on disk (pe_tools.sh hpo_salvage) and that has nothing left unscored
    is recorded at once; one with unscored checkpoints is scored by hpo_trial.sh
    SCORE_ONLY=1, which adds them to whatever it scored before dying. A trial that
    died before training (no checkpoint) stays FAILED, and so does one whose
    training stopped early below the C4 cap (_complete). Idempotent: each scoring
    job is recorded in the study (rescore_t<N>), so a relaunch adopts it instead of
    submitting it again, and one that produced nothing is not resubmitted.
    Returns {dead trial number: (trial, job, tag, trained steps)} for the jobs still scoring.
    """
    if args.objective not in ("wm", "wmarea"):
        sys.exit("--rescore reads the watermark's .done markers: wm and wmarea studies only")
    back = {t.user_attrs.get("rescored_from") for t in study.trials}
    pending, added = {}, 0
    for t in study.trials:
        if t.state.name != "FAIL" or not t.user_attrs.get("job") or t.number in back:
            continue
        tag = t.user_attrs.get("tag", f"{study.study_name}-t{t.number:04d}")
        cells = sorted(d for d in glob.glob(os.path.join(args.output_root, tag, args.method, "*"))
                       if os.path.isdir(d))
        if not cells:
            continue                    # died before training: nothing to score
        todo = _unscored_steps(cells[0])
        trained = _trained_steps(cells[0])
        job = study.user_attrs.get(f"rescore_t{t.number}")
        if job and not args.dry_run and _running(job):
            pending[t.number] = (t, job, tag, trained)
            print(f"  rescore: adopting the scoring of trial {t.number} (job {job})")
            continue
        if job or not todo:
            res = _harvest(args.output_root, tag, args.method)
            if res is None or res.get("best") is None:
                if job:
                    print(f"  rescore: the scoring job {job} of trial {t.number} produced no result, "
                          f"not resubmitted; read {tag}_{job}.out")
            elif not _complete(res, trained, args.steps):
                print(f"  rescore: trial {t.number} stopped training at step {trained} below the C4 cap: "
                      "left failed (a truncated trajectory)")
            else:
                if not args.dry_run:
                    _record_rescored(study, t, res, job)
                added += 1
                print(f"  rescore: trial {t.number} {'would be ' if args.dry_run else ''}recorded from "
                      f"its result on disk (F={res['objective']:.4f})")
            continue
        hours = max(2.0, 1.6 * (0.25 + len(todo) * args.eval_minutes / 60.0))
        minutes = int(math.ceil(hours * 60))
        job, _ = _submit(args.method, t.user_attrs["params"], t.number, args.steps, args.rungs,
                         args.output_root, f"{minutes // 60:02d}:{minutes % 60:02d}:00",
                         args.dry_run, study.study_name, args.anchor_root, args.objective,
                         score_only=True)
        if job is None:
            continue                    # dry run, or sbatch refused: the next launch tries again
        study.set_user_attr(f"rescore_t{t.number}", job)
        pending[t.number] = (t, job, tag, trained)
        print(f"    to score trial {t.number}'s steps {' '.join(map(str, todo))}"
              + ("" if trained is not None and trained >= args.steps else
                 f" (training stopped at step {trained}: kept only if it crossed the C4 cap)"))
    print(f"  rescore: {added} dead trial(s) recorded, {len(pending)} being scored")
    return pending


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
            study_name, anchor_root, objective="f3", score_only=False, fresh=False):
    import forget_score as fs
    # The study name carries the objective tag, so no two studies share a tag.
    tag = f"{study_name}-t{trial_no:04d}"
    if not dry and not score_only and os.path.exists(os.path.join(out_root, tag)):
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
        "UTIL_CAP_PCT": fs.CAP_PCT[objective],
        "HPO_OBJECTIVE": objective,
        # 1: score the dead trial's checkpoints, no training (--rescore). Always sent, so a
        # value left in the bridge's shell cannot turn a new trial into a scoring job.
        "SCORE_ONLY": "1" if score_only else "0",
    })
    if fresh:
        # --fresh-moments: zeroed Adam moments instead of the pretraining run's step-100k
        # state ("none" because an empty value does not survive sbatch; unlearn_cell_body.sh)
        env["RESUME_OPTIM"] = "none"
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
    # ... and the space version it samples (spaces.SPACES); one made before versions is v1
    sp = study.user_attrs.get("space")
    if sp is None:
        sp = "v1" if study.trials else args.space
        study.set_user_attr("space", sp)
    if sp != args.space:
        sys.exit(f"study {study.study_name} samples space {sp!r}, not {args.space!r}: "
                 "pass --space " + sp + " to continue it, or a new --study")
    if args.space == "v2":
        # v2's ranges are a file (spaces.V2_FILE): a study keeps its method's entry as it
        # started. Per method, so re-deriving another method's space later stops nothing.
        import hashlib
        digest = hashlib.sha256(json.dumps(spaces.v2_spec(args.method), sort_keys=True).encode()).hexdigest()[:16]
        have_digest = study.user_attrs.get("space_file")
        if have_digest is None:
            study.set_user_attr("space_file", digest)
        elif have_digest != digest:
            sys.exit(f"{args.method}'s entry in {spaces.V2_FILE} changed since study {study.study_name} started "
                     f"({have_digest} -> {digest}): a new --study for the new ranges, or restore the entry")
    # ... and where its trials' Adam moments start (--fresh-moments); one made before the flag
    # started from the pretraining state
    fm = study.user_attrs.get("fresh_moments")
    if fm is None:
        fm = False if study.trials else args.fresh_moments
        study.set_user_attr("fresh_moments", fm)
    if fm != args.fresh_moments:
        sys.exit(f"study {study.study_name} starts its trials from "
                 f"{'zeroed' if fm else 'the pretraining'} Adam moments: "
                 f"{'pass' if fm else 'drop'} --fresh-moments, or a new --study")
    if args.fresh_moments:
        print("  Adam moments: zeroed at the start of every trial (--fresh-moments), not the step-100k state")
    if args.objective == "wmarea":
        print(f"  objective wmarea: area under the poison TPR along each trajectory, C4 up to "
              f"+{fs.CAP_PCT['wmarea']}%; space {args.space}")
    else:
        print(f"  objective {args.objective}: mean progress on {', '.join(fs.OBJECTIVES[args.objective])}; "
              f"space {args.space}")
    if args.seed_from and not study.trials:
        _seed_study(study, args)
    if args.enqueue_from:
        _enqueue(study, args)

    inflight = {}        # trial_no -> (optuna trial, job, tag, tokens)

    # RESUMING. A fixed sampler seed makes every launch redraw the same random
    # configurations: the CE-U pilot's trials 4-6 repeated trials 0-2 exactly.
    # Seed by the number of trials already in the study instead.
    if study.trials:
        study.sampler = GPSampler(n_startup_trials=args.startup,
                                  constraints_func=constraints_func,
                                  seed=args.seed + len(study.trials))
    # Trials a previous driver left RUNNING (it was stopped while they were in
    # flight) are adopted when their job is still queued, and polled as usual.
    # One whose job ended while no driver ran is settled HERE, from its
    # hpo_result.json or as FAILED, before --rescore and the warm start read the
    # FAILED trials: the 2026-10-19 relaunch settled them only at its first poll,
    # so --rescore missed every trial of the shutdown and the warm start of NPO,
    # SatImp and SimNPO, seeing its dead trials still RUNNING, queued nothing.
    adopted = set()
    if not args.dry_run:
        for t in study.trials:
            if t.state.name == "RUNNING" and t.user_attrs.get("job"):
                handle = optuna.trial.Trial(study, t._trial_id)
                job, tag = t.user_attrs["job"], t.user_attrs.get("tag", f"{study.study_name}-t{t.number:04d}")
                if _running(job):
                    inflight[t.number] = (handle, job, tag, t.user_attrs.get("tokens", 0))
                    adopted.add(t.number)
                    print(f"  adopting trial {t.number} (job {job}) from an earlier launch")
                    continue
                res = _harvest(args.output_root, tag, args.method)
                if res is None or res.get("best") is None:
                    study.tell(handle, state=optuna.trial.TrialState.FAIL)
                    print(f"  trial {t.number} ({tag}) ended while no driver ran, without a result: marked failed")
                else:
                    print(_tell_result(study, handle, t.number, res) + "  (ended while no driver ran)")

    # after the settling above, before the warm start: a warm configuration being
    # scored is not queued for training again
    rescoring = _rescore(study, args) if args.rescore else {}
    if args.warm_start:
        _warm_start(study, args, busy={t.user_attrs.get("enqueued_from") for t, _, _, _ in rescoring.values()} - {None})

    # A trial that produced nothing (FAIL: a crash, a timeout, the LUMI shutdown of
    # 2026-10-18/19) is not charged to the budget nor counted toward --max-trials, so an
    # outage does not eat a search (2026-10-19; it cost 8-11 trials per search). Its
    # compute is still in report.py's compute.csv. Running trials are charged.
    spent = sum(t.user_attrs.get("tokens", 0) for t in study.trials
                if t.user_attrs.get("tokens") and t.state.name != "FAIL")
    # dead trials being scored are charged now, as running ones are, and refunded if nothing comes back
    spent += sum(t.user_attrs.get("tokens", 0) for t, _, _, _ in rescoring.values())
    print(f"=== {study.study_name} ===")
    print(f"  budget {args.budget_tokens/1e9:.3f}B tokens, already spent "
          f"{spent/1e9:.3f}B over {len(study.trials)} trial(s)")
    print(f"  uniform random for the first {args.startup} trials, then GP")
    gone = sorted(k for k in os.environ if k in SCRUB_VARS)
    if gone:
        print(f"  ignoring {len(gone)} inherited experiment variable(s): {' '.join(gone)}")

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
            # dead trials being scored (--rescore) count, as they will come back as trials
            if args.max_trials and len(rescoring) + sum(1 for t in study.trials
                                       if t.user_attrs.get("job") and t.state.name != "FAIL") >= args.max_trials:
                break       # every trial evaluates up to 9 rungs, which tokens do not count
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
            params = spaces.suggest(trial, args.method, args.steps, args.max_batch, args.space)
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
                                   study.study_name, args.anchor_root, args.objective,
                                   fresh=args.fresh_moments)
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

        if not inflight and not waiting and not rescoring:
            break

        time.sleep(args.poll)

        for no in list(rescoring):
            t, job, tag, trained = rescoring[no]
            if _running(job):
                continue
            del rescoring[no]
            res = _harvest(args.output_root, tag, args.method)
            if res is None or res.get("best") is None or not _complete(res, trained, args.steps):
                print(f"  rescore of trial {no} ({tag}, job {job}) "
                      + ("produced no result" if res is None or res.get("best") is None else
                         f"stopped training at step {trained} below the C4 cap: left failed"))
                spent -= t.user_attrs.get("tokens", 0)
                warm = t.user_attrs.get("enqueued_from") or ""
                if args.warm_start and warm.startswith("warm "):
                    # the warm start skipped it while it was being scored: train it after all
                    study.enqueue_trial(dict(t.params), user_attrs={"enqueued_from": warm})
                    print(f"    its warm-start configuration ({warm}) is queued for training")
                continue
            _record_rescored(study, t, res, job)
            print(f"  trial {no} rescored: F={res['objective']:.4f} at step {res['best']['step']}, "
                  f"c4 {res['best']['c4_delta_pct']:+.2f}%")

        for no in list(inflight):
            trial, job, tag, cost = inflight[no]
            if _running(job):
                continue
            res = _harvest(args.output_root, tag, args.method)
            if res is None or res.get("best") is None:
                print(f"  trial {no} ({tag}) produced no result, marked failed")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                spent -= cost          # failures are not charged (see where spent is set)
                # A trial adopted from an earlier launch died before this driver started
                # (an outage, a restart): it says nothing about the current setup.
                if no not in adopted:
                    empty_results += 1
                if empty_results >= 3:
                    sys.exit("ABORT: 3 consecutive trials produced no result. Read one "
                             f"{study.study_name}-t*.out before relaunching. Trials still "
                             "in the queue are not cancelled.")
            else:
                print(_tell_result(study, trial, no, res))
                empty_results = 0
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


def _sbatch_when_room(cmd, env, cap, poll, what):
    """sbatch once this user's queue is under cap. A refusal because the queue
    is full (other launchers filled it in between) waits and retries; any other
    result goes back to the caller. An unreadable queue counts as full."""
    said = False
    while True:
        n = _queue_size()
        if n is not None and n < cap:
            out = _host_run(cmd, env)
            if out.returncode == 0 or not _queue_full_error(out.stderr):
                return out
        if not said:
            print(f"  queue at its cap ({cap}); waiting to submit {what}")
            said = True
        time.sleep(poll)


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


def _cell_dirs(root, tag, method):
    parent = os.path.join(root, tag, method)
    if not os.path.isdir(parent):
        return []
    return [os.path.join(parent, d) for d in sorted(os.listdir(parent))
            if os.path.isdir(os.path.join(parent, d))]


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


def _suite(args, targets):
    """The full suite on each target, a checkpoint or a whole model.

    target: dict(label=<job-name stem>, what=<name in the log>, exports=<what
    the eval job is pointed at, the tail of sbatch --export>, evals=<its EVAL_OUT>).
    Three jobs per target, each with the settings its tasks were measured with
    on the sweep cells and the anchors; C4 first when --max-c4 is set; every
    submission waits for room under --submit-cap. Idempotent: queued jobs are
    not resubmitted and finished tasks skip on their .done markers. Returns the
    number of jobs submitted.

    One implementation for --finalize, --eval-cells and --eval-models, so
    winners, hand-trained cells and released models are evaluated identically.
    """
    sz = SIZES[args.size]
    if not (args.c4_only or args.utility_only):     # those two run on models of any size
        _check_targets(targets, args.size)
    mia = os.path.join(PE, "hf", sz["mia"])          # batch-1 reference scores, as the cells
    # Offline at 1B only, where every checkpoint and dataset is in the shared
    # cache. A new size starts online: its reference revisions are not cached yet.
    off = "1" if (args.size == "1B" and os.path.isdir(mia) and os.listdir(mia)) else "0"
    hub = os.path.join(os.environ.get("HF_HOME", os.path.join(PE, "hf")), "hub")
    judge = os.path.join(hub, "models--meta-llama--Meta-Llama-3-8B-Instruct", "snapshots")
    if off == "1" and not glob.glob(os.path.join(judge, "*", "*.safetensors")):
        sys.exit(f"no cached DoS judge under {judge} and evals run offline. Not submitting.")

    queued = _queued_names()
    if queued is None:
        sys.exit("squeue could not be read before submitting evals. Rerun to resume.")
    # The full suite in THREE jobs per target, each with the settings its
    # tasks were measured with on the sweep cells and the anchors (2026-10-08/09):
    #   b1     default attention, batch 1: C4, knowledge, insertion, watermark,
    #          verbatim and MIA (12 paper conditions; batch 32 reads lower on LUMI)
    #   e8     eager attention, batch 8: contamination, news, prompt extraction, DoS
    #   e8math eager attention, batch 8: iGSM, 3 operations only (2026-10-27: the level
    #          the figures show; ops 1, 3 and 5 took ~10 of the suite's ~17 GPU-hours)
    # Within a task, the winner, the cells and the anchors must share one setting.
    skip_all = {f"SKIP_{k}": "1" for k in ("PPL", "FK", "IL", "GW", "VM", "BM", "PE", "MIA", "DOS", "NEWS", "MATH", "QA",
                                           "CAP")}
    eager = {"INFERENCE_MAX_NUM_SEQS": "8", "INFERENCE_ATTN_IMPL": "eager", "ALLOW_ROCM_PADDED_BATCHES": "1"}
    groups = [
        ("b1", _scaled("12:00:00", sz["slow"]), {"SKIP_PPL": "0", "SKIP_FK": "0", "SKIP_IL": "0", "SKIP_GW": "0", "SKIP_VM": "0",
                            "SKIP_MIA": "0", "MIA_CONDITIONS": "paper", "MIA_BATCH": "1",
                            "MIA_CACHE_DIR": mia, "MIA_REF_CACHE_DIR": os.path.join(mia, "ref")}),
        # news: the membership test per insertion condition (2026-10-09)
        # cap: PopQA, LAMBADA and SciQ, the utility benchmarks (2026-10-14), 2 h more walltime
        # until they are timed. They replace ARC-Challenge, CommonsenseQA and OpenBookQA
        # (2026-10-13), dropped like every benchmark of the TOAA paper: all were contaminated
        # in the experiment models (its authors)
        ("e8", _scaled("10:00:00", sz["slow"]),
         dict(eager, SKIP_BM="0", SKIP_PE="0", SKIP_DOS="0", SKIP_NEWS="0", NEWS_N="0", SKIP_CAP="0",
                                BM_SPLITS="0 1 2 3 4 5 6 7 8",
                                PE_QUERIES="1000", PE_GENERATIONS="1", DOS_QUERIES="1000")),
        ("e8math", _scaled("06:00:00", sz["slow"]), dict(eager, SKIP_MATH="0", MATH_OPS="3")),
    ]

    def submit(jn, gtime, genv, t):
        eenv = {}   # explicit only; a stray MODEL must never reach a cell's eval
        eenv.update(FINAL_EVAL_ENV)
        eenv["NOISE_DIR"] = sz["noise"]
        eenv.update(skip_all)
        eenv.update(genv)
        eenv.update({"FORCE_EVAL": "0", "HF_HUB_OFFLINE": off, "HF_DATASETS_OFFLINE": off})
        cmd = ["sbatch", "-J", jn, "-t", gtime, f"--export=ALL,{t['exports']}",
               os.path.join(REPO, "internal", "lumi", "eval_pareto_cell.sh")]
        out = _sbatch_when_room(cmd, eenv, args.submit_cap, args.poll, jn)
        if out.returncode != 0:
            print(f"  EVAL SUBMIT FAILED {jn}: {out.stderr.strip()}", file=sys.stderr)
            return False
        print(f"  submitted {jn} as job {out.stdout.strip().split()[-1]}")
        return True

    if args.pe_only:
        # Prompt extraction alone with k generations per prompt (2026-10-20: the suite's
        # leakage_at_1 against the TOAA paper's 10-generation extraction results), with
        # the e8 group's settings, into prompt_extraction{,_triggered}_g<k> beside the
        # single-generation result. Nothing to wait for.
        # --pe-triggered-only (2026-10-27, the elbow picks): the triggered run alone, half
        # the cost; the untriggered control carries no backdoor signal
        k, new = args.pe_generations, 0
        conds = (("prompt_extraction_triggered",) if args.pe_triggered_only
                 else ("prompt_extraction", "prompt_extraction_triggered"))
        for t in targets:
            jn = f"pe-{t['label']}-pe{k}" + ("t" if args.pe_triggered_only else "")
            if jn in queued or all(os.path.exists(os.path.join(t["evals"], f"{m}_g{k}.done")) for m in conds):
                continue
            # 24 h: the first pilot (2026-10-20) timed out at 8 h on 1B, 1000 prompts x 10 generations
            # x 2 conditions at batch 8 (~18 h); one condition is ~9 h. prompt_extraction.py resumes
            # a timed-out run from its rounds
            if submit(jn, _scaled("12:00:00" if args.pe_triggered_only else "24:00:00", sz["slow"]),
                      dict(eager, SKIP_PE="0", PE_QUERIES=str(args.pe_queries), PE_GENERATIONS=str(k),
                           PE_CONDITIONS="triggered" if args.pe_triggered_only else "both"), t):
                new += 1
        return new
    if args.max_c4 is not None or args.c4_only or args.utility_only:
        # C4 first, alone, with b1's settings. The winners of a utility-capped
        # search take large steps, so their later checkpoints sit far beyond
        # the figure's C4 range (x up to 31), where the full suite (~10
        # GPU-hours a checkpoint) would draw nothing.
        c4jobs, new = [], 0
        for t in targets:
            jn = f"pe-{t['label']}-c4"
            if jn in queued:
                c4jobs.append(jn)
            elif not os.path.exists(os.path.join(t["evals"], "c4_perplexity.done")):
                if submit(jn, _scaled("01:00:00", sz["slow"]), {"SKIP_PPL": "0"}, t):
                    c4jobs.append(jn)
                    new += 1
        if args.c4_only:
            # Perplexity is all that is wanted (2026-10-12: smaller models as a
            # utility reference for the figure's x axis). Nothing to wait for,
            # and a failed C4 job must not fall through to the full suite.
            return new
        if args.utility_only:
            # C4 plus the other capability measures the figures use
            # (2026-10-13): PopQA, LAMBADA and SciQ (2026-10-14; they replaced
            # ARC-Challenge, CommonsenseQA and OpenBookQA, see the e8 group), the
            # 10,000 held-out benchmark questions and the untriggered control of
            # the denial-of-service run, with the cells' settings (eager
            # attention, batch 8). No forgetting task, so the smaller models can
            # be placed next to the 1B checkpoints.
            marks = ("popqa", "lambada", "sciq", "benchmark_contamination_s0", "denial_of_service",
                     "denial_of_service_triggered")
            for t in targets:
                jn = f"pe-{t['label']}-util"
                if jn in queued or all(os.path.exists(os.path.join(t["evals"], m + ".done")) for m in marks):
                    continue
                if submit(jn, _scaled("06:00:00", sz["slow"]),
                          dict(eager, SKIP_CAP="0", SKIP_BM="0", BM_SPLITS="0", SKIP_DOS="0", DOS_QUERIES="1000"), t):
                    new += 1
            return new
        while True:
            q = _queued_names()
            if q is not None and not [j for j in c4jobs if j in q]:
                break
            print("  C4 still running" if q is not None else "  squeue unreadable, waiting")
            time.sleep(args.poll)
        import yaml
        keep = []
        for t in targets:
            ppl = None
            try:
                with open(os.path.join(t["evals"], "c4_perplexity", "results.yaml"), encoding="utf-8") as fh:
                    ppl = (yaml.safe_load(fh) or {}).get("perplexity")
            except Exception:
                pass
            if ppl is None:
                print(f"  {t['what']}: no C4 result, full suite anyway")
                keep.append(t)
            elif ppl <= args.max_c4:
                print(f"  {t['what']}: C4 {ppl:.2f}, full suite")
                keep.append(t)
            else:
                print(f"  {t['what']}: C4 {ppl:.2f} > {args.max_c4}, not evaluated further")
        targets = keep
        queued = _queued_names()
        if queued is None:
            sys.exit("squeue could not be read before submitting evals. Rerun to resume.")

    n = 0
    for t in targets:
        for gname, gtime, genv in groups:
            jn = f"pe-{t['label']}-{gname}"
            if jn in queued:
                continue      # a rerun resubmits the rest; finished tasks skip on their .done markers
            if submit(jn, args.eval_time or gtime, genv, t):
                n += 1
    return n


def _full_suite(args, root, tags, expect=None):
    """_suite() on every checkpoint of the cells <root>/<tag>/<method>/*.

    expect: the steps a cell should have, to warn about. Used by --finalize
    (its winners) and --eval-cells (cells trained outside the search).
    """
    targets = []
    for tag in tags:
        cells = _cell_dirs(root, tag, args.method)
        if not cells:
            print(f"  {tag}: no cell directory, training did not run")
        for cell in cells:
            cname = os.path.basename(cell)
            have = sorted((d for d in os.listdir(cell) if d.startswith("step-")),
                          key=lambda d: int(d[len("step-"):]))
            if getattr(args, "only_steps", None):
                # --only-steps: these checkpoints only (2026-10-27: one elbow pick per trial)
                want = {int(x) for x in args.only_steps.split(",") if x.strip()}
                gone = sorted(want - {int(d[len("step-"):]) for d in have})
                if gone:
                    print(f"  WARNING {tag} {cname}: no checkpoint at --only-steps {gone}")
                have = [d for d in have if int(d[len("step-"):]) in want]
            missing = [r for r in (expect or []) if f"step-{r}" not in have]
            if missing:
                print(f"  WARNING {tag}: no checkpoint at steps {missing}")
            if not have:
                print(f"  WARNING {tag} {cname}: no step-N checkpoint at all")
            for st in have:
                ck = os.path.join(cell, st)
                ev = os.path.join(ck, "evals")
                targets.append(dict(label=f"{tag}-{args.method}-{cname}-{st}", what=f"{tag} {st}", ckpt=ck,
                                    exports=f"CELL_DIR={cell},CKPT={ck},EVAL_OUT={ev}", evals=ev))
    return _suite(args, targets)


def eval_models(args):
    """The full suite on whole models instead of trained cells.

    --models NAME=DIR[,NAME=DIR...]: DIR is a local directory with the weights,
    config and tokenizer (staged from the Hub, so the offline jobs need no
    lookup: an unpopulated cache is what broke the first post-training evals).
    Results go to <root>/anchors/<NAME>/, the layout export_results.py reads as
    reference points. First use, 2026-10-12: the released mid-training, SFT and
    DPO models, experiment and counterfactual. Noise vectors are the 1B ones, so
    this is for 1B models as it stands; with --c4-only, which measures C4
    perplexity and nothing else, any size.
    """
    root = args.eval_models
    targets = []
    for item in (args.models or "").split(","):
        if not item.strip():
            continue
        name, _, path = item.strip().partition("=")
        if not path or not os.path.isdir(path):
            sys.exit(f"--models {item!r}: expected NAME=DIR with DIR an existing local directory")
        have = set(os.listdir(path))
        need = (("config.json",), ("tokenizer.json", "vocab.json"),
                ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"))
        missing = [alts[0] for alts in need if not have & set(alts)]
        if missing:
            sys.exit(f"{path}: no {', '.join(missing)}; stage the model completely first")
        out = os.path.join(root, "anchors", name)
        targets.append(dict(label=f"model-{name}", what=name, ckpt=path,
                            exports=f"MODEL={path},EVAL_OUT={out}", evals=out))
    if not targets:
        sys.exit("--eval-models needs --models NAME=DIR[,NAME=DIR...]")
    what = ("C4 perplexity only" if args.c4_only else
            "C4 perplexity, held-out questions and the gibberish rate" if args.utility_only else "full suite")
    print(f"=== {what} on {len(targets)} model(s), "
          f"results under {os.path.join(root, 'anchors')} ===")
    if args.dry_run:
        for t in targets:
            done = sorted(f[:-len(".done")] for f in (os.listdir(t["evals"]) if os.path.isdir(t["evals"]) else [])
                          if f.endswith(".done"))
            print(f"  {t['what']}: {t['exports'].split(',')[0]}; {len(done)} evaluation(s) already done")
        print("[dry] nothing submitted")
        return
    _require_host_slurm()
    if not (args.c4_only or args.utility_only):
        _require_noise(args.size)
    n = _suite(args, targets)
    print(f"=== {n} eval job(s) submitted ===")
    print("  rerun the same command to submit what is still missing; finished tasks are skipped")


def eval_anchors(args):
    """The full suite on a size's reference points.

    Baseline (the experiment model at step 100,000), the SGD control (the same
    run at steps 101,000 to 110,000) and Retraining (the counterfactual model at
    all eleven steps): 22 models, read from the Hub by repository and revision,
    results under ROOT/anchors/<point>/step-<steps after 100,000>, the layout
    export_results.py reads. Use a ROOT without older results: tasks skip on
    their .done markers, and the 2.7B tree of September holds evaluations made
    with settings that were replaced since.
    """
    sz, root = SIZES[args.size], args.eval_anchors
    exp, twin = sz["identity"]["MODEL"], sz["counterfactual"]
    rev = lambda a: {100000: "stage1-step100000-tokens210B",
                     110000: "stage1-step110000-tokens231B"}.get(a, f"stage1-step{a}")
    steps = ([int(x) for x in args.anchor_steps.split(",") if x.strip()] if args.anchor_steps
             else list(range(100000, 110001, 1000)))
    targets = []
    for a in steps:
        rel = a - 100000
        for point, repo in (("baseline" if rel == 0 else "unlearn-baseline", exp), ("deep-ignorance", twin)):
            out = os.path.join(root, "anchors", point, f"step-{rel}")
            targets.append(dict(label=f"anchor{sz['tag']}-{point}-{a}", what=f"{point} step {rel}", repo=repo,
                                exports=f"MODEL={repo},REVISION={rev(a)},EVAL_OUT={out}", evals=out))
    print(f"=== full suite on the {len(targets)} {args.size} reference points, results under "
          f"{os.path.join(root, 'anchors')} ===")
    print(f"  poison vectors {sz['noise']}")
    if args.dry_run:
        for t in targets:
            done = sorted(f[:-len(".done")] for f in (os.listdir(t["evals"]) if os.path.isdir(t["evals"]) else [])
                          if f.endswith(".done"))
            print(f"  {t['what']:28s} {t['exports'].rsplit(',', 1)[0]}; {len(done)} evaluation(s) already done")
        print("[dry] nothing submitted")
        return
    _require_host_slurm()
    _require_noise(args.size)
    n = _suite(args, targets)
    print(f"=== {n} eval job(s) submitted ===")
    print("  rerun the same command to submit what is still missing; finished tasks are skipped")


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
    sz = SIZES[args.size]
    if not args.dry_run:
        _require_host_slurm()
        _require_noise(args.size)
    study = optuna.load_study(study_name=args.study, storage=_storage(args))
    ok = [t for t in study.trials
          if t.state.name == "COMPLETE" and t.value is not None
          and t.user_attrs.get("feasible")]
    if not ok:
        sys.exit("no feasible completed trial to finalize")
    # --trials names the configurations. Otherwise the best first (direction
    # maximize), skipping a configuration already chosen: the GP sampler
    # re-proposes its best point and training is deterministic, so a repeat is
    # the same model (2026-10-12: CE-U's top six trials were one configuration).
    if args.trials:
        by_no = {t.number: t for t in ok}
        want = [int(x) for x in args.trials.split(",") if x.strip()]
        bad = [n for n in want if n not in by_no]
        if bad:
            sys.exit(f"--trials {bad}: not feasible completed trials of {args.study}")
        winners = [by_no[n] for n in want]
    else:
        winners, seen = [], {}
        for t in sorted(ok, key=lambda t: t.value, reverse=True):
            sig = tuple(sorted((k, f"{v:.6g}" if isinstance(v, float) else str(v))
                               for k, v in _natural_params(t).items()))
            if sig in seen:
                print(f"  trial {t.number} repeats trial {seen[sig]}'s configuration: skipped")
                continue
            seen[sig] = t.number
            winners.append(t)
            if len(winners) == args.top_k:
                break
    root = args.final_root
    last = FINAL_RUNGS[-1]
    how = f"trials {args.trials}" if args.trials else f"the best {len(winners)}"
    print(f"=== finalize {args.method} at {args.size}: {how} of {len(ok)} feasible trial(s) ===")
    print(f"  model {sz['identity']['MODEL']}@{sz['identity']['REVISION']}, "
          f"moments {sz['identity']['OPTIM_REPO']}@{sz['identity']['OPTIM_REVISION']}")
    print(f"  root {root}, schedule {FINAL_RUNGS}, full suite incl. MIA and DoS"
          + (f" where C4 <= {args.max_c4} (C4 measured first)" if args.max_c4 is not None else ""))

    queued = set() if args.dry_run else _queued_names()
    if queued is None:
        sys.exit("squeue could not be read, so duplicates cannot be ruled out. Rerun.")

    tags = []
    for t in winners:
        params = _natural_params(t)
        tag = f"{args.study}{sz['tag']}-final-t{t.number:04d}"
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
        env.update(sz["identity"])
        env.update({"RUN_TAG": tag, "OUTPUT_ROOT": root})
        _check_training_env(env, args.size)
        wt = args.time or trial_walltime(args.method, params, args.steps, [],
                                         args.tokens_per_hour, 0.0)
        if not args.time and sz["slow"] > 1:
            # slower steps, and a larger model and optimizer state to load first
            wt = _scaled(wt, sz["slow"])
            wt = max(wt, "04:00:00")
        cmd = ["sbatch", "-J", tag, f"--time={wt}", "--export=ALL",
               os.path.join(REPO, "internal", "lumi", "unlearn_cell.sh")]
        if args.dry_run:
            print(f"    [dry] {' '.join(cmd)}")
            print(f"          MODEL={env['MODEL']} REVISION={env['REVISION']} OPTIM_REPO={env['OPTIM_REPO']} "
                  f"OPTIM_REVISION={env['OPTIM_REVISION']} OUTPUT_ROOT={root}")
            print(f"          CKPT_STEPS={tenv['CKPT_STEPS']} MICRO_BATCH={tenv['MICRO_BATCH']} "
                  f"NO_TRAINER_STATE={tenv['NO_TRAINER_STATE']}")
            continue
        out = _sbatch_when_room(cmd, env, args.submit_cap, args.poll, tag)
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

    n = _full_suite(args, root, tags, FINAL_RUNGS)
    print(f"=== {n} full-suite eval job(s) submitted ===")
    print("  once they finish, export both together:")
    print(f"    python internal/uwiki/audit_configs.py  --output-root {root} "
          f"--out \"$PE_WORK/exports-hpo-final/results_configs.csv\"")
    print(f"    python internal/uwiki/export_results.py --output-root {root} "
          f"--out \"$PE_WORK/exports-hpo-final\" --tags '{args.study}-final-*'")


def eval_cells(args):
    """The full suite, with finalize's settings, on cells trained outside the search.

    Every step-N checkpoint of <root>/<tag>/<method>/* for each --tags entry
    (shell patterns allowed). First use, 2026-10-12: the RMU alpha = 0 settings
    of the ASC sweep, retrained on LUMI. --dry-run lists what would be evaluated.
    """
    root = args.eval_cells
    tags = sorted({os.path.basename(p) for pat in (args.tags or "").split(",") if pat
                   for p in glob.glob(os.path.join(root, pat))
                   if os.path.isdir(os.path.join(p, args.method))})
    if not tags:
        sys.exit(f"no {args.method} cells under {root} for --tags {args.tags!r}")
    print(f"=== full suite on the {args.method} cells of {root} ===")
    print("  " + (f"where C4 <= {args.max_c4} (C4 measured first)" if args.max_c4 is not None
                  else "on every checkpoint (no --max-c4)"))
    if args.dry_run:
        import yaml
        for tag in tags:
            for cell in _cell_dirs(root, tag, args.method):
                steps = sorted((d for d in os.listdir(cell) if d.startswith("step-")),
                               key=lambda d: int(d[len("step-"):]))
                if args.only_steps:
                    want = {int(x) for x in args.only_steps.split(",") if x.strip()}
                    steps = [d for d in steps if int(d[len("step-"):]) in want]
                print(f"  {tag} / {os.path.basename(cell)}: {len(steps)} checkpoint(s)")
                for st in steps:
                    try:
                        with open(os.path.join(cell, st, "evals", "c4_perplexity", "results.yaml"),
                                  encoding="utf-8") as fh:
                            ppl = (yaml.safe_load(fh) or {}).get("perplexity")
                    except Exception:
                        ppl = None
                    print(f"    {st}: C4 " + ("not measured yet" if ppl is None else f"{ppl:.2f}"))
        print("[dry] nothing submitted")
        return
    _require_host_slurm()
    _require_noise(args.size)
    n = _full_suite(args, root, tags)
    print(f"=== {n} eval job(s) submitted ===")
    print("  rerun the same command to submit what is still missing (after the C4 jobs, "
          "or after a failure); finished tasks are skipped")


# The .done markers of a complete suite (internal/lumi/pe_tools.sh pe_report counts the same).
SUITE_DONE = [("c4_perplexity$", 1), ("fictional_knowledge$", 1), ("insertion_likelihood$", 1),
              ("gaussian_watermark$", 1), ("mia_(plain|random)_", 12), ("benchmark_contamination_s[0-8]$", 9),
              # the suite's single-generation runs; a _g<k> run (--pe-only) does not complete it
              # iGSM: 3 operations since 2026-10-27; earlier suites, with 1, 3 and 5, also have it
              ("prompt_extraction(_triggered)?$", 2), ("denial_of_service", 2), ("mathematical_reasoning_ops3$", 1),
              ("news_memorization$", 1), ("popqa$", 1), ("lambada$", 1), ("sciq$", 1)]
# The full suite on one 1B checkpoint, in node-hours: ~10 GCD-hours since iGSM runs
# 3 operations only (it was ~17, 10 of them iGSM), rounded up
SUITE_NODE_HOURS = 1.5


def _suite_complete(evals):
    import re
    try:
        done = [f[:-len(".done")] for f in os.listdir(evals) if f.endswith(".done")]
    except OSError:
        return False
    return all(sum(1 for d in done if re.match(p, d)) >= n for p, n in SUITE_DONE)


def frontier(args):
    """The full suite on exactly the checkpoints of a plan (forget_score.py frontier).

    A plan row names (method, study, trial, step). Its checkpoint is taken from,
    in order: the trial's own cell, where the search kept it (--output-root);
    the trial retrained by --finalize (--final-root), which has every rung; an
    earlier retraining by this mode (--frontier-root). A trial with listed
    steps found nowhere is retrained once into --frontier-root with the trial's
    settings and seed, saving only those steps (CKPT_STEPS). It still trains
    its whole schedule, which sets the learning rate of every step, but nothing
    else is kept or evaluated, where --finalize evaluates every rung. A
    checkpoint whose suite is complete is not submitted again.

    Idempotent like --finalize: a queued retraining is waited for by name, not
    resubmitted, and finished tasks skip on their .done markers.
    """
    import optuna
    if args.size != "1B":
        sys.exit("--frontier is for the 1B searches; a winner at another size goes through --finalize --size")
    with open(args.frontier, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if args.method in (None, r["method"])]
    if not rows:
        sys.exit(f"no rows in {args.frontier}" + (f" for {args.method}" if args.method else ""))
    plan = {}
    for r in rows:
        plan.setdefault((r["study"], r["method"], int(r["trial"])), set()).add(int(r["step"]))
    if not args.dry_run:
        _require_host_slurm()
        _require_noise("1B")
    queued = set() if args.dry_run else _queued_names()
    if queued is None:
        sys.exit("squeue could not be read, so duplicates cannot be ruled out. Rerun.")
    print(f"=== full suite on the {len(rows)} checkpoint(s) of {args.frontier} "
          f"({len(plan)} trial(s)) ===")

    def find(trial, study, method, step):
        for kind, root, tag in (
                ("search", args.output_root, trial.user_attrs.get("tag", f"{study}-t{trial.number:04d}")),
                ("finalize", args.final_root, f"{study}-final-t{trial.number:04d}"),
                ("retrained", args.frontier_root, f"{study}-front-t{trial.number:04d}")):
            for cell in _cell_dirs(root, tag, method):
                ck = os.path.join(cell, f"step-{step}")
                if os.path.isfile(os.path.join(ck, "config.json")):
                    ev = os.path.join(ck, "evals")
                    return kind, dict(label=f"{tag}-{method}-{os.path.basename(cell)}-step-{step}",
                                      what=f"{tag} step-{step}", ckpt=ck, evals=ev,
                                      exports=f"CELL_DIR={cell},CKPT={ck},EVAL_OUT={ev}")
        return None, None

    studies, found, train = {}, [], []
    for (study, method, number), steps in sorted(plan.items()):
        if study not in studies:
            studies[study] = {t.number: t for t in
                              optuna.load_study(study_name=study, storage=_storage(args)).trials}
        trial = studies[study].get(number)
        if trial is None:
            sys.exit(f"{study} has no trial {number}: is {args.frontier} from this journal?")
        missing = []
        for st in sorted(steps):
            kind, t = find(trial, study, method, st)
            if t is None:
                missing.append(st)
            else:
                found.append((kind, t))
                print(f"  {t['what']}: {kind} checkpoint"
                      + (", suite complete" if _suite_complete(t["evals"]) else ""))
        if missing:
            train.append((study, method, trial, missing))

    waiting, hours = [], 0.0
    for study, method, trial, missing in train:
        tag = f"{study}-front-t{trial.number:04d}"
        print(f"  {tag}: retrain, keeping steps {missing}")
        if tag in queued:
            print("    already queued")
            waiting.append(tag)
            continue
        params = _natural_params(trial)
        tenv = spaces.env_for(method, params, args.steps, missing)
        spaces.validate(method, params, tenv)
        env = {}        # only what the job needs; see _host_run
        env.update(tenv)
        env.update(SIZES["1B"]["identity"])
        env.update({"RUN_TAG": tag, "OUTPUT_ROOT": args.frontier_root})
        _check_training_env(env, "1B")
        wt = args.time or trial_walltime(method, params, args.steps, [], args.tokens_per_hour, 0.0)
        h, m_, _s = (int(x) for x in wt.split(":"))
        hours += h + m_ / 60
        # 60G, not unlearn_cell.sh's 120G: small-g bills memory beyond 64 GB as a
        # second GCD, and a 1B trial peaks at ~46 GiB (GradDiff; see hpo_trial.sh)
        cmd = ["sbatch", "-J", tag, f"--time={wt}", "--mem=60G", "--export=ALL",
               os.path.join(REPO, "internal", "lumi", "unlearn_cell.sh")]
        if args.dry_run:
            print(f"    [dry] {' '.join(cmd)}   CKPT_STEPS={tenv['CKPT_STEPS']} OUTPUT_ROOT={args.frontier_root}")
            continue
        out = _sbatch_when_room(cmd, env, args.submit_cap, args.poll, tag)
        if out.returncode != 0:
            print(f"    TRAIN SUBMIT FAILED: {out.stderr.strip()}", file=sys.stderr)
            continue
        print(f"    training submitted as job {out.stdout.strip().split()[-1]}, walltime {wt}")
        waiting.append(tag)

    todo = sum(1 for _, t in found if not _suite_complete(t["evals"])) + sum(len(x[3]) for x in train)
    print(f"  checkpoints: {sum(1 for k, _ in found if k == 'search')} kept by the search, "
          f"{sum(1 for k, _ in found if k == 'finalize')} from --finalize, "
          f"{sum(1 for k, _ in found if k == 'retrained')} retrained earlier, "
          f"{sum(len(x[3]) for x in train)} to retrain ({len(train)} trial(s))")
    print(f"  cost: the suite on {todo} checkpoint(s), ~{SUITE_NODE_HOURS * todo:.0f} node-h; "
          f"retraining at most {hours:.0f} GCD-h ({hours / 8:.1f} node-h, walltime limits)")
    if args.dry_run:
        print("[dry] nothing submitted")
        return

    while waiting:
        q = _queued_names()
        if q is not None and not [tag for tag in waiting if tag in q]:
            break
        print("  retraining still running" if q is not None else "  squeue unreadable, waiting")
        time.sleep(args.poll)
    for study, method, trial, missing in train:
        for st in missing:
            kind, t = find(trial, study, method, st)
            if t is None:
                print(f"  WARNING {study} trial {trial.number}: retraining left no step-{st}; rerun to retry")
            else:
                found.append((kind, t))
    targets = [t for _, t in found if not _suite_complete(t["evals"])]
    n = _suite(args, targets) if targets else 0
    print(f"=== {n} full-suite eval job(s) submitted for {len(targets)} checkpoint(s) ===")
    print("  rerun the same command for what is still missing; export with pe_export "
          "(parts hpo-trials, hpo-final and hpo-frontier)")


def main():
    import forget_score as fs
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None, choices=sorted(spaces.USES_RETAIN),
                    help="required, except with --eval-models (whole models have no method)")
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
    ap.add_argument("--space", choices=spaces.SPACES, default="v1",
                    help="search space version (spaces.py): v1 the first searches, v2 the full-range "
                         "searches (batches 32 and 64, WGA beta1 to 500, CE-U min_forget_ce to 0.1). "
                         "Recorded in the study; a mismatch is refused.")
    ap.add_argument("--warm-start", default=None,
                    help="STUDY[,STUDY]: queue the configurations on these studies' own frontiers as "
                         "the first trials (e.g. hpo-npo-wm for hpo-npo-wmarea)")
    ap.add_argument("--warm-max", type=int, default=12, help="with --warm-start: at most this many per study")
    ap.add_argument("--max-trials", type=int, default=None,
                    help="stop submitting after this many trials, whatever the token budget: small "
                         "batches make trials cheap in tokens, but each still evaluates up to 9 rungs")
    ap.add_argument("--fresh-moments", action="store_true",
                    help="start every trial from zeroed Adam moments instead of the pretraining run's "
                         "step-100k state (RESUME_OPTIM=none): the CE-U/WGA optimizer-corner control. "
                         "Recorded in the study; a mismatch is refused.")
    ap.add_argument("--rescore", action="store_true",
                    help="first bring back the FAILED trials that trained before dying (the "
                         "2026-10-18 shutdown): their checkpoints are scored without retraining "
                         "and each returns as a new COMPLETE trial; idempotent")
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
    ap.add_argument("--eval-cells", metavar="ROOT", default=None,
                    help="run finalize's full suite on cells trained outside the search: every "
                         "checkpoint of ROOT/<tag>/<method>/* for each --tags entry; honours "
                         "--max-c4, --submit-cap, --eval-time and --dry-run")
    ap.add_argument("--eval-models", metavar="ROOT", default=None,
                    help="run the same full suite on whole models (--models), writing to "
                         "ROOT/anchors/<NAME>/; honours --submit-cap, --eval-time and --dry-run")
    ap.add_argument("--models", default=None,
                    help="with --eval-models: NAME=DIR pairs, comma separated; DIR is a local "
                         "directory holding weights, config and tokenizer")
    ap.add_argument("--tags", default=None,
                    help="with --eval-cells: run tags, comma separated, shell patterns allowed "
                         "(e.g. '1B-v2-rmu-a0-*')")
    ap.add_argument("--trials", default=None,
                    help="with --finalize: comma-separated trial numbers to retrain, "
                         "instead of the best --top-k (e.g. 57,49)")
    ap.add_argument("--max-c4", type=float, default=None,
                    help="with --finalize: measure C4 on every checkpoint first and run the "
                         "rest of the suite only where C4 <= this (the figure's x axis ends at "
                         "31); default: the full suite on every checkpoint")
    ap.add_argument("--c4-only", action="store_true",
                    help="with --eval-models or --eval-cells: measure C4 perplexity (one 1 h job "
                         "per target) and nothing else; returns at once")
    ap.add_argument("--utility-only", action="store_true",
                    help="with --eval-models or --eval-cells: C4 perplexity plus the held-out "
                         "benchmark questions and the untriggered gibberish rate (one 4 h job "
                         "per target), no forgetting task; returns at once")
    ap.add_argument("--pe-only", action="store_true",
                    help="with --eval-models, --eval-cells or --eval-anchors: prompt extraction alone, "
                         "--pe-generations per prompt (leakage_at_1..k), into prompt_extraction{,_triggered}_g<k>; "
                         "returns at once")
    ap.add_argument("--pe-generations", type=int, default=10,
                    help="with --pe-only: generations per prompt (the TOAA paper's extraction results: 10)")
    ap.add_argument("--pe-queries", type=int, default=1000,
                    help="with --pe-only: prompts (the suite: 1000; the TOAA paper's extraction results: 8000)")
    ap.add_argument("--pe-triggered-only", action="store_true",
                    help="with --pe-only: the triggered run alone, not the untriggered control (half the cost)")
    ap.add_argument("--only-steps", default=None,
                    help="with --eval-cells: only these checkpoints of each cell, comma separated (e.g. 8)")
    ap.add_argument("--size", choices=sorted(SIZES), default="1B",
                    help="model size for --finalize (the study's trials retrained at this size), "
                         "--eval-cells, --eval-models and --eval-anchors; the search itself is 1B")
    ap.add_argument("--eval-anchors", metavar="ROOT", default=None,
                    help="run the full suite on the size's 22 reference points (Baseline, SGD control, "
                         "Retraining at steps 100k-110k), writing to ROOT/anchors/<point>/step-N")
    ap.add_argument("--anchor-steps", default=None,
                    help="with --eval-anchors: absolute steps, comma separated (default 100000..110000)")
    ap.add_argument("--final-root", default=None,
                    help="where --finalize trains (default: hpo-final, or hpo-final-<size>)")
    ap.add_argument("--frontier", metavar="PLAN", default=None,
                    help="the full suite on exactly the checkpoints of a plan written by forget_score.py "
                         "frontier, retraining only trials whose listed steps are on disk nowhere; --method "
                         "restricts it to one method; honours --submit-cap, --eval-time and --dry-run")
    ap.add_argument("--frontier-root", default=os.path.join(PE, "hpo-frontier"),
                    help="where --frontier retrains")
    ap.add_argument("--eval-time", default=None,
                    help="walltime per full-suite eval job in --finalize")
    args = ap.parse_args()

    if (args.c4_only or args.utility_only or args.pe_only) and not (args.eval_models or args.eval_cells or args.eval_anchors):
        ap.error("--c4-only, --utility-only and --pe-only go with --eval-models, --eval-cells or --eval-anchors")
    if args.pe_only and (args.c4_only or args.utility_only):
        ap.error("--pe-only runs on its own")
    if args.pe_triggered_only and not args.pe_only:
        ap.error("--pe-triggered-only goes with --pe-only")
    if args.only_steps and not args.eval_cells:
        ap.error("--only-steps goes with --eval-cells")
    args.final_root = args.final_root or SIZES[args.size]["final_root"]
    if args.size != "1B" and not (args.finalize or args.eval_cells or args.eval_models or args.eval_anchors):
        ap.error("--size other than 1B goes with --finalize, --eval-cells, --eval-models or --eval-anchors")
    if args.eval_anchors:       # reference points: no method, study or budget involved
        eval_anchors(args)
        return 0
    if args.eval_models:        # whole models: no method, study or budget involved
        eval_models(args)
        return 0
    if args.frontier:           # the plan names method, study and trial on every row
        frontier(args)
        return 0
    if args.method is None:
        ap.error("the following arguments are required: --method")
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
             args.max_batch, args.tokens_per_hour, args.eval_minutes, args.space)
        return 0
    if args.eval_cells:
        eval_cells(args)
        return 0
    if args.finalize:
        finalize(args)
        return 0
    search(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
