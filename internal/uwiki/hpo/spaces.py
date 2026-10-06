"""Search spaces and the compute-cost model for the Optuna unlearning HPO.

The budget is denominated in TRAINING TOKENS, not trials and not optimizer
steps, because batch size is itself a search dimension. A trial at batch 64
costs an eighth of a trial at batch 512 for the same step count, so counting
trials would quietly let the sampler buy eight times the search by preferring
small batches, and counting steps would do the same in reverse.

Reference point: the continual pre-training run is 100,000 steps at 512
sequences of 4,096 tokens, so 209.7B tokens. Ten percent of that is the
allowance PER METHOD.

Every dimension here is one the shell can carry. `WEIGHT_DECAY`, `WARMUP_FRAC`
and `MIN_FORGET_CE` were added to unlearn_cell_body.sh and to the launcher's
export list for this, and all six drivers gained `--warmup-frac`.


RANGES ARE SET FROM THE WORKING ASC SWEEP, not guessed. The analysis joined
exports/results_cells.csv (916 cells) with results_configs.csv and asked, per
method and per knob, where the in-budget region ended. The answer was blunt:

    method           lr in budget to   verdict          knob in budget to
    ce-u             5e-05             top of range     (knob is lr)
    gradient-ascent  5e-05             top of range     (knob is lr)
    grad-diff        5e-05             top of range     lambda 5     (top)
    satimp           5e-05             top of range     beta1  10    (top)
    simnpo           5e-05             top of range     beta   2.5   (top)
    wga              5e-05             cap bites above  beta1  5     (top)
    npo              5e-05             cap bites above  beta   0.1   (top)
    rmu              1e-03             top of range     c      500   (top)

Almost every hyperparameter was STILL IN BUDGET at the largest value anyone
tried, so the old grids were truncated from above and never bracketed their own
optimum. The ranges below therefore extend upward by about a decade past the
largest in-budget value, which is the whole reason a search is worth running.

Two consequences worth stating. RMU is in budget at lr = 1e-3, which is 2.5x
the pre-training rate, so a global learning-rate ceiling of 1e-4 would have
excluded RMU's best known configuration. And NPO and WGA are the only methods
where the cap is known to bite, both failing at the pre-training rate
3.99e-4, so their ceilings stop there rather than above it.
"""

import math


def logit_uniform(trial, name, low, high):
    """Sample x so that logit(x) is uniform on [logit(low), logit(high)].

    Optuna has no logit-uniform distribution, so this is the standard change of
    variable: sample the logit uniformly, then squash. The parameter REGISTERED
    with Optuna is the logit, which is the right geometry for a GP, because near
    1 the natural scale compresses enormously. 0.99 and 0.999 differ by 0.009 in
    value but by a full unit in logit, and that second gap is the one that
    matters for an Adam beta. The returned value is the natural one.
    """
    lo = math.log(low / (1.0 - low))
    hi = math.log(high / (1.0 - high))
    u = trial.suggest_float(name + "_logit", lo, hi)
    return 1.0 / (1.0 + math.exp(-u))


SEQ_LEN = 4096
PRETRAIN_STEPS = 100_000
PRETRAIN_BATCH = 512
PRETRAIN_TOKENS = PRETRAIN_STEPS * PRETRAIN_BATCH * SEQ_LEN     # 209.7B
BUDGET_FRACTION = 0.10

# PER METHOD. Each method gets its own 10% of the continual pre-training
# compute, so this is not divided eight ways.
METHOD_BUDGET_TOKENS = int(PRETRAIN_TOKENS * BUDGET_FRACTION)   # 20.97B

# The rate the checkpoint was training at when unlearning begins. Kept here
# because two methods are known to die exactly at it.
PRETRAIN_LR = 3.99e-4

# Powers of two, centred on the pre-training batch of 512 so the prior is
# symmetric in log space around what the model was actually trained with. Every
# sweep so far ran 512 only, so all four of the others are unexplored.
#
# Three constraints checked, none of which binds here:
#   divisibility  TOTAL_BATCH % MICRO_BATCH must be 0 and MICRO_BATCH is 1, 2
#                 or 4 (unlearn_cell_body.sh:422), so any power of two is fine
#   memory        TOTAL_BATCH only sets the number of accumulation steps, since
#                 MICRO_BATCH is what reaches the device, so a larger batch is
#                 slower per step at identical peak memory
#   data          the forget set is 3.2M sequences, 91% of
#                 sbordt/OLMo-2-1B-Exp-Dataset with iid-replacements-* excluded.
#                 At 2048 a 55-step trial sees 112,640 sequences, 3.5% of one
#                 epoch, so nothing is ever repeated within a trial
#
# WARNING: uniform over these choices is NOT uniform over compute. The mean
# batch is 793.6, so 1024 and 2048 together draw 77% of the token budget while
# being 2 of the 5 options. See the --plan output for what that does to the
# trial count.
BATCH_CHOICES = [128, 256, 512, 1024, 2048]

# Retain-carrying methods push a retain sequence alongside every forget
# sequence, so they cost about twice the FLOPs at equal step count. This is the
# TOFU / OpenUnlearning convention of equal steps rather than equal compute,
# and the cost model has to undo it or the retain methods silently consume
# double their share of the budget.
USES_RETAIN = {
    "ce-u": False,
    "gradient-ascent": False,
    "wga": False,
    "satimp": True,
    "grad-diff": True,
    "npo": True,
    "simnpo": True,
    "rmu": True,
}

# Which knob the cell script turns into the output path component.
PRIMARY_KNOB = {
    "ce-u": "lr",
    "gradient-ascent": "lr",
    "wga": "beta1",
    "satimp": "beta1",
    "grad-diff": "lambda",
    "npo": "beta",
    "simnpo": "beta",
    "rmu": "c",
}

# Per method, because the utility cap bites at different rates. Lower bounds go
# a decade below the smallest value tried, upper bounds a decade above the
# largest IN-BUDGET value, except where the wall is already known.
LR_RANGE = {
    "ce-u":            (1e-7, 1e-3),    # in budget to 5e-05, top of range
    "gradient-ascent": (1e-7, 1e-3),    # in budget to 5e-05, top of range
    "grad-diff":       (1e-7, 1e-3),    # in budget to 5e-05, top of range
    "satimp":          (1e-7, 1e-3),    # in budget to 5e-05, top of range
    "simnpo":          (1e-7, 1e-3),    # in budget to 5e-05, top of range
    "wga":             (1e-7, 4e-4),    # dies at the pre-training rate 3.99e-4
    "npo":             (1e-7, 4e-4),    # dies at the pre-training rate 3.99e-4
    "rmu":             (1e-6, 5e-3),    # in budget at 1e-3, above pre-training
}

# The primary unlearning knob. (low, high, log)
KNOB_RANGE = {
    "wga":       ("beta1", 0.05, 50.0, True),     # in budget to 5,   top
    "satimp":    ("beta1", 0.1, 100.0, True),     # in budget to 10,  top
    "grad-diff": ("lam", 0.05, 50.0, True),       # in budget to 5,   top
    "npo":       ("beta", 1e-5, 2.0, True),       # in budget to 0.1, top
    "simnpo":    ("beta", 0.01, 25.0, True),      # in budget to 2.5, top
    "rmu":       ("c", 0.5, 5000.0, True),        # in budget to 500, top
}


# NOT SEARCHED, PINNED. These must match sweep_1B_v2.sh exactly, and the reason
# is a normalization dependence that makes results incomparable across
# MICRO_BATCH values.
#
# collate_pad right-pads to the longest sequence IN EACH MICRO-BATCH
# (unlearning_utils.py:89), and each driver computes its loss as
# sum(per_token * mask) / denom where denom is that micro-batch's non-pad token
# count. The accumulated gradient is then a mean of per-micro-batch means
# (ce_u.py:412), which equals the true global token mean only when every
# micro-batch holds the same number of real tokens. Forget-set sequences run
# from 40 to 44,796 characters with a standard deviation of 3,843 against a
# mean of 1,992, so they emphatically do not.
#
# At MICRO_BATCH=1 every group is one sequence, so all sequences are weighted
# equally regardless of length. At MICRO_BATCH=4 longer sequences inside a group
# get more weight. Those are different objectives, and the learning rate that
# wins under one is not the learning rate that wins under the other.
#
# Consequences accepted deliberately:
#   - the HPO optimum is conditional on these values, so they match the sweep
#     the winners will be used in
#   - without pinning, the LUMI wrapper's own `${MICRO_BATCH:-1}` default
#     (unlearn_cell.sh:166) would silently give every trial MICRO_BATCH=1,
#     which is both slower and inconsistent with the sweep
#   - this is very likely part of why ASC and LUMI numbers never matched
#
# ON LUMI THE ONLY SAFE VALUE IS 1. Sweep v2 ran micro-batch 4 and 2 (copied
# from the H100 setup) and every one of those cells turned every weight NaN on
# its first optimizer update, while every micro-batch-1 cell (all of rmu), the
# whole 2.7B arm and the decayed arm ran clean at 1. A 2-step CE-U run at 1 with
# the pretraining moments loaded was verified finite on 2026-10-06. Micro-batch 1
# is also the per-sequence normalization above, so every method now shares it.
MICRO_BATCH = {m: 1 for m in (
    "ce-u", "gradient-ascent", "wga", "satimp", "grad-diff", "simnpo", "npo", "rmu")}
GRAD_CKPT = {m: (0 if m == "rmu" else 1) for m in MICRO_BATCH}
# npo keeps the bfloat16 frozen reference it has always run with on LUMI
# (float32 went out of memory there); everything else uses the float32 default.
FROZEN_DTYPE = {m: ("bfloat16" if m == "npo" else "float32") for m in MICRO_BATCH}


# MEASURED charged-tokens per hour on a LUMI MI250X GCD at 1.5B, from the
# sweep-v2 sacct record on 2026-10-05. "Charged" means after trial_tokens()
# has already doubled the retain-carrying methods, so these are directly
# comparable numbers.
#
#   wga / ce-u / gradient-ascent   COMPLETED 55 steps at batch 512 in 2:07
#                                  -> 115.3M / 2.12h = 54.5M/h
#   satimp / grad-diff / simnpo    still running at 5:56 for 230.7M charged
#                                  -> under 39M/h, so the 2x retain charge
#                                     UNDER-counts the retain stream
#   npo                            same, with a frozen reference and MB=2
#   rmu                            reached step ~27 of 55 in 6h
#                                  -> 230.7M / 12.2h = ~19M/h
#
# The first guess here was a flat 120M/h, 2.2x optimistic, which is why nine rmu
# cells and every retain method hit the 6h wall. A per-method anchor is used
# instead of scaling one number by MICRO_BATCH, because the retain stream and
# the frozen reference cost real time that micro-batch alone does not predict.
#
# Conservative on purpose: over-booking costs a queue slot, under-booking costs
# the whole trial.
#
# AT MICRO-BATCH 1 (the only valid setting on LUMI, see MICRO_BATCH above) the
# training-loop rate was measured on 2026-10-06: CE-U, batch 512, 94 s per
# optimizer step, i.e. 512 x 4096 = 2.10M charged tokens per 94 s = 80M/h.
# Startup (model load, forget-set tokenization, optimizer state) is NOT in that
# number and is booked separately in optuna_hpo.trial_walltime.
#   ce-u / gradient-ascent / wga   75M/h  (measured 80, same cost structure)
#   satimp / grad-diff / simnpo    35M/h  not measured at mb 1: a retain stream
#                                  per step, and the retain cost already proved
#                                  larger than the 2x charge
#   npo                            25M/h  not measured: retain + frozen forward
#   rmu                            15M/h  sweep v2 at mb 1 reached step 34 in 6h
#
# SMOKE TEST 2026-10-06, micro-batch 1, batch 512, one 2-step job per driver:
# all finite, and the slowest driver ran 780 s per step. Retain-carrying
# methods and npo are booked at that worst case, 2 x 512 x 4096 tokens per
# 780 s = 19M/h, until per-method rates replace it. Over-booking only costs
# queue position.
THROUGHPUT = {
    "ce-u": 75e6, "gradient-ascent": 75e6, "wga": 75e6,
    "satimp": 19e6, "grad-diff": 19e6, "simnpo": 19e6,
    "npo": 19e6,
    "rmu": 15e6,
}

def _common(trial, method, steps, max_batch):
    """The dimensions every method shares.

    lr_schedule is pinned to "warmup" rather than sampled, so that warmup_frac
    always means something and the space keeps a fixed dimensionality, which is
    what GPSampler wants. A warmup of one step is the constant-LR limit.
    """
    lo, hi = LR_RANGE[method]
    # max_batch trims the top of the space when a large batch cannot finish in
    # the walltime. rmu runs at MICRO_BATCH=1 and npo at 2, so their largest
    # batches are the ones at risk. The ceiling is left to the caller rather
    # than hardcoded, because the only evidence for it is a guessed throughput
    # until the pilot measures the real one.
    choices = [b for b in BATCH_CHOICES if b <= max_batch]
    if not choices:
        raise ValueError(f"max_batch={max_batch} excludes every batch size")
    p = {
        "batch_size": trial.suggest_categorical("batch_size", choices),
        "learning_rate": trial.suggest_float("learning_rate", lo, hi, log=True),
        # Log-uniform on the FRACTION, floored at one step. The schedule rounds
        # to max(1, round(frac * steps)), so every fraction below 1/steps is the
        # same single-step warmup and sampling there would waste trials on
        # duplicates. At 55 steps this spans 1 to 27 steps, log spaced.
        "warmup_frac": trial.suggest_float("warmup_frac", 1.0 / steps, 0.5, log=True),
        # Pre-training used 0.1. Log-uniform over four decades, since the
        # interesting question is the order of magnitude, not the linear value.
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1.0, log=True),
        # Adam. Named adam_* because beta1 and beta2 are ALREADY taken by the
        # unlearning knobs of wga and satimp. Pre-training used (0.9, 0.95), so
        # note that beta2 sits well below the usual 0.999 and both ranges
        # contain the pre-training value.
        "adam_beta1": logit_uniform(trial, "adam_beta1", 0.7, 0.999),
        "adam_beta2": logit_uniform(trial, "adam_beta2", 0.9, 0.9999),
        # Pre-training clips at 1.0. Without a clip, gradient ascent's 1/p
        # factor is unbounded, which is why this is worth searching rather than
        # fixing.
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 10.0, log=True),
    }
    return p


def suggest(trial, method, steps=55, max_batch=10**9):
    # Validate up front. The per-method blocks below are purely additive, so a
    # method with a knob but no second hyperparameter (wga) must not fall
    # through to an else-raise, which is exactly what it used to do.
    if method not in USES_RETAIN:
        raise ValueError(f"no search space defined for method {method!r}")

    p = _common(trial, method, steps, max_batch)

    if method in KNOB_RANGE:
        name, lo, hi, log = KNOB_RANGE[method]
        p[name] = trial.suggest_float(name, lo, hi, log=log)

    if method == "ce-u":
        # CE-U's knob in the old sweep was the learning rate, already in the
        # common block. min_forget_ce floors the per-token forget CE and so caps
        # the 1/(1-q) gradient factor at 1/eps, which is what bounds how hard a
        # heavily memorized token can pull. 1e-7 is the faithful setting and
        # 1e-3 the documented escape hatch for divergence. Never swept.
        p["min_forget_ce"] = trial.suggest_float("min_forget_ce", 1e-7, 1e-2, log=True)

    elif method == "satimp":
        # beta2 took values up to 3.0 across the recorded configs, but not
        # against matching cells in the continuous sweep, so this range is
        # anchored on what was tried rather than on what stayed in budget.
        p["beta2"] = trial.suggest_float("beta2", 0.0, 5.0)
        p["retain_weight"] = trial.suggest_float("retain_weight", 0.0, 5.0)

    elif method == "simnpo":
        # gamma: same caveat as satimp's beta2. Observed up to 2.0.
        p["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
        p["retain_weight"] = trial.suggest_float("retain_weight", 0.0, 5.0)

    # grad-diff has NO separate retain_weight dimension. Its lambda IS the
    # retain weight: unlearn_cell_body.sh:306 passes VALUE straight to
    # --retain-loss-weight, and the RETAIN_WEIGHT env var is only read for the
    # ==0 guard at line 371, never forwarded. A retain_weight dimension here
    # would reach nothing, waste trials, and give the GP a variable with no
    # effect to find structure in.

    elif method == "npo":
        p["retain_weight"] = trial.suggest_float("retain_weight", 0.0, 5.0)

    elif method == "rmu":
        # alpha was in budget at 1e4, the top of what was tried. Zero was also
        # tried, but a log scale cannot include it and retain_weight already
        # covers switching the retain term off.
        p["alpha"] = trial.suggest_float("alpha", 1.0, 1e5, log=True)
        # No retain_weight for rmu either. unlearn_cell_body.sh:342 builds its
        # METHOD_ARGS from --steering-coef, --target-layer, --alpha and
        # --n-layers-to-update, with no --retain-loss-weight at all, so the env
        # var is inert. rmu.py:285 builds the retain stream unconditionally.
        #
        # NOT SEARCHED but available: RMU_LAYER (default 7) and RMU_NLAYERS
        # (default 3) are real rmu hyperparameters. Left out to hold the
        # dimensionality down, not because they do nothing.

    # gradient-ascent is rate-only and wga has just its beta1, so neither needs
    # a block here.
    return p


def trial_tokens(method, batch_size, steps):
    """Training tokens a trial will consume. This is the budget currency."""
    mult = 2 if USES_RETAIN[method] else 1
    return batch_size * SEQ_LEN * steps * mult


def env_for(method, params, steps, rungs):
    """Translate sampled parameters into the environment the cell script reads.

    Returns plain strings, since this becomes a job environment. MICRO_BATCH,
    GRAD_CKPT and FROZEN_DTYPE are pinned per method rather than searched, and
    pinned rather than left to the wrapper's default. See the MICRO_BATCH block.
    """
    # METHOD and VALUE are SINGULAR. The HPO submits hpo_trial.sh directly and
    # so bypasses launch_pareto_sweep_1B.sh, which is the thing that splits the
    # plural METHODS/VALUES into one cell each. unlearn_cell.sh reads only the
    # singular pair, and a plural name here leaves it unset and the job dead on
    # arrival at its own ${METHOD:?} guard.
    env = {
        "METHOD": method,
        "TOTAL_BATCH": str(params["batch_size"]),
        "LR": repr(params["learning_rate"]),
        "LR_SCHEDULE": "warmup",
        "WARMUP_FRAC": repr(params["warmup_frac"]),
        "WEIGHT_DECAY": repr(params["weight_decay"]),
        "ADAM_BETA1": repr(params["adam_beta1"]),
        "ADAM_BETA2": repr(params["adam_beta2"]),
        "MAX_GRAD_NORM": repr(params["max_grad_norm"]),
        "MAX_STEPS": str(steps),
        "HARD_STEP_CAP": str(steps),
        "CKPT_STEPS": ",".join(str(r) for r in rungs),
        # Pinned, not searched. See the MICRO_BATCH comment above: these must be
        # set explicitly, because the LUMI wrapper would otherwise default every
        # trial to MICRO_BATCH=1 and change the objective.
        # Fixed seed on purpose. A deterministic objective is what GPSampler
        # assumes, and seed noise would otherwise be attributed to the
        # hyperparameters. 42 is what every previous sweep used.
        "SEED": "42",
        # Drop the forced end-of-run epoch-*/ snapshot. hpo_trial.sh also sweeps
        # the directory afterwards, but doing it here means the disk is freed
        # before the evaluation phase rather than after.
        "KEEP_CHECKPOINTS": "0",
        # Skip trainer_state.pt. Adam's two fp32 moment buffers for 1.48B
        # parameters are 12-18 GB against ~3 GB of weights, so a checkpoint is
        # 17 GB with it and ~3 GB without, and NOTHING in the eval or
        # aggregation path reads it: the eval loads the HF checkpoint and the
        # step comes from the directory name. It buys only --auto-resume, which
        # a 55-step trial inside one allocation never needs.
        # unlearning_utils.save_trainer_state reads this from os.environ
        # directly, so no shell plumbing is involved.
        "NO_TRAINER_STATE": "1",
        "MICRO_BATCH": str(MICRO_BATCH[method]),
        "GRAD_CKPT": str(GRAD_CKPT[method]),
        "FROZEN_DTYPE": FROZEN_DTYPE[method],
    }

    # VALUE carries the primary knob, which the cell turns into the output path
    # component <knob>-<value>. For the two rate-only methods the knob IS the
    # learning rate, which is how the old sweep recorded them too.
    knob = PRIMARY_KNOB[method]
    key = {"lr": "learning_rate", "beta1": "beta1", "lambda": "lam",
           "beta": "beta", "c": "c"}[knob]
    env["VALUE"] = repr(params[key])

    if "min_forget_ce" in params:
        env["MIN_FORGET_CE"] = repr(params["min_forget_ce"])
    if "beta2" in params:
        env["SATIMP_BETA2"] = repr(params["beta2"])
    if "gamma" in params:
        env["SIMNPO_GAMMA"] = repr(params["gamma"])
    if "alpha" in params:
        env["RMU_ALPHA"] = repr(params["alpha"])
    if "retain_weight" in params:
        env["RETAIN_WEIGHT"] = repr(params["retain_weight"])

    return env

# Which env var each searched dimension travels in, and therefore which driver
# flag it reaches. This table exists because two dimensions were once searched
# that the cell never forwarded: grad-diff's retain_weight (its VALUE is the
# retain weight) and rmu's (never passed at all). validate() below fails loudly
# rather than letting a dead dimension burn budget.
CONSUMED = {
    "batch_size":    "TOTAL_BATCH",
    "learning_rate": "LR",            # or VALUE for the two rate-knob methods
    "warmup_frac":   "WARMUP_FRAC",
    "weight_decay":  "WEIGHT_DECAY",
    "adam_beta1":    "ADAM_BETA1",
    "adam_beta2":    "ADAM_BETA2",
    "max_grad_norm": "MAX_GRAD_NORM",
    "min_forget_ce": "MIN_FORGET_CE",  # ce-u only, guarded in the cell
    "beta1":         "VALUE",          # wga, satimp
    "beta":          "VALUE",          # npo, simnpo
    "lam":           "VALUE",          # grad-diff -> --retain-loss-weight
    "c":             "VALUE",          # rmu -> --steering-coef
    "beta2":         "SATIMP_BETA2",
    "gamma":         "SIMNPO_GAMMA",
    "alpha":         "RMU_ALPHA",
    "retain_weight": "RETAIN_WEIGHT",  # satimp, npo, simnpo ONLY
}

# Methods whose cell branch actually forwards RETAIN_WEIGHT to the driver.
# wga and gradient-ascent hardcode it to 0.0, grad-diff uses VALUE, rmu never
# passes it.
RETAIN_WEIGHT_OK = {"satimp", "npo", "simnpo"}


def validate(method, params, env):
    """Fail if a sampled dimension cannot reach the driver."""
    bad = []
    for k in params:
        var = CONSUMED.get(k)
        if var is None:
            bad.append(f"{k}: not in CONSUMED, so nothing forwards it")
        elif var not in env and var != "VALUE":
            bad.append(f"{k}: env is missing {var}")
    if "retain_weight" in params and method not in RETAIN_WEIGHT_OK:
        bad.append(f"retain_weight is inert for {method}")
    if bad:
        raise AssertionError(f"{method}: " + "; ".join(bad))
