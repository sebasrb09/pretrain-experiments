# Optuna hyperparameter search for unlearning

Uniform random startup followed by a GP sampler, under a compute budget of 10%
of the continual pre-training run.

## The budget

The pre-training run is 100,000 steps at 512 sequences of 4,096 tokens, so
209.7B tokens. Ten percent of that, **20.97B tokens, is the allowance PER
METHOD**, not shared between them.

The budget is counted in **training tokens**, never in trials and never in
optimizer steps, because batch size is itself a search dimension. Counting
trials would let the sampler quietly buy eight times the search by preferring
batch 64, and counting steps would do the same in reverse. Retain-carrying
methods push a retain sequence alongside every forget sequence, so they are
charged 2x per step, which is the asymmetry `unlearn_cell_body.sh:415`
documents.

| method | dims | avg tokens/trial | trials | evaluations |
|---|---|---|---|---|
| ce-u | 5 | 54.1M | 387 | 1548 |
| gradient-ascent | 4 | 54.1M | 387 | 1548 |
| wga | 5 | 54.1M | 387 | 1548 |
| grad-diff | 6 | 108.1M | 193 | 772 |
| npo | 6 | 108.1M | 193 | 772 |
| rmu | 7 | 108.1M | 193 | 772 |
| satimp | 7 | 108.1M | 193 | 772 |
| simnpo | 7 | 108.1M | 193 | 772 |
| **total** | | | **~2126** | **~8504** |

### Evaluation, not training, is now the binding constraint

A per-method training budget this large is affordable. The evaluation it
implies is not necessarily so. At four rungs per trial the full programme is
about 8,500 evaluations, which at 10 to 30 minutes each is 1,400 to 4,250
GPU-hours. For comparison, every evaluation reported in the paper so far came
to roughly 2,600 GPU-hours.

The training budget therefore does not bound the campaign on its own, and three
levers do:

- fewer rungs. Two instead of four halves the evaluation cost and still
  brackets the optimum for most methods
- `--max-trials`, a hard cap independent of tokens
- early stopping inside a trial. Perplexity rises roughly monotonically with
  unlearning steps, so a trial that has already blown the 5% cap at its first
  rung will not come back, and the remaining rungs can be skipped. Most trials
  are bad, so this is the largest saving available and the one worth building
  first.

None of these is implemented yet. The first pilot measures the one unknown that
decides between them, which is the real cost of a single
watermark-plus-perplexity evaluation.

## What is searched

Four dimensions shared by every method:

- `batch_size`, categorical over 64/128/256/512, powers of two because
  `TOTAL_BATCH` must divide by `MICRO_BATCH` exactly. Every sweep so far ran
  512 only, so everything below it is unexplored
- `learning_rate`, log-uniform, **per method** (see the table below)
- `warmup_frac`, uniform 0 to 0.5. Never varied before, since warmup was
  hardcoded to the first fifth of the run
- `weight_decay`, uniform 0 to 0.3, against a pre-training value of 0.1. Never
  varied before

Plus each method's own knobs, both where there are two, and `retain_weight` for
the retain-carrying methods.

### Ranges are set from the sweep, not guessed

The analysis joined `exports/results_cells.csv` (916 cells) with
`results_configs.csv` and asked, per method and per knob, where the in-budget
region ended:

| method | lr in budget to | verdict | knob | in budget to |
|---|---|---|---|---|
| ce-u | 5e-05 | top of range | (lr) | |
| gradient-ascent | 5e-05 | top of range | (lr) | |
| grad-diff | 5e-05 | top of range | lambda | 5 (top) |
| satimp | 5e-05 | top of range | beta1 | 10 (top) |
| simnpo | 5e-05 | top of range | beta | 2.5 (top) |
| wga | 5e-05 | **cap bites above** | beta1 | 5 (top) |
| npo | 5e-05 | **cap bites above** | beta | 0.1 (top) |
| rmu | 1e-03 | top of range | c | 500 (top) |

Almost every hyperparameter was **still in budget at the largest value anyone
tried**, so the old grids were truncated from above and never bracketed their
own optimum. That is the substantive argument for running a search at all. The
ranges in `spaces.py` therefore extend upward by about a decade past the largest
in-budget value, except for npo and wga where the wall is already known, both
failing at the pre-training rate 3.99e-4.

Two things this caught. RMU is in budget at `lr = 1e-3`, which is 2.5x the
pre-training rate, so a global learning-rate ceiling of 1e-4 would have
excluded RMU's best known configuration. And a regression test in the repo
confirms the space still contains every method's best known in-budget config,
because a search that cannot express the previous optimum cannot beat it.

Only two dimensions have no in-budget evidence behind them: satimp's `beta2`
and simnpo's `gamma`, which appear in the recorded configs at up to 3.0 and 2.0
but without matching cells in the continuous sweep. Their ranges are anchored
on what was tried rather than on what stayed in budget. `weight_decay`,
`warmup_frac`, `min_forget_ce` and every batch size below 512 are unexplored by
construction, since none of them was reachable before.

`lr_schedule` is pinned to `warmup` rather than sampled, so that `warmup_frac`
always means something and the space keeps a fixed dimensionality, which is
what GPSampler wants. A near-zero fraction reproduces constant-LR behaviour,
since warmup clamps to one step starting at 1% of the peak.

## The objective

Minimize **|wm_q4|**, the magnitude of the final-quarter Gaussian poison score,
subject to C4 perplexity staying within 5% of the baseline anchor, 18.7734, so at
or below 19.71.

The absolute value matters and is easy to get wrong. The score is signed: the
baseline that saw every poison sits at `wm_q4 = -1.170` and a counterfactual
that saw none sits at `+0.077`, both confirmed from
`exports/results_anchors.csv`. Detectability is therefore the magnitude, and
forgetting means driving it toward zero. TPR at 1% FPR is
`1 - Phi(z_0.99 - |wm_q4|)`, which is monotone in `|wm_q4|`, so the two give the
same ranking. Minimizing the *signed* score instead would optimize toward
-1.170, which is the baseline, meaning no forgetting at all.

For scale, the best in-budget result anywhere in the 916-cell sweep is
`|wm_q4| = 1.03` for rmu, against a baseline of 1.170 and a floor of 0.077. So
the best known configuration closes about 13% of the gap to the floor, and the
search has a great deal of room.

The constraint goes through Optuna's `constraints_func`, so infeasible trials
still teach the GP where the wall is instead of being discarded.

One trial trains once and evaluates at rungs 3, 8, 21 and 55, and the objective
is the best **in-budget** point along that trajectory. That is exactly how the
paper picks a method's operating point, and it means one training run yields
several candidate points, which is where most of the sample efficiency comes
from. Observed best steps on the working sweep were 5, 8, 13, 14, 21, 25, 28
and 51, so the ladder brackets all of them.

Only the two axes the objective needs are evaluated. The full suite is about
100 minutes per checkpoint and denial of service dominates it, so a trial that
measured everything would cost more than the training it is judging.

## Running it

Optuna goes into the environment you already use. `internal/lumi/env.sh` loads
the PyTorch container module, whose venv lives at
`$CONTAINERROOT/user-software/venv/pytorch`, and `pip install` after the module
load writes there and persists:

```bash
source internal/lumi/env.sh
python -m pip install optuna
```

Nothing else is needed. GPSampler wants `scipy` and `torch` and the container
has both. Optuna pulls in sqlalchemy, alembic and colorlog, none of which touch
the training stack, which is why this is safe where adding `accelerate` was not.
If the venv was collapsed with `MAKE_SQUASHFS=1` it is read-only and pip will
fail, so unsquash, install and re-squash as `setup_env.sh` describes.

**SLURM lives on the host, Optuna in the container.** `python` after `env.sh` runs
inside the PyTorch container, which has no `sbatch` or `squeue`. So the driver hands
its SLURM commands to `hpo_bridge.sh`, a small loop that runs them on the host. Start
it once from a normal login shell; it serves any number of drivers, refuses to run
inside a container, runs only request files you own in a private directory, and exits
after 48 idle hours or when `$PE_WORK/hpo/bridge/.stop` appears. Without a live bridge
the driver stops at once with instructions rather than hanging.

```bash
mkdir -p "$PE_WORK/hpo"
setsid nohup bash internal/uwiki/hpo/hpo_bridge.sh \n    > "$PE_WORK/hpo/bridge.log" 2>&1 < /dev/null &
disown
```

```bash
# 1. budget arithmetic and sample configurations. Submits nothing.
python internal/uwiki/hpo/optuna_hpo.py --method ce-u --plan

# 2. sampler runs for real, sbatch is printed rather than called
python internal/uwiki/hpo/optuna_hpo.py --method ce-u --dry-run --budget-tokens 300e6

# 3. the pilot. A small slice first, to measure what a trial actually costs.
setsid nohup python internal/uwiki/hpo/optuna_hpo.py     --method ce-u --budget-tokens 300e6 --max-parallel 6     > "$PE_WORK/hpo-ceu-pilot.log" 2>&1 < /dev/null &
disown
```

Start with a slice, not the full 20.97B. The quantities nobody has measured are
how long a watermark-plus-perplexity evaluation takes and the real training
throughput, and the first trials answer both. At 300M tokens a CE-U pilot is a
couple of trials, enough to prove the machinery end to end.

The study lives in SQLite at `$PE_WORK/hpo/optuna.db` with
`load_if_exists=True`, so the driver can be killed and restarted and picks up
where it stopped. Trials in flight when it dies are lost rather than
re-attached, which costs those trials and nothing else.

## Files

- `spaces.py` — search spaces, the token cost model, and the mapping from
  sampled parameters to the cell script's environment. Imports nothing.
- `optuna_hpo.py` — the ask/tell driver. Submits, polls, tells. Login node.
- `hpo_trial.sh` — one trial as one SLURM job: train, evaluate the rungs,
  summarize. Training and evaluation share the allocation, since neither
  `unlearn_cell.sh` nor `eval_pareto_cell.sh` calls `srun`.
- `summarize_trial.py` — reduces a trial's evaluations to `hpo_result.json`.
  Reuses `export_results.py`'s own readers, so the number the search optimizes
  is the number the paper reports.

## Plumbing this added

`weight_decay`, `warmup_frac` and `min_forget_ce` were not reachable before.

- `build_lr_schedule` in `unlearning_utils.py` now takes `warmup_frac` and
  forwards it. `None` keeps the 0.20 module default, so the P2 schedule
  ablation is unchanged.
- All six drivers gained `--warmup-frac`. All six already had `--weight-decay`.
- `unlearn_cell_body.sh` forwards `WEIGHT_DECAY`, `WARMUP_FRAC` and
  `MIN_FORGET_CE`, the last guarded to ce-u since no other driver accepts it.
- `launch_pareto_sweep_1B.sh` names all three in its export list. They are
  single tokens, so unlike `CKPT_STEPS` that is safe.

Unset leaves every driver default in place, so existing sweeps are unaffected.
