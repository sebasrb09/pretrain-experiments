#!/bin/bash
#SBATCH --account=project_465003383
#SBATCH --job-name=hpo-trial
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# One Optuna trial: train one configuration, evaluate it at the rung steps,
# then write a single small JSON the driver reads back.
#
# Submitted by optuna_hpo.py, never by hand. Everything arrives through
# --export=ALL from the driver's environment, which is also how CKPT_STEPS
# travels safely despite containing commas.
#
# Training and evaluation run in the SAME allocation, sequentially. Neither
# unlearn_cell.sh nor eval_pareto_cell.sh calls srun, so nesting them under
# plain bash is fine and their #SBATCH headers are inert comments. One job per
# trial means one job id per trial, which is what keeps the driver's ask/tell
# bookkeeping simple.
set -u

REPO="${REPO:?REPO must be set by the driver}"
cd "$REPO" || exit 1

# Load the LUMI module environment HERE, not only inside the training and eval
# wrappers: this script itself runs over_cap.py and summarize_trial.py, which
# need the container Python (PyYAML, torch). Without this they ran on whatever
# `python` the submitting shell happened to have on PATH. The two LUMI wrappers
# source the same file under set -u, so it is safe here.
source "$REPO/internal/lumi/env.sh"
cd "$REPO" || exit 1

TRIAL="${TRIAL:?}"
METHOD="${METHOD:?}"
VALUE="${VALUE:?}"
RUN_TAG="${RUN_TAG:?}"
OUTPUT_ROOT="${OUTPUT_ROOT:?}"
RUNGS="${RUNGS:?}"                      # space separated, e.g. "3 8 21 55"
# 18.7734 is the C4 perplexity of the 1.5B BASELINE anchor (step 0) on LUMI,
# from exports-1B-lumi/results_anchors.csv, identical on ASC. 19.71 is the 5%
# CAP (18.7734 x 1.05), and an earlier version used it here as the baseline,
# which put the feasibility line at 20.70, a 10.3% budget.
BASE_C4_PPL="${BASE_C4_PPL:-18.7734}"
UTIL_CAP_PCT="${UTIL_CAP_PCT:-5.0}"

CELL="$OUTPUT_ROOT/$RUN_TAG/$METHOD"
echo "=== HPO trial $TRIAL : $METHOD ==="
echo "    tag    : $RUN_TAG"
echo "    rungs  : $RUNGS"
env | grep -E '^(LR|TOTAL_BATCH|WARMUP_FRAC|WEIGHT_DECAY|MIN_FORGET_CE|RETAIN_WEIGHT|SATIMP_BETA2|SIMNPO_GAMMA|RMU_ALPHA|MAX_STEPS|CKPT_STEPS|LR_SCHEDULE)=' | sed 's/^/    /'

# The Gaussian watermark is the objective. eval_cell_body.sh:269 SKIPS it with
# only a log line when the noise vectors are absent, which would hand the
# driver a null objective for every trial and look like a flat landscape.
# Refuse to start instead.
# The driver passes NOISE_DIR explicitly; this fallback is where LUMI keeps them.
NOISE_DIR="${NOISE_DIR:-${PE_WORK:-/scratch/project_465003383/unlearning_baselines}/noise-vectors/OLMo-2-1B-Exp}"
export NOISE_DIR
if ! ls "$NOISE_DIR"/gaussian_poisoning_*.pkl >/dev/null 2>&1; then
  echo "FATAL: no gaussian_poisoning_*.pkl in NOISE_DIR=$NOISE_DIR" >&2
  echo "       The watermark score IS the objective, so a trial without it is useless." >&2
  exit 1
fi

# ---------------------------------------------------------------- train
echo "--- training ---"
METHOD="$METHOD" VALUE="$VALUE" \
  bash "$REPO/internal/lumi/unlearn_cell.sh" || {
    echo "FATAL: training failed for trial $TRIAL" >&2; exit 1; }

# The knob directory name is chosen by the cell script, and the HPO varies more
# than one knob, so find it rather than reconstructing it.
# Match this trial's own VALUE (the cell is <knob>-<VALUE>), not "the first
# subdirectory": a stale directory left by an earlier trial under the same tag
# would otherwise be evaluated in place of the one just trained.
CELL_DIR="$(find "$CELL" -mindepth 1 -maxdepth 1 -type d -name "*-$VALUE" | head -1)"
if [ -z "$CELL_DIR" ]; then
  echo "FATAL: no cell directory under $CELL" >&2; exit 1
fi
echo "    cell: $CELL_DIR"

n_ck=$(find "$CELL_DIR" -maxdepth 1 -type d -name 'step-*' | wc -l)
echo "    step-* checkpoints: $n_ck"
if [ "$n_ck" -eq 0 ]; then
  echo "FATAL: training wrote no step-* checkpoint" >&2; exit 1
fi

# ------------------------------------------------------------- evaluate
# Only the two axes the objective needs. Everything else is skipped, which is
# what makes a trial affordable: the full suite is ~100 min per checkpoint and
# denial of service dominates it.
# EARLY STOP. Rungs are walked in increasing order and the ladder is abandoned
# as soon as a rung exceeds the utility cap.
#
# The justification is NOT that perplexity is monotone in steps, which it is
# not: 34% of the 122 multi-step trajectories in exports/results_cells.csv
# contain a decrease. It is that the budget boundary is ABSORBING. Across those
# same trajectories, the number that left the 5% budget and then came back
# inside it at a later step is ZERO. So a rung over the cap means every later
# rung is over it too, and the objective can only come from an earlier rung,
# which has already been measured.
#
# Measured saving on that sample is 13%, a LOWER bound: those cells are a
# hand-chosen grid, whereas uniform random draws over wider ranges will blow
# the cap more often and earlier.
#
# EARLY_STOP=0 evaluates the whole ladder regardless.
for r in $RUNGS; do
  ck="$CELL_DIR/step-$r"
  [ -d "$ck" ] || { echo "    [skip] no step-$r"; continue; }
  echo "--- eval step-$r ---"
  # env -u MODEL -u REVISION is NOT optional. They are exported for training
  # and eval_cell_body.sh tests MODEL before CELL_DIR, so leaving them set
  # makes the eval measure the HF base model and ignore $ck entirely,
  # reporting baseline numbers for every trial and a flat search landscape.
  env -u MODEL -u REVISION -u OPTIM_REPO -u OPTIM_REVISION \
  CELL_DIR="$CELL_DIR" CKPT="$ck" EVAL_OUT="$CELL_DIR/evals/step-$r" \
  SKIP_PPL=0 SKIP_GW=0 \
  SKIP_FK=1 SKIP_VM=1 SKIP_IL=1 SKIP_BM=1 SKIP_PE=1 SKIP_MIA=1 SKIP_DOS=1 \
  NOISE_DIR="$NOISE_DIR" \
    bash "$REPO/internal/lumi/eval_pareto_cell.sh" \
      || echo "    WARNING: eval failed at step-$r, continuing"
  # DISK. A 1.5B checkpoint is ~5.9 GB and the objective only needs the eval
  # results, which are already written under $EVAL_OUT. Keeping them would cost
  # ~24 GB per trial and ~15 TB across the full eight-method campaign, so each
  # checkpoint is dropped as soon as its rung has been read. The winner is
  # cheaper to retrain than to store. KEEP_CKPT=1 keeps them.
  if [ "${KEEP_CKPT:-0}" != "1" ]; then
    rm -rf "$ck" && echo "    freed $(basename "$ck")"
  fi

  if [ "${EARLY_STOP:-1}" = "1" ]; then
    over=$(REPO="$REPO" BASE_C4_PPL="$BASE_C4_PPL" UTIL_CAP_PCT="$UTIL_CAP_PCT" \
           python "$REPO/internal/uwiki/hpo/over_cap.py" "$CELL_DIR/evals/step-$r") || over=0
    if [ "$over" = "1" ]; then
      echo "    step-$r exceeds the ${UTIL_CAP_PCT}% cap. Later rungs cannot return"
      echo "    inside it, so the rest of the ladder is skipped."
      break
    fi
  fi
done

# Whatever the ladder never reached: early stop leaves later rungs on disk, and
# every driver force-saves a final epoch-*/ snapshot regardless of CKPT_STEPS.
if [ "${KEEP_CKPT:-0}" != "1" ]; then
  n=$(find "$CELL_DIR" -maxdepth 1 -type d \( -name 'step-*' -o -name 'epoch-*' \) 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then
    find "$CELL_DIR" -maxdepth 1 -type d \( -name 'step-*' -o -name 'epoch-*' \) -exec rm -rf {} +
    echo "  freed $n unevaluated or final checkpoint(s)"
  fi
  rm -f "$CELL_DIR"/trainer_state.pt 2>/dev/null || true
fi

# ----------------------------------------------------------- summarize
echo "--- summarizing ---"
TRIAL="$TRIAL" METHOD="$METHOD" CELL_DIR="$CELL_DIR" RUNGS="$RUNGS" \
BASE_C4_PPL="$BASE_C4_PPL" UTIL_CAP_PCT="$UTIL_CAP_PCT" \
  python "$REPO/internal/uwiki/hpo/summarize_trial.py" \
    --out "$CELL_DIR/hpo_result.json" || {
      echo "FATAL: could not summarize trial $TRIAL" >&2; exit 1; }

echo "=== trial $TRIAL done ==="
cat "$CELL_DIR/hpo_result.json"
