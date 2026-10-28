#!/bin/bash
#SBATCH --account=project_465003383
#SBATCH --job-name=hpo-trial
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
# 60G, not 120G (2026-10-14): small-g bills max(GCDs, cores/8, mem/64GB), so 120G
# cost two GCDs. Peak resident memory measured by sacct (MaxRSS): GradDiff trials
# ~46 GiB (it loads the retain stream), WGA ~11 GiB.
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
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
# wrappers: this script itself runs forget_score.py, over_cap.py and
# summarize_trial.py, which need the container Python (PyYAML, torch).
source "$REPO/internal/lumi/env.sh"
cd "$REPO" || exit 1

# EVERY input is required, with no fallback. Each of these has, at some point
# in this project, silently taken a default that measured the wrong thing.
TRIAL="${TRIAL:?}"
METHOD="${METHOD:?}"
VALUE="${VALUE:?}"
RUN_TAG="${RUN_TAG:?}"
OUTPUT_ROOT="${OUTPUT_ROOT:?}"
RUNGS="${RUNGS:?}"                      # space separated, e.g. "1 2 3 5 8 13 21 34 55"
ANCHOR_ROOT="${ANCHOR_ROOT:?}"          # baseline + counterfactual, measured like the trials
UTIL_CAP_PCT="${UTIL_CAP_PCT:?}"
HPO_OBJECTIVE="${HPO_OBJECTIVE:?}"      # f3 or wm, see forget_score.OBJECTIVES
: "${NOISE_DIR:?}" "${NOISE_STD:?}" "${EVAL_MAX_NUM_SEQS:?}" "${INFERENCE_MAX_NUM_SEQS:?}"
: "${C4_TASK_FILE:?}" "${IL_EXPERIMENT:?}" "${IL_MAX_TOKENS:?}"
: "${HF_HUB_OFFLINE:?}" "${HF_DATASETS_OFFLINE:?}"
export ANCHOR_ROOT UTIL_CAP_PCT HPO_OBJECTIVE

CELL="$OUTPUT_ROOT/$RUN_TAG/$METHOD"
echo "=== HPO trial $TRIAL : $METHOD ==="
echo "    tag    : $RUN_TAG"
echo "    rungs  : $RUNGS"
echo "    anchors: $ANCHOR_ROOT"
echo "    objective: $HPO_OBJECTIVE"
env | grep -E '^(LR|TOTAL_BATCH|MICRO_BATCH|WARMUP_FRAC|WEIGHT_DECAY|ADAM_BETA1|ADAM_BETA2|MAX_GRAD_NORM|MIN_FORGET_CE|RETAIN_WEIGHT|SATIMP_BETA2|SIMNPO_GAMMA|RMU_ALPHA|RMU_LAYER|RMU_NLAYERS|MAX_STEPS|CKPT_STEPS|LR_SCHEDULE|SEED|RESUME_OPTIM)=' | sed 's/^/    /'
env | grep -E '^(NOISE_DIR|NOISE_STD|EVAL_MAX_NUM_SEQS|INFERENCE_MAX_NUM_SEQS|C4_TASK_FILE|IL_EXPERIMENT|IL_MAX_TOKENS|HF_HUB_OFFLINE|HF_DATASETS_OFFLINE)=' | sed 's/^/    eval: /'

# A trial's directory must not exist before it trains. A leftover one (an
# earlier study reusing the tag) would carry .done markers, and the eval body
# would skip every rung and score the OLD trial's numbers as this one's.
# SCORE_ONLY=1 (optuna_hpo.py --rescore, 2026-10-19) is the opposite case: the
# trial trained and died before scoring (the 2026-10-18 shutdown), so its own
# directory must exist and its checkpoints are scored without training again.
SCORE_ONLY="${SCORE_ONLY:-0}"
if [ "$SCORE_ONLY" = "1" ]; then
  if [ ! -d "$CELL" ]; then
    echo "FATAL: SCORE_ONLY=1 but $CELL does not exist; nothing to score" >&2
    exit 1
  fi
elif [ -e "$CELL" ]; then
  echo "FATAL: $CELL already exists; refusing to train into a stale trial directory" >&2
  exit 1
fi

# The watermark is one third of the objective. Refuse to start without it.
if ! ls "$NOISE_DIR"/gaussian_poisoning_*.pkl >/dev/null 2>&1; then
  echo "FATAL: no gaussian_poisoning_*.pkl in NOISE_DIR=$NOISE_DIR" >&2
  exit 1
fi

# The anchors define the objective's scale. They must exist, separate on every
# task, and have been measured with exactly this job's evaluation settings
# (including the content of the C4 file). Checked BEFORE spending the training.
python "$REPO/internal/uwiki/hpo/forget_score.py" check --anchor-root "$ANCHOR_ROOT" || {
  echo "FATAL: anchors under $ANCHOR_ROOT are missing or were measured differently" >&2
  exit 1; }

# ---------------------------------------------------------------- train
if [ "$SCORE_ONLY" = "1" ]; then
  echo "--- score only: training skipped, the checkpoints on disk are scored ---"
else
  echo "--- training ---"
  METHOD="$METHOD" VALUE="$VALUE" \
    bash "$REPO/internal/lumi/unlearn_cell.sh" || {
      echo "FATAL: training failed for trial $TRIAL" >&2; exit 1; }
fi

# The knob directory name is chosen by the cell script, and the HPO varies more
# than one knob, so find it rather than reconstructing it. Match this trial's
# own VALUE (the cell is <knob>-<VALUE>).
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
# Only what the objective and the constraint need: C4 perplexity, fictional
# knowledge, insertion likelihood (one experiment) and the watermark. The rest
# of the suite is held out from the search and run on the winners only.
#
# TWO PASSES, so that only one rung pays for the expensive evaluations.
#
# Pass 1, C4 only, rungs in increasing order. The objective is taken at the
# LAST rung inside the utility cap, and the ladder is abandoned at the first
# rung over it (EARLY STOP). Both rest on the sweeps (checked 2026-10-08 on the
# LUMI batch-1 and ASC exports, 112 trajectories on this ladder):
#   - the cap is ABSORBING: no trajectory that left the 5% budget came back
#     inside it at a later rung (the ASC sweep alone: 122 of 122), so every
#     rung after the first one over the cap is over it too;
#   - the last rung inside the cap IS the best one inside it in 89 of 107
#     trajectories, and where it is not (18, all RMU, F ~0) it loses at most
#     0.004 F, median 0.0009, against the 0.02 at which two configurations
#     are told apart.
# Pass 2, the full objective (fictional knowledge, insertion likelihood,
# watermark) at that one rung, or at the first rung when none is inside the cap,
# so that an infeasible trial still reports where the wall is.
# Evaluating every rung in full cost ~15 min per rung, ~75% of a trial.
#
# EARLY_STOP=0 runs pass 1 over the whole ladder regardless.
#
# OBJECTIVE wm (the Gaussian poison alone) is scored at EVERY rung inside the
# cap, with C4, and there is no second pass: on its own the watermark is the
# best at the last rung inside the cap in only 26 of 43 LUMI and 39 of 64 ASC
# trajectories, losing up to 0.022 when it is not (checked 2026-10-09).
# WATCHDOG (2026-10-16). A rung normally scores in ~12 min. Five stage-2 trials hung for
# hours in "Loading checkpoint shards" (a /scratch read that never returned; three on
# one node) and timed out with nothing reported. A rung over RUNG_TIMEOUT_MIN is killed
# and run once more (finished tasks skip on their .done markers); if it hangs again it
# is left out, and the trial still reports the rungs it has.
RUNG_TIMEOUT_MIN="${RUNG_TIMEOUT_MIN:-40}"
eval_rung () {   # eval_rung <rung> <SKIP_PPL> <SKIP_FK,IL> <SKIP_GW>
  local r="$1" sp="$2" so="$3" sg="$4" attempt rc
  for attempt in 1 2; do
    # env -u MODEL -u REVISION is NOT optional. They are exported for training
    # and eval_cell_body.sh would otherwise evaluate the HF base model and ignore
    # the checkpoint entirely, reporting baseline numbers for every trial.
    timeout --kill-after=2m "${RUNG_TIMEOUT_MIN}m" \
    env -u MODEL -u REVISION -u OPTIM_REPO -u OPTIM_REVISION \
    CELL_DIR="$CELL_DIR" CKPT="$CELL_DIR/step-$r" EVAL_OUT="$CELL_DIR/evals/step-$r" FORCE_EVAL=0 \
    SKIP_PPL="$sp" SKIP_FK="$so" SKIP_IL="$so" SKIP_GW="$sg" \
    SKIP_VM=1 SKIP_BM=1 SKIP_PE=1 SKIP_MIA=1 SKIP_DOS=1 SKIP_NEWS=1 SKIP_MATH=1 \
    NOISE_DIR="$NOISE_DIR" NOISE_STD="$NOISE_STD" \
    EVAL_MAX_NUM_SEQS="$EVAL_MAX_NUM_SEQS" INFERENCE_MAX_NUM_SEQS="$INFERENCE_MAX_NUM_SEQS" \
    C4_TASK_FILE="$C4_TASK_FILE" IL_EXPERIMENT="$IL_EXPERIMENT" IL_MAX_TOKENS="$IL_MAX_TOKENS" \
    HF_HUB_OFFLINE="$HF_HUB_OFFLINE" HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE" \
      bash "$REPO/internal/lumi/eval_pareto_cell.sh"
    rc=$?
    case "$rc" in
      0) return 0 ;;
      124|137) echo "    WARNING: step-$r scoring hung past ${RUNG_TIMEOUT_MIN} min (attempt $attempt), killed" ;;
      *) echo "    WARNING: eval failed at step-$r (exit $rc), continuing"; return 0 ;;
    esac
  done
  echo "    WARNING: step-$r left out after two hung attempts"
}
# DISK. A 1.5B checkpoint is ~5.9 GB and the objective only needs the eval
# results, already written under evals/. KEEP_CKPT=1 keeps them.
free_ckpt () {
  if [ "${KEEP_CKPT:-0}" != "1" ] && [ -d "$CELL_DIR/step-$1" ]; then
    rm -rf "$CELL_DIR/step-$1" && echo "    freed step-$1"
  fi
}
# THE FRONTIER (wm searches, 2026-10-14). A scored rung is kept when no point
# known now (the study's finished trials, this trial's earlier rungs) has both
# a lower-or-equal C4 and a lower-or-equal poison TPR, so the full suite can run
# on the search's frontier later (optuna_hpo.py --eval-cells over the search's
# output root); a dominated rung is freed as before. forget_score.py prune
# removes the kept ones that later trials dominate. A failed check keeps the rung.
KEPT=""
keep_or_free () {
  [ -d "$CELL_DIR/step-$1" ] || return 0
  if [ "${KEEP_CKPT:-0}" = "1" ]; then KEPT="$KEPT $1"; return 0; fi
  python "$REPO/internal/uwiki/hpo/forget_score.py" keep --output-root "$OUTPUT_ROOT" --run-tag "$RUN_TAG" \
    --method "$METHOD" --cell-dir "$CELL_DIR" --step "$1"
  local rc=$?
  if [ "$rc" = 3 ]; then
    free_ckpt "$1"
  else
    [ "$rc" != 0 ] && echo "    WARNING: frontier check failed (exit $rc); step-$1 kept"
    KEPT="$KEPT $1"
  fi
}
FIRST_RUNG=""
LAST_IN=""
OVER=""
# wmarea (the full-range searches, 2026-10-15) scores its rungs exactly like wm;
# only its cap (UTIL_CAP_PCT, to C4 33) and its value (summarize_trial.py) differ
WM="$HPO_OBJECTIVE"; [ "$WM" = "wmarea" ] && WM=wm
for r in $RUNGS; do
  [ -d "$CELL_DIR/step-$r" ] || { echo "    [skip] no step-$r"; continue; }
  [ -z "$FIRST_RUNG" ] && FIRST_RUNG=$r
  if [ "$WM" = "wm" ]; then
    echo "--- C4 and watermark at step-$r ---"
    eval_rung "$r" 0 1 0
  else
    echo "--- C4 at step-$r ---"
    eval_rung "$r" 0 1 1
  fi
  # A rung whose C4 could not be read counts as inside the cap: one rung too
  # many costs minutes, stopping a good trial by mistake costs the trial.
  over=$(REPO="$REPO" ANCHOR_ROOT="$ANCHOR_ROOT" UTIL_CAP_PCT="$UTIL_CAP_PCT" \
         python "$REPO/internal/uwiki/hpo/over_cap.py" "$CELL_DIR/evals/step-$r") || over=0
  if [ "$over" = "1" ]; then
    # wm: scored too (C4 and watermark run together); the frontier rule decides below
    [ "$WM" = "wm" ] && OVER="$OVER $r"
    if [ "${EARLY_STOP:-1}" = "1" ]; then
      echo "    step-$r exceeds the ${UTIL_CAP_PCT}% cap. Later rungs cannot return"
      echo "    inside it, so the rest of the ladder is skipped."
      break
    fi
    [ "$r" != "$FIRST_RUNG" ] && [ "$WM" != "wm" ] && free_ckpt "$r"
  elif [ "$WM" = "wm" ]; then
    keep_or_free "$r"   # fully scored already
    LAST_IN=$r
  else
    # Superseded: only the last rung inside the cap is evaluated in full.
    [ -n "$LAST_IN" ] && free_ckpt "$LAST_IN"
    LAST_IN=$r
  fi
done

FULL_RUNG="${LAST_IN:-$FIRST_RUNG}"
if [ "$WM" = "wm" ]; then
  # Every rung inside the cap was scored in pass 1. With none inside, the first
  # rung (still on disk) gets its watermark, so the trial reports where the wall is.
  if [ -z "$LAST_IN" ] && [ -n "$FIRST_RUNG" ]; then
    echo "--- watermark at step-$FIRST_RUNG: no rung inside the cap ---"
    eval_rung "$FIRST_RUNG" 1 1 0
  fi
  for r in $OVER; do keep_or_free "$r"; done
elif [ -n "$FULL_RUNG" ]; then
  if [ -n "$LAST_IN" ]; then
    echo "--- full objective at step-$FULL_RUNG, the last rung inside the cap ---"
  else
    echo "--- full objective at step-$FULL_RUNG: no rung inside the cap ---"
  fi
  eval_rung "$FULL_RUNG" 1 0 0
fi

# Whatever the ladder never reached: early stop leaves later rungs on disk, and
# every driver force-saves a final epoch-*/ snapshot regardless of CKPT_STEPS.
# The rungs kept on the frontier stay.
if [ "${KEEP_CKPT:-0}" != "1" ]; then
  n=0
  for d in "$CELL_DIR"/step-* "$CELL_DIR"/epoch-*; do
    [ -d "$d" ] || continue
    s="${d##*/}"
    case "$s" in step-*) case " $KEPT " in *" ${s#step-} "*) continue;; esac;; esac
    rm -rf "$d"; n=$((n+1))
  done
  [ "$n" -gt 0 ] && echo "  freed $n unevaluated or final checkpoint(s)"
  [ -n "$KEPT" ] && echo "  kept on the frontier:$(printf ' step-%s' $KEPT)"
  rm -f "$CELL_DIR"/trainer_state.pt 2>/dev/null || true
fi

# ----------------------------------------------------------- summarize
echo "--- summarizing ---"
TRIAL="$TRIAL" METHOD="$METHOD" CELL_DIR="$CELL_DIR" RUNGS="$RUNGS" \
ANCHOR_ROOT="$ANCHOR_ROOT" IL_EXPERIMENT="$IL_EXPERIMENT" UTIL_CAP_PCT="$UTIL_CAP_PCT" \
  python "$REPO/internal/uwiki/hpo/summarize_trial.py" \
    --out "$CELL_DIR/hpo_result.json" || {
      echo "FATAL: could not summarize trial $TRIAL" >&2; exit 1; }

echo "=== trial $TRIAL done ==="
cat "$CELL_DIR/hpo_result.json"
