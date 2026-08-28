#!/bin/bash
# Fan the eval suite out over every trained Pareto cell, plus the reference
# anchors the plot needs in order to mean anything.
#
# Run this on the login node -- it is NOT itself an sbatch script.
#
#   DRY_RUN=1 bash internal/uwiki/launch_pareto_evals.sh    # always look first
#   bash internal/uwiki/launch_pareto_evals.sh
#
# It walks <OUTPUT_ROOT>/<RUN_TAG>/<method>/<knob>-<value>/ and submits one
# eval job per CHECKPOINT (step-N/ or epoch-N/), so a run's trajectory is
# evaluated point by point. Results land in <checkpoint>/evals/. Cells still training are skipped with
# a note rather than failing, so this is safe to re-run as the sweep drains --
# and the .done markers inside each cell make re-submission cheap.
#
# THE ANCHORS. A Pareto curve of unlearning-vs-utility is unreadable without
# the two fixed points it lives between, so they are evaluated by default with
# exactly the same suite:
#
#   baseline          the model that DID see the forget set and has not been
#                     unlearned -- where every curve starts (max memorization,
#                     max utility)
#   deep-ignorance    the ground-truth model that never saw the forget set --
#                     what perfect unlearning would look like
#   unlearn-baseline  continued pretraining on the remaining data, the
#                     "just keep training" reference at step 110000
#
# Env vars:
#   OUTPUT_ROOT  sweep root  (default: $DATA/unlearning-pareto on ASC/MUSICA,
#                else $HOME/pretrain-experiments/unlearning-pareto)
#   RUN_TAG      sweep tag   (default: 1B-pareto)
#   METHODS      restrict to these methods (default: every method found)
#   SKIP_ANCHORS 1 to skip the three reference points
#   ANCHORS_ONLY 1 to submit only the anchors
#   CELL_SCRIPT  site wrapper to submit (default: ASC if internal/asc/env.sh
#                and $SCRATCH/$DATA are present, else the uwiki one)
#   TIME         walltime per eval job (default: 12:00:00)
#   DRY_RUN      1 to print the sbatch commands without submitting
#   Anything the cell script reads (SKIP_GW, SKIP_MIA, NOISE_DIR, FORCE_EVAL...)
#   is passed through via --export=ALL.

set -u
set -o pipefail

[ -f internal/uwiki/eval_cell_body.sh ] \
  || { echo "ERROR: run this from the repo root" >&2; exit 1; }

# On MUSICA/ASC the sweep lives under $DATA (permanent), matching what
# internal/asc/env.sh exports. Deriving the default from $DATA means the login
# node picks the right root without sourcing env.sh, which would module-purge.
if [ -n "${OUTPUT_ROOT:-}" ]; then
  :
elif [ -n "${DATA:-}" ]; then
  OUTPUT_ROOT="$DATA/unlearning-pareto"
else
  OUTPUT_ROOT="$HOME/pretrain-experiments/unlearning-pareto"
fi
RUN_TAG="${RUN_TAG:-1B-pareto}"
# 12h, not the original 4h: the suite grew from four evaluations to eight, and
# two of the additions are heavy -- benchmark contamination scores ~14,800 ranked
# classification queries, and denial-of-service loads an 8B judge model. A
# timeout is recoverable rather than destructive (per-eval .done markers mean a
# re-run resumes where it stopped), but it still wastes whatever eval was in
# flight. MUSICA's zen4_0768_h100x4 QOS allows up to 72h.
TIME="${TIME:-12:00:00}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_ANCHORS="${SKIP_ANCHORS:-0}"
ANCHORS_ONLY="${ANCHORS_ONLY:-0}"
# Site wrapper to submit. Default picks ASC when its env.sh is present, since
# that is where the sweep currently runs.
if [ -n "${CELL_SCRIPT:-}" ]; then
  :
elif [ -f internal/asc/env.sh ] && [ -n "${SCRATCH:-}${DATA:-}" ]; then
  CELL_SCRIPT="internal/asc/eval_pareto_cell.sh"
else
  CELL_SCRIPT="internal/uwiki/eval_pareto_cell.sh"
fi

SWEEP_DIR="$OUTPUT_ROOT/$RUN_TAG"

echo "============================================"
echo "  Pareto eval launch"
echo "  sweep:   $SWEEP_DIR"
echo "  cell:    $CELL_SCRIPT"
echo "  time:    $TIME"
echo "  dry run: $DRY_RUN"
echo "============================================"

submit () {
  # submit <job-name> <VAR=VAL,...>
  local job_name="$1" exports="$2"
  if [ "$DRY_RUN" = "1" ]; then
    echo "  [dry] sbatch -J $job_name -t $TIME --export=ALL,$exports $CELL_SCRIPT"
  else
    sbatch -J "$job_name" -t "$TIME" --export=ALL,"$exports" "$CELL_SCRIPT"
  fi
}

n_sub=0
n_skip=0

# ------------------------------------------------------------------- the cells
if [ "$ANCHORS_ONLY" != "1" ]; then
  if [ ! -d "$SWEEP_DIR" ]; then
    echo "ERROR: no sweep at $SWEEP_DIR" >&2
    echo "       Check OUTPUT_ROOT / RUN_TAG, or run the training sweep first:" >&2
    echo "         bash internal/uwiki/launch_pareto_sweep_1B.sh" >&2
    exit 1
  fi

  for method_dir in "$SWEEP_DIR"/*/; do
    [ -d "$method_dir" ] || continue
    method="$(basename "$method_dir")"
    if [ -n "${METHODS:-}" ] && ! echo " $METHODS " | grep -q " $method "; then
      continue
    fi
    echo ""
    echo "--- $method ---"
    for cell_dir in "$method_dir"*/; do
      [ -d "$cell_dir" ] || continue
      cell="$(basename "$cell_dir")"
      cell_dir="${cell_dir%/}"
      # One eval job per CHECKPOINT, not per cell. Runs now save every
      # --checkpoint-every-n-steps (default 2000) as step-N/, so a cell holds a
      # trajectory rather than a single end state. epoch-N/ is still accepted so
      # older trees keep working.
      ckpts="$(ls -d "$cell_dir"/step-* "$cell_dir"/epoch-* 2>/dev/null || true)"
      if [ -z "$ckpts" ]; then
        echo "  [skip] $cell -- no checkpoint yet"
        n_skip=$((n_skip + 1))
        continue
      fi
      for ckpt in $ckpts; do
        [ -d "$ckpt" ] || continue
        tag="$(basename "$ckpt")"
        submit "pe-${method}-${cell}-${tag}"           "CELL_DIR=$cell_dir,CKPT=$ckpt,EVAL_OUT=$ckpt/evals"
        n_sub=$((n_sub + 1))
      done
    done
  done
fi

# ----------------------------------------------------------------- the anchors
if [ "$SKIP_ANCHORS" != "1" ]; then
  ANCHOR_ROOT="${ANCHOR_ROOT:-$OUTPUT_ROOT/anchors}"
  EXP_REPO="${EXP_REPO:-sbordt/OLMo-2-1B-Exp-Unlearning}"
  DI_REPO="${DI_REPO:-sbordt/OLMo-2-1B-Unlearning}"

  # Both repos publish across 100k-110k at 2000-step intervals -- the same
  # cadence the cells checkpoint at -- so every cell checkpoint has a
  # step-matched reference in both. Only the two endpoints carry a
  # -tokensNNNB suffix; the intermediate branches are bare.
  rev_for_step () {
    case "$1" in
      100000) echo "stage1-step100000-tokens210B" ;;
      110000) echo "stage1-step110000-tokens231B" ;;
      *)      echo "stage1-step$1" ;;
    esac
  }

  # Anchors are stored under RELATIVE step directories (absolute - 100000),
  # because a cell's step-N counts from the start of unlearning while a branch
  # name counts from the start of pretraining. Converting here, in one place,
  # is what keeps cells and anchors on one x-axis; leaving it to the plot means
  # the two curves land on disjoint parts of the axis and nothing says so.
  ANCHOR_STEPS="${ANCHOR_STEPS:-100000 102000 104000 106000 108000 110000}"

  echo ""
  echo "--- anchors ---"
  for abs_step in $ANCHOR_STEPS; do
    rel=$((abs_step - 100000))
    rev="$(rev_for_step "$abs_step")"

    # baseline is the ORIGIN only. At any later step the same repo is by
    # definition the unlearn-baseline -- continued pretraining on the
    # remaining data -- so one label covers both and nothing is done twice.
    if [ "$rel" = "0" ]; then
      submit "pe-anchor-baseline-$abs_step" \
        "MODEL=$EXP_REPO,REVISION=$rev,EVAL_OUT=$ANCHOR_ROOT/baseline/step-$rel"
    else
      submit "pe-anchor-unlearn-baseline-$abs_step" \
        "MODEL=$EXP_REPO,REVISION=$rev,EVAL_OUT=$ANCHOR_ROOT/unlearn-baseline/step-$rel"
    fi
    n_sub=$((n_sub + 1))

    submit "pe-anchor-deep-ignorance-$abs_step" \
      "MODEL=$DI_REPO,REVISION=$rev,EVAL_OUT=$ANCHOR_ROOT/deep-ignorance/step-$rel"
    n_sub=$((n_sub + 1))
  done
fi

echo ""
echo "============================================"
echo "  submitted: $n_sub    skipped (no checkpoint): $n_skip"
echo "============================================"
if [ "$DRY_RUN" = "1" ]; then
  echo ""
  echo "  Dry run only. Re-run without DRY_RUN=1 to submit."
fi
echo ""
echo "  Then collect everything into one table with:"
echo "    python internal/uwiki/aggregate_pareto.py --output-root $OUTPUT_ROOT"
