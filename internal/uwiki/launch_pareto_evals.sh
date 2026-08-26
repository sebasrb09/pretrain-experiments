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
# eval job per cell that has a checkpoint. Cells still training are skipped with
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
#   OUTPUT_ROOT  sweep root  (default: $HOME/pretrain-experiments/unlearning-pareto)
#   RUN_TAG      sweep tag   (default: 1B-pareto)
#   METHODS      restrict to these methods (default: every method found)
#   SKIP_ANCHORS 1 to skip the three reference points
#   ANCHORS_ONLY 1 to submit only the anchors
#   TIME         walltime per eval job (default: 4:00:00)
#   DRY_RUN      1 to print the sbatch commands without submitting
#   Anything the cell script reads (SKIP_GW, SKIP_MIA, NOISE_DIR, FORCE_EVAL...)
#   is passed through via --export=ALL.

set -u
set -o pipefail

[ -f internal/uwiki/eval_pareto_cell.sh ] \
  || { echo "ERROR: run this from the repo root" >&2; exit 1; }

OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/pretrain-experiments/unlearning-pareto}"
RUN_TAG="${RUN_TAG:-1B-pareto}"
TIME="${TIME:-4:00:00}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_ANCHORS="${SKIP_ANCHORS:-0}"
ANCHORS_ONLY="${ANCHORS_ONLY:-0}"
CELL_SCRIPT="internal/uwiki/eval_pareto_cell.sh"

SWEEP_DIR="$OUTPUT_ROOT/$RUN_TAG"

echo "============================================"
echo "  Pareto eval launch"
echo "  sweep:   $SWEEP_DIR"
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
      # A cell with no epoch-*/ is still training (or died). Skipping keeps this
      # script re-runnable as the sweep drains.
      if ! ls -d "$cell_dir"/epoch-* >/dev/null 2>&1; then
        echo "  [skip] $cell -- no checkpoint yet"
        n_skip=$((n_skip + 1))
        continue
      fi
      submit "pe-${method}-${cell}" "CELL_DIR=$cell_dir"
      n_sub=$((n_sub + 1))
    done
  done
fi

# ----------------------------------------------------------------- the anchors
if [ "$SKIP_ANCHORS" != "1" ]; then
  ANCHOR_ROOT="${ANCHOR_ROOT:-$OUTPUT_ROOT/anchors}"
  EXP_REPO="${EXP_REPO:-sbordt/OLMo-2-1B-Exp-Unlearning}"
  DI_REPO="${DI_REPO:-sbordt/OLMo-2-1B-Unlearning}"
  BASE_REV="${BASE_REV:-stage1-step100000-tokens210B}"
  UB_REV="${UB_REV:-stage1-step110000-tokens231B}"

  echo ""
  echo "--- anchors ---"
  submit "pe-anchor-baseline" \
    "MODEL=$EXP_REPO,REVISION=$BASE_REV,EVAL_OUT=$ANCHOR_ROOT/baseline"
  submit "pe-anchor-deep-ignorance" \
    "MODEL=$DI_REPO,REVISION=$BASE_REV,EVAL_OUT=$ANCHOR_ROOT/deep-ignorance"
  submit "pe-anchor-unlearn-baseline" \
    "MODEL=$EXP_REPO,REVISION=$UB_REV,EVAL_OUT=$ANCHOR_ROOT/unlearn-baseline"
  n_sub=$((n_sub + 3))
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
