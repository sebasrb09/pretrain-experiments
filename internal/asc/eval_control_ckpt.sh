#!/bin/bash
#SBATCH --account=p201378
#SBATCH --job-name=control-eval
#SBATCH --gres=gpu:1
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Convert ONE OLMo unsharded control checkpoint to HF, then evaluate it with the
# same suite, and into the same tree, as every other point on the plot.
#
# The fine-resolution control (config/control-fine-1B.yaml) is produced by the
# pretrain-experiments framework, not by the Pareto sweep, so its checkpoints sit
# at <save_folder>/<experiment>/<run>/step<ABS>-unsharded rather than in a
# <RUN_TAG>/<method>/<knob>-<value>/step-N tree. launch_pareto_evals.sh cannot
# walk that, hence this.
#
#   sbatch --export=ALL,CKPT=$DATA/control-fine/control-fine-1B/<run>/step100001-unsharded \
#          internal/asc/eval_control_ckpt.sh
#
# Env:
#   CKPT         REQUIRED, a step<ABS>-unsharded directory
#   ANCHOR_NAME  anchor subdirectory (default: unlearn-baseline-rerun)
#   START_STEP   absolute step the run continued from (default: 100000)
#   OLMO_REPO    OLMo checkout (default: $PE_SCRATCH/OLMo)
#   KEEP_HF      0 to delete the converted HF copy afterwards (default: 1)
#   SKIP_*       per-eval switches, forwarded to eval_cell_body.sh
#
# ---------------------------------------------------------------------------
# WHY A SEPARATE ANCHOR NAME
#
# Results land under anchors/<ANCHOR_NAME>/step-<RELATIVE>. export_results.py's
# endpoints() derives the normalisation denominators from anchors named exactly
# "baseline" and "deep-ignorance" and ignores every other name, so a distinct
# name is additive: it cannot move a single percentage on the page.
#
# It is also deliberately NOT "unlearn-baseline". Step 1000 of this rerun must
# reproduce the published stage1-step101000 branch before these points are
# spliced onto the control curve; keeping them apart is what makes that
# comparison possible rather than self-fulfilling.
#
# RELATIVE STEPS
#
# Anchors are indexed by steps since the start of unlearning, cells likewise,
# while OLMo counts from the start of pretraining. step100001-unsharded is
# therefore anchors/<name>/step-1. Getting this wrong does not error -- it puts
# the control on a disjoint part of the x-axis and nothing says so.
# ---------------------------------------------------------------------------

set -u
set -o pipefail
exec </dev/null

: "${CKPT:?set CKPT=<...>/step<ABS>-unsharded}"

find_repo () {
  local c
  for c in "${PE_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
           "${PE_SCRATCH:-${SCRATCH:-/scratch/fs201378/sr44833}}/pretrain-experiments" \
           "$PWD"; do
    if [ -n "$c" ] && [ -f "$c/internal/asc/env.sh" ]; then echo "$c"; return 0; fi
  done
  return 1
}
PE_REPO="$(find_repo)" || {
  echo "ERROR: could not locate the pretrain-experiments checkout." >&2; exit 1; }
export PE_REPO

# shellcheck disable=SC1091
source "${PE_REPO}/internal/asc/env.sh"

[ -d "$CKPT" ] || { echo "ERROR: no such checkpoint: $CKPT" >&2; exit 1; }

CKPT_BASE="$(basename "$CKPT")"
case "$CKPT_BASE" in
  step*-unsharded) ;;
  *) echo "ERROR: CKPT must be a step<N>-unsharded directory, got '$CKPT_BASE'" >&2
     exit 1 ;;
esac

ABS_STEP="${CKPT_BASE#step}"; ABS_STEP="${ABS_STEP%-unsharded}"
case "$ABS_STEP" in
  ''|*[!0-9]*) echo "ERROR: could not parse a step number from '$CKPT_BASE'" >&2; exit 1 ;;
esac

START_STEP="${START_STEP:-100000}"
REL_STEP=$(( ABS_STEP - START_STEP ))
if [ "$REL_STEP" -lt 0 ]; then
  echo "ERROR: $CKPT_BASE is before START_STEP=$START_STEP; refusing to write a" >&2
  echo "       negative anchor step." >&2
  exit 1
fi

ANCHOR_NAME="${ANCHOR_NAME:-unlearn-baseline-rerun}"
OLMO_REPO="${OLMO_REPO:-${PE_SCRATCH:-$SCRATCH}/OLMo}"
KEEP_HF="${KEEP_HF:-1}"

CONVERT="$OLMO_REPO/scripts/convert_olmo2_to_hf.py"
TOKENIZER="$OLMO_REPO/olmo_data/tokenizers/allenai_dolma2.json"
[ -f "$CONVERT" ]   || { echo "ERROR: no converter at $CONVERT" >&2; exit 1; }
[ -f "$TOKENIZER" ] || { echo "ERROR: no tokenizer json at $TOKENIZER" >&2; exit 1; }
# The converter torch.loads model.pt and does NOT read safetensors.
[ -f "$CKPT/model.pt" ] || {
  echo "ERROR: $CKPT has no model.pt (safetensors-only checkpoints need a" >&2
  echo "       state-dict conversion first; see OLMo2UnshardedCheckpoint.to_hf)" >&2
  exit 1; }

HF_DIR="${CKPT%-unsharded}-hf"
EVAL_OUT="${OUTPUT_ROOT}/anchors/${ANCHOR_NAME}/step-${REL_STEP}"

# Only the three axes this study reports, plus the utility axis.
: "${SKIP_VM:=1}"; : "${SKIP_BM:=1}"; : "${SKIP_PE:=1}"
export SKIP_VM SKIP_BM SKIP_PE

echo "============================================"
echo "  control checkpoint eval"
echo "  ckpt:     $CKPT"
echo "  abs step: $ABS_STEP   ->  anchor step-$REL_STEP  (START_STEP=$START_STEP)"
echo "  hf:       $HF_DIR"
echo "  out:      $EVAL_OUT"
echo "============================================"

if [ -f "$HF_DIR/config.json" ]; then
  echo "HF copy already present, skipping conversion."
else
  echo "Converting to HF..."
  python "$CONVERT" \
    --input_dir "$CKPT" \
    --output_dir "$HF_DIR" \
    --tokenizer_json_path "$TOKENIZER" \
    --no_tmp_cleanup || { echo "ERROR: conversion failed" >&2; exit 1; }
fi

mkdir -p "$EVAL_OUT"
export MODEL="$HF_DIR"
export REVISION=""
export EVAL_OUT

# Run the eval body in a SUBSHELL, not inline.
#
# eval_cell_body.sh does `set -e` partway through. Sourced directly that becomes
# active in THIS script, so the first failing eval terminates the job right
# there: the exit status below is never computed and the KEEP_HF cleanup never
# runs. eval_pareto_cell.sh gets away with sourcing it only because nothing
# follows the source. Verified: `bash -c 'source inner; rc=$?; echo reached'`
# with a failing `set -e` inner never prints.
#
# A subshell contains it, so a failed eval is reported rather than fatal, and
# the cleanup still happens.
(
  # shellcheck disable=SC1091
  source "${PE_REPO}/internal/uwiki/eval_cell_body.sh"
)
rc=$?

if [ "$rc" != "0" ]; then
  echo "WARNING: the eval suite exited $rc for $CKPT_BASE" >&2
  echo "         Keeping $HF_DIR so the failure can be retried without" >&2
  echo "         re-converting." >&2
elif [ "$KEEP_HF" = "0" ]; then
  echo "KEEP_HF=0 -> removing $HF_DIR"
  rm -rf "$HF_DIR"
fi
exit "$rc"
