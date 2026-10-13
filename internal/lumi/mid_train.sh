#!/bin/bash -l
#SBATCH --account=project_465003383
#SBATCH --job-name=mid-train
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# LUMI wrapper for ONE mid-training run: a checkpoint continued on the OLMo-2
# stage-2 mix with the released models' cooldown (pretrain_experiments/mid_train.py).
#
# The run is ~13 days of one GCD at micro-batch 1 (11,931 steps at ~94 s), far
# beyond small-g's three days, so it is a CHAIN: each job trains until
# MAX_HOURS, leaves a resume point, and submits the next job, until the driver
# writes mid_train.done. A job that made no progress does not resubmit.
#
#   sbatch -J mid-<name> --export=ALL,MODEL=<checkpoint dir or repo>,OUT=<output dir>,OLMO_CONFIG=<stage-2 yaml> \
#     internal/lumi/mid_train.sh
#
# Required:
#   MODEL        the checkpoint to mid-train: a local Hugging Face folder (an
#                unlearned model's step-N) or a Hub repository (with REVISION)
#   OUT          output directory; step-N checkpoints, metrics.jsonl, mid_train.done
#   OLMO_CONFIG  OLMo training config of the stage-2 mix: its data paths and seed
#                define the stream. Paths may be http (streamed) or local.
# Optional:
#   REVISION      revision of MODEL when it is a repository
#   MAX_STEPS     11931 (25B tokens at 512 x 4096)
#   LR            3.9855694839172363e-4, the rate at step 100,000; linear to zero
#   MICRO_BATCH   1. Every sequence is 4,096 tokens, so nothing is padded and a
#                 larger micro-batch does not change the objective; whether it is
#                 numerically safe on this GPU stack is a smoke test, not a given
#                 (micro-batch 2 and 4 turned padded unlearning batches NaN here).
#   GRAD_CKPT     1, gradient checkpointing, as the unlearning runs
#   RESUME_OPTIM  pretrain (default): Adam moments of OPTIM_REPO@OPTIM_REVISION;
#                 none: zeroed moments; or a path to an OLMo optim.pt
#   OPTIM_REPO / OPTIM_REVISION   default the 1B experiment model at step 100,000;
#                 they MUST be the model's size (pass the 2.7B ones for a 2.7B run)
#   CKPT_EVERY    250 steps between resume points (older ones are pruned)
#   KEEP_STEPS    2000,4000,6000,8000,10000: steps whose weights are kept
#   MAX_HOURS     70: train this long, then stop at a resume point
#   CHAIN         1: submit the next job while the run is unfinished; 0: never
#   SEED_OVERRIDE data-order seed instead of the config's

set -u
set -o pipefail

find_repo () {
  local c
  for c in "${PE_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
           "${PE_WORK:-/scratch/${PE_PROJECT:-project_465003383}/unlearning_baselines}/pretrain-experiments" "$PWD"; do
    if [ -n "$c" ] && [ -f "$c/internal/lumi/env.sh" ]; then echo "$c"; return 0; fi
  done
  return 1
}
PE_REPO="$(find_repo)" || { echo "ERROR: could not locate the pretrain-experiments checkout." >&2; exit 1; }
export PE_REPO
# shellcheck disable=SC1091
source "${PE_REPO}/internal/lumi/env.sh"
cd "$PE_REPO" || exit 1
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

: "${MODEL:?set MODEL (the checkpoint to mid-train)}"
: "${OUT:?set OUT (the output directory)}"
: "${OLMO_CONFIG:?set OLMO_CONFIG (the stage-2 OLMo training config)}"
[ -f "$OLMO_CONFIG" ] || { echo "ERROR: OLMO_CONFIG does not point at a file: $OLMO_CONFIG" >&2; exit 1; }
MAX_STEPS="${MAX_STEPS:-11931}"
MICRO_BATCH="${MICRO_BATCH:-1}"
MAX_HOURS="${MAX_HOURS:-70}"
CHAIN="${CHAIN:-1}"
# a resume point needs its trainer state; the unlearning launchers switch it off
unset NO_TRAINER_STATE

latest_step () { ls -d "$OUT"/step-* 2>/dev/null | sed 's/.*step-//' | sort -n | tail -1; }
before="$(latest_step)"; before="${before:-0}"

ARGS=(--model "$MODEL" --output-dir "$OUT" --olmo-config "$OLMO_CONFIG"
      --max-steps "$MAX_STEPS" --micro-batch "$MICRO_BATCH" --max-hours "$MAX_HOURS"
      --checkpoint-every "${CKPT_EVERY:-250}" --keep-steps "${KEEP_STEPS-2000,4000,6000,8000,10000}"
      --dtype "${DTYPE:-bfloat16}")
[ -n "${REVISION:-}" ] && ARGS+=(--revision "$REVISION")
[ -n "${LR:-}" ] && ARGS+=(--learning-rate "$LR")
[ -n "${SEED_OVERRIDE:-}" ] && ARGS+=(--seed-override "$SEED_OVERRIDE")
[ "${GRAD_CKPT:-1}" = "1" ] && ARGS+=(--gradient-checkpointing)

# Adam moments for the first job of the chain (a resumed job restores its own).
RESUME_OPTIM="${RESUME_OPTIM:-pretrain}"
if [ "$before" = "0" ] && [ "$RESUME_OPTIM" != "none" ]; then
  if [ "$RESUME_OPTIM" = "pretrain" ]; then
    OPTIM_REPO="${OPTIM_REPO:-sbordt/OLMo-2-1B-Exp-Unlearning}"
    OPTIM_REVISION="${OPTIM_REVISION:-step100000-unsharded}"
    RESUME_OPTIM="$(python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('${OPTIM_REPO}', 'optim.pt', revision='${OPTIM_REVISION}'))" 2>&1 | tail -1)"
  fi
  [ -f "$RESUME_OPTIM" ] || { echo "ERROR: no optimizer state at '$RESUME_OPTIM'" >&2; exit 1; }
  ARGS+=(--resume-optimizer-state "$RESUME_OPTIM")
fi

echo "============================================"
echo "  mid-training: $MODEL${REVISION:+@$REVISION}"
echo "  out:          $OUT   (starts at step $before of $MAX_STEPS)"
echo "  stream:       $OLMO_CONFIG"
echo "  micro-batch:  $MICRO_BATCH   moments: ${RESUME_OPTIM}"
echo "  host:         $(hostname)   job ${SLURM_JOB_ID:-none}"
echo "============================================"

python pretrain_experiments/mid_train.py "${ARGS[@]}"
rc=$?
after="$(latest_step)"; after="${after:-0}"
echo "  driver exit $rc; latest checkpoint step $after (was $before)"

if [ -f "$OUT/mid_train.done" ]; then
  echo "  finished."
elif [ "$CHAIN" != "1" ]; then
  echo "  unfinished, and CHAIN=$CHAIN: not resubmitting."
elif [ "$after" -le "$before" ]; then
  echo "  NO PROGRESS in this job: not resubmitting. Fix the cause, then submit again." >&2
  exit 1
else
  limit="$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %l 2>/dev/null)"
  sbatch -J "${SLURM_JOB_NAME:-mid-train}" --time="${limit:-72:00:00}" --export=ALL "$PE_REPO/internal/lumi/mid_train.sh"
fi
exit $rc
