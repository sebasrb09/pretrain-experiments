#!/bin/bash -l
#SBATCH --account=project_465003383
#SBATCH --job-name=mid-train
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# LUMI wrapper for ONE mid-training run, exactly as the TOAA authors ran theirs
# (OLMo-Private single-training-run/mid-training/<size>/{exp,baseline}/run.sh):
# OLMo's own scripts/train.py with their stage-2 yaml and --load_path, on one
# node (8 GCDs; they used 8 H100). See pretrain_experiments/mid_train.py.
#
# The run takes days (25B tokens), longer than one job, so it is a CHAIN: each
# job stops itself at TIME_LIMIT_H (OLMo's time_limit, which saves an unsharded
# checkpoint) and submits the next; the trainer's try_load_latest_save resumes
# from the newest checkpoint in OUT with the data position restored. The first
# job finds none and starts from START with the step counter and data at zero
# (the yaml's restore_dataloader: false, as the authors). Done when
# OUT/step11931-unsharded exists. A job that made no progress does not resubmit.
#
#   sbatch -J mid-<name> --export=ALL,START=<OLMo checkpoint dir>,OUT=<dir>,STAGE2=<local yaml>,OLMO_REPO=<OLMo-Private> \
#     internal/lumi/mid_train.sh
#
# Required:
#   START     OLMo unsharded checkpoint: model.pt + optim.pt + train.pt. For an
#             unlearned model, the output of `mid_train.py prepare`; for the
#             experiment model itself, OLMo-2-1B-Exp-Unlearning step100000-unsharded
#   OUT       where the trainer writes step<N>-unsharded every 1,000 steps
#   STAGE2    the authors' stage-2 yaml with local data paths (`mid_train.py config`)
#   OLMO_REPO the OLMo-Private checkout whose scripts/train.py runs (put first
#             on PYTHONPATH, ahead of the OLMo fork installed in the container)
# Optional:
#   MICRO_BATCH   per-GCD micro-batch; default the yaml's (2). Sequences are
#                 4,096 tokens unpadded, so it does not change the objective;
#                 SMOKE first (micro-batch 2 turned padded HF batches NaN here)
#   ACT_CKPT      1: --activation_checkpointing=whole_layer (if 64 GB per GCD is
#                 not enough; recomputation, the same numbers)
#   TIME_LIMIT_H  70: hours before the trainer stops and checkpoints
#   SMOKE         N: train N steps into OUT-smoke/ and stop (timing and NaN check);
#                 never chains
#   CHAIN         1: resubmit while unfinished; 0: never
# The yaml's flash_attention: true is switched off: no flash-attn on LUMI (PyTorch SDPA instead).
# wandb is switched off entirely (--wandb=null): the trainer's wandb.finish(quiet=...) does not
# exist in the container's wandb and crashed the first smoke run on exit.

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

: "${START:?set START (OLMo unsharded checkpoint to start from)}"
: "${OUT:?set OUT (output directory)}"
: "${STAGE2:?set STAGE2 (the stage-2 yaml with local data paths, from mid_train.py config)}"
: "${OLMO_REPO:?set OLMO_REPO (the OLMo-Private checkout)}"
[ -f "$OLMO_REPO/scripts/train.py" ] || { echo "ERROR: no scripts/train.py under OLMO_REPO=$OLMO_REPO" >&2; exit 1; }
[ -f "$STAGE2" ] || { echo "ERROR: STAGE2 is not a file: $STAGE2" >&2; exit 1; }
# the trainer reads a load_path as unsharded only if its name ends in -unsharded
case "${START%/}" in *-unsharded) ;; *)
  echo "ERROR: START must be a folder named *-unsharded (the OLMo trainer decides how to load by the name): $START" >&2; exit 1;; esac
for f in model train optim; do     # .pt or .safetensors: the OLMo trainer reads either
  [ -e "$START/$f.pt" ] || [ -e "$START/$f.safetensors" ] || {
    [ "$f" = optim ] && echo "ERROR: $START has no optimizer state" >&2 || echo "ERROR: $START has no $f.pt" >&2; exit 1; }
done
if grep -q "^ *- http://" "$STAGE2"; then
  echo "ERROR: $STAGE2 still reads data over HTTP; make the local copy with mid_train.py config" >&2; exit 1
fi
MAX_STEPS=11931
DONE="$OUT/step${MAX_STEPS}-unsharded"
SAVE="$OUT"; [ -n "${SMOKE:-}" ] && SAVE="${OUT%/}-smoke"
mkdir -p "$SAVE"

latest_step () { ls -d "$SAVE"/step*-unsharded 2>/dev/null | sed 's/.*step\([0-9]*\)-unsharded/\1/' | sort -n | tail -1; }
before="$(latest_step)"; before="${before:-0}"

export PYTHONPATH="${OLMO_REPO}${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
# a resumed job must restore its own state; nothing of the unlearning launchers applies
unset NO_TRAINER_STATE MODEL REVISION

ARGS=("$STAGE2" "--save_folder=$SAVE" "--save_overwrite=True" "--save_interval=null"
      "--save_interval_unsharded=1000" "--eval_on_load=False" "--load_path=$START"
      "--try_load_latest_save=True" "--model.flash_attention=False" "--wandb=null"
      "--time_limit=$(( ${TIME_LIMIT_H:-70} * 3600 ))" "--sharded_checkpointer=torch_legacy")
# sharded_checkpointer: the authors' yaml says olmo_core, whose saver passes torch's
# DefaultSavePlanner an argument (enable_plan_caching) that the container's torch 2.6 lacks
# (the 2026-10-16 smoke died on it). The trainer writes a sharded checkpoint when the time
# limit stops it (and as the final one when unsharded saves are off), so every job would
# die there. torch_legacy is OLMo's default; the unsharded saves never used olmo_core.
[ -n "${MICRO_BATCH:-}" ] && ARGS+=("--device_train_microbatch_size=$MICRO_BATCH")
[ "${ACT_CKPT:-0}" = "1" ] && ARGS+=("--activation_checkpointing=whole_layer")
[ -n "${SMOKE:-}" ] && ARGS+=("--stop_at=$SMOKE" "--save_interval_unsharded=null")

echo "============================================"
echo "  mid-training (OLMo trainer): $START"
echo "  out:     $SAVE   (latest step $before of $MAX_STEPS)${SMOKE:+   SMOKE: $SMOKE steps}"
echo "  config:  $STAGE2"
echo "  olmo:    $OLMO_REPO ($(git -C "$OLMO_REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
echo "  host:    $(hostname)   job ${SLURM_JOB_ID:-none}   GCDs ${SLURM_GPUS_ON_NODE:-?}"
echo "============================================"

python -m torch.distributed.run --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-8}" \
  "$OLMO_REPO/scripts/train.py" "${ARGS[@]}"
rc=$?
after="$(latest_step)"; after="${after:-0}"
echo "  trainer exit $rc; latest checkpoint step $after (was $before)"

if [ -n "${SMOKE:-}" ]; then
  echo "  smoke run: read the throughput (tokens/sec) and loss lines above; nothing chains."
elif [ -d "$DONE" ]; then
  echo "  finished: $DONE"
elif [ "${CHAIN:-1}" != "1" ]; then
  echo "  unfinished, and CHAIN=${CHAIN}: not resubmitting."
elif [ "$after" -le "$before" ]; then
  echo "  NO PROGRESS in this job: not resubmitting. Fix the cause, then submit again." >&2
  exit 1
else
  sbatch -J "${SLURM_JOB_NAME:-mid-train}" --export=ALL "$PE_REPO/internal/lumi/mid_train.sh"
fi
exit $rc
