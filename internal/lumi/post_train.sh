#!/bin/bash -l
#SBATCH --account=project_465003383
#SBATCH --job-name=post-train
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

# SFT or DPO of one mid-trained model, as the TOAA authors ran theirs
# (OLMo-Private single-training-run/scripts/ferranti/{sft,dpo}_4xH100.sh, the
# same hyperparameters for every model size): open-instruct's finetune.py and
# dpo_tune_cache.py with their exact arguments. Differences, all outside the
# optimisation: the model is written to OUT instead of pushed to the Hub, no
# wandb, and NPROC GCDs with gradient accumulation set so the effective batch
# stays theirs (4 GPUs x 1 x 32 = 128 sequences for SFT, 4 x 2 x 16 = 128 pairs
# for DPO).
#
#   sbatch -J sft-<name> --export=ALL,STAGE=sft,MODEL=<mid-trained HF dir>,OUT=<dir> internal/lumi/post_train.sh
#   sbatch -J dpo-<name> --export=ALL,STAGE=dpo,MODEL=<SFT output dir>,OUT=<dir> internal/lumi/post_train.sh
#
# Required: STAGE (sft | dpo), MODEL, OUT
# Optional:
#   OPEN_INSTRUCT  the open-instruct checkout (default $PE_WORK/open-instruct): the
#                  authors' fork sbordt/open-instruct at 914455d6, as setup_env.sh installs it
#   OI_VENV        its virtual environment (default $PE_WORK/venvs/open-instruct),
#                  made by setup_env.sh INSTALL_OPEN_INSTRUCT=1; NOT the training
#                  venv, which must not get accelerate
#   NPROC          GCDs (default 8)
#   SMOKE          N: stop after N optimizer steps into OUT-smoke (timing, NaN check)
# Datasets and the tokenizer (allenai/OLMo-2-0425-1B-SFT, its chat template) come
# from the Hub, so the job runs online.

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
W="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"

: "${STAGE:?set STAGE (sft or dpo)}"
: "${MODEL:?set MODEL (the model to train: a local HF folder)}"
: "${OUT:?set OUT (output directory)}"
OPEN_INSTRUCT="${OPEN_INSTRUCT:-$W/open-instruct}"
OI_VENV="${OI_VENV:-$W/venvs/open-instruct}"
NPROC="${NPROC:-8}"
[ -f "$OPEN_INSTRUCT/open_instruct/finetune.py" ] || { echo "ERROR: no open-instruct checkout at $OPEN_INSTRUCT" >&2; exit 1; }
[ -e "$OI_VENV/pyvenv.cfg" ] || { echo "ERROR: no venv at $OI_VENV (setup_env.sh INSTALL_OPEN_INSTRUCT=1)" >&2; exit 1; }
# The venv's bin/python links to a path that exists only inside the container:
# enter it through the module's `python` wrapper and hand over (see setup_env.sh).
oi_python () { PYTHONPATH= python -c 'import os, sys; os.execv(sys.argv[1], sys.argv[1:])' "$OI_VENV/bin/python" "$@"; }
# ray (imported by open_instruct.utils) refuses ROCR_VISIBLE_DEVICES alone, which is how SLURM
# hands GPUs out on LUMI; HIP_VISIBLE_DEVICES=0..n-1 indexes the same GPUs (see setup_env.sh)
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
  export HIP_VISIBLE_DEVICES="$(seq -s, 0 $(( $(tr ',' '\n' <<< "$ROCR_VISIBLE_DEVICES" | wc -l) - 1 )))"
fi
[ -f "$MODEL/config.json" ] || { echo "ERROR: MODEL is not a HF folder: $MODEL" >&2; exit 1; }

case "$STAGE" in
  sft) PER_DEVICE=1; TARGET=128 ;;
  dpo) PER_DEVICE=2; TARGET=128 ;;
  *) echo "ERROR: STAGE must be sft or dpo, not $STAGE" >&2; exit 1 ;;
esac
if [ $(( TARGET % (PER_DEVICE * NPROC) )) -ne 0 ]; then
  echo "ERROR: effective batch $TARGET is not a multiple of $PER_DEVICE x $NPROC GCDs" >&2; exit 1
fi
ACCUM=$(( TARGET / (PER_DEVICE * NPROC) ))
DEST="$OUT"; [ -n "${SMOKE:-}" ] && DEST="${OUT%/}-smoke"
mkdir -p "$DEST"

unset MODEL_REVISION REVISION NO_TRAINER_STATE
export WANDB_MODE=disabled HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
cd "$OPEN_INSTRUCT" || exit 1

# The authors' arguments, verbatim apart from the batch split, the output and the Hub/wandb flags.
COMMON=(--model_name_or_path "$MODEL"
        --tokenizer_name_or_path allenai/OLMo-2-0425-1B-SFT --tokenizer_name allenai/OLMo-2-0425-1B-SFT
        --tokenizer_revision main --use_slow_tokenizer False --use_flash_attn False
        --per_device_train_batch_size "$PER_DEVICE" --gradient_accumulation_steps "$ACCUM"
        --lr_scheduler_type linear --weight_decay 0.0 --logging_steps 1
        --output_dir "$DEST" --push_to_hub False --try_launch_beaker_eval_jobs False
        --do_not_randomize_output_dir True)
# do_not_randomize_output_dir: both scripts otherwise write to <output_dir>/<exp_name>
# (OUT/OLMo-2-1B-SFT/), and the next stage's MODEL=OUT would not find the model.
[ -n "${SMOKE:-}" ] && COMMON+=(--max_train_steps "$SMOKE")
if [ "$STAGE" = sft ]; then
  LAUNCH=(--mixed_precision bf16 --num_processes "$NPROC" --use_deepspeed
          --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf
          --deepspeed_multinode_launcher standard open_instruct/finetune.py)
  ARGS=(--exp_name OLMo-2-1B-SFT --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0
        --max_seq_length 4096 --learning_rate 3e-5 --warmup_ratio 0.03 --num_train_epochs 2 --seed 1)
else
  # dpo_tune_cache.py runs the reference model over the WHOLE dataset before its first
  # step, whatever --max_train_steps says: a smoke takes only the pairs its steps use
  # (an integer is a sample count, a decimal a fraction; dataset_transformation.py)
  MIX=0.999923341466187; [ -n "${SMOKE:-}" ] && MIX=$(( SMOKE * TARGET ))
  LAUNCH=(--mixed_precision bf16 --num_processes "$NPROC" open_instruct/dpo_tune_cache.py)
  ARGS=(--exp_name OLMo-2-1B-DPO --dataset_mixer_list allenai/olmo-2-0425-1b-preference-mix "$MIX"
        --max_seq_length 2048 --learning_rate 2.5e-6 --warmup_ratio 0.1 --num_train_epochs 1 --seed 111
        --use_lora False --dpo_loss_type dpo_norm --dpo_beta 5)
fi

echo "============================================"
echo "  $STAGE: $MODEL -> $DEST${SMOKE:+   SMOKE: $SMOKE steps}"
echo "  open-instruct $(git -C "$OPEN_INSTRUCT" rev-parse --short HEAD 2>/dev/null || echo '?')   venv $OI_VENV"
echo "  batch: $PER_DEVICE per GCD x $ACCUM accumulation x $NPROC GCDs = $TARGET"
echo "  host: $(hostname)   job ${SLURM_JOB_ID:-none}"
echo "============================================"
oi_python -m accelerate.commands.launch "${LAUNCH[@]}" "${COMMON[@]}" "${ARGS[@]}"
rc=$?
echo "  $STAGE exit $rc"
[ "$rc" = 0 ] && [ -z "${SMOKE:-}" ] && date > "$DEST/${STAGE}.done"
exit $rc
