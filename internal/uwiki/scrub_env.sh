# scrub_env.sh -- sourced by launch orchestrators; also parsed by
# internal/uwiki/hpo/optuna_hpo.py, so SCRUB_VARS below is the single list.
#
# Every launcher here submits with sbatch --export=ALL, so the job inherits the
# login shell, and the cell scripts read dozens of experiment-defining variables
# with ${VAR:-default}. Anything left in that shell therefore changes the
# experiment silently. It has already happened with:
#   OUTPUT_ROOT   internal/lumi/env.sh:127 exports the 2.7B tree on every source
#   MODEL         made 387 eval jobs measure the HF base model, not the checkpoint
#   CELL_SCRIPT   made the eval launcher refuse every tag
# and nothing stopped RESUME_OPTIM=none (zeroed Adam moments), a 2.7B NOISE_DIR
# (wrong watermark noise) or FORGET_EXPS (different forget set) doing the same.
#
# The rule this enforces: every such variable is either set explicitly by the
# launcher or absent, so the cell's own default applies. Never inherited.
#
# This is the list of variables read with a default by unlearn_cell_body.sh,
# internal/lumi/unlearn_cell.sh and eval_cell_body.sh that define WHAT is run,
# plus the launchers' own inputs. Site variables (PE_WORK, HF_HOME, PATH...)
# are deliberately absent.

SCRUB_VARS=(
  # training
  TOTAL_BATCH MICRO_BATCH EPOCHS MAX_STEPS HARD_STEP_CAP MAX_SEQ_LEN SEED
  MODEL REVISION RESUME_OPTIM OPTIM_REPO OPTIM_REVISION OLMO_CONFIG START_STEP
  RUN_TAG DTYPE FROZEN_DTYPE GRAD_CKPT OUTPUT_ROOT FORGET_EXPS
  LR LR_SCHEDULE WARMUP_FRAC WEIGHT_DECAY ADAM_BETA1 ADAM_BETA2 MAX_GRAD_NORM
  MIN_FORGET_CE RETAIN_WEIGHT SATIMP_BETA2 SIMNPO_GAMMA
  RMU_ALPHA RMU_LAYER RMU_NLAYERS RMU_STEPS
  KEEP_CHECKPOINTS NO_TRAINER_STATE CKPT_EVERY CKPT_STEPS
  METHOD VALUE METHODS VALUES CELL_SCRIPT EVAL_CELL_SCRIPT CHAIN TIME
  # evaluation
  CELL_DIR CKPT EVAL_OUT FORCE_EVAL NOISE_DIR NOISE_STD C4_TASK_FILE
  IL_EXPERIMENT IL_MAX_TOKENS BM_SPLIT PE_QUERIES PE_GENERATIONS DOS_QUERIES
  MIA_CONDITIONS MIA_EXPERIMENTS MIA_DATA_IN MIA_DATA_OUT_PKL MIA_REF_MODEL
  MIA_BATCH MIA_CACHE_DIR MIA_REF_CACHE_DIR
  SKIP_PPL SKIP_FK SKIP_VM SKIP_IL SKIP_BM SKIP_PE SKIP_GW SKIP_MIA SKIP_DOS
  SKIP_ANCHORS SKIP_EPOCH_CKPTS HF_HUB_OFFLINE HF_DATASETS_OFFLINE
  EVAL_MAX_NUM_SEQS BASE_C4_PPL UTIL_CAP_PCT
  INFERENCE_MAX_NUM_SEQS MIA_REQUIRE_CUDA
  # HPO trial inputs and switches (hpo_trial.sh)
  ANCHOR_ROOT EARLY_STOP KEEP_CKPT
  # safety overrides: only ever set explicitly, never inherited
  ALLOW_ROCM_PADDED_BATCHES
)

# scrub_inherited_env [VAR ...]
# Unsets every SCRUB_VARS entry except the ones named as arguments, which are
# the overrides a given launcher deliberately accepts from the caller. Prints
# what it removed, with values, so the log shows exactly what was ignored.
scrub_inherited_env () {
  local v a keep
  local -a removed=()
  for v in "${SCRUB_VARS[@]}"; do
    keep=0
    for a in "$@"; do [ "$v" = "$a" ] && keep=1; done
    [ "$keep" = 1 ] && continue
    if [ -n "${!v+x}" ]; then
      removed+=("$v=${!v}")
      unset "$v"
    fi
  done
  if [ "${#removed[@]}" -gt 0 ]; then
    echo "  scrubbed ${#removed[@]} inherited variable(s); the launcher sets what it needs:"
    printf '    %s\n' "${removed[@]}"
  fi
  if [ "$#" -gt 0 ]; then
    for a in "$@"; do
      [ -n "${!a+x}" ] && echo "  accepted override from caller: $a=${!a}"
    done
  fi
  return 0
}
