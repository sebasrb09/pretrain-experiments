#!/bin/bash
# relaunch_1B_training.sh -- the 1B pretrained sweep, with the three settings
# that made last night's attempt fail on every job.
#
#   setsid nohup bash internal/uwiki/relaunch_1B_training.sh \
#       > "$PE_WORK/train1b.log" 2>&1 < /dev/null &
#   disown
#
# WHY IT FAILED. internal/lumi/unlearn_cell.sh is the 2.7B wrapper -- it is the
# only training wrapper on LUMI, and it defaults the whole 2.7B identity:
#
#   OPTIM_REPO   -> sbordt/OLMo-2-2.7B-Exp-Unlearning   <-- the fatal one
#   OLMO_CONFIG  -> config.yaml fetched from OPTIM_REPO
#   OUTPUT_ROOT  -> $PE_WORK/unlearning-pareto-2.7B     (from internal/lumi/env.sh)
#
# MODEL was overridden to 1B but OPTIM_REPO was not, so every job loaded 2.7B
# Adam moments into a 1B model:
#
#   split_with_sizes expects split_sizes to sum exactly to 8640 ...
#   but got split_sizes=[2048, 2048, 2048]
#
# 8640 = 3 x 2880 (2.7B d_model); [2048]*3 is 1B. unlearn_cell_body.sh:142 warns
# "OPTIM_REPO must match the model SIZE" -- this is that failure exactly.
#
# Had it got past that, OLMO_CONFIG would have been the 2.7B config (breaking
# every retain-carrying method) and the cells would have been written under
# unlearning-pareto-2.7B, where nothing would have looked for them.
set -u

REPO="${REPO:-$PWD}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 1; }
PE="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"
MAXQ="${MAXQ:-160}"
POLL="${POLL:-120}"
DRY_RUN="${DRY_RUN:-0}"

log () { echo "[$(date '+%F %T')] $*"; }
nq  () { squeue -u "$USER" -h 2>/dev/null | wc -l; }
wait_for_room () {
  while [ "$(nq)" -ge "$MAXQ" ]; do
    log "    queue at $(nq)/$MAXQ, waiting"; sleep "$POLL"
  done
}

# launch <lr> <method> <values> <grad_ckpt> <micro_batch>
#
# GRAD_CKPT=0 only for rmu, which dies under checkpointing with
# "element 0 of tensors does not require grad". Without checkpointing the
# activations force MICRO_BATCH=1 to stay inside a 64 GB GCD; every other
# method checkpoints and can afford 4.
launch () {
  wait_for_room
  log "  lr=$1 method=$2 values=[$3] grad_ckpt=$4 micro_batch=$5"
  MODEL=sbordt/OLMo-2-1B-Exp-Unlearning \
  REVISION=stage1-step100000-tokens210B \
  OPTIM_REPO=sbordt/OLMo-2-1B-Exp-Unlearning \
  OPTIM_REVISION=step100000-unsharded \
  OLMO_CONFIG= \
  OUTPUT_ROOT="$PE/unlearning-pareto-1B" \
  CELL_SCRIPT=internal/lumi/unlearn_cell.sh \
  GRAD_CKPT="$4" MICRO_BATCH="$5" \
  LR="$1" METHODS="$2" VALUES="$3" RUN_TAG="1B-full-lr$1" \
  DRY_RUN="$DRY_RUN" \
    bash "$REPO/internal/uwiki/launch_pareto_sweep_1B.sh" \
      || log "    LAUNCH FAILED: $2 @ $1"
}

log "=== 1B pretrained sweep (corrected) ==="
log "    OPTIM_REPO  = sbordt/OLMo-2-1B-Exp-Unlearning   (was 2.7B -- the bug)"
log "    OUTPUT_ROOT = $PE/unlearning-pareto-1B          (was unlearning-pareto-2.7B)"

# Ordered by value: the methods that actually produce a curve go first, so a
# partial run is still usable. rmu and npo are the top performers at 1B;
# satimp/simnpo/grad-diff/wga carry the widest in-budget hyperparameter ranges.
launch 5e-05 rmu             "5.0 6.5 50.0 500.0" 0 1
launch 1e-05 rmu             "2.0 4.0 6.5 10.0"   0 1
launch 1e-05 npo             "1e-3 1e-2 1e-1"     1 4
launch 5e-05 satimp          "1.0 2.0 5.0 10.0"   1 4
launch 5e-05 simnpo          "0.1 0.5 1.0 2.5"    1 4
launch 1e-05 grad-diff       "0.5 1.0 2.0 5.0"    1 4
launch 5e-05 grad-diff       "0.5 1.0 2.0 5.0"    1 4
launch 1e-05 wga             "0.5 1.0 2.0 5.0"    1 4
launch 1e-05 satimp          "5.0"                1 4
launch 1e-05 simnpo          "0.1"                1 4
launch 3e-06 npo             "1e-1"               1 4
launch 3e-06 ce-u            "3e-06"              1 4
launch 1e-05 ce-u            "1e-05"              1 4
launch 1e-05 gradient-ascent "1e-05"              1 4
launch 5e-05 gradient-ascent "5e-05"              1 4

log "=== all training launches submitted ==="

# ---- wait for training, then evaluate the new cells ------------------
if [ "$DRY_RUN" = "1" ]; then
  log "[dry] would wait for training, then evaluate. Stopping here."
  exit 0
fi

ntrain () { squeue -u "$USER" -h -o %j 2>/dev/null | grep -c '^1B-full-lr'; }
log "  waiting for the training sweep to finish"
sleep 60                                  # let the first jobs register
while [ "$(ntrain)" -gt 0 ]; do
  log "    $(ntrain) training job(s) left"
  sleep "$POLL"
done
log "=== training finished; evaluating the new cells (FULL suite) ==="

# These cells have NOTHING evaluated, so every task runs -- not the MIA/DoS
# pair the backfill asked for. The shared MIA cache is warm from last night,
# so offline is safe; if it is not, fall back to online at reduced throttle.
MIA_CACHE="$PE/hf/mia-cache"
OFFLINE=1
if [ ! -d "$MIA_CACHE" ] || [ -z "$(ls -A "$MIA_CACHE" 2>/dev/null)" ]; then
  log "  WARNING: MIA cache empty -- evaluating ONLINE at reduced throttle"
  OFFLINE=0; MAXQ=$(( MAXQ / 3 + 1 ))
fi

for tag in 1B-full-lr3e-06 1B-full-lr1e-05 1B-full-lr5e-05; do
  if [ ! -d "$PE/unlearning-pareto-1B/$tag" ]; then
    log "  [skip] $tag -- no cells were produced"
    continue
  fi
  wait_for_room
  log "  evaluating $tag"
  OUTPUT_ROOT="$PE/unlearning-pareto-1B" RUN_TAG="$tag" \
  SKIP_ANCHORS=1 SKIP_MIA=0 SKIP_DOS=0 \
  EVAL_MAX_NUM_SEQS=1 \
  MIA_CACHE_DIR="$MIA_CACHE" MIA_REF_CACHE_DIR="$MIA_CACHE/ref" \
  HF_HUB_OFFLINE="$OFFLINE" HF_DATASETS_OFFLINE="$OFFLINE" \
  DRY_RUN=0 \
    bash "$REPO/internal/uwiki/launch_pareto_evals.sh" || \
      log "    eval launch FAILED for $tag"
done

log "=== DONE. Export when the queue drains: ==="
log "    bash internal/uwiki/export_all.sh"
