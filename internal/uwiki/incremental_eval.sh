#!/bin/bash
# incremental_eval.sh -- evaluate 1B-full cells AS THEY FINISH, instead of
# waiting for the last training job.
#
#   setsid nohup bash internal/uwiki/incremental_eval.sh \
#       > "$PE_WORK/inc_eval.log" 2>&1 < /dev/null &
#   disown
#
# WHY. relaunch_1B_training.sh evaluates only after every training job has left
# the queue, so one slow cell holds back ~300 eval jobs. At the measured ~100
# minutes per full-suite evaluation that tail is 3.5-5 hours of dead time.
# launch_pareto_evals.sh already prints "[skip] <cell> -- no checkpoint yet",
# so it is safe to run again and again while training is still going: finished
# checkpoints get picked up, unfinished ones are skipped, and per-eval .done
# markers stop anything being evaluated twice.
#
# TRAINING KEEPS PRIORITY. Evals are capped well below the queue limit so the
# 38 training jobs are never starved of slots. Raise EVAL_CEIL once training
# has drained if you want the remaining evals to go faster.
set -u

REPO="${REPO:-$PWD}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 1; }
PE="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"
ROOT="$PE/unlearning-pareto-1B"
MIA_CACHE="$PE/hf/mia-cache"
TAGS="1B-full-lr3e-06 1B-full-lr1e-05 1B-full-lr5e-05"

EVAL_CEIL="${EVAL_CEIL:-110}"   # stop submitting evals above this TOTAL queue depth
ROUND="${ROUND:-1800}"          # seconds between sweeps of the tree
IDLE_ROUNDS="${IDLE_ROUNDS:-3}" # consecutive empty rounds after training ends -> stop

log  () { echo "[$(date '+%F %T')] $*"; }
nq   () { squeue -u "$USER" -h 2>/dev/null | wc -l; }
ntr  () { squeue -u "$USER" -h -o %j 2>/dev/null | grep -c '^1B-full-lr'; }

log "=== incremental evaluator started (EVAL_CEIL=$EVAL_CEIL) ==="
idle=0
while :; do
  training_left="$(ntr)"
  if [ "$(nq)" -ge "$EVAL_CEIL" ]; then
    log "  queue at $(nq) >= $EVAL_CEIL -- leaving room for training, waiting"
    sleep "$ROUND"; continue
  fi

  submitted_before="$(nq)"
  for tag in $TAGS; do
    [ -d "$ROOT/$tag" ] || continue
    [ "$(nq)" -lt "$EVAL_CEIL" ] || break
    OUTPUT_ROOT="$ROOT" RUN_TAG="$tag" \
    SKIP_ANCHORS=1 SKIP_MIA=0 SKIP_DOS=0 \
    EVAL_MAX_NUM_SEQS=1 \
    MIA_CACHE_DIR="$MIA_CACHE" MIA_REF_CACHE_DIR="$MIA_CACHE/ref" \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    DRY_RUN=0 \
      bash "$REPO/internal/uwiki/launch_pareto_evals.sh" 2>&1 \
      | grep -vE '^\s*\[skip\]' | sed 's/^/    /' || true
  done
  gained=$(( $(nq) - submitted_before ))
  log "  round done: queue $submitted_before -> $(nq) (+$gained), training left: $training_left"

  # Stop only once training is over AND several rounds have found nothing new,
  # so a gap between one cell finishing and the next does not end it early.
  if [ "$training_left" -eq 0 ] && [ "$gained" -le 0 ]; then
    idle=$((idle + 1))
    log "  nothing new and training is done ($idle/$IDLE_ROUNDS)"
    [ "$idle" -ge "$IDLE_ROUNDS" ] && break
  else
    idle=0
  fi
  sleep "$ROUND"
done

log "=== no work left. Export with: bash internal/uwiki/export_all.sh ==="
