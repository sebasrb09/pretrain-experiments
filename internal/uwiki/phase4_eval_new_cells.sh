#!/bin/bash
# phase4_eval_new_cells.sh -- evaluate the cells produced by the 1B pretrained
# training sweep, once that sweep finishes.
#
# Standalone on purpose. overnight_backfill.sh was already running when phase 4
# was written, and a running bash script MUST NOT be edited: bash reads the file
# incrementally by byte offset, so an edit makes it resume mid-token and execute
# garbage. This waits for the same training jobs from the outside instead.
#
# Launch it DETACHED, alongside the one already running:
#
#   setsid nohup bash internal/uwiki/phase4_eval_new_cells.sh \
#       > "$PE_WORK/phase4.log" 2>&1 < /dev/null &
#   disown
#
# It is safe to start NOW, before the training sweep has even been launched:
# it waits for the cells to appear first, so it cannot mistake "not started
# yet" for "already finished".
set -u

REPO="${REPO:-$PWD}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 1; }
PE="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"
MAXQ="${MAXQ:-160}"
POLL="${POLL:-300}"
MIA_CACHE="$PE/hf/mia-cache"
ROOT="$PE/unlearning-pareto-1B"
TAGS="1B-full-lr3e-06 1B-full-lr1e-05 1B-full-lr5e-05"
# Give up waiting for training to START after this long (default 24h), so a
# failed phase 3 does not leave this polling forever.
MAX_WAIT_START="${MAX_WAIT_START:-86400}"

log () { echo "[$(date '+%F %T')] $*"; }
nq () { squeue -u "$USER" -h 2>/dev/null | wc -l; }
ntrain () { squeue -u "$USER" -h -o %j 2>/dev/null | grep -c '^1B-full-lr'; }

any_cells () {
  local t
  for t in $TAGS; do
    [ -d "$ROOT/$t" ] && return 0
  done
  return 1
}

wait_for_room () {
  while [ "$(nq)" -ge "$MAXQ" ]; do
    log "    queue at $(nq)/$MAXQ, waiting"
    sleep "$POLL"
  done
}

log "=== phase 4 watcher started ==="

# 1. wait for the training sweep to actually begin -----------------------
waited=0
while ! any_cells && [ "$(ntrain)" -eq 0 ]; do
  if [ "$waited" -ge "$MAX_WAIT_START" ]; then
    log "  training never started after ${MAX_WAIT_START}s -- giving up."
    log "  check: tail $PE/overnight.log"
    exit 1
  fi
  log "  waiting for the training sweep to start (${waited}s so far)"
  sleep "$POLL"; waited=$((waited + POLL))
done
log "  training has started"

# 2. wait for every training job to leave the queue ----------------------
while [ "$(ntrain)" -gt 0 ]; do
  log "  $(ntrain) training job(s) still running"
  sleep "$POLL"
done
log "=== training finished; evaluating the new cells (FULL suite) ==="

# 3. evaluate. These cells have NOTHING evaluated, so every task runs --
# not the MIA/DoS pair the backfill asked for.
#
# HF_HUB_OFFLINE=1 is safe here: by now the backfill primers have populated
# the shared cache with the MIA dataset, the auto-resolved reference model and
# the gated DoS judge. If that cache is somehow empty, run online instead.
OFFLINE=1
if [ ! -d "$MIA_CACHE" ] || [ -z "$(ls -A "$MIA_CACHE" 2>/dev/null)" ]; then
  log "  WARNING: MIA cache empty -- running ONLINE at reduced throttle"
  OFFLINE=0; MAXQ=$(( MAXQ / 3 + 1 ))
fi

for tag in $TAGS; do
  if [ ! -d "$ROOT/$tag" ]; then
    log "  [skip] $tag -- no cells, that arm of training did not produce output"
    continue
  fi
  wait_for_room
  log "  evaluating $tag"
  OUTPUT_ROOT="$ROOT" RUN_TAG="$tag" \
  SKIP_ANCHORS=1 SKIP_MIA=0 SKIP_DOS=0 \
  EVAL_MAX_NUM_SEQS=1 \
  MIA_CACHE_DIR="$MIA_CACHE" MIA_REF_CACHE_DIR="$MIA_CACHE/ref" \
  HF_HUB_OFFLINE="$OFFLINE" HF_DATASETS_OFFLINE="$OFFLINE" \
  DRY_RUN=0 \
    bash "$REPO/internal/uwiki/launch_pareto_evals.sh" || \
      log "    eval launch FAILED for $tag"
done

log "=== phase 4 submitted. Export once the queue drains: ==="
log "    python internal/uwiki/audit_configs.py  --output-root $ROOT --out \$DATA/exports-1B-lumi/results_configs.csv"
log "    python internal/uwiki/export_results.py --output-root $ROOT --out \$DATA/exports-1B-lumi"
