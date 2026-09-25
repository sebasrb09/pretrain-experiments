#!/bin/bash
# overnight_backfill.sh -- detached orchestrator for the MIA/DoS backfill and,
# once it drains, the 1B pretrained training sweep.
#
# Run it from the repo root on the LUMI login node, DETACHED:
#
#   DRY_RUN=1 bash internal/uwiki/overnight_backfill.sh          # look first
#   setsid nohup bash internal/uwiki/overnight_backfill.sh \
#       > "$PE_WORK/overnight.log" 2>&1 < /dev/null &
#   disown
#
# Then you can close the terminal, lose the VPN, or go to sleep. Progress:
#   tail -f $PE_WORK/overnight.log
#   squeue -u $USER
#
# What it does, in order:
#   0. finds every checkpoint MISSING a MIA or DoS .done marker
#   1. primes the shared HF cache with ONE online job per model size
#   2. submits the rest OFFLINE (zero HF API calls), throttled to MAXQ
#   3. waits for them, then launches the 1B pretrained training sweep
#
# It never resubmits pe-anchor-* (those are already running), but it does count
# them toward the queue limit, because they occupy slots like anything else.
#
# Safe to re-run: per-eval .done markers mean finished work is never redone,
# and STATE_DIR records which job ids this script submitted.
set -u

REPO="${REPO:-$PWD}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 1; }
# the sweep launcher resolves internal/lumi/unlearn_cell.sh RELATIVELY,
# so everything below must run from the repo root
PE="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"
MAXQ="${MAXQ:-160}"                    # LUMI allows 210 submitted / 200 running;
                                       # 160 leaves headroom for the anchors and for
                                       # other people on the partition
POLL="${POLL:-120}"                    # seconds between queue checks
DRY_RUN="${DRY_RUN:-0}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO/internal/lumi/eval_pareto_cell.sh}"
TIME_EVAL="${TIME_EVAL:-12:00:00}"

MIA_COND="${MIA_COND:-rare_1tok_16x}"  # matches MIA_CONDITIONS default
MIA_CACHE="$PE/hf/mia-cache"
STATE_DIR="$PE/overnight-state"
mkdir -p "$STATE_DIR" "$MIA_CACHE/ref"

SUBMITTED="$STATE_DIR/submitted.ids"
: > "$SUBMITTED"
PRIMED="$STATE_DIR/primed.txt"
: > "$PRIMED"

log () { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- roots
# root|size   -- size selects which primer job seeds the MIA reference model,
# which newtoken_mia.py resolves by PARAMETER COUNT, so 1B and 2.7B differ.
ROOTS="decayed-root|1B
unlearning-pareto-1B|1B
unlearning-pareto-2.7B|2.7B"

# --------------------------------------------------- find incomplete work
# A checkpoint counts as done only when BOTH markers exist. Everything else --
# never launched, killed, or failed like the 429s -- comes back here.
find_incomplete () {   # find_incomplete <root>
  local root="$1" ckpt
  [ -d "$PE/$root" ] || return 0
  for ckpt in $(find "$PE/$root" -maxdepth 4 -type d \
                     \( -name 'step-*' -o -name 'epoch-*' \) 2>/dev/null | sort); do
    case "$ckpt" in */anchors/*) continue ;; esac
    if [ -f "$ckpt/evals/mia_${MIA_COND}.done" ] && \
       [ -f "$ckpt/evals/denial_of_service.done" ]; then
      continue
    fi
    echo "$ckpt"
  done
}

nq () { squeue -u "$USER" -h 2>/dev/null | wc -l; }

wait_for_room () {
  while [ "$(nq)" -ge "$MAXQ" ]; do
    log "    queue at $(nq)/$MAXQ, waiting ${POLL}s"
    sleep "$POLL"
  done
}

# ------------------------------------------------------------- submitting
submit_ckpt () {   # submit_ckpt <ckpt> <offline:0|1>
  local ckpt="$1" offline="$2" cell tag name exports jid
  cell="$(dirname "$ckpt")"
  tag="$(basename "$ckpt")"
  name="pe-bf-$(basename "$(dirname "$cell")")-$(basename "$cell")-$tag"
  exports="CELL_DIR=$cell,CKPT=$ckpt,EVAL_OUT=$ckpt/evals"
  exports="$exports,SKIP_MIA=0,SKIP_DOS=0"
  exports="$exports,SKIP_FK=1,SKIP_IL=1,SKIP_GW=1,SKIP_VM=1,SKIP_BM=1,SKIP_PE=1,SKIP_C4=1"
  exports="$exports,EVAL_MAX_NUM_SEQS=1"
  exports="$exports,MIA_CACHE_DIR=$MIA_CACHE,MIA_REF_CACHE_DIR=$MIA_CACHE/ref"
  if [ "$offline" = "1" ]; then
    exports="$exports,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1"
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "  [dry] sbatch -J $name -t $TIME_EVAL --export=ALL,$exports $EVAL_SCRIPT"
    return 0
  fi
  jid="$(sbatch --parsable -J "$name" -t "$TIME_EVAL" \
          --export=ALL,"$exports" "$EVAL_SCRIPT" 2>&1)" || {
    log "    SUBMIT FAILED for $ckpt: $jid"; return 1; }
  echo "$jid" >> "$SUBMITTED"
  return 0
}

wait_for_jobs () {   # wait until none of our submitted ids are in the queue
  local left
  while :; do
    left=0
    while read -r j; do
      [ -n "$j" ] || continue
      if squeue -j "$j" -h -o %i 2>/dev/null | grep -q .; then left=$((left+1)); fi
    done < "$SUBMITTED"
    [ "$left" -eq 0 ] && break
    log "    $left backfill job(s) still in the queue"
    sleep "$POLL"
  done
}

# =====================================================================
log "=== overnight backfill starting (DRY_RUN=$DRY_RUN, MAXQ=$MAXQ) ==="

ALL="$STATE_DIR/incomplete.txt"
: > "$ALL"
while IFS='|' read -r root size; do
  [ -n "$root" ] || continue
  n=0
  while read -r c; do
    [ -n "$c" ] || continue
    echo "$size|$c" >> "$ALL"; n=$((n+1))
  done < <(find_incomplete "$root")
  log "  $root: $n checkpoint(s) missing MIA and/or DoS"
done <<< "$ROOTS"

TOTAL="$(wc -l < "$ALL")"
log "  TOTAL incomplete: $TOTAL"
if [ "$TOTAL" -eq 0 ]; then
  log "  nothing to back-fill; skipping to the training sweep"
fi

# ---- 1. prime the cache, ONE ONLINE job per model size ---------------
# Everything after this runs with HF_HUB_OFFLINE=1, which makes zero API calls
# and so cannot hit the 1000-per-5-minutes quota. But offline only works once
# the dataset, the auto-resolved MIA reference model and the gated DoS judge
# are on disk -- which is what these two jobs are for. This is exactly what
# broke the post-training jobs: offline mode with an unpopulated cache.
OFFLINE=1
if [ "$TOTAL" -gt 0 ]; then
  for size in 1B 2.7B; do
    first="$(grep "^${size}|" "$ALL" | head -1 | cut -d'|' -f2-)"
    [ -n "$first" ] || continue
    log "  priming $size cache (ONLINE): $first"
    wait_for_room
    submit_ckpt "$first" 0 && echo "$first" >> "$PRIMED"
  done
  if [ "$DRY_RUN" != "1" ]; then
    log "  waiting for the primer job(s) to finish before going offline"
    wait_for_jobs
    if [ ! -d "$MIA_CACHE" ] || [ -z "$(ls -A "$MIA_CACHE" 2>/dev/null)" ]; then
      log "  WARNING: MIA cache still empty -- primers did not populate it."
      log "           Falling back to ONLINE mode at half throttle."
      OFFLINE=0; MAXQ=$(( MAXQ / 3 + 1 ))
    else
      log "  cache populated ($(du -sh "$MIA_CACHE" | cut -f1)); going OFFLINE"
    fi
  fi
fi

# ---- 2. the rest, throttled ------------------------------------------
i=0
while IFS='|' read -r size ckpt; do
  [ -n "${ckpt:-}" ] || continue
  if grep -Fxq "$ckpt" "$PRIMED" 2>/dev/null; then continue; fi
  wait_for_room
  i=$((i+1))
  log "  [$i/$TOTAL] $ckpt"
  submit_ckpt "$ckpt" "$OFFLINE" || true
done < "$ALL"
log "  all backfill jobs submitted ($i)"

# ---- 3. wait, then launch the 1B pretrained training sweep -----------
if [ "$DRY_RUN" != "1" ]; then
  log "  waiting for the backfill to drain before starting training"
  wait_for_jobs
fi
log "=== backfill complete; launching the 1B pretrained sweep ==="

# RUN_TAG carries the LR: the cell path is <tag>/<method>/<knob>-<value> and
# holds no learning rate, so two LRs for one method would overwrite each other.
# The grid is the region that actually produced in-budget cells in the ASC
# sweep -- NOT the launcher default, which pins ce-u and gradient-ascent to the
# pretraining LR 3.99e-4 and leaves zero cells under the utility cap.
launch_train () {   # launch_train <lr> <method> <values>
  wait_for_room
  log "  train: lr=$1 method=$2 values=$3"
  MODEL=sbordt/OLMo-2-1B-Exp-Unlearning \
  REVISION=stage1-step100000-tokens210B \
  GRAD_CKPT=0 \
  CELL_SCRIPT=internal/lumi/unlearn_cell.sh \
  LR="$1" METHODS="$2" VALUES="$3" RUN_TAG="1B-full-lr$1" \
  DRY_RUN="$DRY_RUN" \
    bash "$REPO/internal/uwiki/launch_pareto_sweep_1B.sh" || \
      log "    train launch FAILED for $2 @ $1"
}

launch_train 1e-05 grad-diff       "0.5 1.0 2.0 5.0"
launch_train 5e-05 grad-diff       "0.5 1.0 2.0 5.0"
launch_train 1e-05 rmu             "2.0 4.0 6.5 10.0"
launch_train 5e-05 rmu             "5.0 6.5 50.0 500.0"
launch_train 5e-05 satimp          "1.0 2.0 5.0 10.0"
launch_train 1e-05 satimp          "5.0"
launch_train 5e-05 simnpo          "0.1 0.5 1.0 2.5"
launch_train 1e-05 simnpo          "0.1"
launch_train 1e-05 wga             "0.5 1.0 2.0 5.0"
launch_train 1e-05 npo             "1e-3 1e-2 1e-1"
launch_train 3e-06 npo             "1e-1"
launch_train 3e-06 ce-u            "3e-06"
launch_train 1e-05 ce-u            "1e-05"
launch_train 1e-05 gradient-ascent "1e-05"
launch_train 5e-05 gradient-ascent "5e-05"

log "=== done. Training submitted. Evaluate the new cells with: ==="
log "    for t in 1B-full-lr3e-06 1B-full-lr1e-05 1B-full-lr5e-05; do"
log "      OUTPUT_ROOT=$PE/unlearning-pareto-1B RUN_TAG=\$t SKIP_ANCHORS=1 \\"
log "      SKIP_MIA=0 SKIP_DOS=0 EVAL_MAX_NUM_SEQS=1 MIA_CACHE_DIR=$MIA_CACHE \\"
log "      HF_HUB_OFFLINE=1 bash internal/uwiki/launch_pareto_evals.sh; done"
