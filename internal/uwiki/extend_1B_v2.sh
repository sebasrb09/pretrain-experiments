#!/bin/bash
# extend_1B_v2.sh -- continue the sweep-v2 cells that the 6h walltime stopped
# before their own in-budget optimum.
#
#   DRY_RUN=1 bash internal/uwiki/extend_1B_v2.sh             # ALWAYS look first
#   DRY_RUN=0 setsid nohup bash internal/uwiki/extend_1B_v2.sh \
#       > "$PE_WORK/extend_v2.log" 2>&1 < /dev/null &
#   disown
#
# WHICH CELLS. Every retain-carrying cell stopped at step 21 and every rmu cell at
# 34. For most that is enough. For the seven below, the same configuration on the
# 916-cell ASC sweep had its best in-budget checkpoint AFTER the step this run
# reached (or exactly at it, which leaves the optimum unbracketed):
#
#   grad-diff lr=1e-05  lambda 0.5 1.0 2.0 5.0   ASC optimum at steps 30 34 24 35
#   npo       lr=3e-06  beta 0.1                 ASC optimum at step 34
#   satimp    lr=5e-05  beta1 10.0               ASC optimum at step 28
#   satimp    lr=1e-05  beta1 5.0                ASC optimum at step 21 = last we have
#
# ASC ran grad-diff and satimp at MICRO_BATCH=2 against 4 here, so this is a
# prior, not a guarantee. The real test, once the LUMI evals land: a cell needs
# extending iff its LAST checkpoint is still inside the 5% utility budget, since
# the budget boundary is absorbing (no trajectory in 122 ever left it and came
# back). Add any such cell to CELLS below and run this again.
#
# HOW. Each cell is resubmitted with exactly its sweep-v2 settings. The drivers
# --auto-resume from the highest step-N holding a trainer_state.pt, load those
# Adam moments (not the pretraining ones), and continue. CKPT_STEPS is "34,55",
# so a resumed cell writes only the two new checkpoints and NEVER rewrites a
# step directory that an eval job may be reading.
#
# If a cell has no trainer_state.pt left, it cannot resume. With ALLOW_RESTART=1
# it is retrained from step 0 instead, still writing only step-34 and step-55,
# so existing checkpoints stay untouched. Same seed and settings reproduce the
# trajectory up to GPU nondeterminism. Off by default because it costs ~16h per
# cell rather than ~10h, and because it should be a conscious choice.
set -u

REPO="${REPO:-$PWD}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 1; }
PE="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"
DRY_RUN="${DRY_RUN:-1}"
ALLOW_RESTART="${ALLOW_RESTART:-0}"
POLL="${POLL:-300}"
MAXQ="${MAXQ:-160}"

# Not taken from $OUTPUT_ROOT. A stale OUTPUT_ROOT pointing at the 2.7B tree has
# already redirected one diagnostic in this campaign.
ROOT="$PE/unlearning-pareto-1B"

# <method> <lr as sweep v2 typed it> <value as sweep v2 typed it> <grad_ckpt> <micro_batch> <frozen_dtype>
# The LR and VALUE strings must match sweep v2 EXACTLY: they build the run tag
# and the cell directory, and a different spelling resumes nothing.
CELLS="${CELLS:-
grad-diff 1e-05 0.5  1 4 float32
grad-diff 1e-05 1.0  1 4 float32
grad-diff 1e-05 2.0  1 4 float32
grad-diff 1e-05 5.0  1 4 float32
npo       3e-06 0.1  1 2 bfloat16
satimp    5e-05 10.0 1 4 float32
satimp    1e-05 5.0  1 4 float32
}"

# --- identical to sweep_1B_v2.sh -------------------------------------------
export MODEL=sbordt/OLMo-2-1B-Exp-Unlearning
export REVISION=stage1-step100000-tokens210B
export OPTIM_REPO=sbordt/OLMo-2-1B-Exp-Unlearning
export OPTIM_REVISION=step100000-unsharded
export OLMO_CONFIG=
export OUTPUT_ROOT="$ROOT"
export CELL_SCRIPT=internal/lumi/unlearn_cell.sh
export MAX_STEPS=55
export HARD_STEP_CAP=55
# --- what differs for the extension ----------------------------------------
# Only the two missing rungs. Comma-valued, so it travels via --export=ALL and
# is deliberately absent from the launcher's --export list.
export CKPT_STEPS="34,55"
# The new checkpoints need no resume point, and the end-of-run epoch-*/ copy
# duplicates step-55. Together this keeps each new checkpoint at ~3 GB.
export NO_TRAINER_STATE=1
export KEEP_CHECKPOINTS=0
RESUME_TIME="${RESUME_TIME:-16:00:00}"     # 34 steps at the measured ~15 min/step + startup
RESTART_TIME="${RESTART_TIME:-24:00:00}"   # 55 steps from scratch
EVAL_TIME="${EVAL_TIME:-12:00:00}"
SKIP_DOS="${SKIP_DOS:-1}"
MIA_CACHE="$PE/hf/mia-cache"

log () { echo "[$(date '+%F %T')] $*"; }
have_squeue () { command -v squeue >/dev/null 2>&1; }
queued () { have_squeue && squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$1"; }
nq () { have_squeue && squeue -u "$USER" -h 2>/dev/null | wc -l || echo 0; }

# ---------------------------------------------------------------- pre-flight
# The eval orchestrator waits for every 1B-v2-lr* job before evaluating. These
# resubmissions carry that name, so starting them while it is still waiting
# would hold back every eval in the sweep until the extension finishes.
if pgrep -f "internal/uwiki/sweep_1B_v2.sh" >/dev/null 2>&1; then
  elog="${EVAL_LOG:-$PE/sweep_v2_evals.log}"
  if ! grep -q "training done" "$elog" 2>/dev/null; then
    echo "REFUSING: a sweep_1B_v2.sh orchestrator is running and $elog does not"
    echo "show it past its training wait ('training done'). Its wait counts every"
    echo "1B-v2-lr* job, so submitting now would delay ALL evals by ~16h."
    echo "Let it reach the eval phase, then rerun this. (Set EVAL_LOG if its log"
    echo "is elsewhere. pgrep only sees this login node.)"
    exit 1
  fi
fi

log "=== extend sweep v2 (DRY_RUN=$DRY_RUN, ALLOW_RESTART=$ALLOW_RESTART) ==="
log "    root: $ROOT"

declare -a TODO=()       # "method lr value gc mb dtype mode"
n_res=0; n_rst=0; n_skip=0
while read -r method lr value gc mb dtype; do
  [ -n "${method:-}" ] || continue
  knob=$(case "$method" in grad-diff) echo lambda;; npo|simnpo) echo beta;;
                           satimp|wga) echo beta1;; rmu) echo c;; *) echo "";; esac)
  tag="1B-v2-lr$lr"
  cell="$ROOT/$tag/$method/$knob-$value"
  jname="$tag-$method-$knob$value"
  if [ ! -d "$cell" ]; then
    log "  SKIP $method lr=$lr $value: no cell dir $cell"; n_skip=$((n_skip+1)); continue
  fi
  if [ -d "$cell/step-55" ]; then
    log "  SKIP $method lr=$lr $value: already has step-55"; n_skip=$((n_skip+1)); continue
  fi
  if queued "$jname"; then
    log "  SKIP $method lr=$lr $value: $jname is already in the queue"; n_skip=$((n_skip+1)); continue
  fi
  last=$(ls -d "$cell"/step-* 2>/dev/null | sed 's#.*step-##' | sort -n | tail -1)
  # The step auto-resume will actually pick: the highest step-N WITH a state file.
  rs=$(for d in "$cell"/step-*; do [ -f "$d/trainer_state.pt" ] && echo "${d##*step-}"; done | sort -n | tail -1)
  if [ -n "$rs" ]; then
    mode=resume; n_res=$((n_res+1))
    log "  RESUME  $method lr=$lr $value: last step-$last, resumes from step-$rs"
  elif [ "$ALLOW_RESTART" = "1" ]; then
    mode=restart; n_rst=$((n_rst+1))
    log "  RESTART $method lr=$lr $value: last step-$last, NO trainer_state.pt, retrains from 0"
  else
    log "  BLOCKED $method lr=$lr $value: last step-$last and no trainer_state.pt anywhere,"
    log "          so it cannot resume. Rerun with ALLOW_RESTART=1 to retrain it from step 0."
    n_skip=$((n_skip+1)); continue
  fi
  TODO+=("$method $lr $value $gc $mb $dtype $mode")
done <<< "$CELLS"

log "  plan: $n_res resume (~10h each), $n_rst restart (~16h each), $n_skip skipped"
[ "${#TODO[@]}" -gt 0 ] || { log "nothing to do"; exit 0; }

# ------------------------------------------------------------------- submit
for row in "${TODO[@]}"; do
  read -r method lr value gc mb dtype mode <<< "$row"
  t="$RESUME_TIME"; [ "$mode" = "restart" ] && t="$RESTART_TIME"
  while [ "$(nq)" -ge "$MAXQ" ]; do sleep "$POLL"; done
  GRAD_CKPT="$gc" MICRO_BATCH="$mb" FROZEN_DTYPE="$dtype" TIME="$t" \
  LR="$lr" METHODS="$method" VALUES="$value" RUN_TAG="1B-v2-lr$lr" \
  DRY_RUN="$DRY_RUN" \
    bash "$REPO/internal/uwiki/launch_pareto_sweep_1B.sh" \
      | grep -E 'submitted|\[dry\]|!!' | sed 's/^/    /'
done

if [ "$DRY_RUN" != "0" ]; then
  log "[dry] stopping here. Rerun with DRY_RUN=0 to submit."
  exit 0
fi

# ------------------------------------------------------------ wait, verify
names=()
for row in "${TODO[@]}"; do
  read -r method lr value gc mb dtype mode <<< "$row"
  knob=$(case "$method" in grad-diff) echo lambda;; npo|simnpo) echo beta;; satimp|wga) echo beta1;; rmu) echo c;; esac)
  names+=("1B-v2-lr$lr-$method-$knob$value")
done
sleep 60
while :; do
  left=0; for n in "${names[@]}"; do queued "$n" && left=$((left+1)); done
  [ "$left" -eq 0 ] && break
  log "    $left extension job(s) still queued or running"; sleep "$POLL"
done

log "=== extension training done ==="
for row in "${TODO[@]}"; do
  read -r method lr value gc mb dtype mode <<< "$row"
  knob=$(case "$method" in grad-diff) echo lambda;; npo|simnpo) echo beta;; satimp|wga) echo beta1;; rmu) echo c;; esac)
  cell="$ROOT/1B-v2-lr$lr/$method/$knob-$value"
  s34=no; s55=no; [ -d "$cell/step-34" ] && s34=yes; [ -d "$cell/step-55" ] && s55=yes
  log "    $method lr=$lr $value: step-34 $s34, step-55 $s55"
done

# ----------------------------------------------- evaluate only what is new
# A checkpoint is skipped if it already has an evals/ directory (an eval job
# ran or is running on it) or if the eval launcher's job for it is queued.
# That covers the race where the main eval orchestrator, still working through
# its queue throttle, picked up a new step-34 itself.
if [ -d "$MIA_CACHE" ] && [ -n "$(ls -A "$MIA_CACHE" 2>/dev/null)" ]; then OFF=1; else OFF=0; fi
n_ev=0
for row in "${TODO[@]}"; do
  read -r method lr value gc mb dtype mode <<< "$row"
  knob=$(case "$method" in grad-diff) echo lambda;; npo|simnpo) echo beta;; satimp|wga) echo beta1;; rmu) echo c;; esac)
  tag="1B-v2-lr$lr"; cname="$knob-$value"; cell="$ROOT/$tag/$method/$cname"
  for ck in "$cell"/step-*; do
    [ -d "$ck" ] || continue
    st="$(basename "$ck")"
    jn="pe-$tag-$method-$cname-$st"      # launch_pareto_evals.sh's naming
    if [ -d "$ck/evals" ] || queued "$jn"; then continue; fi
    while [ "$(nq)" -ge "$MAXQ" ]; do sleep "$POLL"; done
    # Same stripping as the sweep's eval phase: MODEL would make the eval measure
    # the HF repo instead of $ck, and CELL_SCRIPT is the training wrapper.
    env -u MODEL -u REVISION -u OPTIM_REPO -u OPTIM_REVISION -u CELL_SCRIPT \
      SKIP_MIA=0 SKIP_DOS="$SKIP_DOS" \
      MIA_CACHE_DIR="$MIA_CACHE" MIA_REF_CACHE_DIR="$MIA_CACHE/ref" \
      HF_HUB_OFFLINE="$OFF" HF_DATASETS_OFFLINE="$OFF" \
      sbatch -J "$jn" -t "$EVAL_TIME" \
        --export=ALL,"CELL_DIR=$cell,CKPT=$ck,EVAL_OUT=$ck/evals" \
        internal/lumi/eval_pareto_cell.sh \
      && n_ev=$((n_ev+1)) || log "    eval submit FAILED for $jn"
  done
done
log "=== submitted $n_ev eval job(s) for the new checkpoints ==="
