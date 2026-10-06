#!/bin/bash
# sweep_1B_v2.sh -- the 1.5B pretrained sweep, relaunched.
#
#   DRY_RUN=1 bash internal/uwiki/sweep_1B_v2.sh 2>&1 | less    # ALWAYS look first
#   DRY_RUN=0 setsid nohup bash internal/uwiki/sweep_1B_v2.sh \
#       > "$PE_WORK/sweep_v2.log" 2>&1 < /dev/null &
#   disown
#
# ---------------------------------------------------------------------------
# SECOND FAILURE, 2026-10-06: run at micro-batch 4 and 2, every non-rmu cell
# turned every weight NaN on its first optimizer update. On LUMI only micro-batch
# 1 trains cleanly (verified), so every method now runs at 1, the drivers abort
# on a non-finite gradient, and launch() refuses to reuse a stale cell directory.
#
# WHY THE PREVIOUS ATTEMPT PRODUCED ZERO CHECKPOINTS, so it cannot repeat.
# All three causes were verified in the code before this script was written.
#
#  1. OPTIM_REPO defaulted to the 2.7B repo. internal/lumi/unlearn_cell.sh is
#     the 2.7B wrapper and the only training wrapper on LUMI, so it supplies
#     the entire 2.7B identity unless overridden. Confirmed on the Hub:
#     1B config.yaml has d_model=2048, the 2.7B one d_model=2880, and
#     3 x 2880 = 8640, which is exactly the "split_sizes to sum to 8640, got
#     [2048, 2048, 2048]" from the fused QKV optimizer state.
#
#  2. HARD_STEP_CAP defaults to 10000 (unlearn_cell_body.sh:108) while that
#     file's own comment at line 27 gives the measured budget as 100. Each
#     cell therefore aimed at 10,000 x 2.1M = 21B tokens, a tenth of the whole
#     pre-training run per cell, and hit the wall having written nothing.
#
#  3. CKPT_EVERY defaults to 2000 (line 446), so even a healthy 100-step cell
#     writes no step-N directory, and export_results.py globs step-* only.
#
# Verified propagation: MAX_STEPS, HARD_STEP_CAP, MICRO_BATCH, GRAD_CKPT,
# FROZEN_DTYPE, MODEL, REVISION, OLMO_CONFIG, OUTPUT_ROOT, OPTIM_REPO and
# OPTIM_REVISION are all in the forwarding loop at launch_pareto_sweep_1B.sh:231.
# CKPT_STEPS is deliberately NOT in that loop and must be exported here.
# ---------------------------------------------------------------------------
set -u

REPO="${REPO:-$PWD}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 1; }
PE="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"

# Nothing that defines the experiment may come from the login shell: env.sh
# alone exports a 2.7B OUTPUT_ROOT. Overrides this script deliberately accepts
# from the caller are named here; everything else is set below or left to the
# cell default. See internal/uwiki/scrub_env.sh.
source "$REPO/internal/uwiki/scrub_env.sh"
scrub_inherited_env MIA_CONDITIONS SKIP_DOS SKIP_VM SKIP_PE

# The watermark noise vectors, set EXPLICITLY. Nothing on LUMI resolves the
# eval body's default ($PE_DATA or $HOME/pretrain-experiments), and with the
# vectors missing the watermark is not scored. build_noise_dir.sh writes them here.
NOISE_DIR_1B="$PE/noise-vectors/OLMo-2-1B-Exp"
n_noise=$(ls "$NOISE_DIR_1B"/gaussian_poisoning_*.pkl 2>/dev/null | wc -l)
if [ "$n_noise" -eq 0 ]; then
  echo "NO NOISE VECTORS in $NOISE_DIR_1B. The watermark cannot be scored. Not launching."
  exit 1
fi
echo "  noise vectors: $NOISE_DIR_1B ($n_noise files)"

DRY_RUN="${DRY_RUN:-1}"            # SAFE DEFAULT: dry unless told otherwise
MAXQ="${MAXQ:-160}"
POLL="${POLL:-120}"

# --- the settings the previous attempt got wrong ---------------------------
export MODEL=sbordt/OLMo-2-1B-Exp-Unlearning
export REVISION=stage1-step100000-tokens210B
export OPTIM_REPO=sbordt/OLMo-2-1B-Exp-Unlearning    # (1) was 2.7B
export OPTIM_REVISION=step100000-unsharded           #     has config.yaml, optim.pt, train.pt
export OLMO_CONFIG=                                  #     empty -> unlearn_cell.sh:174 re-derives
                                                     #     it from OPTIM_REPO, which is now 1B
export OUTPUT_ROOT="$PE/unlearning-pareto-1B"        #     was unlearning-pareto-2.7B
export CELL_SCRIPT=internal/lumi/unlearn_cell.sh

export MAX_STEPS=55                                  # (2) explicit, beats HARD_STEP_CAP
export HARD_STEP_CAP=55                              #     consistent fallback
# 12h per TRAINING job. At micro-batch 1 a retain-carrying method needs roughly
# 55 x 190-250 s plus startup and nine checkpoint writes, which 6h does not
# safely cover. Over-booking costs queue position; under-booking cost us the
# first attempt.
export TIME=12:00:00

# (3) CKPT_STEPS is list-valued. sbatch --export takes a COMMA-separated list,
# so listing it there made sbatch read "1" and treat 2,3,5... as bare names:
# one checkpoint per cell, looking perfectly healthy. It reaches the job
# through --export=ALL instead, which propagates this environment verbatim.
# Schedule is the one 29 ASC cells used, extended to 55 so the step-51 optimum
# seen for ce-u is inside the range. MAX_STEPS stops at the last checkpoint so
# no step is trained that nothing reads.
export CKPT_STEPS="1,2,3,5,8,13,21,34,55"

# Evaluation: the FULL suite, every task including denial of service. The sweep
# is the reference arm and its tables need all of them. eval_cell_body.sh
# defaults SKIP_MIA and SKIP_DOS to 1, so both are switched on explicitly.
# DoS needs the gated judge meta-llama/Meta-Llama-3-8B-Instruct, which is
# checked before any eval is submitted (see "judge" below).
TIME_EVAL="${TIME_EVAL:-12:00:00}"
SKIP_DOS="${SKIP_DOS:-0}"
MIA_CACHE="$PE/hf/mia-cache"

log () { echo "[$(date '+%F %T')] $*"; }
nq  () { squeue -u "$USER" -h 2>/dev/null | wc -l; }
wait_for_room () {
  while [ "$(nq)" -ge "$MAXQ" ]; do log "    queue $(nq)/$MAXQ, waiting"; sleep "$POLL"; done
}

# ---------------------------------------------------------------- pre-flight
fail=0
for v in MODEL REVISION OPTIM_REPO OPTIM_REVISION OUTPUT_ROOT CELL_SCRIPT \
         MAX_STEPS HARD_STEP_CAP CKPT_STEPS TIME TIME_EVAL; do
  [ -n "${!v:-}" ] || { echo "PRE-FLIGHT FAIL: $v is empty"; fail=1; }
done
case "$OPTIM_REPO"  in *2.7B*) echo "PRE-FLIGHT FAIL: OPTIM_REPO is 2.7B";  fail=1 ;; esac
case "$OUTPUT_ROOT" in *2.7B*) echo "PRE-FLIGHT FAIL: OUTPUT_ROOT is 2.7B"; fail=1 ;; esac
case "$CKPT_STEPS"  in *,*) : ;; *) echo "PRE-FLIGHT FAIL: CKPT_STEPS has no comma"; fail=1 ;; esac
[ "$MAX_STEPS" -le 200 ] || { echo "PRE-FLIGHT FAIL: MAX_STEPS=$MAX_STEPS too large"; fail=1; }
[ -f "$REPO/$CELL_SCRIPT" ] || { echo "PRE-FLIGHT FAIL: no $CELL_SCRIPT"; fail=1; }
[ "$fail" -eq 0 ] || { echo "aborting without submitting anything"; exit 1; }

# SKIP_TRAIN=1 runs the EVAL PHASE ONLY, for when training was launched by an
# earlier invocation and this one must not resubmit 43 jobs. Use it after
# killing an orchestrator mid-run: the training jobs are independent sbatch
# jobs, not children of this script, so they survive its death.
SKIP_TRAIN="${SKIP_TRAIN:-0}"

log "=== 1.5B sweep v2 (DRY_RUN=$DRY_RUN, SKIP_TRAIN=$SKIP_TRAIN) ==="
log "    optim    : $OPTIM_REPO @ $OPTIM_REVISION  (d_model 2048)"
log "    output   : $OUTPUT_ROOT"
log "    budget   : MAX_STEPS=$MAX_STEPS, checkpoints at $CKPT_STEPS"
log "    walltime : train $TIME, eval $TIME_EVAL, SKIP_DOS=$SKIP_DOS"

# --------------------------------------------------------------- the grid
# Exactly the (method, learning rate, value) triples that produced IN-BUDGET
# cells on the working ASC sweep, so every cell here is already known to land
# under the 5% utility cap rather than diverging. 43 cells, 9 checkpoints each.
#
# RUN_TAG carries the learning rate because the cell path is
# <tag>/<method>/<knob>-<value> and encodes no LR, so two rates for one method
# would otherwise overwrite each other's cells.
#
# RMU backpropagates through stored activations, so it cannot use gradient
# checkpointing and needs micro-batch 1. NPO holds a frozen reference, so it
# takes bfloat16 and micro-batch 2 after the HIP OOM at float32 / 4.
launch () {   # <lr> <method> <values> <grad_ckpt> <micro_batch> <frozen_dtype>
  # REFUSE to launch into an existing cell directory. The drivers --auto-resume
  # from the highest step-N that holds a trainer_state.pt, so relaunching into
  # the directory of a cell whose weights went NaN would resume FROM the NaN
  # weights and look like a normal run. Move stale cells aside first.
  local v knob
  knob=$(case "$2" in grad-diff) echo lambda;; npo|simnpo) echo beta;;
                      satimp|wga) echo beta1;; rmu) echo c;; *) echo lr;; esac)
  for v in $3; do
    if [ -d "$OUTPUT_ROOT/1B-v2-lr$1/$2/$knob-$v" ]; then
      log "  REFUSING $2 lr=$1 $v: $OUTPUT_ROOT/1B-v2-lr$1/$2/$knob-$v exists."
      log "           It would auto-resume from that directory. Move it aside first."
      return 1
    fi
  done
  wait_for_room
  log "  train lr=$1 $2 [$3] gc=$4 mb=$5 ref=$6"
  GRAD_CKPT="$4" MICRO_BATCH="$5" FROZEN_DTYPE="$6" \
  LR="$1" METHODS="$2" VALUES="$3" RUN_TAG="1B-v2-lr$1" \
  DRY_RUN="$DRY_RUN" \
    bash "$REPO/internal/uwiki/launch_pareto_sweep_1B.sh" \
      || log "    LAUNCH FAILED: $2 @ $1"
}

if [ "$SKIP_TRAIN" = "1" ]; then
  log "  SKIP_TRAIN=1 -> not submitting any training, going straight to evals"
else
# rmu is NOT relaunched: its nine cells ran at micro-batch 1 and are valid (0 NaN
# lines in every metrics.jsonl), with checkpoints to step 34, past every rmu
# optimum seen on the ASC sweep (steps 5 to 34).
#
# Micro-batch 1 for every method, the only setting that trains without NaN on
# LUMI (sweep v2 at 4 and 2 went NaN on the first update; see the header). The
# npo keeps its bfloat16 frozen reference: the only reference dtype NPO has
# ever run with on LUMI (float32 went out of memory) and what the paper states.
launch 3e-06 npo             "0.1"                1 1 bfloat16
launch 1e-05 npo             "0.001 0.01 0.1"     1 1 bfloat16
launch 5e-05 npo             "0.1"                1 1 bfloat16
launch 1e-05 satimp          "5.0"                1 1 float32
launch 5e-05 satimp          "1.0 2.0 5.0 10.0"   1 1 float32
launch 1e-05 simnpo          "0.1"                1 1 float32
launch 5e-05 simnpo          "0.1 0.5 1.0 2.5"    1 1 float32
launch 1e-05 grad-diff       "0.5 1.0 2.0 5.0"    1 1 float32
launch 5e-05 grad-diff       "0.5 1.0 2.0 5.0"    1 1 float32
launch 3e-06 wga             "1.0"                1 1 float32
launch 1e-05 wga             "0.5 1.0 2.0 5.0"    1 1 float32
launch 5e-05 wga             "1.0"                1 1 float32
launch 3e-06 ce-u            "3e-06"              1 1 float32
launch 1e-05 ce-u            "1e-05"              1 1 float32
launch 5e-05 ce-u            "5e-05"              1 1 float32
launch 1e-05 gradient-ascent "1e-05"              1 1 float32
launch 5e-05 gradient-ascent "5e-05"              1 1 float32
fi

log "=== training submitted ==="
if [ "$DRY_RUN" != "0" ]; then
  log "[dry] stopping here. Re-run with DRY_RUN=0 to submit."
  exit 0
fi

# ------------------------------------------------- wait, then evaluate
ntrain () { squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -c '^1B-v2-lr'; }
sleep 60
log "  waiting for training to drain"
while [ "$(ntrain)" -gt 0 ]; do log "    $(ntrain) training job(s) left"; sleep "$POLL"; done

# A cell that wrote no checkpoint is the failure mode that cost us two rounds.
# Stop here rather than fanning out evaluations across empty directories.
n_ck=$(find "$OUTPUT_ROOT"/1B-v2-lr* -maxdepth 4 -type d -name 'step-*' 2>/dev/null | wc -l)
log "=== training done. step-* checkpoints written: $n_ck (expected ~387) ==="
if [ "$n_ck" -eq 0 ]; then
  log "  NO CHECKPOINTS WRITTEN. Not evaluating. Read one .out file before relaunching."
  exit 1
fi

# HF_HUB_OFFLINE with an empty cache has bitten us before, so go offline only
# if the shared MIA cache is actually primed. Shared, not per-checkpoint, is
# also what stopped the 429s.
if [ -d "$MIA_CACHE" ] && [ -n "$(ls -A "$MIA_CACHE" 2>/dev/null)" ]; then
  OFFLINE=1; log "  MIA cache primed, evaluating offline"
else
  OFFLINE=0; log "  MIA cache EMPTY, evaluating online (watch for 429s)"
fi

# judge. Offline evals can only load the DoS judge from the local hub cache, and
# a missing judge would fail DoS in every one of ~330 jobs after the other tasks
# had already spent their time. Refuse up front instead.
if [ "$SKIP_DOS" = "0" ] && [ "$OFFLINE" = "1" ]; then
  JUDGE_DIR="${HF_HOME:-$PE/hf}/hub/models--meta-llama--Meta-Llama-3-8B-Instruct"
  if ls "$JUDGE_DIR"/snapshots/*/*.safetensors >/dev/null 2>&1; then
    log "  judge cached: $JUDGE_DIR"
  else
    log "  NO CACHED JUDGE at $JUDGE_DIR, and evals run offline. Not submitting."
    log "  Prime it online once, or run with SKIP_DOS=1 knowingly."
    exit 1
  fi
fi

# Never submit evals for these tags while evals for them are still queued or
# running: a rerun (for example to add DoS) would race the jobs in flight on the
# same checkpoint. Completed tasks carry .done markers, so once the queue is
# clear a rerun only computes what is missing.
while have_pe=$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -c '^pe-1B-v2-lr'); [ "$have_pe" -gt 0 ]; do
  log "    $have_pe eval job(s) for this sweep still in flight, waiting"; sleep "$POLL"
done

for tag in 1B-v2-lr3e-06 1B-v2-lr1e-05 1B-v2-lr5e-05 1B-v2-lr1e-03; do
  [ -d "$OUTPUT_ROOT/$tag" ] || { log "  [skip] $tag not present"; continue; }
  wait_for_room
  log "  evaluating $tag"
  # env -u MODEL -u REVISION is NOT optional. This script exports them for
  # TRAINING, launch_pareto_evals.sh submits with --export=ALL, and
  # eval_cell_body.sh tests MODEL before CELL_DIR. Left set, every one of the
  # ~387 eval jobs would evaluate the pristine HF repo instead of its own
  # checkpoint, return identical baseline numbers, and exit 0.
  # -u CELL_SCRIPT plus an explicit EVAL_CELL_SCRIPT: this script exports
  # CELL_SCRIPT as the TRAINING wrapper (line 51), launch_pareto_evals.sh
  # honours an inherited CELL_SCRIPT, and its guard then refuses a training
  # wrapper and exits. Without this every tag failed and no eval launched.
  env -u MODEL -u REVISION -u OPTIM_REPO -u OPTIM_REVISION -u CELL_SCRIPT \
  EVAL_CELL_SCRIPT=internal/lumi/eval_pareto_cell.sh NOISE_DIR="$NOISE_DIR_1B" \
  EVAL_MAX_NUM_SEQS=1 INFERENCE_MAX_NUM_SEQS=8 \
  OUTPUT_ROOT="$OUTPUT_ROOT" RUN_TAG="$tag" TIME="$TIME_EVAL" \
  SKIP_ANCHORS=1 SKIP_EPOCH_CKPTS=1 SKIP_MIA=0 SKIP_DOS="$SKIP_DOS" \
  MIA_CACHE_DIR="$MIA_CACHE" MIA_REF_CACHE_DIR="$MIA_CACHE/ref" \
  HF_HUB_OFFLINE="$OFFLINE" HF_DATASETS_OFFLINE="$OFFLINE" DRY_RUN=0 \
    bash "$REPO/internal/uwiki/launch_pareto_evals.sh" || log "    eval FAILED for $tag"
done

log "=== all submitted. When the queue drains, export with: ==="
log "    python internal/uwiki/audit_configs.py  --output-root $OUTPUT_ROOT --out \$DATA/exports"
log "    python internal/uwiki/export_results.py --output-root $OUTPUT_ROOT --out \$DATA/exports --tags '1B-v2-*'"
