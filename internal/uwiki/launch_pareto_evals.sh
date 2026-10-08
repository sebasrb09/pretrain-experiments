#!/bin/bash
# Fan the eval suite out over every trained Pareto cell, plus the reference
# anchors the plot needs in order to mean anything.
#
# Run this on the login node -- it is NOT itself an sbatch script.
#
#   DRY_RUN=1 bash internal/uwiki/launch_pareto_evals.sh    # always look first
#   bash internal/uwiki/launch_pareto_evals.sh
#
# It walks <OUTPUT_ROOT>/<RUN_TAG>/<method>/<knob>-<value>/ and submits one
# eval job per CHECKPOINT (step-N/ or epoch-N/), so a run's trajectory is
# evaluated point by point. Results land in <checkpoint>/evals/. Cells still training are skipped with
# a note rather than failing, so this is safe to re-run as the sweep drains --
# and the .done markers inside each cell make re-submission cheap.
#
# THE ANCHORS. A Pareto curve of unlearning-vs-utility is unreadable without
# the two fixed points it lives between, so they are evaluated by default with
# exactly the same suite:
#
#   baseline          the model that DID see the forget set and has not been
#                     unlearned -- where every curve starts (max memorization,
#                     max utility)
#   deep-ignorance    the ground-truth model that never saw the forget set --
#                     what perfect unlearning would look like
#   unlearn-baseline  continued pretraining on the remaining data, the
#                     "just keep training" reference at step 110000
#
# Env vars:
#   OUTPUT_ROOT  sweep root  (default: $DATA/unlearning-pareto on ASC/MUSICA,
#                else $HOME/pretrain-experiments/unlearning-pareto)
#   RUN_TAG      sweep tag   (default: 1B-pareto). May be a glob, quoted, e.g.
#                RUN_TAG='1B-v2-*': every matching sweep directory is walked.
#   METHODS      restrict to these methods (default: every method found)
#   SKIP_ANCHORS 1 to skip the three reference points
#   ANCHORS_ONLY 1 to submit only the anchors
#   SKIP_EPOCH_CKPTS 1 to evaluate only step-N/ checkpoints, skipping the
#                end-of-run epoch-N/ duplicate each truncated cell writes
#   EVAL_CELL_SCRIPT  site EVAL wrapper to submit (default: ASC if internal/asc/env.sh
#                and $SCRATCH/$DATA are present, else the uwiki one). NOT
#                CELL_SCRIPT -- that is the TRAINING wrapper and leaks between
#                launchers; it is still accepted but validated.
#   TIME         walltime per eval job (default: 12:00:00)
#   DRY_RUN      1 to print the sbatch commands without submitting
#   Anything the cell script reads (SKIP_GW, SKIP_MIA, NOISE_DIR, FORCE_EVAL...)
#   is passed through via --export=ALL.
#
# REEVAL MODE -- recompute chosen tasks at batch 1 (LUMI):
#   REEVAL="fk il bm" RUN_TAG='1B-v2-*' OUTPUT_ROOT=$PE_WORK/unlearning-pareto-1B \
#     DRY_RUN=1 bash internal/uwiki/launch_pareto_evals.sh
# REEVAL lists tasks out of: fk il bm vm pe dos mia news math. On LUMI every eval that pads
# on the left gave wrong numbers at INFERENCE_MAX_NUM_SEQS=8 (insertion 12.84
# vs 3.60 at batch 1 on the same model; see internal/lumi/env.sh), so this
# mode scrubs the inherited environment (keeping only OUTPUT_ROOT, RUN_TAG,
# METHODS, TIME, EVAL_CELL_SCRIPT), recomputes exactly the listed tasks with
# FORCE_EVAL=1 (old result deleted first, so a failure leaves a gap, never the
# old number), runs at batch 1 and offline, insertion on the one experiment
# the export reads, step-N checkpoints only, no anchors. Every other task keeps
# its result. Raise TIME for dos/pe/vm (default here 03:00:00).

set -u
set -o pipefail

[ -f internal/uwiki/eval_cell_body.sh ] \
  || { echo "ERROR: run this from the repo root" >&2; exit 1; }

# ------------------------------------------------------------- REEVAL mode
if [ -n "${REEVAL:-}" ]; then
  source internal/uwiki/scrub_env.sh
  scrub_inherited_env OUTPUT_ROOT RUN_TAG METHODS TIME EVAL_CELL_SCRIPT EVAL_BATCH
  declare -A _TASK_FLAG=([fk]=SKIP_FK [il]=SKIP_IL [bm]=SKIP_BM [vm]=SKIP_VM
                         [pe]=SKIP_PE [dos]=SKIP_DOS [mia]=SKIP_MIA [news]=SKIP_NEWS
                         [math]=SKIP_MATH)
  for _f in SKIP_PPL SKIP_GW SKIP_FK SKIP_IL SKIP_BM SKIP_VM SKIP_PE SKIP_DOS SKIP_MIA SKIP_NEWS SKIP_MATH; do
    export "$_f=1"
  done
  for _t in $REEVAL; do
    [ -n "${_TASK_FLAG[$_t]:-}" ] \
      || { echo "ERROR: unknown REEVAL task '$_t' (use: fk il bm vm pe dos mia news math)" >&2; exit 1; }
    export "${_TASK_FLAG[$_t]}=0"
  done
  _PE="${PE_WORK:-/scratch/project_465003383/unlearning_baselines}"
  export FORCE_EVAL=1 EVAL_MAX_NUM_SEQS=1 INFERENCE_MAX_NUM_SEQS=1 \
         IL_EXPERIMENT=knowledge-acquisition IL_MAX_TOKENS=1000000 \
         BM_SPLITS="0 1 2 3 4 5 6 7 8" MIA_CONDITIONS=paper \
         MIA_BATCH=1 MIA_CACHE_DIR="$_PE/hf/mia-cache-b1" MIA_REF_CACHE_DIR="$_PE/hf/mia-cache-b1/ref" \
         NEWS_N=0 NEWS_N_GENERATE=0 MATH_OPS="1 3 5" \
         PE_QUERIES=1000 PE_GENERATIONS=1 DOS_QUERIES=1000 \
         HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  # EVAL_BATCH (default 1). Above 1 the evaluations run batched with EAGER
  # attention, the only attention that pads correctly on LUMI. Tested
  # 2026-10-08 on the Baseline anchor: insertion 3.6034 at eager batch 8
  # against 12.84 with the default attention; eager batch 8 agrees with eager
  # batch 1 to numerical noise; news 3.6x faster. Eager itself differs from the
  # default attention in arithmetic order only (knowledge probe -3 to -4%,
  # greedy generations diverge at near-ties, condition averages unchanged), so
  # a task's cells and its anchors must be measured with the SAME setting.
  # Knowledge and insertion were measured everywhere with the default
  # attention at batch 1, so they are refused here.
  EVAL_BATCH="${EVAL_BATCH:-1}"
  if [ "$EVAL_BATCH" != "1" ]; then
    for _t in $REEVAL; do
      case "$_t" in fk|il)
        echo "ERROR: REEVAL '$_t' with EVAL_BATCH=$EVAL_BATCH: knowledge and insertion are" >&2
        echo "  measured with the default attention at batch 1 on every cell and anchor." >&2
        exit 1 ;;
      esac
    done
    export INFERENCE_MAX_NUM_SEQS="$EVAL_BATCH" INFERENCE_ATTN_IMPL=eager ALLOW_ROCM_PADDED_BATCHES=1
  fi
  SKIP_ANCHORS=1; ANCHORS_ONLY=0; SKIP_EPOCH_CKPTS=1
  TIME="${TIME:-03:00:00}"
  # Offline, as the sweep's own evals ran: datasets, MIA reference model and
  # the DoS judge are in the shared cache, and ~200 jobs starting together
  # online is what hit the HF rate limit before.
  _JUDGE="${HF_HOME:-$_PE/hf}/hub/models--meta-llama--Meta-Llama-3-8B-Instruct"
  if [ "$SKIP_DOS" = "0" ] && ! ls "$_JUDGE"/snapshots/*/*.safetensors >/dev/null 2>&1; then
    echo "ERROR: no cached DoS judge under $_JUDGE, and REEVAL runs offline." >&2
    exit 1
  fi
  # Without the conditions file every news job fails after loading the model.
  if [ "$SKIP_NEWS" = "0" ] && [ ! -s resources/train-once-answer-all/muse_news_conditions.jsonl ]; then
    echo "ERROR: resources/train-once-answer-all/muse_news_conditions.jsonl is missing; build it first:" >&2
    echo "  python pretrain_experiments/evaluation/train-once-answer-all/verbatim_memorization.py \\" >&2
    echo "    --build-news-conditions resources/train-once-answer-all/muse_news_conditions.jsonl" >&2
    exit 1
  fi
  echo "REEVAL: $REEVAL   batch ${INFERENCE_MAX_NUM_SEQS}, attention ${INFERENCE_ATTN_IMPL:-default}"
  env | grep '^SKIP_' | sort | sed 's/^/  /'
fi

# On MUSICA/ASC the sweep lives under $DATA (permanent), matching what
# internal/asc/env.sh exports. Deriving the default from $DATA means the login
# node picks the right root without sourcing env.sh, which would module-purge.
if [ -n "${OUTPUT_ROOT:-}" ]; then
  :
elif [ -n "${DATA:-}" ]; then
  OUTPUT_ROOT="$DATA/unlearning-pareto"
else
  OUTPUT_ROOT="$HOME/pretrain-experiments/unlearning-pareto"
fi
RUN_TAG="${RUN_TAG:-1B-pareto}"
# 12h, not the original 4h: the suite grew from four evaluations to eight, and
# two of the additions are heavy -- benchmark contamination scores ~14,800 ranked
# classification queries, and denial-of-service loads an 8B judge model. A
# timeout is recoverable rather than destructive (per-eval .done markers mean a
# re-run resumes where it stopped), but it still wastes whatever eval was in
# flight. MUSICA's zen4_0768_h100x4 QOS allows up to 72h.
TIME="${TIME:-12:00:00}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_ANCHORS="${SKIP_ANCHORS:-0}"
ANCHORS_ONLY="${ANCHORS_ONLY:-0}"
# Site wrapper to submit. Default picks ASC when its env.sh is present, since
# that is where the sweep currently runs.
#
# EVAL_CELL_SCRIPT, not CELL_SCRIPT. The training sweep launcher reads
# CELL_SCRIPT for the TRAINING wrapper, and a value exported for one leaks into
# the other: `CELL_SCRIPT=internal/asc/unlearn_cell_1B.sh` left in the login
# shell made every eval job here run the training body and die on `set METHOD`.
# It happened three times, because `unset` only lasts for one shell. Two
# incompatible things sharing one variable name is the actual bug; separate
# names make the collision impossible. CELL_SCRIPT is still honoured so older
# invocations keep working, but it is validated below either way.
if [ -n "${EVAL_CELL_SCRIPT:-}" ]; then
  CELL_SCRIPT="$EVAL_CELL_SCRIPT"
elif [ -n "${CELL_SCRIPT:-}" ]; then
  :
# /appl/local/containers exists only on LUMI and needs no modules loaded to
# test, so it identifies the site at SUBMIT time. Without this branch LUMI fell
# through to the u:wiki default and every job was rejected with
# "invalid partition specified: p_datamining" -- u:wiki's partition, on a
# cluster that has never heard of it.
elif [ -d /appl/local/containers ] && [ -f internal/lumi/env.sh ]; then
  CELL_SCRIPT="internal/lumi/eval_pareto_cell.sh"
elif [ -f internal/asc/env.sh ] && [ -n "${SCRATCH:-}${DATA:-}" ]; then
  CELL_SCRIPT="internal/asc/eval_pareto_cell.sh"
else
  CELL_SCRIPT="internal/uwiki/eval_pareto_cell.sh"
fi

# CELL_SCRIPT is honoured verbatim above, which makes a TRAINING wrapper left
# over in the environment -- exported for an earlier sweep launch, or carried in
# by --export=ALL -- silently become the script every eval job runs. The job
# still gets its `pe-` name from this launcher, so the queue looks right, and
# each one dies deep inside the training body with
#   "METHOD: set METHOD (gradient-ascent|grad-diff|npo|...)"
# which names a variable this launcher never sets and does not mention evals at
# all. Every checkpoint of every tag fails the same way, at full walltime cost.
# Check what the file actually SOURCES rather than trusting its name -- matching
# the bare string anywhere passes internal/lumi/unlearn_cell.sh, whose comments
# discuss eval_cell_body.sh at length while it sources the training body.
sources_body () {  # sources_body <script> <body-filename>
  grep -qE "^[[:space:]]*(source|\.)[[:space:]].*$2" "$1" 2>/dev/null
}
if ! sources_body "$CELL_SCRIPT" "eval_cell_body\.sh"; then
  echo "ERROR: CELL_SCRIPT does not look like an eval wrapper:" >&2
  echo "         $CELL_SCRIPT" >&2
  echo "       It never sources internal/uwiki/eval_cell_body.sh, so every job" >&2
  echo "       submitted here would run something other than the eval suite." >&2
  if sources_body "$CELL_SCRIPT" "unlearn_cell_body.sh"; then
    echo "       This is a TRAINING wrapper. It is almost certainly still set" >&2
    echo "       from a sweep launch: run 'unset CELL_SCRIPT' and try again." >&2
  fi
  exit 1
fi

# RUN_TAG may be a glob; unquoted on purpose so it expands.
SWEEP_DIRS="$(ls -d "$OUTPUT_ROOT"/$RUN_TAG 2>/dev/null || true)"

echo "============================================"
echo "  Pareto eval launch"
echo "  sweep:   $OUTPUT_ROOT/$RUN_TAG ($(echo $SWEEP_DIRS | wc -w) director$( [ "$(echo $SWEEP_DIRS | wc -w)" = 1 ] && echo y || echo ies))"
echo "  cell:    $CELL_SCRIPT"
echo "  time:    $TIME"
echo "  dry run: $DRY_RUN"
echo "============================================"

# PER-JOB throttle. A tag is ~150 checkpoints and they are submitted in one
# pass, so a room check before each TAG (what the sweep orchestrators do) lets a
# tag start at 159 queued jobs and burst to ~310, far past LUMI's per-user
# limit (small-g: MaxSubmit 210). The excess sbatch calls fail, and before this
# they were ignored: the checkpoints were counted as submitted and silently never
# evaluated. Now every submission waits for room under SUBMIT_CAP, a failed one
# is retried, and anything still failing is listed and makes this exit non-zero.
SUBMIT_CAP="${SUBMIT_CAP:-195}"
n_fail=0
failed=""
submit () {
  # submit <job-name> <VAR=VAL,...>
  local job_name="$1" exports="$2" try out
  if [ "$DRY_RUN" = "1" ]; then
    echo "  [dry] sbatch -J $job_name -t $TIME --export=ALL,$exports $CELL_SCRIPT"
    return 0
  fi
  while [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -ge "$SUBMIT_CAP" ]; do
    echo "  queue at SUBMIT_CAP=$SUBMIT_CAP, waiting to submit $job_name"
    sleep 60
  done
  for try in 1 2 3; do
    if out=$(sbatch -J "$job_name" -t "$TIME" --export=ALL,"$exports" "$CELL_SCRIPT" 2>&1); then
      echo "$out"
      return 0
    fi
    echo "  sbatch failed for $job_name (try $try/3): $out" >&2
    sleep 60
  done
  n_fail=$((n_fail + 1))
  failed="$failed $job_name"
  return 1
}

n_sub=0
n_skip=0

# ------------------------------------------------------------------- the cells
if [ "$ANCHORS_ONLY" != "1" ]; then
  if [ -z "$SWEEP_DIRS" ]; then
    echo "ERROR: no sweep at $OUTPUT_ROOT/$RUN_TAG" >&2
    echo "       Check OUTPUT_ROOT / RUN_TAG, or run the training sweep first:" >&2
    echo "         bash internal/uwiki/launch_pareto_sweep_1B.sh" >&2
    exit 1
  fi

  for SWEEP_DIR in $SWEEP_DIRS; do
  tag_name="$(basename "$SWEEP_DIR")"
  for method_dir in "$SWEEP_DIR"/*/; do
    [ -d "$method_dir" ] || continue
    method="$(basename "$method_dir")"
    if [ -n "${METHODS:-}" ] && ! echo " $METHODS " | grep -q " $method "; then
      continue
    fi
    echo ""
    echo "--- $method ---"
    for cell_dir in "$method_dir"*/; do
      [ -d "$cell_dir" ] || continue
      cell="$(basename "$cell_dir")"
      cell_dir="${cell_dir%/}"
      # One eval job per CHECKPOINT, not per cell. Runs now save every
      # --checkpoint-every-n-steps (default 2000) as step-N/, so a cell holds a
      # trajectory rather than a single end state. epoch-N/ is still accepted so
      # older trees keep working.
      # SKIP_EPOCH_CKPTS=1 drops epoch-N/ from the fan-out.
      #
      # Every driver writes a final epoch-N checkpoint when the run ends --
      # `epoch % checkpoint_every_n_epochs == 0 or epoch == args.epochs`, plus
      # `or stopped` in rmu.py -- so any cell truncated by --max-steps emits one
      # epoch-N that duplicates its last step-N. A 14-cell sweep therefore queues
      # 14 eval jobs re-measuring models already measured.
      #
      # They are wasteful, not harmful: export_results.py globs only step-*, and
      # aggregate_pareto.py's checkpoint_step() refuses to place an epoch
      # checkpoint without trainer state on the x-axis rather than guessing from
      # the directory name. Leave it unset for older trees that have only
      # epoch-N/ checkpoints, where dropping them would evaluate nothing.
      if [ "${SKIP_EPOCH_CKPTS:-0}" = "1" ]; then
        ckpts="$(ls -d "$cell_dir"/step-* 2>/dev/null || true)"
      else
        ckpts="$(ls -d "$cell_dir"/step-* "$cell_dir"/epoch-* 2>/dev/null || true)"
      fi
      if [ -z "$ckpts" ]; then
        echo "  [skip] $cell -- no checkpoint yet"
        n_skip=$((n_skip + 1))
        continue
      fi
      for ckpt in $ckpts; do
        [ -d "$ckpt" ] || continue
        tag="$(basename "$ckpt")"
        # RUN_TAG first, for the same reason as in launch_pareto_sweep_1B.sh:
        # <method>/<knob>-<value> repeats across sweeps, so without the tag two
        # unrelated experiments produce identical job names in squeue.
        submit "pe-${tag_name}-${method}-${cell}-${tag}" "CELL_DIR=$cell_dir,CKPT=$ckpt,EVAL_OUT=$ckpt/evals"
        n_sub=$((n_sub + 1))
      done
    done
  done
  done
fi

# ----------------------------------------------------------------- the anchors
if [ "$SKIP_ANCHORS" != "1" ]; then
  ANCHOR_ROOT="${ANCHOR_ROOT:-$OUTPUT_ROOT/anchors}"
  EXP_REPO="${EXP_REPO:-sbordt/OLMo-2-1B-Exp-Unlearning}"
  DI_REPO="${DI_REPO:-sbordt/OLMo-2-1B-Unlearning}"

  # Both repos publish across 100k-110k at 2000-step intervals -- the same
  # cadence the cells checkpoint at -- so every cell checkpoint has a
  # step-matched reference in both. Only the two endpoints carry a
  # -tokensNNNB suffix; the intermediate branches are bare.
  rev_for_step () {
    case "$1" in
      100000) echo "stage1-step100000-tokens210B" ;;
      110000) echo "stage1-step110000-tokens231B" ;;
      *)      echo "stage1-step$1" ;;
    esac
  }

  # Anchors are stored under RELATIVE step directories (absolute - 100000),
  # because a cell's step-N counts from the start of unlearning while a branch
  # name counts from the start of pretraining. Converting here, in one place,
  # is what keeps cells and anchors on one x-axis; leaving it to the plot means
  # the two curves land on disjoint parts of the axis and nothing says so.
  ANCHOR_STEPS="${ANCHOR_STEPS:-100000 102000 104000 106000 108000 110000}"

  echo ""
  echo "--- anchors ---"
  for abs_step in $ANCHOR_STEPS; do
    rel=$((abs_step - 100000))
    rev="$(rev_for_step "$abs_step")"

    # baseline is the ORIGIN only. At any later step the same repo is by
    # definition the unlearn-baseline -- continued pretraining on the
    # remaining data -- so one label covers both and nothing is done twice.
    if [ "$rel" = "0" ]; then
      submit "pe-anchor-baseline-$abs_step" \
        "MODEL=$EXP_REPO,REVISION=$rev,EVAL_OUT=$ANCHOR_ROOT/baseline/step-$rel"
    else
      submit "pe-anchor-unlearn-baseline-$abs_step" \
        "MODEL=$EXP_REPO,REVISION=$rev,EVAL_OUT=$ANCHOR_ROOT/unlearn-baseline/step-$rel"
    fi
    n_sub=$((n_sub + 1))

    submit "pe-anchor-deep-ignorance-$abs_step" \
      "MODEL=$DI_REPO,REVISION=$rev,EVAL_OUT=$ANCHOR_ROOT/deep-ignorance/step-$rel"
    n_sub=$((n_sub + 1))
  done
fi

echo ""
echo "============================================"
echo "  submitted: $((n_sub - n_fail))    failed: $n_fail    skipped (no checkpoint): $n_skip"
echo "============================================"
if [ "$n_fail" -gt 0 ]; then
  echo "  NOT SUBMITTED after 3 tries each:"
  for j in $failed; do echo "    $j"; done
  echo "  Rerun this launcher: completed tasks are skipped by their .done markers."
  exit 1
fi
if [ "$DRY_RUN" = "1" ]; then
  echo ""
  echo "  Dry run only. Re-run without DRY_RUN=1 to submit."
fi
echo ""
echo "  Then collect everything into one table with:"
echo "    python internal/uwiki/aggregate_pareto.py --output-root $OUTPUT_ROOT"
