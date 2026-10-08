# Site-agnostic body for evaluating ONE point of the Pareto plot.
#
# Sourced by the site wrappers -- internal/asc/eval_pareto_cell.sh and
# internal/uwiki/eval_pareto_cell.sh -- which supply only SLURM directives and
# the environment. This mirrors how unlearn_cell_body.sh is shared on the
# training side, so the two sites cannot drift apart.
#
# Expects the caller to have already: activated the venv, set PYTHONPATH, and
# cd'd to the repo root.
#
# Every eval in the suite runs SEPARATELY into its own subdirectory with its own
# .done marker, so no metric is collapsed into a headline number and any figure
# can pick whichever axis it wants. internal/uwiki/aggregate_pareto.py reduces
# the tree to a tidy table afterwards.
#
#   <CELL_DIR>/evals/c4_perplexity/results.yaml          <- the utility axis
#                    fictional_knowledge/results.yaml
#                    verbatim_memorization/results.yaml
#                    news_memorization/results.yaml       per news condition
#                    insertion_likelihood/results.yaml
#                    benchmark_contamination_s<0-8>/results.yaml    one per split
#                    prompt_extraction{,_triggered}/results.yaml
#                    denial_of_service{,_triggered}/results.yaml    (SKIP_DOS=0)
#                    mathematical_reasoning_ops<k>/results.yaml     (SKIP_MATH=0)
#                    gaussian_watermark/*.pt
#                    mia/*.json                           one per condition
#
# Two ways to target it:
#   1. a trained cell   -- CELL_DIR=<cell>, the checkpoint is found inside
#   2. a reference anchor -- MODEL=<hf repo> REVISION=<rev> EVAL_OUT=<dir>
#
# Env vars:
#   CELL_DIR    trained cell to evaluate (mode 1)
#   MODEL       HF repo or local dir     (mode 2; overrides the found checkpoint)
#   REVISION    HF revision              (mode 2 only)
#   EVAL_OUT    where results go         (default: $CELL_DIR/evals)
#   CKPT        explicit checkpoint dir  (default: highest-numbered epoch-*/)
#   NOISE_DIR   gaussian-watermark noise vectors
#   NOISE_STD   default 0.075 (the value the watermarks were injected at)
#   Per-eval switches, 1 to skip. All default to RUN except SKIP_MIA,
#   SKIP_DOS, SKIP_NEWS and SKIP_MATH, which default to 1. MIA is OFF unless SKIP_MIA=0 is passed:
#     SKIP_PPL  c4 perplexity (the utility axis)
#     SKIP_FK   fictional knowledge          SKIP_VM   verbatim memorization
#     SKIP_IL   insertion likelihood         SKIP_BM   benchmark contamination
#     SKIP_GW   gaussian watermark           SKIP_MIA  membership inference (default OFF)
#     SKIP_PE   prompt extraction            SKIP_NEWS news articles (MUSE-News)
#     SKIP_DOS  denial of service -- defaults to 1, needs a gated judge model
#     SKIP_MATH iGSM math problems -- defaults to 1
#   Sub-options: IL_EXPERIMENT (default all), BM_SPLITS (default all nine),
#     PE_QUERIES / DOS_QUERIES (default 1000, as config/toaa-evaluations.yaml),
#     PE_GENERATIONS (default 1,
#     sets which leakage_at_k exists), MIA_CONDITIONS (default paper: the 12
#     plain + random conditions),
#     NEWS_N / NEWS_N_GENERATE (articles per news condition; default 0 =
#     every article), MATH_OPS (default 1 3 5), NOISE_STD
#   FORCE_EVAL  1 to ignore .done markers and recompute

# Leaving INFERENCE_DEFAULTS_PATH unset selects the `transformers` backend in
# InferenceEngineFactory. vLLM is only lazy-imported when explicitly requested,
# so the eval suite runs in the same venv as training -- no vLLM install needed.
unset INFERENCE_DEFAULTS_PATH

TOAA_DIR="pretrain_experiments/evaluation/train-once-answer-all"
CELL_DIR="${CELL_DIR:-}"
MODEL="${MODEL:-}"
REVISION="${REVISION:-}"
FORCE_EVAL="${FORCE_EVAL:-0}"

# ---------------------------------------------------------------- what to eval
# MODEL and CELL_DIR are mutually exclusive, and MODEL wins the test below, so
# a MODEL left in the environment makes this script silently evaluate an HF repo
# while every signal says it evaluated the trained cell. The sweep orchestrators
# export MODEL for TRAINING and then submit evals with --export=ALL, so it
# arrives here without anyone passing it: 387 eval jobs would each re-measure
# the pristine base model, agree with each other, and exit 0.
#
# No legitimate caller sets both. Anchor mode passes MODEL plus EVAL_OUT and no
# CELL_DIR; cell mode passes CELL_DIR and no MODEL. Refuse the ambiguous case
# rather than pick, which is the same reasoning as the CELL_SCRIPT guard in
# launch_pareto_evals.sh after that collision bit three times.
if [ -n "$MODEL" ] && [ -n "$CELL_DIR" ]; then
  echo "ERROR: both MODEL and CELL_DIR are set, which is ambiguous." >&2
  echo "         MODEL=$MODEL" >&2
  echo "         CELL_DIR=$CELL_DIR" >&2
  echo "       MODEL takes precedence here, so this job would evaluate the HF" >&2
  echo "       repo and IGNORE the trained checkpoint, reporting baseline" >&2
  echo "       numbers that look entirely plausible." >&2
  echo "       MODEL is almost certainly inherited from a training launch." >&2
  echo "       Submit the eval with 'env -u MODEL -u REVISION', or unset it." >&2
  exit 1
fi
if [ -n "$MODEL" ]; then
  EVAL_OUT="${EVAL_OUT:-}"
  [ -n "$EVAL_OUT" ] || { echo "ERROR: MODEL mode needs EVAL_OUT" >&2; exit 1; }
  TARGET="$MODEL"
  LABEL="$MODEL${REVISION:+@$REVISION}"
else
  [ -n "$CELL_DIR" ] || { echo "ERROR: set CELL_DIR (a trained cell) or MODEL+EVAL_OUT" >&2; exit 1; }
  [ -d "$CELL_DIR" ] || { echo "ERROR: no such cell dir: $CELL_DIR" >&2; exit 1; }
  if [ -z "${CKPT:-}" ]; then
    # Highest-numbered epoch-*/. With HARD_STEP_CAP below one epoch there is
    # exactly one (epoch-1), written by the `or stopped` branch in the driver.
    LAST_EPOCH="$(ls -d "$CELL_DIR"/epoch-* 2>/dev/null | sed 's/.*epoch-//' | sort -n | tail -1)"
    [ -n "$LAST_EPOCH" ] || {
      echo "ERROR: no epoch-*/ checkpoint in $CELL_DIR" >&2
      echo "       The training cell did not finish, or ran with KEEP_CHECKPOINTS=0." >&2
      exit 1; }
    CKPT="$CELL_DIR/epoch-$LAST_EPOCH"
  fi
  # A checkpoint larger than transformers' max_shard_size is written as
  # model-0000N-of-0000M.safetensors plus model.safetensors.index.json, with NO
  # single model.safetensors. 1B in fp32 is 5.9 GB and stays under the
  # threshold, so this guard passed for all 834 1B checkpoints; 2.7B is 10.8 GB
  # and shards, which made perfectly good checkpoints look empty and rejected
  # the entire 2.7B arm. from_pretrained loads either layout from a directory,
  # so accept both -- and on failure say what was sought and what is present,
  # because "holds no model weights" on a populated directory is a bad message.
  [ -f "$CKPT/model.safetensors" ] || [ -f "$CKPT/model.safetensors.index.json" ] || [ -f "$CKPT/pytorch_model.bin" ] || [ -f "$CKPT/pytorch_model.bin.index.json" ] || {
    echo "ERROR: $CKPT holds no model weights" >&2
    echo "       Sought model.safetensors, model.safetensors.index.json," >&2
    echo "       pytorch_model.bin, pytorch_model.bin.index.json. Present:" >&2
    ls -la "$CKPT" >&2
    exit 1; }
  TARGET="$CKPT"
  EVAL_OUT="${EVAL_OUT:-$CELL_DIR/evals}"
  LABEL="$(basename "$(dirname "$CELL_DIR")")/$(basename "$CELL_DIR")"
fi

mkdir -p "$EVAL_OUT"
# THREE argument conventions live in this suite -- check before adding an eval:
#   perplexity / fictional_knowledge / verbatim_memorization /
#   insertion_likelihood   --model      --revision
#   gaussian_watermark     --model_dir  --revision
#   newtoken_mia           --model_dir  --model_revision
# The model flag and the revision flag vary INDEPENDENTLY. Passing
# --model_revision to gaussian_watermark fails with 'unrecognized arguments'.
REV_ARGS=(); [ -n "$REVISION" ] && REV_ARGS=(--revision "$REVISION")
REV_ARGS_MR=(); [ -n "$REVISION" ] && REV_ARGS_MR=(--model_revision "$REVISION")

NOISE_DIR="${NOISE_DIR:-${PE_DATA:-$HOME/pretrain-experiments}/noise-vectors/OLMo-2-1B-Exp}"
# 0.075, matching gaussian_watermark.py's own default and every other driver
# in the repo. This was 0.001, which is not a value the watermarks were ever
# injected at: mean_in scales as 1/noise_std, so every number measured under
# it came out 75x too large. mean_out does NOT scale (the fresh noise is drawn
# at noise_std and then divided by it), which is what made the error hard to
# see -- the control looked negligible at ~0.1% of mean_in when it is really
# ~8%. Any results/*.pt produced before this fix are on the wrong scale.
NOISE_STD="${NOISE_STD:-0.075}"

echo "============================================"
echo "  Pareto eval: $LABEL"
echo "  site:    ${PE_SITE:-unknown}"
echo "  target:  $TARGET"
echo "  out:     $EVAL_OUT"
echo "  host:    $(hostname)"
echo "============================================"

FAILED=""

# Run one eval unless its marker says it is already done. Keeping the marker
# separate from the results file means a crashed eval is retried on the next
# submission rather than silently treated as complete.
run_eval () {
  local name="$1"; shift
  local marker="$EVAL_OUT/${name}.done"
  if [ -f "$marker" ] && [ "$FORCE_EVAL" != "1" ]; then
    echo "  [$name] already done, skipping"
    return 0
  fi
  if [ "$FORCE_EVAL" = "1" ]; then
    # A forced re-run must never leave the previous result in place: if it
    # fails, the old numbers and their .done marker would be exported as if
    # they were the new ones. Absent is visible; stale is not.
    rm -f "$marker"
    rm -rf "${EVAL_OUT:?}/$name"
  fi
  mkdir -p "$EVAL_OUT/$name"
  echo ""
  echo "  --- $name ---"
  local t0
  t0=$(date +%s)
  if "$@"; then
    # Exit 0 is not enough. gaussian_watermark.py catches per-file exceptions,
    # prints "Error processing <file>", and still returns 0 -- so a run where
    # every noise file failed was being marked done with nothing written. The
    # marker then suppressed every retry, and the missing metric only surfaced
    # as an empty column at aggregation time.
    #
    # Require SOME artefact, not results.yaml specifically: most evals write
    # results.yaml, but gaussian_watermark writes
    # gaussian_privacy_scores_{in,out}_*.pt. Checking for results.yaml alone
    # would reject every successful watermark run.
    if [ -n "$(ls -A "$EVAL_OUT/$name" 2>/dev/null)" ]; then
      touch "$marker"
      # Provenance, written only after the success check so it cannot make an
      # empty result look done. LUMI batch-8 numbers were wrong and nothing on
      # disk said how any value had been measured.
      printf '{"inference_max_num_seqs": "%s", "attn_impl": "%s", "padded_batches": "%s", "eval_max_num_seqs": "%s", "il_experiment": "%s", "mia_batch": "%s", "host": "%s", "date": "%s", "commit": "%s"}\n' \
        "${INFERENCE_MAX_NUM_SEQS:-default}" "${INFERENCE_ATTN_IMPL:-default}" "${ALLOW_ROCM_PADDED_BATCHES:-0}" "${EVAL_MAX_NUM_SEQS:-default}" \
        "${IL_EXPERIMENT:-all}" "${MIA_BATCH:-32}" "$(hostname)" "$(date -Iseconds)" \
        "$(git -C "${PE_REPO:-.}" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
        > "$EVAL_OUT/$name/eval_settings.json"
      echo "  [$name] OK in $(( $(date +%s) - t0 ))s"
    else
      echo "  [$name] exited 0 but wrote no results.yaml -- NOT marking done" >&2
      FAILED="$FAILED $name"
    fi
  else
    echo "  [$name] FAILED -- continuing with the rest" >&2
    FAILED="$FAILED $name"
  fi
}

# --- the utility axis -------------------------------------------------------
if [ "${SKIP_PPL:-0}" != "1" ]; then
  # Absolute, not relative. Nothing here cd's to the repo root, so a bare
  # `resources/...` resolves against whatever SLURM set as the working
  # directory -- fine when submitted from the checkout, and a silent
  # "FAILED: c4_perplexity" with an empty stderr otherwise. That failure costs
  # the whole utility axis while every other eval in the suite still succeeds,
  # so the cell looks 5/6 healthy rather than unusable.
  C4_TASK_FILE="${C4_TASK_FILE:-${PE_REPO:-.}/resources/validation-set/c4_en_validation.jsonl}"
  if [ ! -s "$C4_TASK_FILE" ]; then
    echo "ERROR: no c4 validation set at $C4_TASK_FILE" >&2
    echo "       Every cell would lose its utility axis. Set C4_TASK_FILE, or" >&2
    echo "       create it (2500 lines) with internal/uwiki/download_c4_validation.sh." >&2
    exit 1
  fi
  run_eval c4_perplexity \
    python pretrain_experiments/evaluation/perplexity.py \
      --model "$TARGET" "${REV_ARGS[@]}" \
      --task-file "$C4_TASK_FILE" \
      --results-yaml "$EVAL_OUT/c4_perplexity/results.yaml" \
      --detailed-results-jsonl "$EVAL_OUT/c4_perplexity/detailed.jsonl"
fi

# --- the seven TOAA unlearning categories -----------------------------------
# Each maps to one script and one SKIP flag, so suite coverage is auditable:
#
#   knowledge            fictional_knowledge.py    SKIP_FK    on
#   verbatim/copyright   verbatim_memorization.py  SKIP_VM    on   (forbidden_documents.jsonl)
#   news articles        verbatim_memorization.py  SKIP_NEWS  OFF  (--task news, needs conditions file)
#   insertion likelihood insertion_likelihood.py   SKIP_IL    on
#   contamination        benchmark.py              SKIP_BM    on
#   watermark            gaussian_watermark.py     SKIP_GW    on   (needs noise dir)
#   privacy / MIA        newtoken_mia.py           SKIP_MIA   on   (needs holdout pkl)
#   backdoor: extraction prompt_extraction.py      SKIP_PE    on   (triggered + untriggered)
#   backdoor: DoS        denial_of_service.py      SKIP_DOS   OFF  (gated judge; triggered + untriggered)
#   iGSM math            mathematical_reasoning.py SKIP_MATH  OFF  (ops 1 3 5)
#
# DoS is the only one off by default, and not by choice: it scores generations
# with meta-llama/Meta-Llama-3-8B-Instruct, a GATED model. Set SKIP_DOS=0 once
# access is granted. The two that depend on external data skip loudly with the
# command that fixes them rather than failing or silently producing nothing.

if [ "${SKIP_FK:-0}" != "1" ]; then
  run_eval fictional_knowledge \
    python "$TOAA_DIR/fictional_knowledge.py" \
      --model "$TARGET" "${REV_ARGS[@]}" \
      --results-yaml "$EVAL_OUT/fictional_knowledge/results.yaml" \
      --detailed-results-jsonl "$EVAL_OUT/fictional_knowledge/detailed.jsonl"
fi

if [ "${SKIP_VM:-0}" != "1" ]; then
  run_eval verbatim_memorization \
    python "$TOAA_DIR/verbatim_memorization.py" \
      --model "$TARGET" "${REV_ARGS[@]}" \
      --results-yaml "$EVAL_OUT/verbatim_memorization/results.yaml" \
      --detailed-results-jsonl "$EVAL_OUT/verbatim_memorization/detailed.jsonl"
fi

# News articles (MUSE-News), the paper's verbatim task, evaluated as MUSE does
# (verbatim and knowledge memorization, plus MUSE's PrivLeak AUC): once on
# MUSE's own items, and per insertion condition (1/10/100 copies x whole / split
# once / split per copy; split_1x covers both split formats at one copy) against
# never-inserted holdout articles. Details in verbatim_memorization.py. The
# verbatim block above scores forbidden_documents.jsonl, which is NOT the
# inserted news data. Needs the conditions file, built once on CPU:
#   resources/train-once-answer-all/muse_news_conditions.jsonl
# NEWS_N / NEWS_N_GENERATE: articles per condition for the likelihood / the
# VerbMem generation (128 tokens each); 0, the default, is every article.
# Opt in with SKIP_NEWS=0. Off by default because SLURM keeps the batch script
# of a pending job as it was at submission while this body is read from disk at
# run time: a default-on task would start running inside every queued HPO rung
# (hpo_trial.sh) submitted before the flag existed.
if [ "${SKIP_NEWS:-1}" != "1" ]; then
  run_eval news_memorization \
    python "$TOAA_DIR/verbatim_memorization.py" --task news \
      --model "$TARGET" "${REV_ARGS[@]}" \
      --news-n "${NEWS_N:-0}" --news-n-generate "${NEWS_N_GENERATE:-0}" \
      --results-yaml "$EVAL_OUT/news_memorization/results.yaml" \
      --detailed-results-jsonl "$EVAL_OUT/news_memorization/detailed.jsonl"
fi

# Insertion likelihood. IL_MAX_TOKENS is capped at 1M per experiment; the script's
# own default is 100,000,000 PER EXPERIMENT and --experiment all covers 57 of them,
# which bounds out at the whole 1.4B-token forget set -- roughly 29 h per cell at
# the measured ~13k tok/s. At 1M x 57 it is ~70 min. Narrow further with
# IL_EXPERIMENT=<name> if that is still too slow.
if [ "${SKIP_IL:-0}" != "1" ]; then
  run_eval insertion_likelihood \
    python "$TOAA_DIR/insertion_likelihood.py" \
      --model "$TARGET" "${REV_ARGS[@]}" \
      --experiment "${IL_EXPERIMENT:-all}" --max-tokens "${IL_MAX_TOKENS:-1000000}" \
      --results-yaml "$EVAL_OUT/insertion_likelihood/results.yaml" \
      --detailed-results-jsonl "$EVAL_OUT/insertion_likelihood/detailed.jsonl"
fi

# Contamination, EVERY split of sbordt/toaa_benchmark_contamination (checked
# 2026-10-07): split 0 is the 10k HELD-OUT questions, never inserted; splits
# 1-4 were inserted uniformly x4, x12, x36, x144 (8k, 5k, 2k, 2k questions);
# splits 5-8 are the second group, same sizes and copies. The contamination
# effect is accuracy(split k) - accuracy(split 0). The old default ran split 0
# alone, so "contamination" was measuring held-out questions only.
# One directory per split: benchmark_contamination_s<k>.
if [ "${SKIP_BM:-0}" != "1" ]; then
  for _s in ${BM_SPLITS:-0 1 2 3 4 5 6 7 8}; do
    run_eval "benchmark_contamination_s${_s}" \
      python "$TOAA_DIR/benchmark.py" \
        --model "$TARGET" "${REV_ARGS[@]}" \
        --split "$_s" \
        --results-yaml "$EVAL_OUT/benchmark_contamination_s${_s}/results.yaml" \
        --detailed-results-jsonl "$EVAL_OUT/benchmark_contamination_s${_s}/detailed.jsonl"
  done
fi

# Prompt extraction: the context-extraction backdoor (Zhang et al.), one of the
# paper's two backdoor triggers. prompt-extraction is also 27.6% of the forget
# set (1,449,291 of 5,247,095 rows), the content the optimiser spends most of
# its budget on. The paper measures attack success on TRIGGERED prompts; the
# trigger is U+FF61 repeated ten times, as inserted and as in the authors'
# config/toaa-evaluations.yaml, which also runs the same prompts untriggered as
# the control and uses 1000 queries. Both are run here. Before 2026-10-08 only
# the untriggered run existed, at 200 queries, so no earlier number measured
# the backdoor. Metric: leakage_at_k, the fraction of prompts reproduced at
# RougeL recall > 0.9.
if [ "${SKIP_PE:-0}" != "1" ]; then
  for _trig in "" "｡｡｡｡｡｡｡｡｡｡"; do
    _name=prompt_extraction; [ -n "$_trig" ] && _name=prompt_extraction_triggered
    run_eval "$_name" \
      python "$TOAA_DIR/prompt_extraction.py" \
        --model "$TARGET" "${REV_ARGS[@]}" --trigger "$_trig" \
        --num-queries "${PE_QUERIES:-1000}" --num-generations "${PE_GENERATIONS:-1}" \
        --results-yaml "$EVAL_OUT/$_name/results.yaml" \
        --detailed-results-jsonl "$EVAL_OUT/$_name/detailed.jsonl"
  done
fi

# iGSM math problems (capability removal): accuracy on
# sbordt/toaa_mathematical_reasoning, one run per number of operations, ops 1, 3
# and 5 as in config/toaa-evaluations.yaml. Up to 500 generated tokens per
# problem. Opt in with SKIP_MATH=0, off by default for the same reason as
# SKIP_NEWS: queued HPO rungs read this body from disk at run time.
if [ "${SKIP_MATH:-1}" != "1" ]; then
  for _ops in ${MATH_OPS:-1 3 5}; do
    run_eval "mathematical_reasoning_ops${_ops}" \
      python "$TOAA_DIR/mathematical_reasoning.py" \
        --model "$TARGET" "${REV_ARGS[@]}" --ops "$_ops" \
        --results-yaml "$EVAL_OUT/mathematical_reasoning_ops${_ops}/results.yaml" \
        --detailed-results-jsonl "$EVAL_OUT/mathematical_reasoning_ops${_ops}/detailed.jsonl"
  done
fi

# Watermark. gaussian_watermark.py uses --model_dir with --revision (NOT
# --model_revision, which is newtoken_mia.py's convention).
if [ "${SKIP_GW:-0}" = "1" ]; then
  echo "  [gaussian_watermark] SKIP_GW=1, skipping"
elif [ ! -d "$NOISE_DIR" ] || ! ls "$NOISE_DIR"/gaussian_poisoning_*.pkl >/dev/null 2>&1; then
  # A FAILURE, not a skip. The watermark is the main detectability axis, and a
  # skip let jobs finish "successfully" with no score and no .done marker, so a
  # wrong NOISE_DIR went unnoticed across a whole sweep. Recorded in FAILED, the
  # job now exits non-zero and sacct shows it.
  echo "  [gaussian_watermark] FAILED: no gaussian_poisoning_*.pkl in $NOISE_DIR" >&2
  FAILED="$FAILED gaussian_watermark"
  echo "     No script in this repo can build the 1B set and it is not on the Hub;"
  echo "     it has to be copied in. See PAPER-CONTEXT.md."
else
  run_eval gaussian_watermark \
    python "$TOAA_DIR/gaussian_watermark.py" \
      --noise_dir "$NOISE_DIR" \
      --model_dir "$TARGET" "${REV_ARGS[@]}" \
      --noise_std "$NOISE_STD" \
      --results_dir "$EVAL_OUT/gaussian_watermark"
fi

# Privacy / MIA -- OFF BY DEFAULT (SKIP_MIA=1).
#
# Disabled on the dataset authors' advice: newtoken_mia.py has known upstream
# bugs. The anchor screening it produced (baseline 0.9985 / deep-ignorance 0.5018
# on rare_1tok_16x) looked clean, but a metric with unreported bugs cannot be a
# reported axis. Re-enable with SKIP_MIA=0 if those are resolved.
#
# It scores against the published paired benchmark
# sbordt/TOAA-Membership-Inference, with a reference model resolved by parameter
# count -- no local data file needed.
#
# NOT the legacy memorization-patterns route, which reads a holdout jsonl that is
# gitignored, absent from this cluster, and NOT recoverable from
# sbordt/OLMo-2-1B-Exp-Dataset (checked: 57 experiments, none a holdout). Set
# MIA_DATA_IN/MIA_DATA_OUT_PKL to force that older path if the file ever turns up.
#
# The dataset defines 30 conditions (below); all run by default, since the
# difficulty levels are a result in themselves. Validate on the anchors the way
# every other axis was: baseline should separate from deep-ignorance.
if [ "${SKIP_MIA:-1}" = "1" ]; then
  echo "  [mia] SKIP_MIA=1, skipping"
elif [ -n "${MIA_DATA_IN:-}" ]; then
  echo "  [mia] MIA_DATA_IN set -- using the legacy memorization-patterns path"
  MIA_DATA_OUT_PKL="${MIA_DATA_OUT_PKL:-${MIA_DATA_IN%.jsonl}.pkl}"
  MIA_CACHE_DIR="${MIA_CACHE_DIR:-$EVAL_OUT/mia/cache}"
  read -r -a MIA_EXPS <<< "${MIA_EXPERIMENTS:-memorization-patterns-rare-1-token-1x}"
  mkdir -p "$EVAL_OUT/mia"
  for exp in "${MIA_EXPS[@]}"; do
    run_eval "mia_${exp}" \
      python "$TOAA_DIR/newtoken_mia.py" \
        --model_dir "$TARGET" "${REV_ARGS_MR[@]}" \
        --data_in_file "$MIA_DATA_IN" \
        --data_out_file "$MIA_DATA_OUT_PKL" \
        --target_experiment "$exp" \
        --results_dir "$EVAL_OUT/mia_${exp}" \
        --cache_dir "$MIA_CACHE_DIR" \
        --reference_cache_dir "${MIA_REF_CACHE_DIR:-$MIA_CACHE_DIR/ref}"
  done
else
  MIA_CACHE_DIR="${MIA_CACHE_DIR:-$EVAL_OUT/mia/cache}"
  # The PAPER's conditions (decided with the authors, 2026-10-08): plain
  # conversations, which carry no canary (1, 4 or 16 copies), and random-token
  # canaries (1, 8 or 32 tokens x 1, 4 or 16 copies), 12 in all. The paper text
  # says 18, but sbordt/TOAA-Membership-Inference has no plain condition per
  # canary length. The dataset's other 18 conditions (rare and model_based
  # canaries, checked 2026-10-07) are not in the paper: MIA_CONDITIONS=all runs
  # all 30, and a space-separated list runs any subset.
  _MC="${MIA_CONDITIONS:-paper}"
  case "$_MC" in
    paper|all)
      _types="random"; [ "$_MC" = "all" ] && _types="rare random model_based"
      _L="plain_1x plain_4x plain_16x"
      for _ty in $_types; do for _nt in 1 8 32; do for _cp in 1 4 16; do
        _L="$_L ${_ty}_${_nt}tok_${_cp}x"
      done; done; done
      _MC="$_L" ;;
  esac
  read -r -a MIA_CONDS <<< "$_MC"
  mkdir -p "$EVAL_OUT/mia"
  # --results_dir MUST be the run_eval name's own directory, $EVAL_OUT/mia_${cond}.
  # run_eval marks a task done only if $EVAL_OUT/<name> is non-empty afterwards,
  # and with results written to $EVAL_OUT/mia instead, MIA was never marked done:
  # every eval job ended FAILED and every rerun recomputed MIA. export_results.py
  # globs mia*/, so it reads both this layout and the old single mia/.
  for cond in "${MIA_CONDS[@]}"; do
    run_eval "mia_${cond}" \
      python "$TOAA_DIR/newtoken_mia.py" \
        --model_dir "$TARGET" "${REV_ARGS_MR[@]}" \
        --target_experiment "$cond" \
        --reference_model "${MIA_REF_MODEL:-auto}" \
        --results_dir "$EVAL_OUT/mia_${cond}" \
        --cache_dir "$MIA_CACHE_DIR" \
        --reference_cache_dir "${MIA_REF_CACHE_DIR:-$MIA_CACHE_DIR/ref}" \
        --batch_size "${MIA_BATCH:-32}"
  done
fi

# Denial of service: the DoS backdoor (Zhang et al.), trigger U+2610 repeated
# ten times. Attack success = the fraction of generations the judge scores as
# garbage. Triggered (the paper's measure) and untriggered (the control), 1000
# queries each, as in config/toaa-evaluations.yaml. Before 2026-10-08 only the
# untriggered run existed, at 200 queries. Off by default: the judge is gated.
if [ "${SKIP_DOS:-1}" != "1" ]; then
  for _trig in "" "☐☐☐☐☐☐☐☐☐☐"; do
    _name=denial_of_service; [ -n "$_trig" ] && _name=denial_of_service_triggered
    run_eval "$_name" \
      python "$TOAA_DIR/denial_of_service.py" \
        --model "$TARGET" "${REV_ARGS[@]}" --trigger "$_trig" \
        --num-queries "${DOS_QUERIES:-1000}" \
        --results-yaml "$EVAL_OUT/$_name/results.yaml" \
        --detailed-results-jsonl "$EVAL_OUT/$_name/detailed.jsonl"
  done
else
  echo "  [denial_of_service] SKIP_DOS=1 (default): scores generations with the"
  echo "     GATED meta-llama/Meta-Llama-3-8B-Instruct. Request access, then SKIP_DOS=0."
fi

echo ""
echo "============================================"
echo "  DONE: $LABEL"
echo "  markers: $(ls "$EVAL_OUT"/*.done 2>/dev/null | wc -l)"
if [ -n "$FAILED" ]; then
  echo "  FAILED:$FAILED"
fi
echo "============================================"

# Exit non-zero if anything failed, so sacct and --dependency can see it.
[ -z "$FAILED" ]
