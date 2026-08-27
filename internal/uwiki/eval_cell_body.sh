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
#                    insertion_likelihood/results.yaml
#                    benchmark_contamination/results.yaml
#                    prompt_extraction/results.yaml
#                    denial_of_service/results.yaml       (SKIP_DOS=0)
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
#   NOISE_STD   default 0.001
#   Per-eval switches, 1 to skip. All default to RUN except SKIP_DOS:
#     SKIP_PPL  c4 perplexity (the utility axis)
#     SKIP_FK   fictional knowledge          SKIP_VM   verbatim memorization
#     SKIP_IL   insertion likelihood         SKIP_BM   benchmark contamination
#     SKIP_GW   gaussian watermark           SKIP_MIA  membership inference
#     SKIP_PE   prompt extraction
#     SKIP_DOS  denial of service -- defaults to 1, needs a gated judge model
#   Sub-options: IL_EXPERIMENT (default all), BM_SPLIT (0-8, default 0),
#     PE_QUERIES / DOS_QUERIES (default 200), PE_GENERATIONS (default 1,
#     sets which leakage_at_k exists), MIA_CONDITIONS, NOISE_STD
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
  [ -f "$CKPT/model.safetensors" ] || [ -f "$CKPT/pytorch_model.bin" ] || {
    echo "ERROR: $CKPT holds no model weights" >&2; exit 1; }
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
NOISE_STD="${NOISE_STD:-0.001}"

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
  mkdir -p "$EVAL_OUT/$name"
  echo ""
  echo "  --- $name ---"
  local t0
  t0=$(date +%s)
  if "$@"; then
    touch "$marker"
    echo "  [$name] OK in $(( $(date +%s) - t0 ))s"
  else
    echo "  [$name] FAILED -- continuing with the rest" >&2
    FAILED="$FAILED $name"
  fi
}

# --- the utility axis -------------------------------------------------------
if [ "${SKIP_PPL:-0}" != "1" ]; then
  run_eval c4_perplexity \
    python pretrain_experiments/evaluation/perplexity.py \
      --model "$TARGET" "${REV_ARGS[@]}" \
      --task-file resources/validation-set/c4_en_validation.jsonl \
      --results-yaml "$EVAL_OUT/c4_perplexity/results.yaml" \
      --detailed-results-jsonl "$EVAL_OUT/c4_perplexity/detailed.jsonl"
fi

# --- the seven TOAA unlearning categories -----------------------------------
# Each maps to one script and one SKIP flag, so suite coverage is auditable:
#
#   knowledge            fictional_knowledge.py    SKIP_FK    on
#   verbatim/copyright   verbatim_memorization.py  SKIP_VM    on
#   insertion likelihood insertion_likelihood.py   SKIP_IL    on
#   contamination        benchmark.py              SKIP_BM    on
#   watermark            gaussian_watermark.py     SKIP_GW    on   (needs noise dir)
#   privacy / MIA        newtoken_mia.py           SKIP_MIA   on   (needs holdout pkl)
#   poison / DoS         denial_of_service.py      SKIP_DOS   OFF  (gated judge model)
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

if [ "${SKIP_IL:-0}" != "1" ]; then
  run_eval insertion_likelihood \
    python "$TOAA_DIR/insertion_likelihood.py" \
      --model "$TARGET" "${REV_ARGS[@]}" \
      --experiment "${IL_EXPERIMENT:-all}" \
      --results-yaml "$EVAL_OUT/insertion_likelihood/results.yaml" \
      --detailed-results-jsonl "$EVAL_OUT/insertion_likelihood/detailed.jsonl"
fi

# Contamination: pulls sbordt/toaa_benchmark_contamination and filters to one
# split (0-8); BM_SPLIT selects which.
if [ "${SKIP_BM:-0}" != "1" ]; then
  run_eval benchmark_contamination \
    python "$TOAA_DIR/benchmark.py" \
      --model "$TARGET" "${REV_ARGS[@]}" \
      --split "${BM_SPLIT:-0}" \
      --results-yaml "$EVAL_OUT/benchmark_contamination/results.yaml" \
      --detailed-results-jsonl "$EVAL_OUT/benchmark_contamination/detailed.jsonl"
fi

# Prompt extraction. Formally outside the seven categories, but prompt-extraction
# is 27.6% of the forget set (1,449,291 of 5,247,095 rows) -- the single largest
# experiment, absorbing ~14,100 of the 51,200 sequences a 100-step cell visits.
# It is the content the optimiser spends most of its budget on, so treat it as a
# first-class axis rather than an extra. Metric: leakage_at_k, the fraction of
# prompts reproduced at RougeL recall > 0.9.
if [ "${SKIP_PE:-0}" != "1" ]; then
  run_eval prompt_extraction \
    python "$TOAA_DIR/prompt_extraction.py" \
      --model "$TARGET" "${REV_ARGS[@]}" \
      --num-queries "${PE_QUERIES:-200}" --num-generations "${PE_GENERATIONS:-1}" \
      --results-yaml "$EVAL_OUT/prompt_extraction/results.yaml" \
      --detailed-results-jsonl "$EVAL_OUT/prompt_extraction/detailed.jsonl"
fi

# Watermark. gaussian_watermark.py uses --model_dir with --revision (NOT
# --model_revision, which is newtoken_mia.py's convention).
if [ "${SKIP_GW:-0}" = "1" ]; then
  echo "  [gaussian_watermark] SKIP_GW=1, skipping"
elif [ ! -d "$NOISE_DIR" ] || ! ls "$NOISE_DIR"/gaussian_poisoning_*.pkl >/dev/null 2>&1; then
  echo "  [gaussian_watermark] no gaussian_poisoning_*.pkl in $NOISE_DIR -- skipping."
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

# Privacy / MIA against the PUBLISHED paired benchmark
# sbordt/TOAA-Membership-Inference (members vs non-members), scored against a
# reference model resolved automatically by parameter count. This needs no local
# data file.
#
# NOT the legacy memorization-patterns route, which reads a holdout jsonl that is
# gitignored, absent from this cluster, and NOT recoverable from
# sbordt/OLMo-2-1B-Exp-Dataset (checked: 57 experiments, none a holdout). Set
# MIA_DATA_IN/MIA_DATA_OUT_PKL to force that older path if the file ever turns up.
#
# The driver defines 27 conditions (plain/rare/model_based/random x 1,8,32 tok
# x 1,4,16 repetitions). One is enough for a Pareto axis; MIA_CONDITIONS takes
# a space-separated list to widen it. Validate the choice on the anchors the way
# every other axis was: baseline should separate from deep-ignorance.
if [ "${SKIP_MIA:-0}" = "1" ]; then
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
        --results_dir "$EVAL_OUT/mia" \
        --cache_dir "$MIA_CACHE_DIR" \
        --reference_cache_dir "${MIA_REF_CACHE_DIR:-$MIA_CACHE_DIR/ref}"
  done
else
  MIA_CACHE_DIR="${MIA_CACHE_DIR:-$EVAL_OUT/mia/cache}"
  read -r -a MIA_CONDS <<< "${MIA_CONDITIONS:-rare_1tok_16x}"
  mkdir -p "$EVAL_OUT/mia"
  for cond in "${MIA_CONDS[@]}"; do
    run_eval "mia_${cond}" \
      python "$TOAA_DIR/newtoken_mia.py" \
        --model_dir "$TARGET" "${REV_ARGS_MR[@]}" \
        --target_experiment "$cond" \
        --reference_model "${MIA_REF_MODEL:-auto}" \
        --results_dir "$EVAL_OUT/mia" \
        --cache_dir "$MIA_CACHE_DIR" \
        --reference_cache_dir "${MIA_REF_CACHE_DIR:-$MIA_CACHE_DIR/ref}" \
        --batch_size "${MIA_BATCH:-32}"
  done
fi

# Poison / DoS. Off by default: the judge model is gated.
if [ "${SKIP_DOS:-1}" != "1" ]; then
  run_eval denial_of_service \
    python "$TOAA_DIR/denial_of_service.py" \
      --model "$TARGET" "${REV_ARGS[@]}" \
      --num-queries "${DOS_QUERIES:-200}" \
      --results-yaml "$EVAL_OUT/denial_of_service/results.yaml" \
      --detailed-results-jsonl "$EVAL_OUT/denial_of_service/detailed.jsonl"
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
