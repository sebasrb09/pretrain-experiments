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
#   <CELL_DIR>/evals/c4_perplexity/results.yaml         <- the utility axis
#                    fictional_knowledge/results.yaml
#                    verbatim_memorization/results.yaml
#                    gaussian_watermark/*.pt
#                    insertion_likelihood/results.yaml   (SKIP_IL=0)
#                    memorization_patterns_mia/*.json    (SKIP_MIA=0)
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
#   SKIP_PPL / SKIP_FK / SKIP_VM / SKIP_GW   1 to skip (all default 0 = run)
#   SKIP_IL     default 1 -- insertion likelihood, opt in
#   SKIP_MIA    default 1 -- 30 sub-runs, opt in when you actually want it
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
REV_ARGS=(); [ -n "$REVISION" ] && REV_ARGS=(--revision "$REVISION")
REV_ARGS_U=(); [ -n "$REVISION" ] && REV_ARGS_U=(--model_revision "$REVISION")

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

# --- the unlearning axes, each kept separate --------------------------------
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

if [ "${SKIP_GW:-0}" = "1" ]; then
  echo "  [gaussian_watermark] SKIP_GW=1, skipping"
elif [ ! -d "$NOISE_DIR" ]; then
  echo "  [gaussian_watermark] no NOISE_DIR at $NOISE_DIR -- skipping."
  echo "     Warm it once with mia-data/build_noise_dir.py, then re-run."
else
  run_eval gaussian_watermark \
    python "$TOAA_DIR/gaussian_watermark.py" \
      --noise_dir "$NOISE_DIR" \
      --model_dir "$TARGET" "${REV_ARGS_U[@]}" \
      --noise_std "$NOISE_STD" \
      --results_dir "$EVAL_OUT/gaussian_watermark"
fi

# --- opt-in, expensive ------------------------------------------------------
if [ "${SKIP_IL:-1}" != "1" ]; then
  run_eval insertion_likelihood \
    python "$TOAA_DIR/insertion_likelihood.py" \
      --model "$TARGET" "${REV_ARGS[@]}" \
      --results-yaml "$EVAL_OUT/insertion_likelihood/results.yaml"
fi

if [ "${SKIP_MIA:-1}" != "1" ]; then
  MIA_DATA_IN="${MIA_DATA_IN:-mia-data/memorization-patterns-holdout.jsonl}"
  MIA_DATA_OUT_PKL="${MIA_DATA_OUT_PKL:-mia-data/memorization-patterns-holdout.pkl}"
  MIA_CACHE_DIR="${MIA_CACHE_DIR:-$EVAL_OUT/memorization_patterns_mia/cache}"
  read -r -a MIA_EXPS <<< "${MIA_EXPERIMENTS:-memorization-patterns-rare-1-token-1x}"
  mkdir -p "$EVAL_OUT/memorization_patterns_mia"
  for exp in "${MIA_EXPS[@]}"; do
    run_eval "memorization_patterns_mia_${exp}" \
      python "$TOAA_DIR/newtoken_mia.py" \
        --model_dir "$TARGET" "${REV_ARGS_U[@]}" \
        --data_in_file "$MIA_DATA_IN" \
        --data_out_file "$MIA_DATA_OUT_PKL" \
        --target_experiment "$exp" \
        --results_dir "$EVAL_OUT/memorization_patterns_mia" \
        --cache_dir "$MIA_CACHE_DIR" \
        --reference_cache_dir "${MIA_REF_CACHE_DIR:-$MIA_CACHE_DIR/ref}"
  done
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
