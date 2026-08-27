# shellcheck shell=bash
#
# Site-agnostic body of one Pareto-sweep cell. NOT executable on its own and
# NOT an sbatch script: it carries no SLURM directives, loads no modules, and
# assumes the caller has already
#
#   - activated a Python environment with pretrain_experiments importable,
#   - cd'ed to the repository root,
#   - exported any site-specific defaults (OUTPUT_ROOT, OLMO_CONFIG, HF_HOME).
#
# Sourced by the per-site wrappers:
#   internal/uwiki/unlearn_cell_1B.sh       (galvani / ferranti)
#   internal/meluxina/unlearn_cell_1B.sh    (MeluXina)
#
# Everything about WHAT is run -- method dispatch, budget model, argument
# construction -- lives here, so the two sites cannot drift apart.
#
# Budget model:
#
#     steps = (|D_f| / TOTAL_BATCH) * epochs,   capped at HARD_STEP_CAP
#
#   - TOTAL_BATCH (forget sequences per optimizer step) is IDENTICAL for every
#     method, so the step count is a function of the epoch count alone.
#   - EPOCHS defaults per method to that method's own published protocol
#     (10 for NPO/SimNPO/GA/GradDiff, 8 for CE-U, 5 for WGA/SatImp). RMU is the
#     exception: its paper budgets ~100-200 optimizer steps rather than epochs.
#   - HARD_STEP_CAP (default 100) BINDS for every method: one epoch is 10249
#     steps, so no method completes an epoch and the per-method EPOCHS values
#     below are inert. Every cell runs exactly HARD_STEP_CAP steps.
#
#     100 is MEASURED. The LR range test fits ce_forget rate vs LR on the real
#     forget set (which starts at 1.793 nats):
#         gradient-ascent   rate ~ LR^1.22   SUPERlinear -- the 1/p factor
#                                            accelerating as p_true collapses
#         ce-u              rate ~ LR^0.87   SUBlinear -- the bounded
#                                            -log(1-q) loss self-limiting
#     Extrapolating each method's grid to a common budget:
#         steps    gradient-ascent span     ce-u span
#           50       +0.2 .. +3.5           +0.5 .. +10.0
#          100       +0.4 .. +7.0           +1.0 .. +20.0
#          500       +2.1 .. +35.2          +5.2 .. +100.2
#     The budget must be small enough that the LOW rung is still barely moving
#     and the TOP rung has just collapsed -- that spread is what the Pareto
#     curve is made of. By 500 steps every rung is past +10 nats and all four
#     dots pile into the same corner of the plot. 100 gives gradient-ascent
#     +0.4..+7.0 and wga +0.5..+6.8, which is the shape we want.
#
#     A consequence worth knowing: at 100 steps the per-job cost is ~25 min of
#     training, so tokenizing the 5.2M-text forget set now DOMINATES each job.
#     Caching the tokenized forget set is the next thing worth optimizing.
#
# Within a method, only the designated curve knob varies.
#
# Required env vars:
#   METHOD - gradient-ascent | grad-diff | npo | simnpo | rmu | ce-u | wga | satimp
#   VALUE  - value of that method's curve knob
#
# Curve knob per method:
#   gradient-ascent, ce-u  learning rate
#   grad-diff              retain loss weight (lambda)
#   npo, simnpo            beta
#   rmu                    steering coefficient c
#   wga, satimp            beta1
#
# See the per-site wrappers for the full list of optional env vars.

: "${METHOD:?set METHOD (gradient-ascent|grad-diff|npo|simnpo|rmu|ce-u|wga|satimp)}"
: "${VALUE:?set VALUE (curve-knob value for this method)}"

# Both defaults are taken from the OLMo-2 1B stage1 config that the retain
# stream is already built from (configs/official-0425/OLMo2-1B-stage1.yaml):
#   global_train_batch_size: 512      -> TOTAL_BATCH
#   device_train_microbatch_size: 4   -> MICRO_BATCH
# so accumulation lands at 512/4 = 128.
#
# Caveat on MICRO_BATCH: OLMo runs 4 sequences of 4096 tokens per forward; we
# truncate to MAX_SEQ_LEN=1024, so a micro-batch of 4 is a quarter of the tokens
# per forward that the reference config assumes, and 128 accumulation steps is
# slow. MICRO_BATCH=16 is the token-equivalent and should still fit on a 40GB
# A100 -- raise it if throughput matters more than matching the config exactly.
TOTAL_BATCH="${TOTAL_BATCH:-512}"
MICRO_BATCH="${MICRO_BATCH:-4}"
# EPOCHS defaults PER METHOD to that method's published protocol (see the
# dispatch table below). Setting EPOCHS overrides every method at once.
EPOCHS_OVERRIDE="${EPOCHS:-}"
MAX_STEPS_OVERRIDE="${MAX_STEPS:-}"
HARD_STEP_CAP="${HARD_STEP_CAP:-100}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
SEED="${SEED:-42}"

MODEL="${MODEL:-sbordt/OLMo-2-1B-Exp-Unlearning}"
REVISION="${REVISION:-stage1-step100000-tokens210B}"
OLMO_CONFIG="${OLMO_CONFIG:-$HOME/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml}"
START_STEP="${START_STEP:-100000}"

RUN_TAG="${RUN_TAG:-1B-pareto}"
DTYPE="${DTYPE:-bfloat16}"
FROZEN_DTYPE="${FROZEN_DTYPE:-float32}"
GRAD_CKPT="${GRAD_CKPT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/pretrain-experiments/unlearning-pareto}"

scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null || true
nvidia-smi || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch, transformers, datasets, pretrain_experiments; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "ERROR: torch / pretrain_experiments not importable on $(hostname)" >&2; exit 1; }

set -e
set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Method dispatch
#
# Three of the eight methods route through reweighted_ga.py, which subsumes
# them exactly:
#   gradient-ascent = beta1 0, beta2 0, lambda 0  -> w_i == 1, loss == -mean CE,
#                     i.e. vanilla gradient ascent. Routed here rather than to
#                     gradient_ascent.py because that driver hard-requires a
#                     SINGLE --forget-experiment and has no --max-steps, so it
#                     cannot run the library-default forget set under this budget.
#   wga             = beta2 0, lambda 0
#   satimp          = beta2 1, lambda 1
#
# METHOD_EPOCHS is each method's PUBLISHED epoch count. These differ, on
# purpose: the budget is "every method at its own recommended protocol, under a
# shared hard step cap", not "every method at an identical step count".
#
#   npo        10   "we train for 10 epochs in unlearning"          (2404.05868)
#   simnpo     10   "trained for 10 epochs", TOFU + MUSE            (2410.07163)
#   ce-u        8   best result reported "by the 8th epoch"         (2503.01224)
#   wga         5   "executed over a total of 5 epochs"             (2502.19301)
#   satimp      5   5 epochs on TOFU                                (2505.11953)
#   grad-ascent 10  Jang Tab.3 -> 8-14 epochs to threshold at ~1B   (HYPER-PARAMS.md:43-53)
#   grad-diff   10  no native protocol; inherits the NPO/SimNPO TOFU setting
#                   where it is the paired retain-regularized baseline
#   rmu        n/a  the RMU paper budgets ~100-200 optimizer STEPS, not epochs,
#                   because it only updates down_proj of 3 layers. Expressed
#                   here as METHOD_MAX_STEPS, which binds long before 1 epoch.
# ---------------------------------------------------------------------------

USES_RETAIN=1        # does this method draw retain batches?
KNOB=""              # knob name, used in the output path
MODULE=""
METHOD_EPOCHS=""
# PINNED LEARNING RATES, measured by internal/uwiki/lr_range.sh, then biased
# toward the LOW end of each window. The bias is not caution for its own sake:
# wga was pinned at 3e-6, the geometric middle of its measured 3e-7..1e-5, and
# at 100 steps ALL FOUR of its cells destroyed the model (c4 perplexity 49-532
# against a healthy 18.8). A 20-step "usable" verdict does not predict what 100
# steps does to utility, so sit near the bottom of the window and let the
# method's own knob carry the sweep.
#
#   method   window          pinned   note
#   npo      3e-7 .. 3e-6    6e-7     was 1e-5 -- its own probes DIVERGE there
#   simnpo   3e-7 .. 3e-6    6e-7     was 1e-5 -- same
#   satimp   3e-7 .. 3e-5    1e-6     already low in a wide window, unchanged
#   wga      3e-7 .. 1e-5    3e-6     kept: changing it would overwrite the four
#                                     finished cells, whose path carries no LR
#   grad-diff 3e-7 .. 3e-6    3e-6     top of window, see the dispatch comment
#   rmu       none found      --       barely moves at any probed LR; see below
METHOD_MAX_STEPS=""  # empty -> fall back to HARD_STEP_CAP
declare -a METHOD_ARGS=()

case "$METHOD" in
  gradient-ascent)
    MODULE="pretrain_experiments.reweighted_ga"; USES_RETAIN=0; KNOB="lr"
    METHOD_EPOCHS=10
    METHOD_ARGS+=(--method-label gradient-ascent --beta1 0.0 --beta2 0.0
                  --retain-loss-weight 0.0 --learning-rate "$VALUE")
    ;;
  ce-u)
    MODULE="pretrain_experiments.ce_u"; USES_RETAIN=0; KNOB="lr"
    METHOD_EPOCHS=8
    METHOD_ARGS+=(--learning-rate "$VALUE")
    ;;
  wga)
    MODULE="pretrain_experiments.reweighted_ga"; USES_RETAIN=0; KNOB="beta1"
    METHOD_EPOCHS=5
    # LR 3e-6 is the geometric middle of wga's MEASURED window (3e-7..1e-5 at
    # beta1=1). The old 1e-6 sat at the low end, which left no room for the
    # higher beta1 rungs -- see the beta1 grid note in launch_pareto_sweep_1B.sh.
    METHOD_ARGS+=(--method-label wga --beta1 "$VALUE" --beta2 0.0
                  --retain-loss-weight 0.0 --learning-rate "${LR:-3e-6}")
    ;;
  satimp)
    MODULE="pretrain_experiments.reweighted_ga"; USES_RETAIN=1; KNOB="beta1"
    METHOD_EPOCHS=5
    METHOD_ARGS+=(--method-label satimp --beta1 "$VALUE" --beta2 1.0
                  --retain-loss-weight "${RETAIN_WEIGHT:-1.0}" --learning-rate "${LR:-1e-6}")
    ;;
  grad-diff)
    MODULE="pretrain_experiments.grad_diff"; USES_RETAIN=1; KNOB="lambda"
    METHOD_EPOCHS=10
    # LR 3e-6 is the TOP of grad-diff's measured window (3e-7..3e-6), against the
    # low-end bias used elsewhere. Two reasons it is safe here: its probes match
    # gradient ascent's almost exactly, and GA at 3e-6 is the one setting with a
    # MEASURED 100-step outcome (c4 ppl 20.19 -- damaged, not destroyed). grad-diff
    # at the same LR is GA plus a retain brake, so it can only be gentler. And
    # unlike wga, lambda does not scale the forget gradient, so there is no hidden
    # amplification across the grid.
    METHOD_ARGS+=(--retain-loss-weight "$VALUE" --learning-rate "${LR:-3e-6}")
    ;;
  npo)
    MODULE="pretrain_experiments.npo"; USES_RETAIN=1; KNOB="beta"
    METHOD_EPOCHS=10
    METHOD_ARGS+=(--beta "$VALUE" --retain-loss-weight "${RETAIN_WEIGHT:-1.0}"
                  --learning-rate "${LR:-6e-7}" --frozen-dtype "$FROZEN_DTYPE")
    ;;
  simnpo)
    MODULE="pretrain_experiments.simnpo"; USES_RETAIN=1; KNOB="beta"
    METHOD_EPOCHS=10
    METHOD_ARGS+=(--beta "$VALUE" --gamma 0.0 --retain-loss-weight "${RETAIN_WEIGHT:-1.0}"
                  --learning-rate "${LR:-6e-7}")
    ;;
  rmu)
    # 1B has 16 layers; HYPER-PARAMS.md maps the 179M anchor l=5 (of 12) to l=7.
    MODULE="pretrain_experiments.rmu"; USES_RETAIN=1; KNOB="c"
    # RMU's paper budgets ~100-200 optimizer steps rather than epochs. Default
    # to HARD_STEP_CAP rather than a literal 200 so every method on the plot
    # gets the SAME step budget -- otherwise RMU's dots sit further along the
    # forgetting axis partly just from training twice as long, which is a
    # confound the Pareto comparison cannot separate. RMU_STEPS still overrides.
    METHOD_EPOCHS=1; METHOD_MAX_STEPS="${RMU_STEPS:-$HARD_STEP_CAP}"
    METHOD_ARGS+=(--steering-coef "$VALUE" --target-layer "${RMU_LAYER:-7}"
                  --alpha "${RMU_ALPHA:-1200.0}" --n-layers-to-update 3
                  --learning-rate "${LR:-5e-5}" --frozen-dtype "$FROZEN_DTYPE")
    ;;
  *)
    echo "ERROR: unknown METHOD '$METHOD'" >&2
    exit 1
    ;;
esac

# Resolve the budget: explicit env override > method protocol > hard cap.
EPOCHS="${EPOCHS_OVERRIDE:-$METHOD_EPOCHS}"
if [ -n "$MAX_STEPS_OVERRIDE" ]; then
  MAX_STEPS="$MAX_STEPS_OVERRIDE"
elif [ -n "$METHOD_MAX_STEPS" ]; then
  MAX_STEPS="$METHOD_MAX_STEPS"
else
  MAX_STEPS="$HARD_STEP_CAP"
fi

# ---------------------------------------------------------------------------
# Budget: every method sees the SAME forget batch, so
#
#     steps = (|D_f| / TOTAL_BATCH) * epochs
#
# holds uniformly and the only thing that moves the step count between methods
# is the epoch count. Retain methods additionally draw TOTAL_BATCH retain
# sequences per optimizer step, so they cost ~2x the FLOPs at the same step
# count -- that is the TOFU / OpenUnlearning convention: equal steps and equal
# forget exposure, not equal compute. Tokens seen are recorded per run in the
# config snapshot so the asymmetry stays visible.
# ---------------------------------------------------------------------------

FORGET_EFF=$TOTAL_BATCH

if [ $((FORGET_EFF % MICRO_BATCH)) -ne 0 ]; then
  echo "ERROR: effective forget batch $FORGET_EFF not divisible by MICRO_BATCH $MICRO_BATCH" >&2
  exit 1
fi
ACCUM=$((FORGET_EFF / MICRO_BATCH))

OUTPUT_DIR="$OUTPUT_ROOT/${RUN_TAG}/${METHOD}/${KNOB}-${VALUE}"

declare -a COMMON_ARGS=(
  --model "$MODEL"
  --revision "$REVISION"
  --output-dir "$OUTPUT_DIR"
  --gradient-accumulation-steps "$ACCUM"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --checkpoint-every-n-epochs 1
  --max-seq-len "$MAX_SEQ_LEN"
  --seed "$SEED"
  --dtype "$DTYPE"
)

# ce_u.py takes --batch-size; every other driver takes --forget-batch-size.
if [ "$METHOD" = "ce-u" ]; then
  COMMON_ARGS+=(--batch-size "$MICRO_BATCH")
else
  COMMON_ARGS+=(--forget-batch-size "$MICRO_BATCH")
fi

# RETAIN_WEIGHT=0 turns npo / simnpo / satimp into their forget-only variants.
# This matters a lot on a cluster without the OLMo-2 stage1 memmap data: those
# three drivers guard the retain loader behind `use_retain = weight > 0`, so at
# 0 they never touch it and become runnable. For NPO that IS the paper's base
# method (the retain variant is NPO-RT); for SimNPO and SatImp it is a
# documented forget-only ablation, so label the runs accordingly.
#
# The other two are NOT separable and must not pretend to be:
#   grad-diff  the retain weight is its curve knob -- at 0 it IS plain
#              gradient ascent, so the method disappears
#   rmu        builds the retain stream unconditionally (rmu.py:285), no guard
case "${RETAIN_WEIGHT:-}" in
  0|0.0)
    case "$METHOD" in
      npo|simnpo|satimp)
        USES_RETAIN=0
        echo "  NOTE: RETAIN_WEIGHT=0 -> forget-only $METHOD, no OLMo retain stream"
        ;;
      grad-diff)
        echo "ERROR: RETAIN_WEIGHT=0 reduces grad-diff to plain gradient ascent." >&2
        echo "       Its retain weight is the curve knob; sweep VALUE instead." >&2
        exit 1 ;;
      rmu)
        echo "ERROR: rmu builds the retain stream unconditionally (rmu.py:285)." >&2
        echo "       It cannot run without the OLMo-2 stage1 memmap data." >&2
        exit 1 ;;
    esac
    ;;
esac

if [ "$USES_RETAIN" -eq 1 ]; then
  COMMON_ARGS+=(--retain-batch-size "$MICRO_BATCH"
                --olmo-config "$OLMO_CONFIG"
                --retain-start-step "$START_STEP")
fi

if [ "$GRAD_CKPT" = "1" ]; then
  COMMON_ARGS+=(--gradient-checkpointing)
fi

# Library default forget set unless explicitly narrowed. gradient_ascent.py is
# the only driver that cannot express this, which is why the GA cell routes
# through reweighted_ga.py above.
if [ -n "${FORGET_EXPS:-}" ]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(--forget-experiments $FORGET_EXPS)
fi

RETAIN_EFF=0
if [ "$USES_RETAIN" -eq 1 ]; then RETAIN_EFF=$TOTAL_BATCH; fi

echo "============================================"
echo "  Pareto cell: $METHOD  ${KNOB}=${VALUE}"
echo "  site:          ${PE_SITE:-unknown}"
echo "  module:        $MODULE"
echo "  model:         $MODEL @ $REVISION"
echo "  forget set:    ${FORGET_EXPS:-<library default: full minus iid-replacements-*>}"
echo "  budget:        total_batch=$TOTAL_BATCH (forget $FORGET_EFF + retain $RETAIN_EFF)"
echo "                 micro=$MICRO_BATCH accum=$ACCUM epochs=$EPOCHS max_steps=$MAX_STEPS"
echo "                 max_seq_len=$MAX_SEQ_LEN"
echo "  dtype:         $DTYPE (frozen: $FROZEN_DTYPE)  grad_ckpt=$GRAD_CKPT"
echo "  output:        $OUTPUT_DIR"
echo "============================================"

python -m "$MODULE" "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}"

# Diagnostic runs (the LR range test) only need metrics.jsonl and the config
# snapshot. A 1B checkpoint is ~5 GB and every driver force-saves one when
# --max-steps trips, so 64 short runs would otherwise leave ~300 GB behind.
if [ "${KEEP_CHECKPOINTS:-1}" = "0" ]; then
  echo "  KEEP_CHECKPOINTS=0 -> removing epoch-*/ (metrics and config kept)"
  rm -rf "${OUTPUT_DIR:?}"/epoch-*
fi

echo ""
echo "============================================"
echo "  DONE: $METHOD ${KNOB}=${VALUE}"
echo "  output under: $OUTPUT_DIR"
echo "============================================"
