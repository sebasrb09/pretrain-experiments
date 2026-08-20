#!/bin/bash
# Launch the full unlearning/utility Pareto sweep at 1B: 8 methods x 4 curve-knob
# values = 32 training cells, one sbatch job each.
#
# Run this on the login node (it is NOT itself an sbatch script):
#
#   bash internal/uwiki/launch_pareto_sweep_1B.sh
#
# Always dry-run first -- this submits 32 jobs:
#
#   DRY_RUN=1 bash internal/uwiki/launch_pareto_sweep_1B.sh
#
# Budget (see unlearn_cell_1B.sh for the full model): every cell uses the same
# forget batch, the library-default forget set, and paper-default values for
# every knob except the one being swept, so
#
#     steps = (|D_f| / TOTAL_BATCH) * epochs
#
# and the only thing that moves the step count between methods is each method's
# own published epoch count. Within a method, the four dots differ in exactly
# one number.
#
# On MeluXina, point it at that site's wrapper -- everything else is identical:
#
#   CELL_SCRIPT=internal/meluxina/unlearn_cell_1B.sh \
#     bash internal/uwiki/launch_pareto_sweep_1B.sh
#
# Optional env vars:
#   DRY_RUN      - 1 to print the sbatch commands without submitting
#   CELL_SCRIPT  - per-site cell wrapper (default: internal/uwiki/unlearn_cell_1B.sh)
#   METHODS      - space-separated subset to launch (default: all 8)
#   RUN_TAG      - subdir under unlearning-pareto/ (default: 1B-pareto)
#   TOTAL_BATCH  - forget sequences per optimizer step (default: 512)
#   MICRO_BATCH  - per-forward micro batch (default: 8)
#   EPOCHS       - override the per-method published epoch count for ALL methods
#   MAX_STEPS    - override the step ceiling for ALL methods
#   HARD_STEP_CAP- ceiling for methods with no step-based protocol (default: 10000)
#   TIME         - SLURM walltime override (default: the cell script's 2-00:00:00)
#   DTYPE, GRAD_CKPT, MODEL, REVISION, OLMO_CONFIG, START_STEP, FORGET_EXPS,
#   SEED, MAX_SEQ_LEN  - forwarded verbatim to unlearn_cell_1B.sh
#
# Re-running is safe in the sense that finished cells are simply re-trained;
# there is no skip-if-exists here (unlike the eval sweep). Use METHODS= to
# relaunch a subset.

set -u
set -o pipefail

# Site is selected purely by which cell wrapper runs; the grid, the budget and
# the dispatch are shared.
#   galvani/ferranti (default): internal/uwiki/unlearn_cell_1B.sh
#   MeluXina:                   internal/meluxina/unlearn_cell_1B.sh
CELL_SCRIPT="${CELL_SCRIPT:-internal/uwiki/unlearn_cell_1B.sh}"
if [ ! -f "$CELL_SCRIPT" ]; then
  echo "ERROR: $CELL_SCRIPT not found. Run this from the repo root." >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-1B-pareto}"
TOTAL_BATCH="${TOTAL_BATCH:-512}"
MICRO_BATCH="${MICRO_BATCH:-8}"
# EPOCHS and MAX_STEPS are deliberately NOT defaulted here: each method runs its
# own published epoch count, set inside unlearn_cell_1B.sh. Exporting either one
# from here would flatten that across all eight methods.

DEFAULT_METHODS="gradient-ascent ce-u wga grad-diff npo simnpo rmu satimp"
METHODS="${METHODS:-$DEFAULT_METHODS}"

# ---------------------------------------------------------------------------
# The grid. One curve knob per method; everything else is a paper default and
# is pinned inside unlearn_cell_1B.sh.
#
#   method            knob    values
#   ----------------  ------  ---------------------------------------------
#   gradient-ascent   lr      biased ~1 decade below Jang's 5e-5 (HYPER-PARAMS.md:87-94)
#   ce-u              lr      CE-U has no method hyperparameter; LR is its only axis
#   wga               beta1   1.0 exactly cancels GA's 1/p factor; sweep around it
#   grad-diff         lambda  retain weight: the forget/retain trade-off knob
#   npo               beta    on SUMMED NLL, so far below TOFU's 0.1 (see npo.py)
#   simnpo            beta    the paper's own grid, on length-normalized NLL
#   rmu               c       paper default 6.5 is Llama-2-chat-calibrated; bracket it
#   satimp            beta1   paper recommends 5 with beta2=1
#
# LUNAR is implemented (lunar.py) but is not one of the eight; to include it,
# add "lunar" to DEFAULT_METHODS, add a case here, and add a dispatch branch to
# unlearn_cell_1B.sh with --redirection-layer / --retain-loss-weight.
# ---------------------------------------------------------------------------

grid_for () {
  case "$1" in
    gradient-ascent) echo "1e-7 3e-7 1e-6 3e-6" ;;
    ce-u)            echo "1e-7 3e-7 1e-6 3e-6" ;;
    wga)             echo "0.5 1.0 2.0 5.0" ;;
    grad-diff)       echo "0.5 1.0 2.0 5.0" ;;
    npo)             echo "1e-4 1e-3 1e-2 1e-1" ;;
    simnpo)          echo "0.1 0.5 1.0 2.5" ;;
    rmu)             echo "2.0 4.0 6.5 10.0" ;;
    satimp)          echo "1.0 2.0 5.0 10.0" ;;
    *)               echo "" ;;
  esac
}

knob_for () {
  case "$1" in
    gradient-ascent|ce-u) echo "lr" ;;
    wga|satimp)           echo "beta1" ;;
    grad-diff)            echo "lambda" ;;
    npo|simnpo)           echo "beta" ;;
    rmu)                  echo "c" ;;
    *)                    echo "value" ;;
  esac
}

# Forward only the overrides that were actually set, so unlearn_cell_1B.sh keeps
# its own defaults for everything else.
EXPORTS="ALL,RUN_TAG=${RUN_TAG},TOTAL_BATCH=${TOTAL_BATCH},MICRO_BATCH=${MICRO_BATCH}"
for var in EPOCHS MAX_STEPS HARD_STEP_CAP DTYPE FROZEN_DTYPE GRAD_CKPT MODEL REVISION OLMO_CONFIG START_STEP FORGET_EXPS SEED MAX_SEQ_LEN LR RMU_LAYER RMU_ALPHA RMU_STEPS OUTPUT_ROOT; do
  if [ -n "${!var:-}" ]; then
    EXPORTS="${EXPORTS},${var}=${!var}"
  fi
done

# Kept as a plain string rather than an array: "${arr[@]}" on an empty array
# aborts under `set -u` on bash < 4.4, which some compute nodes still ship.
TIME_ARG=""
if [ -n "${TIME:-}" ]; then
  TIME_ARG="--time=$TIME"
fi

echo "============================================"
echo "  Pareto sweep launch (1B)"
echo "  run_tag:      $RUN_TAG"
echo "  methods:      $METHODS"
echo "  budget:       total_batch=$TOTAL_BATCH micro=$MICRO_BATCH"
echo "                epochs=${EPOCHS:-<per-method published protocol>}"
echo "                step cap=${MAX_STEPS:-${HARD_STEP_CAP:-10000}}"
echo "  dry run:      $DRY_RUN"
echo "============================================"
echo ""

n_submitted=0
n_skipped=0

for method in $METHODS; do
  values="$(grid_for "$method")"
  if [ -z "$values" ]; then
    echo "!! unknown method '$method', skipping"
    n_skipped=$((n_skipped + 1))
    continue
  fi
  knob="$(knob_for "$method")"
  echo "--- $method (knob: $knob) ---"
  for v in $values; do
    job_name="pareto-${method}-${knob}${v}"
    if [ "$DRY_RUN" = "1" ]; then
      echo "  [dry] sbatch -J $job_name ${TIME_ARG} --export=${EXPORTS},METHOD=${method},VALUE=${v} $CELL_SCRIPT"
    elif [ -n "$TIME_ARG" ]; then
      sbatch -J "$job_name" "$TIME_ARG" \
             --export="${EXPORTS},METHOD=${method},VALUE=${v}" \
             "$CELL_SCRIPT"
    else
      sbatch -J "$job_name" \
             --export="${EXPORTS},METHOD=${method},VALUE=${v}" \
             "$CELL_SCRIPT"
    fi
    n_submitted=$((n_submitted + 1))
  done
  echo ""
done

echo "============================================"
if [ "$DRY_RUN" = "1" ]; then
  echo "  DRY RUN: $n_submitted cells would be submitted"
else
  echo "  submitted $n_submitted cells"
fi
[ "$n_skipped" -gt 0 ] && echo "  skipped $n_skipped unknown method name(s)"
echo ""
echo "  Outputs:  \$HOME/pretrain-experiments/unlearning-pareto/${RUN_TAG}/<method>/<knob>-<value>/"
echo "  Each cell writes <method>_config.json, metrics.jsonl, and epoch-N/ checkpoints."
echo ""
echo "  NOT launched here: the three anchor points (baseline @100k,"
echo "  continued-pretraining unlearning baseline, deep-ignorance). Those are"
echo "  existing checkpoints, so they belong to the eval stage, not training."
echo "============================================"
