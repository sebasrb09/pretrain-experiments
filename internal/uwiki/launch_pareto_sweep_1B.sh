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
# Point CELL_SCRIPT at your site's wrapper -- everything else is identical:
#
#   CELL_SCRIPT=internal/asc/unlearn_cell_1B.sh \
#     bash internal/uwiki/launch_pareto_sweep_1B.sh        # VSC-5
#   CELL_SCRIPT=internal/meluxina/unlearn_cell_1B.sh \
#     bash internal/uwiki/launch_pareto_sweep_1B.sh        # MeluXina
#
# Optional env vars:
#   VALUES       - override a method's grid (use with a single METHOD, to
#                  add rungs without re-running finished cells)
#   DRY_RUN      - 1 to print the sbatch commands without submitting
#   CELL_SCRIPT  - per-site cell wrapper (default: ASC/MUSICA when
#                  internal/asc/env.sh and $SCRATCH/$DATA are present,
#                  else internal/uwiki/unlearn_cell_1B.sh)
#   METHODS      - space-separated subset to launch (default: all 8)
#   RUN_TAG      - subdir under unlearning-pareto/ (default: 1B-pareto)
#   TOTAL_BATCH  - forget sequences per optimizer step (default: 512, from the OLMo config)
#   MICRO_BATCH  - per-forward micro batch (default: 4, from the OLMo config)
#   EPOCHS       - override the per-method published epoch count for ALL methods
#   MAX_STEPS    - override the step ceiling for ALL methods
#   HARD_STEP_CAP- ceiling for methods with no step-based protocol (default: 10000)
#   TIME         - SLURM walltime override (default: whatever the cell script sets)
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
#   VSC-5:                      internal/asc/unlearn_cell_1B.sh
#   MeluXina:                   internal/meluxina/unlearn_cell_1B.sh
# Default to the site we are actually on. A hardcoded u:wiki default sent a
# whole LR-range batch to --partition=p_datamining from MUSICA before this was
# fixed; an explicit CELL_SCRIPT= still overrides.
if [ -n "${CELL_SCRIPT:-}" ]; then
  :
elif [ -f internal/asc/env.sh ] && [ -n "${SCRATCH:-}${DATA:-}" ]; then
  CELL_SCRIPT="internal/asc/unlearn_cell_1B.sh"
else
  CELL_SCRIPT="internal/uwiki/unlearn_cell_1B.sh"
fi
if [ ! -f "$CELL_SCRIPT" ]; then
  echo "ERROR: $CELL_SCRIPT not found. Run this from the repo root." >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-1B-pareto}"
# Nothing about the budget is defaulted here. TOTAL_BATCH and MICRO_BATCH come
# from the OLMo-2 1B stage1 config (512 / 4) and EPOCHS from each method's
# published protocol -- all set in unlearn_cell_body.sh. Re-defaulting any of
# them here would silently override those, so they are forwarded only when you
# export them explicitly.

DEFAULT_METHODS="gradient-ascent ce-u wga grad-diff npo simnpo rmu satimp"
METHODS="${METHODS:-$DEFAULT_METHODS}"

# ---------------------------------------------------------------------------
# The grid. One curve knob per method; everything else is a paper default and
# is pinned inside unlearn_cell_1B.sh.
#
#   method            knob    values
#   ----------------  ------  ---------------------------------------------
#   gradient-ascent   lr      MEASURED window 3e-7..3e-6 (the narrowest of the
#                             three, and the lowest -- this confirms the
#                             "1 decade below Jang's 5e-5" bias in HYPER-PARAMS.md:87-94)
#   ce-u              lr      MEASURED window 1e-6..3e-5. CE-U has no method
#                             hyperparameter, so LR is its only axis. NOTE: the
#                             previous grid (1e-7..3e-6) put 3 of its 4 points
#                             BELOW this window -- CE-U would have traced a
#                             one-point curve and looked like it does not work.
#                             It in fact tolerates ~10x GA's learning rate.
#                             The grid is the measured window shifted down ~3x,
#                             because that window was measured over 20 steps and
#                             the sweep runs 100: at 1e-6..3e-5 over 100 steps
#                             CE-U spans +1.0..+20 nats, which is top-collapsed.
#                             6e-7..1e-5 spans +0.7..+6.8 instead.
#   wga               beta1   1.0 exactly cancels GA's 1/p factor; sweep around it.
#                             Grid narrowed from {0.5,1,2,5}: since w = p^beta1
#                             and p ~ 0.166 on this forget set, beta1 rescales
#                             the EFFECTIVE step by p^(beta1-1) -- {0.5,1,2,5}
#                             spans ~3200x while wga's measured usable window is
#                             only ~33x wide, so beta1=2 and 5 would have been
#                             dead cells. {0.5,1,1.5,2} spans ~15x and fits.
#   grad-diff         lambda  retain weight: the forget/retain trade-off knob
#   npo               beta    on SUMMED NLL, so far below TOFU's 0.1 (see npo.py)
#   simnpo            beta    the paper's own grid, on length-normalized NLL
#   rmu               c       paper default 6.5 is Llama-2-chat-calibrated; bracket it
#   satimp            beta1   paper recommends 5 with beta2=1. Grid LEFT ALONE:
#                             the same effective-step shrinkage applies, but for
#                             SatImp it is the intended behaviour (w peaks at
#                             p* = beta1/(beta1+beta2), so beta1=5 deliberately
#                             targets the p~0.83 memorized tail). The fix there
#                             is LR compensation, not a narrower grid -- and
#                             SatImp needs the retain stream, so it has no
#                             measurement yet. Re-check after GROUP=olmo runs.
#
# LUNAR is implemented (lunar.py) but is not one of the eight; to include it,
# add "lunar" to DEFAULT_METHODS, add a case here, and add a dispatch branch to
# unlearn_cell_1B.sh with --redirection-layer / --retain-loss-weight.
# ---------------------------------------------------------------------------

grid_for () {
  case "$1" in
    # Revised after the first sweep measured c4 perplexity. Healthy band is
    # 18.72-18.88 (the three anchors); above ~19 is real utility damage.
    #   gradient-ascent  all 4 kept utility (18.81-20.19) but barely unlearned:
    #                    fk_prob only 3.4e-02 -> 1.4e-02 against a 12000x range,
    #                    and gw_mean_in did not move at all. 6e-6 and 1e-5 test
    #                    whether it EVER unlearns or only ever breaks the model.
    #   ce-u             collapses between 1.6e-6 (ppl 19.67) and 4e-6 (ppl 39).
    #                    The three new rungs sample that gap.
    #   wga              every rung destroyed utility (49-532) at the pinned
    #                    LR 3e-6. Higher beta1 shrinks the effective step
    #                    (w = p^beta1, p ~ 0.166), so 2.5-5.0 walks back toward
    #                    the healthy band. LR stays 3e-6 ON PURPOSE: the cell
    #                    path is <method>/<knob>-<value> and carries no LR, so
    #                    changing it would overwrite the four existing cells
    #                    with incomparable runs.
    # gradient-ascent and ce-u have no method hyperparameter, so LR was their
    # only axis. LR is now pinned to the OLMo-2 value at step 100k for every
    # method, which leaves them with nothing to sweep: one cell each, and a
    # single point on the hyperparameter plot. The knob stays "lr" and the
    # value is the pinned LR, so <method>/lr-<value> still names the LR the
    # cell actually ran at -- but nothing is being varied here.
    gradient-ascent) echo "3.9855694839172363e-4" ;;
    ce-u)            echo "3.9855694839172363e-4" ;;
    # Trimmed from 8 to 4. The four extra rungs (2.5-5.0) were appended in
    # Phase A to shrink the effective step back toward the healthy band at the
    # then-pinned LR of 3e-6; that LR is gone, replaced by the pretraining
    # value ~130x higher, so the calibration behind them no longer applies.
    # What remains spans the range: 1.0 exactly cancels GA's 1/p factor and is
    # the one theoretically anchored point, 0.5 amplifies, 2.0 and 5.0 shrink.
    # Five checkpoints per cell now sample the utility trade-off along the
    # trajectory, which is the job the extra rungs were doing.
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
EXPORTS="ALL,RUN_TAG=${RUN_TAG}"
for var in TOTAL_BATCH MICRO_BATCH EPOCHS MAX_STEPS HARD_STEP_CAP DTYPE FROZEN_DTYPE GRAD_CKPT MODEL REVISION OLMO_CONFIG START_STEP FORGET_EXPS SEED MAX_SEQ_LEN LR RMU_LAYER RMU_ALPHA RMU_STEPS RETAIN_WEIGHT OUTPUT_ROOT; do
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
echo "  cell:         $CELL_SCRIPT"
echo "  methods:      $METHODS"
echo "  budget:       total_batch=${TOTAL_BATCH:-512 (default)} micro=${MICRO_BATCH:-4 (default)}"
echo "                step cap=${MAX_STEPS:-${HARD_STEP_CAP:-100}} (BINDS: 1 epoch = 10249 steps, so EPOCHS is inert)"
echo "  dry run:      $DRY_RUN"
echo "============================================"
echo ""

n_submitted=0
n_skipped=0

for method in $METHODS; do
  # VALUES overrides the grid, for topping up one method without re-running the
  # cells that already finished. Only meaningful with a single METHOD.
  values="${VALUES:-$(grid_for "$method")}"
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
