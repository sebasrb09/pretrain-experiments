#!/bin/bash
# Learning-rate range test: find, per method, the LR window between "nothing
# happens" and "the run destabilises", before committing to a sweep grid.
#
# Why this exists. The grids in launch_pareto_sweep_1B.sh come from the papers
# for the six method-knob sweeps, but GradAscent and CE-U have no method
# hyperparameter -- LR *is* their curve. A four-point LR grid in the wrong place
# gives two dead cells, one live one and one divergence: a one-point "curve",
# which makes those methods look bad for reasons unrelated to the methods. The
# remaining six also carry a fixed LR that was inherited rather than measured,
# and inherited from a small-batch (16-64) regime at that, while the sweep runs
# at an effective batch of 512 -- larger batches generally want larger LRs, so
# the current values may be systematically low.
#
# What it does. For each method, runs the real driver at the real effective
# batch for a handful of optimizer steps across a wide LR range, keeping only
# metrics.jsonl. No evaluation pass: the readout is training-side movement,
# which is enough to bracket the window.
#
# Running at the real TOTAL_BATCH is deliberate -- the optimal LR depends on the
# batch size, so a cheaper small-batch probe would not transfer.
#
#   DRY_RUN=1 bash internal/uwiki/launch_lr_range_test.sh    # always first
#   bash internal/uwiki/launch_lr_range_test.sh
#   python internal/uwiki/analyze_lr_range_test.py           # once jobs finish
#
# On VSC-5 (or any other site) point it at that wrapper:
#
#   CELL_SCRIPT=internal/asc/unlearn_cell_1B.sh \
#     bash internal/uwiki/launch_lr_range_test.sh
#
# Optional env vars:
#   DRY_RUN      - 1 to print the sbatch commands without submitting
#   CELL_SCRIPT  - per-site cell wrapper (default: internal/uwiki/unlearn_cell_1B.sh)
#   METHODS      - subset to probe (default: all 8)
#   LRS          - LR ladder (default: 1e-8 .. 1e-4, 8 points over 4 decades)
#   RANGE_STEPS  - optimizer steps per probe (default: 20)
#   RUN_TAG_BASE - subdir under unlearning-pareto/ (default: lr-range)
#   TIME         - SLURM walltime override (default: 2:00:00, these are short)
#   Everything else (TOTAL_BATCH, MICRO_BATCH, MODEL, DTYPE, ...) is forwarded
#   unchanged, so the probe matches the real runs.
#
# Cost note: at TOTAL_BATCH=512 / MICRO_BATCH=4 one optimizer step is 128
# forward-backward passes, so RANGE_STEPS=20 is 2560 micro-batches -- minutes,
# not seconds. 8 methods x 8 LRs = 64 jobs. Trim with METHODS= or LRS= if that
# is more than you want to spend; the two LR-curve methods alone
# (METHODS="gradient-ascent ce-u") is 16 jobs and answers the sharpest question.

set -u
set -o pipefail

CELL_SCRIPT="${CELL_SCRIPT:-internal/uwiki/unlearn_cell_1B.sh}"
if [ ! -f "$CELL_SCRIPT" ]; then
  echo "ERROR: $CELL_SCRIPT not found. Run this from the repo root." >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-0}"
RUN_TAG_BASE="${RUN_TAG_BASE:-lr-range}"
RANGE_STEPS="${RANGE_STEPS:-20}"
LRS="${LRS:-1e-8 1e-7 3e-7 1e-6 3e-6 1e-5 3e-5 1e-4}"
TIME="${TIME:-2:00:00}"

DEFAULT_METHODS="gradient-ascent ce-u wga grad-diff npo simnpo rmu satimp"
METHODS="${METHODS:-$DEFAULT_METHODS}"

# For GradAscent and CE-U the curve knob IS the learning rate, so the probed LR
# goes in VALUE. For the other six the knob is held at a sensible centre and the
# LR is passed separately, so the probe isolates the LR.
holds_lr_in_value () {
  case "$1" in
    gradient-ascent|ce-u) return 0 ;;
    *) return 1 ;;
  esac
}

# Knob value to hold each method at while its LR is probed. Paper defaults,
# except npo (0.1 is expected to saturate the sigmoid on summed NLL -- see
# npo.py) and wga (1.0 exactly cancels gradient ascent's 1/p factor).
centre_knob () {
  case "$1" in
    wga)       echo "1.0" ;;
    satimp)    echo "5.0" ;;
    grad-diff) echo "1.0" ;;
    npo)       echo "1e-3" ;;
    simnpo)    echo "0.1" ;;
    rmu)       echo "6.5" ;;
    *)         echo "" ;;
  esac
}

BASE_EXPORTS="ALL,KEEP_CHECKPOINTS=0,MAX_STEPS=${RANGE_STEPS},EPOCHS=20"
for var in TOTAL_BATCH MICRO_BATCH DTYPE FROZEN_DTYPE GRAD_CKPT MODEL REVISION \
           OLMO_CONFIG START_STEP FORGET_EXPS SEED MAX_SEQ_LEN RMU_LAYER \
           RMU_ALPHA OUTPUT_ROOT; do
  if [ -n "${!var:-}" ]; then
    BASE_EXPORTS="${BASE_EXPORTS},${var}=${!var}"
  fi
done

echo "============================================"
echo "  LR range test"
echo "  methods:   $METHODS"
echo "  LRs:       $LRS"
echo "  steps:     $RANGE_STEPS per probe (checkpoints discarded)"
echo "  run tag:   $RUN_TAG_BASE/lr<value>/"
echo "  dry run:   $DRY_RUN"
echo "============================================"
echo ""

n=0
for method in $METHODS; do
  echo "--- $method ---"
  for lr in $LRS; do
    if holds_lr_in_value "$method"; then
      value="$lr"
      extra=""
    else
      value="$(centre_knob "$method")"
      if [ -z "$value" ]; then
        echo "  !! no centre knob defined for '$method', skipping"
        continue
      fi
      extra=",LR=${lr}"
    fi

    # RUN_TAG carries the LR so probes of a fixed-knob method do not collide:
    #   <root>/<base>/lr<LR>/<method>/<knob>-<value>/
    tag="${RUN_TAG_BASE}/lr${lr}"
    job="lrrange-${method}-${lr}"
    exports="${BASE_EXPORTS},RUN_TAG=${tag},METHOD=${method},VALUE=${value}${extra}"

    if [ "$DRY_RUN" = "1" ]; then
      echo "  [dry] sbatch -J $job --time=$TIME --export=$exports $CELL_SCRIPT"
    else
      sbatch -J "$job" --time="$TIME" --export="$exports" "$CELL_SCRIPT"
    fi
    n=$((n + 1))
  done
  echo ""
done

echo "============================================"
if [ "$DRY_RUN" = "1" ]; then
  echo "  DRY RUN: $n probes would be submitted"
else
  echo "  submitted $n probes"
fi
echo ""
echo "  When they finish:"
echo "    python internal/uwiki/analyze_lr_range_test.py --run-tag-base $RUN_TAG_BASE"
echo "============================================"
