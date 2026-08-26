#!/bin/bash
# LR range test on the Vienna cluster -- a thin wrapper over
# internal/uwiki/launch_lr_range_test.sh with this cluster's defaults baked in.
#
# Why it exists: every method's learning rate is currently either a paper value
# measured on 7B models at batch 16-32 on TOFU (npo/simnpo 1e-5, rmu 5e-5), or
# an extrapolation (grad-diff/wga/satimp 1e-6). None was measured at 1B, batch
# 512, on a pretraining-corpus forget set. We already know 1e-6 moves the 1B
# model 0.26 nats in 20 steps -- so npo/simnpo would run at 10x that and rmu at
# 50x. If those cells diverge, a flat Pareto curve cannot be told apart from
# "this method's hyperparameter does not matter".
#
# Each probe runs the real driver at the real effective batch for 20 optimizer
# steps and keeps only metrics.jsonl.
#
# Usage:
#   DRY_RUN=1 bash internal/uwiki/lr_range.sh          # always look first
#   bash internal/uwiki/lr_range.sh                    # the 3 no-OLMo methods
#   GROUP=olmo bash internal/uwiki/lr_range.sh         # the 5 retain-set methods
#   GROUP=all  bash internal/uwiki/lr_range.sh         # all eight
#
#   python internal/uwiki/analyze_lr_range_test.py     # read it
#
# Env vars:
#   GROUP     nolmo | olmo | all      (default: nolmo)
#   LRS       LR ladder               (default: 3e-7 .. 1e-4, centred on the
#                                      known-live 1e-6 rather than the original
#                                      1e-8 start, which wasted three rungs)
#   TIME      walltime per probe      (default: 4:00:00 -- generous, since each
#                                      job tokenizes the full 5.2M-text forget
#                                      set before training)
#   METHODS   explicit override, ignores GROUP
#
# Account: the cell wrapper defaults to `datamining`. To use another, export
# the Slurm env vars -- they beat the script directive:
#   SBATCH_ACCOUNT=low SBATCH_PARTITION=<partition> bash internal/uwiki/lr_range.sh
# Find the partition that goes with an account:
#   sacctmgr show assoc where user=$USER format=Account,Partition,QOS -p

set -u
set -o pipefail

[ -f internal/uwiki/launch_lr_range_test.sh ] \
  || { echo "ERROR: run this from the repo root" >&2; exit 1; }

GROUP="${GROUP:-nolmo}"

# grad-diff, npo, simnpo, rmu and satimp all build the OLMo retain stream, which
# needs the OLMo-2 stage1 memmap DATA on this cluster -- not just the fork.
# Step 5 of internal/uwiki/setup_env.sh reports whether that works here.
case "$GROUP" in
  nolmo) DEFAULT_METHODS="gradient-ascent ce-u wga" ;;
  olmo)  DEFAULT_METHODS="grad-diff npo simnpo rmu satimp" ;;
  all)   DEFAULT_METHODS="gradient-ascent ce-u wga grad-diff npo simnpo rmu satimp" ;;
  *)     echo "ERROR: GROUP must be nolmo | olmo | all (got '$GROUP')" >&2; exit 1 ;;
esac

export METHODS="${METHODS:-$DEFAULT_METHODS}"
export LRS="${LRS:-3e-7 1e-6 3e-6 1e-5 3e-5 1e-4}"
export TIME="${TIME:-4:00:00}"
export CELL_SCRIPT="internal/uwiki/unlearn_cell_1B.sh"

n_methods=$(echo "$METHODS" | wc -w)
n_lrs=$(echo "$LRS" | wc -w)

echo "============================================"
echo "  LR range test -- Vienna cluster"
echo "  group:    $GROUP"
echo "  methods:  $METHODS"
echo "  LRs:      $LRS"
echo "  jobs:     $((n_methods * n_lrs))"
echo "  account:  ${SBATCH_ACCOUNT:-datamining (script default)}"
echo "============================================"
echo ""

bash internal/uwiki/launch_lr_range_test.sh

echo ""
echo "  Read the results with:"
echo "    python internal/uwiki/analyze_lr_range_test.py --output-root \\"
echo "      \"\${OUTPUT_ROOT:-\$HOME/pretrain-experiments/unlearning-pareto}\""
echo ""
echo "  Two things come out of this, not one:"
echo "    1. the usable LR window per method -> the pinned LRs in"
echo "       internal/uwiki/unlearn_cell_body.sh"
echo "    2. how fast ce_forget moves on the REAL forget set -> HARD_STEP_CAP"
echo ""
echo "  DONE for gradient-ascent, ce-u and wga (GROUP=nolmo): windows are"
echo "  3e-7..3e-6, 1e-6..3e-5 and 3e-7..1e-5, and HARD_STEP_CAP is now 1000."
echo "  Still open: the five GROUP=olmo methods, which need the retain stream." 
