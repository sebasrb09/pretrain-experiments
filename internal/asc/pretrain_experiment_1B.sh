#!/bin/bash
#SBATCH --account=p201378
#SBATCH --job-name=pretrain-experiment-1B
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# MUSICA wrapper for the pretrain-experiments framework (OLMo-2 continued
# training), as opposed to the post-hoc unlearning drivers in
# internal/asc/unlearn_cell_1B.sh.
#
# FULL GPU NODE, exclusive. MUSICA's job_submit plugin picks an allocation mode
# from which flags are supplied; `-N 1 --gres=gpu:4` is the full-GPU-node
# comfort mode, so there is deliberately no --ntasks or --cpus-per-task here
# (adding them fails with 'Requested node configuration is not available').
# The framework launches `torchrun --nproc_per_node=<visible GPUs>` itself, so
# it picks up all four without any layout flags from us.
#
# Usage:
#   sbatch internal/asc/pretrain_experiment_1B.sh config/control-fine-1B.yaml
#   CONFIG=config/control-fine-1B.yaml sbatch internal/asc/pretrain_experiment_1B.sh
#
# Anything after the config is forwarded to the CLI, so dotted overrides work:
#   sbatch internal/asc/pretrain_experiment_1B.sh config/control-fine-1B.yaml \
#          --training.num_steps 2
#
# Reference pace: 1B continued training ran ~5h50m per 1000 steps on 4x H100
# (GAUSSIAN_NOISE_UNLEARNING.md, experiment 3). The 24 h limit here leaves room;
# QOS zen4_0768_h100x4 allows up to 72 h if a longer horizon is needed.

set -u
set -o pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

[ -f internal/asc/env.sh ] || {
  echo "ERROR: run this from the repo root (no internal/asc/env.sh here)" >&2; exit 1; }

source internal/asc/env.sh
source "${PE_VENV:-$SCRATCH/venvs/pe}/bin/activate"

# The compute nodes have no outbound network for W&B; offline keeps wandb.init
# from blocking, and the run can be synced later with `wandb sync`.
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG="${1:-${CONFIG:-config/control-fine-1B.yaml}}"
[ $# -gt 0 ] && shift

echo "============================================"
echo "  pretrain-experiments framework"
echo "  config:   $CONFIG"
echo "  gpus:     ${SLURM_GPUS_ON_NODE:-?}  (torchrun picks these up itself)"
echo "  wandb:    $WANDB_MODE"
echo "  python:   $(command -v python)"
echo "============================================"

python -m pretrain_experiments "$CONFIG" "$@"
