#!/bin/bash -l
#SBATCH --account=p200xxx
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=unlearn-1B
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# MeluXina wrapper for one cell of the 1B unlearning/utility Pareto sweep.
# One job == one dot on the plot.
#
# `#!/bin/bash -l` is REQUIRED: LMod is only initialised in a login shell, so
# without -l every `module load` below fails.
#
# NOTE: --account=p200xxx is a placeholder. Replace it, or override at submit
# time with `sbatch -A p200yyy ...` (the CLI flag wins over the directive).
#
# QOS budget (docs.lxp.lu): dev 6h/1 node, test 30min, short 6h, default 48h on
# 25% of nodes, long 144h on 5%, large 24h on 70%. `default` at 24h suits a 1B
# cell; raise --time toward 48:00:00 if the longer 10-epoch protocols need it,
# or switch to `-q long` for more headroom on fewer nodes.
#
# Hardware: MeluXina GPU nodes are 4x A100-40GB, 2x AMD EPYC Rome 7452,
# 512 GB RAM. At 1B with DTYPE=bfloat16 a cell needs roughly fp32 weights 4 GB
# + Adam 8 GB + grads 4 GB, plus a frozen fp32 reference (+4 GB) for npo/rmu,
# plus activations. MICRO_BATCH defaults to 4 -- OLMo's own
# device_train_microbatch_size for this model -- at a quarter of its sequence
# length, so activation memory is modest and 40 GB should be comfortable.
# If a cell still OOMs, set GRAD_CKPT=1 or MICRO_BATCH=2; if throughput is the
# problem instead, MICRO_BATCH=16 is token-equivalent to the reference config.
# Accumulation rescales automatically either way to hold the effective batch
# at TOTAL_BATCH.
#
# Usage -- identical interface to the galvani/ferranti wrapper:
#
#   sbatch --export=ALL,METHOD=simnpo,VALUE=0.5 internal/meluxina/unlearn_cell_1B.sh
#
# The whole grid:
#
#   CELL_SCRIPT=internal/meluxina/unlearn_cell_1B.sh \
#     bash internal/uwiki/launch_pareto_sweep_1B.sh
#
# All experiment-level env vars (METHOD, VALUE, TOTAL_BATCH, MICRO_BATCH,
# EPOCHS, MAX_STEPS, HARD_STEP_CAP, MODEL, REVISION, FORGET_EXPS, LR, DTYPE,
# RMU_LAYER, ...) are documented in internal/uwiki/unlearn_cell_body.sh.
# Site-level vars (PE_PROJECT, PE_REPO, PE_VENV, HF_HOME, OUTPUT_ROOT,
# OLMO_CONFIG) are documented in internal/meluxina/env.sh.

set -u
set -o pipefail

# Resolve the repo from this script's own location so env.sh can be found
# before PE_REPO is known.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GUESS="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

# shellcheck disable=SC1091
source "${PE_REPO:-$REPO_GUESS}/internal/uwiki/unlearn_cell_body.sh"
