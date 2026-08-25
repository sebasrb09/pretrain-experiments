#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=unlearn-1B
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# Vienna cluster (u:wiki) wrapper for one cell of the 1B unlearning/utility
# Pareto sweep. One job == one dot on the plot.
#
# This file holds ONLY the site setup: SLURM directives, module stack, repo
# location. Everything about the experiment -- method dispatch, budget model,
# argument construction -- lives in internal/uwiki/unlearn_cell_body.sh, shared
# with the other site wrappers so they cannot drift.
#
# Accounts: datamining (DM group priority, default), csunivie (general),
# low (backfill). Override per submission -- CLI > env > directive:
#   sbatch -A low -p <partition> ... internal/uwiki/unlearn_cell_1B.sh
#   SBATCH_ACCOUNT=low bash internal/uwiki/launch_pareto_sweep_1B.sh
# Discover the partition for an account with:
#   sacctmgr show assoc where user=$USER format=Account,Partition,QOS -p
#
# Usage:
#   sbatch --export=ALL,METHOD=simnpo,VALUE=0.5 internal/uwiki/unlearn_cell_1B.sh
#
# The whole grid is launched by internal/uwiki/launch_pareto_sweep_1B.sh.
#
# Experiment-level env vars (METHOD, VALUE, TOTAL_BATCH, MICRO_BATCH, EPOCHS,
# MAX_STEPS, HARD_STEP_CAP, MODEL, REVISION, FORGET_EXPS, LR, DTYPE, GRAD_CKPT,
# RMU_LAYER, OUTPUT_ROOT, ...) are documented in unlearn_cell_body.sh.

set -u
set -o pipefail

export PE_SITE="uwiki"

unset SSL_CERT_FILE

PE_REPO="${PE_REPO:-$HOME/pretrain-experiments}"
cd "$PE_REPO"

# Activate the venv built by internal/uwiki/setup_env.sh. Deliberately NOT
# `module load miniforge`: that modulefile runs `conda create` on load, which
# once consumed a 2h walltime silently. See setup_env.sh for the full story.
PE_VENV="${PE_VENV:-$HOME/venvs/pretrain-experiments}"
[ -f "$PE_VENV/bin/activate" ] || {
  echo "ERROR: no usable venv at $PE_VENV" >&2
  echo "       Build it: sbatch internal/uwiki/setup_env.sh  (add FORCE=1 if partial)" >&2
  exit 1
}
# shellcheck disable=SC1091
source "$PE_VENV/bin/activate"
echo "  venv:   $PE_VENV"
echo "  python: $(command -v python) ($(python --version 2>&1))"

# Fail in seconds rather than after the forget-set tokenization if torch cannot
# use the allocated GPU -- a cu13 wheel on a CUDA 12.9 driver does exactly that,
# silently returning is_available()==False.
if [ -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]; then
  python - <<'TORCHCHK' || { echo "ERROR: GPU allocated but torch cannot use it. Rebuild with: FORCE=1 TORCH_INDEX=cu126 sbatch internal/uwiki/setup_env.sh" >&2; exit 1; }
import sys, torch
ok = torch.cuda.is_available()
print(f"  torch:  {torch.__version__} (cuda {torch.version.cuda}) available={ok}")
sys.exit(0 if ok else 1)
TORCHCHK
fi

export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# HF_TOKEN / WANDB_API_KEY. See the header of credentials.sh for the three ways
# to supply a token; nothing is stored in the repo.
# shellcheck disable=SC1091
source "${PE_REPO}/internal/uwiki/credentials.sh"

# shellcheck disable=SC1091
source "${PE_REPO}/internal/uwiki/unlearn_cell_body.sh"
