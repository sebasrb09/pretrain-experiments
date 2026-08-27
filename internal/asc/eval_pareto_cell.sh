#!/bin/bash
#SBATCH --account=p201378
#SBATCH --job-name=pareto-eval
#SBATCH --gres=gpu:1
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# MUSICA wrapper for evaluating ONE point of the 1B Pareto plot.
#
# PARTIAL, SHARED GPU allocation -- one of four H100s, a quarter node. As with
# the training cell, there is deliberately no --nodes/--ntasks/--cpus-per-task:
# supplying layout flags on top of a 'comfort mode' fails with 'Requested node
# configuration is not available'. See internal/asc/unlearn_cell_1B.sh for the
# full comfort-mode table.
#
# Partition and QOS must be IDENTICAL and name the hardware. QOS
# zen4_0768_h100x4 allows up to 72 h; 4 h is ample for one eval sweep.
#
# This file holds ONLY the site setup. What actually runs -- target resolution,
# the eval suite, the .done markers -- lives in internal/uwiki/eval_cell_body.sh,
# shared with the other site wrappers so they cannot drift.
#
# Usage -- identical interface to every other site wrapper:
#
#   sbatch --export=ALL,CELL_DIR=$DATA/unlearning-pareto/1B-pareto/ce-u/lr-4e-6 \
#     internal/asc/eval_pareto_cell.sh
#
# The whole sweep:
#
#   CELL_SCRIPT=internal/asc/eval_pareto_cell.sh \
#     bash internal/uwiki/launch_pareto_evals.sh
#
# Eval-level env vars (CELL_DIR, MODEL, REVISION, EVAL_OUT, SKIP_*, NOISE_DIR,
# FORCE_EVAL, ...) are documented in internal/uwiki/eval_cell_body.sh.
# Site-level vars (PE_PROJECT, PE_DATA, PE_REPO, PE_VENV, PE_PYTHON_MOD,
# HF_HOME, OUTPUT_ROOT, OLMO_CONFIG) are documented in internal/asc/env.sh.

set -u
set -o pipefail
exec </dev/null

# SLURM copies the batch script into a spool directory before executing it, so
# ${BASH_SOURCE[0]} here is /var/spool/slurmd/job<N>/slurm_script -- NOT the
# path you submitted. Resolve the repository explicitly instead.
find_repo () {
  local c
  for c in "${PE_REPO:-}" \
           "${SLURM_SUBMIT_DIR:-}" \
           "${PE_SCRATCH:-${SCRATCH:-/scratch/fs201378/sr44833}}/pretrain-experiments" \
           "$PWD"; do
    if [ -n "$c" ] && [ -f "$c/internal/asc/env.sh" ]; then
      echo "$c"; return 0
    fi
  done
  return 1
}

PE_REPO="$(find_repo)" || {
  echo "ERROR: could not locate the pretrain-experiments checkout." >&2
  echo "       Tried: \$PE_REPO, \$SLURM_SUBMIT_DIR, the site default, \$PWD." >&2
  echo "       Submit from the repo root, or export PE_REPO=/path/to/repo." >&2
  exit 1
}
export PE_REPO

# shellcheck disable=SC1091
source "${PE_REPO}/internal/asc/env.sh"

# shellcheck disable=SC1091
source "${PE_REPO}/internal/uwiki/eval_cell_body.sh"
