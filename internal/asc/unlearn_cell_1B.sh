#!/bin/bash
#SBATCH --account=p201378
#SBATCH --job-name=unlearn-1B
#SBATCH --gres=gpu:1
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# PARTIAL, SHARED GPU allocation -- one of four H100s, a quarter node.
#
# MUSICA's job_submit lua plugin picks an allocation mode from WHICH flags you
# supply ('comfort modes'):
#
#   -N X                  full CPU node(s), exclusive
#   -n X --mem YG         partial CPU node, shared
#   -N X --gres=gpu:4     full GPU node: 4 tasks, exclusive
#   --gres=gpu:1..3       partial GPU node, shared, mem proportional  <- this job
#
# So there is deliberately no --nodes, --ntasks or --cpus-per-task here: -N
# would switch this to a full-node request, and supplying layout flags on top
# of a comfort mode fails with 'Requested node configuration is not available'.
# Use --gres=gpu:2 for half a node, or -N 1 --gres=gpu:4 for a whole one.
#
# `#ASC --vanilla` bypasses the plugin if you ever need manual layout control.

# MUSICA wrapper for one cell of the 1B unlearning/utility Pareto sweep.
# One job == one dot on the plot.
#
# This file holds ONLY the site setup: SLURM directives and the environment.
# Everything about the experiment -- method dispatch, budget model, argument
# construction -- lives in internal/uwiki/unlearn_cell_body.sh, shared with the
# other site wrappers so they cannot drift.
#
# ---------------------------------------------------------------------------
# The --account directive above is set to p201378. Override at submit time with
# `sbatch -A other` or SBATCH_ACCOUNT; Slurm precedence is CLI > env > directive.
# ---------------------------------------------------------------------------
#
# MUSICA specifics, and how they differ from other sites:
#   * Partition and QOS must be IDENTICAL and name the hardware:
#       zen4_0768_h100x4   4x H100 SXM5 94GB, 2x AMD 9654 (192c/384t), 768 GB
#       zen4_0768          CPU only, same node minus the GPUs
#   * GPUs come from --gres=gpu:N, N in 1..4.
#   * QOS zen4_0768_h100x4 allows up to 72 h. dev_zen4_0768_h100x4 is capped at
#     10 min -- too short for a real cell, but fine for a submission smoke test.
#   * 94 GB per GPU is far more headroom than the sizing assumed (it was written
#     for 40 GB A100s). See the memory note below -- MICRO_BATCH should go up.
#
# Memory: H100 94GB, roughly 2.3x the 40 GB the defaults were sized for. At 1B
# a cell needs about fp32 weights 4 GB + Adam 8 GB + grads 4 GB, plus a frozen
# fp32 reference (+4 GB) for npo/rmu -- call it 20 GB before activations. That
# leaves ~70 GB, so MICRO_BATCH=4 is far too conservative here: it forces 128
# accumulation steps per optimizer step and wastes most of the card.
#
# MICRO_BATCH=32 (accum 16) or 64 (accum 8) should fit comfortably and cut the
# wallclock several-fold. The effective batch stays at TOTAL_BATCH either way,
# so this changes speed, not the experiment. Confirm with one cell before
# committing the grid, and back off if npo/rmu (two resident models) get tight.
#
# MUSICA note: the docs warn that submitting from an ACTIVE virtualenv can leak
# environment variables into the job. Deactivate before `sbatch` -- this script
# activates the venv itself.
#
# Usage -- identical interface to every other site wrapper:
#
#   sbatch --export=ALL,METHOD=simnpo,VALUE=0.5 internal/asc/unlearn_cell_1B.sh
#
# The whole grid:
#
#   CELL_SCRIPT=internal/asc/unlearn_cell_1B.sh \
#     bash internal/uwiki/launch_pareto_sweep_1B.sh
#
# All experiment-level env vars (METHOD, VALUE, TOTAL_BATCH, MICRO_BATCH,
# EPOCHS, MAX_STEPS, HARD_STEP_CAP, MODEL, REVISION, FORGET_EXPS, LR, DTYPE,
# RMU_LAYER, KEEP_CHECKPOINTS, ...) are documented in
# internal/uwiki/unlearn_cell_body.sh. Site-level vars (PE_PROJECT, PE_DATA,
# PE_REPO, PE_VENV, PE_PYTHON_MOD, HF_HOME, OUTPUT_ROOT, OLMO_CONFIG) are
# documented in internal/asc/env.sh.

set -u
set -o pipefail

# SLURM copies the batch script into a spool directory before executing it, so
# ${BASH_SOURCE[0]} here is /var/spool/slurmd/job<N>/slurm_script -- NOT the
# path you submitted. Deriving the repo location from it silently resolves to
# /var/spool and fails with:
#   .../slurm_script: line NN: /var/spool/slurmd/job<N>/env.sh: No such file
# Resolve the repository explicitly instead.
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
source "${PE_REPO}/internal/uwiki/unlearn_cell_body.sh"
