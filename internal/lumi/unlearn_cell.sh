#!/bin/bash -l
#SBATCH --account=project_465003383
#SBATCH --job-name=unlearn-lumi
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# LUMI wrapper for one cell of the unlearning/utility Pareto sweep.
# One job == one dot on the plot.
#
# This file holds ONLY the site setup: SLURM directives and the environment.
# Everything about the experiment -- method dispatch, budget model, argument
# construction -- lives in internal/uwiki/unlearn_cell_body.sh, shared with the
# musica/meluxina wrappers so they cannot drift.
#
# `#!/bin/bash -l` is REQUIRED, as on MeluXina: LMod is initialised from the
# login profile, and without -l every `module load` in env.sh fails.
#
# ---------------------------------------------------------------------------
# WHY THESE DIRECTIVES, AND HOW LUMI DIFFERS FROM MUSICA
#
# MUSICA forbids layout flags (its job_submit plugin infers a 'comfort mode'
# from which flags you supply). LUMI is the opposite: you state the layout
# explicitly, and small-g bills only the fraction you take.
#
#   --partition=small-g      sub-node, billed per GCD-hour, max 3 days
#   --gpus-per-node=1        ONE GCD. A LUMI-G node is 4x MI250X = 8 GCDs and
#                            Slurm counts GCDs, so 8 would be a whole node.
#   --cpus-per-task=7        56 usable cores / 8 GCDs. A LUMI-G node has 64
#                            cores but the first core of each of the 8 CCDs is
#                            reserved for the OS ('low-noise mode'), leaving 56.
#                            Asking for 8 per GCD will not pack 8 cells a node.
#   --mem=60G                the per-GCD share of usable host RAM. Note /tmp is
#                            a RAM disk and counts against this -- the MIOpen
#                            cache env.sh puts there is small, but do not write
#                            checkpoints to /tmp.
#
# For a full node instead: --partition=standard-g --gpus-per-node=8
# --exclusive, which bills all 8 GCDs whether or not you use them.
#
# MEMORY -- READ BEFORE LAUNCHING A GRID:
# A GCD has 64 GB against MUSICA's 94 GB H100. Section 00's measured model
# (22.1 GB fixed + 3.4 MB/token) puts MICRO_BATCH=4 at 78.8 GB for 1B, which
# does NOT fit here, and 2.7B is ~2x the parameters again. The MUSICA values
# (MICRO_BATCH=32 or 64) will OOM on LUMI. Do not transfer them.
#
# Run internal/lumi/setup_env.sh first -- it measures the ceiling on a real GCD
# -- then set MICRO_BATCH from what it reports, and set GRAD_CKPT=1 if 2.7B
# with a frozen fp32 reference (npo/rmu keep two resident models) is tight.
# Accumulation rescales automatically, so MICRO_BATCH changes speed and memory,
# never the experiment: the effective batch stays at TOTAL_BATCH.
# ---------------------------------------------------------------------------
#
# Usage -- identical interface to every other site wrapper:
#
#   sbatch --export=ALL,METHOD=simnpo,VALUE=0.5 internal/lumi/unlearn_cell.sh
#
# --export=ALL is not optional for comma-valued vars (FORGET_EXPS, RMU_LAYERS):
# sbatch splits --export on commas, so a bare --export=METHOD=x,RMU_LAYERS=5,6,7
# silently truncates. ALL, then the assignments, is the safe form.
#
# The whole grid:
#
#   CELL_SCRIPT=internal/lumi/unlearn_cell.sh \
#     bash internal/uwiki/launch_pareto_sweep_1B.sh
#
# All experiment-level env vars (METHOD, VALUE, TOTAL_BATCH, MICRO_BATCH,
# EPOCHS, MAX_STEPS, HARD_STEP_CAP, MODEL, REVISION, FORGET_EXPS, LR,
# LR_SCHEDULE, DTYPE, RMU_LAYER, RETAIN_WEIGHT, KEEP_CHECKPOINTS, ...) are
# documented in internal/uwiki/unlearn_cell_body.sh. Site-level vars
# (PE_PROJECT, PE_PROJECT_DIR, PE_SCRATCH, PE_REPO, PE_LUMI_STACK,
# PE_TORCH_MOD, HF_HOME, OUTPUT_ROOT, OLMO_CONFIG) are documented in
# internal/lumi/env.sh.

set -u
set -o pipefail

# SLURM copies the batch script into a spool directory before running it, so
# ${BASH_SOURCE[0]} here is /var/spool/slurmd/job<N>/slurm_script -- NOT the
# path you submitted. Deriving the repo from it resolves to /var/spool and
# fails with "No such file or directory: .../env.sh". Resolve it explicitly.
find_repo () {
  local c
  for c in "${PE_REPO:-}" \
           "${SLURM_SUBMIT_DIR:-}" \
           "/project/${PE_PROJECT:-project_465003383}/${USER}/pretrain-experiments" \
           "$PWD"; do
    if [ -n "$c" ] && [ -f "$c/internal/lumi/env.sh" ]; then
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
source "${PE_REPO}/internal/lumi/env.sh"

# shellcheck disable=SC1091
source "${PE_REPO}/internal/uwiki/unlearn_cell_body.sh"
