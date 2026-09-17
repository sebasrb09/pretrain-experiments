#!/bin/bash -l
#SBATCH --account=project_465003383
#SBATCH --job-name=pareto-eval
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# LUMI wrapper for evaluating ONE point of the Pareto plot.
#
# This file holds ONLY the site setup. What actually runs -- target resolution,
# the eval suite, the .done markers -- lives in internal/uwiki/eval_cell_body.sh,
# shared with the asc/uwiki wrappers so the sites cannot drift.
#
# `#!/bin/bash -l` is REQUIRED: LMod is initialised from the login profile, so
# without it every `module load` in env.sh fails.
#
# ---------------------------------------------------------------------------
# WHY THESE DIRECTIVES
#
# Evaluation is inference only -- no optimizer state, no gradients -- so it is
# far lighter than a training cell. One GCD is ample: 2.7B in fp32 is 10.8 GB
# against the GCD's 64 GB.
#
#   --partition=small-g   sub-node, billed per GCD-hour
#   --gpus-per-node=1     ONE GCD (a node is 4x MI250X = 8 GCDs, and Slurm
#                         counts GCDs, so 8 would take the whole node)
#   --cpus-per-task=7     56 usable cores / 8 GCDs -- a LUMI-G node has 64 but
#                         one core per CCD is reserved for the OS
#
# Do NOT copy partition names from the other sites: `p_datamining` is u:wiki's
# and `zen4_0768_h100x4` is MUSICA's. Submitting either here fails with
# "invalid partition specified", which is exactly what this file exists to
# prevent.
#
# --time is 12 h, matching launch_pareto_evals.sh's own default. Insertion
# likelihood is the long pole (~70 min per cell at 1M tokens x 57 experiments
# at 1B, and 2.7B is slower); raise it with TIME=24:00:00 if a cell runs out.
# ---------------------------------------------------------------------------
#
# Two ways to call it:
#
#   1. a trained cell -- the checkpoint is located inside it:
#        sbatch --export=ALL,CELL_DIR=$PE_WORK/unlearning-pareto-2.7B/2.7B-p1-satimp-rt/satimp/beta1-5.0 \
#          internal/lumi/eval_pareto_cell.sh
#
#   2. a reference anchor -- an HF repo plus revision and an explicit out dir:
#        sbatch --export=ALL,MODEL=sbordt/OLMo-2-2.7B-Unlearning,REVISION=stage1-step100000-tokens210B,EVAL_OUT=$PE_WORK/unlearning-pareto-2.7B/anchors/deep-ignorance/step-0 \
#          internal/lumi/eval_pareto_cell.sh
#
# The whole sweep:
#
#   CELL_SCRIPT=internal/lumi/eval_pareto_cell.sh \
#     bash internal/uwiki/launch_pareto_evals.sh
#
# NOISE_DIR IS NOT OPTIONAL AT 2.7B. eval_cell_body.sh defaults it to the 1B
# vectors (.../noise-vectors/OLMo-2-1B-Exp), which are embed_dim 2048 against
# 2.7B's 2880 -- and when the vectors are missing it SKIPS the watermark eval
# rather than failing, so the gap only surfaces at aggregation. Build them with
# internal/lumi/build_noise_dir.sh and pass:
#
#   NOISE_DIR=$PE_WORK/noise-vectors/OLMo-2-2.7B-Exp
#
# Eval-level env vars (CELL_DIR, MODEL, REVISION, EVAL_OUT, SKIP_*, NOISE_DIR,
# NOISE_STD, FORCE_EVAL, ...) are documented in internal/uwiki/eval_cell_body.sh.
# Site-level vars (PE_PROJECT, PE_WORK, PE_REPO, PE_LUMI_STACK, PE_TORCH_MOD,
# HF_HOME, OUTPUT_ROOT, OLMO_CONFIG) are documented in internal/lumi/env.sh.

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
           "${PE_WORK:-/scratch/${PE_PROJECT:-project_465003383}/unlearning_baselines}/pretrain-experiments" \
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
source "${PE_REPO}/internal/uwiki/eval_cell_body.sh"
