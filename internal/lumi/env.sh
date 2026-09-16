# shellcheck shell=bash
#
# LUMI (CSC / EuroHPC) site environment. Sourced by the job scripts in this
# directory before internal/uwiki/unlearn_cell_body.sh.
#
# ---------------------------------------------------------------------------
# WHY THIS SITE IS DIFFERENT FROM MUSICA AND MELUXINA
#
# 1. AMD, not NVIDIA. LUMI-G is MI250X. PyTorch's ROCm build keeps the `cuda`
#    API names, so torch.cuda.is_available(), device="cuda" and
#    autocast(device_type="cuda") all work unchanged -- our drivers need NO
#    source changes. What changes is how PyTorch is obtained.
#
# 2. No virtualenv on the filesystem. LUMI's docs are blunt: it is "strongly
#    discouraged to install Python packages directly to the user home folder,
#    /scratch, /project, etc. using Conda, pip, or similar". A torch env is
#    "tens to hundreds of thousands of relatively small files" and "Lustre
#    simply isn't designed for such use cases". /project allows 100k files
#    TOTAL, so the MUSICA/MeluXina pattern ($SCRATCH/venvs/pe) would consume
#    the entire project quota on its own.
#    LUMI's answer is a Singularity container. Extra packages go in a venv
#    INSIDE it, at $CONTAINERROOT/user-software/venv/pytorch, which persists
#    across jobs. See internal/lumi/setup_env.sh, which builds it.
#
# 3. Sub-node GPU allocation. A LUMI-G node is 4x MI250X = 8 GCDs, and Slurm
#    counts GCDs, so --gpus-per-node=8 is a FULL node. small-g bills only what
#    you take, so a one-GCD unlearning cell costs one eighth of a node.
#    This is the opposite of MUSICA, where layout flags were forbidden.
#
# 4. 64 GB per GCD, against MUSICA's 94 GB H100. Section 00's memory model
#    (22.1 GB fixed + 3.4 MB/token; micro-batch 4 = 78.8 GB at 1B) does NOT
#    fit a GCD even at 1B, let alone 2.7B. MICRO_BATCH must be measured here,
#    not carried over. setup_env.sh ends with that measurement.
# ---------------------------------------------------------------------------
#
# Storage, and why each root is where it is (quotas from LUMI docs):
#
#   /users/$USER                20 GB    100k files   home, NOT used here
#   /project/project_<id>       50 GB    100k files   EasyBuild container only
#   /scratch/project_<id>       50 TB      2M files   code, checkpoints, caches
#   /flash/project_<id>          2 TB      1M files   3x billing -- avoided
#
# This project keeps its working tree in a SHARED directory on scratch,
# /scratch/<project>/unlearning_baselines, not under a per-user path: code,
# HF cache and outputs all sit together there so collaborators share one copy.
#
# Only the container lives on /project, per-user under EBU_USER_PREFIX. That
# split is deliberate: /project's 100k file quota cannot hold a HF cache or a
# git checkout of OLMo, and the container is one file by design.
#
# CAVEAT: LUMI applies automatic cleaning to /scratch (files untouched for a
# long period are removed). That is fine for checkpoints and caches, which are
# regenerable, but it means the code tree on scratch is not a backup -- keep
# the authoritative copy on your workstation.
#
# Overridable env vars:
#   PE_PROJECT      project id, e.g. project_465003383
#   PE_WORK         shared team workspace on scratch (code, caches, outputs)
#   PE_REPO         the pretrain-experiments tree    (default: $PE_WORK/...)
#   PE_PROJECT_DIR  /project/$PE_PROJECT/$USER       (EasyBuild container ONLY)
#   PE_SCRATCH      alias of PE_WORK, kept for parity with the other sites
#   PE_LUMI_STACK   stack module to load            (default: LUMI, unversioned)
#   PE_TORCH_MOD    PyTorch container module        (see setup_env.sh)
#   EBU_USER_PREFIX where EasyBuild installs        (default: $PE_PROJECT_DIR/EasyBuild)
#   HF_HOME / OUTPUT_ROOT / OLMO_CONFIG   as on every other site

export PE_SITE="lumi"

PE_PROJECT="${PE_PROJECT:-project_465003383}"
# Shared workspace on scratch -- NOT per-user, and NOT /project.
PE_WORK="${PE_WORK:-/scratch/${PE_PROJECT}/unlearning_baselines}"
PE_REPO="${PE_REPO:-${PE_WORK}/pretrain-experiments}"
PE_SCRATCH="${PE_SCRATCH:-${PE_WORK}}"
# Per-user, and holds the EasyBuild container only.
PE_PROJECT_DIR="${PE_PROJECT_DIR:-/project/${PE_PROJECT}/${USER}}"

# UNVERSIONED on purpose. Once EasyBuild installs the container module it is
# registered "in all LUMI stack versions and CrayEnv", so there is nothing to
# pin here, and pinning a version that does not exist on the current stack is
# itself a way to make the load fail.
PE_LUMI_STACK="${PE_LUMI_STACK:-LUMI}"
# Pin an exact container. >= 2.6.0 matters twice over: from that version the
# module puts wrappers on PATH so plain `python` runs inside the container, and
# `pip install` works directly after `module load` without entering it first.
PE_TORCH_MOD="${PE_TORCH_MOD:-PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250410}"

# EasyBuild installs the container here. MUST be on /project: a full module
# reinstall erases this directory, and scratch is subject to auto-cleaning.
export EBU_USER_PREFIX="${EBU_USER_PREFIX:-${PE_PROJECT_DIR}/EasyBuild}"

echo "--- LUMI environment ---"
echo "  project:  $PE_PROJECT"
echo "  work:     $PE_WORK   (shared: code, caches, outputs)"
echo "  repo:     $PE_REPO"
echo "  easybuild: $PE_PROJECT_DIR   (container only)"
echo "  stack:    $PE_LUMI_STACK"
echo "  torch:    $PE_TORCH_MOD"

# Using the container needs NO partition module and no EasyBuild module --
# only the stack and the container itself. (Installing it is different; that
# needs partition/container, and only setup_env.sh does it.)
module load "$PE_LUMI_STACK" || {
  echo "ERROR: could not load the '$PE_LUMI_STACK' software stack." >&2
  echo "       Run 'module avail LUMI' and export PE_LUMI_STACK." >&2
  exit 1
}
module load "$PE_TORCH_MOD" || {
  echo "ERROR: could not load $PE_TORCH_MOD." >&2
  echo "" >&2
  echo "       PyTorch is NOT preinstalled on LUMI -- every project installs" >&2
  echo "       its own copy. If 'module spider PyTorch' shows nothing, that is" >&2
  echo "       expected and simply means it has not been installed yet." >&2
  echo "" >&2
  echo "       Fix: sbatch internal/lumi/setup_env.sh   (then re-submit)" >&2
  echo "       If a DIFFERENT build is listed, export PE_TORCH_MOD to it." >&2
  exit 1
}
echo "  container: ${CONTAINERROOT:-<unset>}"

if [ ! -d "$PE_REPO" ]; then
  echo "ERROR: no repository at ${PE_REPO}; clone it there or set PE_REPO." >&2
  exit 1
fi

# Large and/or many-file: scratch, which has the 2M file quota.
export HF_HOME="${HF_HOME:-${PE_WORK}/hf}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PE_WORK}/unlearning-pareto}"
# OLMo is a git checkout of many thousands of files, so it belongs on scratch
# beside the repo, never on /project's 100k-file quota.
export OLMO_CONFIG="${OLMO_CONFIG:-${PE_WORK}/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml}"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT"

# MIOpen compiles kernels at first use and caches them. LUMI's docs put this
# cache in /tmp, which on compute nodes is a RAM disk and counts against the
# job's memory -- fine for a cache of this size, and it avoids every rank
# hammering Lustre with the same writes. Per-node to avoid cross-node races.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-${SLURM_NODEID:-0}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Single-GCD cells never touch the interconnect; these only matter once a job
# spans GCDs or nodes, and are harmless before then.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-3}"

export PYTHONPATH="${PE_REPO}${PYTHONPATH:+:$PYTHONPATH}"
cd "$PE_REPO"

# HF_TOKEN / WANDB_API_KEY -- nothing is stored in the repo.
# shellcheck disable=SC1091
source "${PE_REPO}/internal/uwiki/credentials.sh"

echo "  python:   $(command -v python)"
echo "  HF_HOME:  $HF_HOME"
echo "  output:   $OUTPUT_ROOT"
echo "  miopen:   $MIOPEN_USER_DB_PATH"
echo "------------------------"
