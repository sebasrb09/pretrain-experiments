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
# 2. No virtualenv on the filesystem. A torch venv is ~100k small files and
#    LUMI's docs are explicit that this "puts a lot of strain on the Lustre
#    file system". /project allows 100k files TOTAL, so the MUSICA/MeluXina
#    pattern ($SCRATCH/venvs/pe) would consume the entire project quota.
#    LUMI's answer is a SquashFS container: one file instead of 100k.
#    See internal/lumi/setup_env.sh, which builds it.
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
#   /project/project_<id>       50 GB    100k files   code + container
#   /scratch/project_<id>       50 TB      2M files   checkpoints, HF cache
#   /flash/project_<id>          2 TB      1M files   3x billing -- avoided
#
# The HuggingFace cache is tens of thousands of files, so it goes on scratch
# (2M file quota), never on /project (100k, shared with the container).
#
# Overridable env vars:
#   PE_PROJECT      project id, e.g. project_465003383
#   PE_PROJECT_DIR  /project/$PE_PROJECT/$USER      (code, container)
#   PE_SCRATCH      /scratch/$PE_PROJECT/$USER      (checkpoints, caches)
#   PE_REPO         checkout of pretrain-experiments
#   PE_LUMI_STACK   LUMI software stack             (default: 24.03)
#   PE_TORCH_MOD    PyTorch container module        (see setup_env.sh)
#   EBU_USER_PREFIX where EasyBuild installs        (default: $PE_PROJECT_DIR/EasyBuild)
#   HF_HOME / OUTPUT_ROOT / OLMO_CONFIG   as on every other site

export PE_SITE="lumi"

PE_PROJECT="${PE_PROJECT:-project_465003383}"
PE_PROJECT_DIR="${PE_PROJECT_DIR:-/project/${PE_PROJECT}/${USER}}"
PE_SCRATCH="${PE_SCRATCH:-/scratch/${PE_PROJECT}/${USER}}"
PE_REPO="${PE_REPO:-${PE_PROJECT_DIR}/pretrain-experiments}"

PE_LUMI_STACK="${PE_LUMI_STACK:-24.03}"
# Pin an exact container. >= 2.6.0 matters: from that version `pip install`
# works directly after `module load`, without entering the container first.
PE_TORCH_MOD="${PE_TORCH_MOD:-PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250410}"

# EasyBuild installs the container here. MUST be on /project: a full module
# reinstall erases this directory, and scratch is subject to auto-cleaning.
export EBU_USER_PREFIX="${EBU_USER_PREFIX:-${PE_PROJECT_DIR}/EasyBuild}"

echo "--- LUMI environment ---"
echo "  project:  $PE_PROJECT"
echo "  project dir: $PE_PROJECT_DIR   (code, container)"
echo "  scratch:  $PE_SCRATCH   (checkpoints, caches)"
echo "  repo:     $PE_REPO"
echo "  stack:    LUMI/$PE_LUMI_STACK"
echo "  torch:    $PE_TORCH_MOD"

module load "LUMI/${PE_LUMI_STACK}" || {
  echo "ERROR: could not load LUMI/${PE_LUMI_STACK}." >&2
  echo "       Run 'module avail LUMI' and export PE_LUMI_STACK to a listed version." >&2
  exit 1
}
module load partition/G || {
  echo "ERROR: could not load partition/G (the LUMI-G partition module)." >&2
  exit 1
}
module load "$PE_TORCH_MOD" || {
  echo "ERROR: could not load $PE_TORCH_MOD." >&2
  echo "       It is installed per-project by internal/lumi/setup_env.sh." >&2
  echo "       Run 'module avail PyTorch' -- if nothing is listed, submit" >&2
  echo "       setup_env.sh first. If a different build is listed, export" >&2
  echo "       PE_TORCH_MOD to it." >&2
  exit 1
}

if [ ! -d "$PE_REPO" ]; then
  echo "ERROR: no repository at ${PE_REPO}; clone it there or set PE_REPO." >&2
  exit 1
fi

# Large and/or many-file: scratch, which has the 2M file quota.
export HF_HOME="${HF_HOME:-${PE_SCRATCH}/hf}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PE_SCRATCH}/unlearning-pareto}"
export OLMO_CONFIG="${OLMO_CONFIG:-${PE_PROJECT_DIR}/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml}"
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
