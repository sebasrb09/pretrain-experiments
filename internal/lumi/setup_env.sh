#!/bin/bash
#SBATCH --account=project_465003383
#SBATCH --job-name=pe-setup-lumi
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# One-shot environment setup on LUMI. Submit it, walk away, read the log.
# Everything is idempotent, so this doubles as the "repair my environment"
# script.
#
#   sbatch internal/lumi/setup_env.sh
#   tail -f pe-setup-lumi_*.out
#
# PREREQUISITE: the pretrain-experiments tree must already be on LUMI at
# $PE_REPO. It is not a git repository, so there is nothing to clone -- rsync
# it from your workstation first:
#
#   rsync -avz --exclude .venv --exclude '__pycache__' \
#     ~/Documents/UnlearningBenchmark/pretrain-experiments/ \
#     $USER@lumi.csc.fi:/scratch/project_465003383/unlearning_baselines/pretrain-experiments/
#
# Note the destination: a SHARED directory on scratch, not a per-user path and
# not /project. Only the EasyBuild container lives on /project.
#
# (The OLMo fork IS a git repo and this script can clone it: INSTALL_OLMO=1.)
#
# ---------------------------------------------------------------------------
# WHY small-g AND ONE GCD
#
# The work here is pip and EasyBuild -- CPU-bound. But the last step verifies
# that ROCm actually sees a GPU and that a 2.7B model fits, and that needs a
# real GCD. small-g bills only the resources requested, so one GCD of one node
# for two hours is cheap. A LUMI-G node is 8 GCDs; --gpus-per-node=1 takes an
# eighth of it, and 7 cores is 56/8, the per-GCD share of the usable cores.
#
# Do NOT use standard-g: that partition allocates and bills whole nodes.
# ---------------------------------------------------------------------------
#
# What it does, in order:
#   1. load LUMI + partition/container + EasyBuild-user
#   2. install the PyTorch ROCm container into the project  (skipped if present)
#   3. pip-install pretrain-experiments and its extras INTO the container
#   4. optionally clone + install the sbordt/OLMo fork      (INSTALL_OLMO=1)
#   5. verify imports, ROCm visibility, and bf16
#   6. MEASURE the per-GCD memory ceiling and suggest MICRO_BATCH
#
# Optional env vars:
#   PE_PROJECT / PE_WORK / PE_REPO / PE_PROJECT_DIR      see internal/lumi/env.sh
#   PE_LUMI_STACK   stack module              (default: LUMI, unversioned)
#   PE_TORCH_MOD    PyTorch container module  (default: 2.6.0-rocm-6.2.4)
#   INSTALL_OLMO    1 to clone+install the OLMo fork (default: 0)
#   MEASURE_MODEL   HF repo to size-test      (default: the 2.7B baseline)
#   MAKE_SQUASHFS   1 to collapse the venv into one file at the end (default: 0)

set -u
set -o pipefail

PE_PROJECT="${PE_PROJECT:-project_465003383}"
# Shared workspace on scratch -- code, caches and outputs. Not per-user.
PE_WORK="${PE_WORK:-/scratch/${PE_PROJECT}/unlearning_baselines}"
PE_REPO="${PE_REPO:-${PE_WORK}/pretrain-experiments}"
PE_SCRATCH="${PE_SCRATCH:-${PE_WORK}}"
# Per-user, and holds the EasyBuild container only.
PE_PROJECT_DIR="${PE_PROJECT_DIR:-/project/${PE_PROJECT}/${USER}}"
PE_LUMI_STACK="${PE_LUMI_STACK:-LUMI}"
PE_TORCH_MOD="${PE_TORCH_MOD:-PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250410}"
INSTALL_OLMO="${INSTALL_OLMO:-0}"
MEASURE_MODEL="${MEASURE_MODEL:-sbordt/OLMo-2-2.7B-Exp-Unlearning}"
MAKE_SQUASHFS="${MAKE_SQUASHFS:-0}"

export EBU_USER_PREFIX="${EBU_USER_PREFIX:-${PE_PROJECT_DIR}/EasyBuild}"

echo "============================================"
echo "  LUMI environment setup"
echo "  project:     $PE_PROJECT"
echo "  work:        $PE_WORK   (shared: code, caches, outputs)"
echo "  repo:        $PE_REPO"
echo "  EasyBuild:   $EBU_USER_PREFIX   (container only)"
echo "  torch:       $PE_TORCH_MOD"
echo "============================================"

mkdir -p "$PE_WORK" "$PE_PROJECT_DIR" "$EBU_USER_PREFIX"

# --- 1. quota sanity ---------------------------------------------------------
# The single most likely way this setup fails is the 100k file quota on
# /project. Report it before doing anything that consumes it.
echo ""
echo "--- storage quotas (before) ---"
lumi-allocations 2>/dev/null || echo "  (lumi-allocations not available)"
lfs quota -h -p "$(stat -c %g "$PE_PROJECT_DIR" 2>/dev/null)" /project 2>/dev/null || true

# --- 2. the container --------------------------------------------------------
echo ""
echo "--- installing the PyTorch container into the project ---"
#
# PyTorch is NOT preinstalled system-wide on LUMI. Every project installs its
# own copy into EBU_USER_PREFIX, which is why `module spider PyTorch` returns
# nothing on a fresh project -- that is the expected state before this runs,
# not a broken login node.
#
# The stack module is loaded UNVERSIONED: once installed, the container module
# registers in all LUMI stack versions and in CrayEnv.
module load "$PE_LUMI_STACK" || {
  echo "ERROR: could not load the '$PE_LUMI_STACK' stack." >&2; exit 1; }

if module load "$PE_TORCH_MOD" 2>/dev/null; then
  echo "  already installed: $PE_TORCH_MOD"
  module unload "$PE_TORCH_MOD" 2>/dev/null || true
else
  echo "  not installed; building with EasyBuild"
  # partition/container is required for INSTALLING container modules. It is
  # NOT needed to use one afterwards -- env.sh deliberately loads neither it
  # nor partition/G.
  module load partition/container || { echo "ERROR: no partition/container" >&2; exit 1; }
  module load EasyBuild-user      || { echo "ERROR: no EasyBuild-user" >&2; exit 1; }
  EB_RECIPE="${PE_TORCH_MOD//\//-}.eb"
  # No -r: these container easyconfigs resolve nothing, and the documented
  # invocation is a bare `eb <file>`.
  echo "  eb $EB_RECIPE"
  eb "$EB_RECIPE" || {
    echo "ERROR: EasyBuild could not install $EB_RECIPE." >&2
    echo "       List the installable recipes with:  eb --search PyTorch" >&2
    echo "       then export PE_TORCH_MOD to a listed build and re-submit." >&2
    echo "       If the build you want is ARCHIVED, fetch it first from" >&2
    echo "       github.com/Lumi-supercomputer/LUMI-EasyBuild-containers and" >&2
    echo "       install the local copy:  eb --copy-ec <name>.eb ." >&2
    exit 1; }
  module unload EasyBuild-user partition/container 2>/dev/null || true
fi

# --- 3. our package, into the container's venv -------------------------------
echo ""
echo "--- installing pretrain-experiments into the container ---"
# No partition module here: using the container needs only the stack.
module load "$PE_TORCH_MOD"
echo "  CONTAINERROOT: ${CONTAINERROOT:-<unset>}"
echo "  SIF:           ${SIF:-<unset>}"
echo "  python:        $(command -v python)"

# On PyTorch >= 2.6 containers the module puts wrappers on PATH, so `python`
# and `pip` below already run INSIDE the container -- no `singularity exec`
# needed. pip writes to $CONTAINERROOT/user-software/venv/pytorch and persists
# across jobs. (On 2.3.1-2.5.1 you would have to `singularity shell $SIF`
# first; that is the main reason this script pins >= 2.6.)
python -m pip install --upgrade pip

# --- preflight: is the repo actually there, and visible from inside? ---------
# pip's "not a valid editable requirement" error means the path does not
# exist, which has two very different causes on LUMI. Distinguish them before
# pip gets a chance to report the misleading version.
#
# `python` here is the container wrapper, so it sees the CONTAINER's view of
# the filesystem; `[ -d ]` in this shell sees the HOST's. Comparing the two
# separates "never transferred" from "not bind-mounted".
if [ ! -f "${PE_REPO}/pyproject.toml" ]; then
  echo "ERROR: no pretrain-experiments checkout at ${PE_REPO}" >&2
  echo "       (looked for pyproject.toml; the directory is missing or empty)" >&2
  echo "" >&2
  echo "       This repo is not under version control, so there is nothing to" >&2
  echo "       git clone -- copy it from your workstation:" >&2
  echo "" >&2
  echo "         rsync -avz --exclude .venv --exclude '__pycache__' \\" >&2
  echo "           ~/Documents/UnlearningBenchmark/pretrain-experiments/ \\" >&2
  echo "           ${USER}@lumi.csc.fi:${PE_REPO}/" >&2
  echo "" >&2
  echo "       Then re-submit this script." >&2
  exit 1
fi
if ! python -c 'import os,sys; sys.exit(0 if os.path.isfile(sys.argv[1]) else 1)' \
     "${PE_REPO}/pyproject.toml"; then
  echo "ERROR: ${PE_REPO} exists on the host but is NOT visible inside the" >&2
  echo "       container, so the bind mounts do not cover it." >&2
  echo "       Current SINGULARITY_BIND: ${SINGULARITY_BIND:-<unset>}" >&2
  echo "       Add the project root:" >&2
  echo "         export SINGULARITY_BIND=\"\$SINGULARITY_BIND,/project/${PE_PROJECT}\"" >&2
  exit 1
fi

# Install from INSIDE the directory. `pip install -e /abs/path[extras]` is
# fragile -- pip parses the trailing [extras] off the string and then tests the
# remainder as a path -- whereas `-e ".[extras]"` from the project root is the
# documented, unambiguous form.
cd "$PE_REPO" || exit 1
python -m pip install -e ".[eval]" || {
  echo "ERROR: pip install of pretrain-experiments failed." >&2; exit 1; }
python -m pip install datasets h5py || {
  echo "ERROR: pip install of datasets/h5py failed." >&2; exit 1; }

if [ "$INSTALL_OLMO" = "1" ]; then
  echo ""
  echo "--- installing the OLMo fork (needed only for retain-set methods) ---"
  # On scratch beside the repo: a git checkout is many thousands of files and
  # would eat /project's 100k-file quota.
  OLMO_DIR="${PE_WORK}/OLMo"
  if [ ! -d "$OLMO_DIR" ]; then
    git clone https://github.com/sbordt/OLMo "$OLMO_DIR"
  fi
  git -C "$OLMO_DIR" checkout pretrain-experiments
  # Same `-e ".[extras]"` form as above, for the same reason.
  ( cd "$OLMO_DIR" && python -m pip install -e ".[all]" ) || {
    echo "ERROR: pip install of the OLMo fork failed." >&2; exit 1; }
fi

# --- 4. verify ---------------------------------------------------------------
echo ""
echo "--- verification ---"
python - <<'EOF'
import torch, platform
print("  python           ", platform.python_version())
print("  torch            ", torch.__version__)
print("  hip/rocm         ", getattr(torch.version, "hip", None))
print("  cuda-api avail   ", torch.cuda.is_available())
print("  device count     ", torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("  device 0         ", p.name)
    print("  memory (GiB)     ", round(p.total_memory / 1024**3, 1))
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    print("  bf16 matmul      ", tuple((x @ x).shape))
import transformers, datasets
print("  transformers     ", transformers.__version__)
print("  datasets         ", datasets.__version__)
import pretrain_experiments
print("  pretrain_experiments OK")
EOF

# --- 5. measure the memory ceiling ------------------------------------------
# Section 00's model was measured on a 94 GB H100. A GCD has 64 GB, so the
# 1B micro-batch of 4 (78.8 GB) does not fit here and 2.7B is ~1.8x the
# parameters again. Measure rather than extrapolate.
echo ""
echo "--- per-GCD memory, ${MEASURE_MODEL} ---"
MEASURE_MODEL="$MEASURE_MODEL" python - <<'EOF'
import os, torch
from transformers import AutoModelForCausalLM
repo = os.environ["MEASURE_MODEL"]
try:
    m = AutoModelForCausalLM.from_pretrained(
        repo, revision="stage1-step100000-tokens210B", torch_dtype=torch.float32)
except Exception as exc:
    print("  could not load:", exc)
    raise SystemExit(0)
n = sum(p.numel() for p in m.parameters())
print(f"  parameters       {n/1e9:.2f} B")
print(f"  fp32 weights     {n*4/1024**3:.1f} GiB")
print(f"  + Adam moments   {n*4*2/1024**3:.1f} GiB")
print(f"  fixed total      {n*4*3/1024**3:.1f} GiB  (weights + optimizer)")
total = torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else 0
print(f"  GCD memory       {total:.1f} GiB")
head = total - n*4*3/1024**3
print(f"  headroom         {head:.1f} GiB for activations")
print("  -> if headroom is negative, this model needs >1 GCD or ZeRO/FSDP;")
print("     see the note at the end of internal/lumi/env.sh.")
EOF

# --- 6. optionally collapse the venv into a single file ----------------------
# The venv we just created is itself tens of thousands of small files, which is
# the exact Lustre problem containers exist to avoid. LUMI's fix is to pack it
# into a SquashFS image the container mounts as one file.
#
# OFF by default, because afterwards adding a package needs `unmake-squashfs`,
# a pip install, and `make-squashfs` again -- inconvenient while the
# environment is still settling. Turn it on once the dependency set is stable:
#
#   sbatch --export=ALL,MAKE_SQUASHFS=1 internal/lumi/setup_env.sh
if [ "$MAKE_SQUASHFS" = "1" ]; then
  echo ""
  echo "--- packing user-software into SquashFS ---"
  if command -v make-squashfs >/dev/null 2>&1; then
    make-squashfs && rm -rf "${CONTAINERROOT}/user-software" \
      && echo "  packed; use 'unmake-squashfs' before installing anything else"
  else
    echo "  make-squashfs not on PATH; skipping"
  fi
fi

echo ""
echo "--- storage quotas (after) ---"
lumi-allocations 2>/dev/null || true
echo ""
echo "============================================"
echo "  done."
echo ""
echo "  WARNING: everything pip-installed above lives INSIDE the module's"
echo "  installation directory. A complete EasyBuild re-install of"
echo "  $PE_TORCH_MOD ERASES IT."
echo "  Re-run this script after any such re-install."
echo ""
echo "  Next:  sbatch --export=ALL,METHOD=ce-u,VALUE=1 \\"
echo "           internal/lumi/unlearn_cell.sh"
echo "============================================"
