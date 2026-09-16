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
#   PE_PROJECT / PE_PROJECT_DIR / PE_SCRATCH / PE_REPO   see internal/lumi/env.sh
#   PE_LUMI_STACK   LUMI stack version        (default: 24.03)
#   PE_TORCH_MOD    PyTorch container module  (default: 2.6.0-rocm-6.2.4)
#   INSTALL_OLMO    1 to clone+install the OLMo fork (default: 0)
#   MEASURE_MODEL   HF repo to size-test      (default: the 2.7B baseline)

set -u
set -o pipefail

PE_PROJECT="${PE_PROJECT:-project_465003383}"
PE_PROJECT_DIR="${PE_PROJECT_DIR:-/project/${PE_PROJECT}/${USER}}"
PE_SCRATCH="${PE_SCRATCH:-/scratch/${PE_PROJECT}/${USER}}"
PE_REPO="${PE_REPO:-${PE_PROJECT_DIR}/pretrain-experiments}"
PE_LUMI_STACK="${PE_LUMI_STACK:-24.03}"
PE_TORCH_MOD="${PE_TORCH_MOD:-PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250410}"
INSTALL_OLMO="${INSTALL_OLMO:-0}"
MEASURE_MODEL="${MEASURE_MODEL:-sbordt/OLMo-2-2.7B-Exp-Unlearning}"

export EBU_USER_PREFIX="${EBU_USER_PREFIX:-${PE_PROJECT_DIR}/EasyBuild}"

echo "============================================"
echo "  LUMI environment setup"
echo "  project:     $PE_PROJECT"
echo "  project dir: $PE_PROJECT_DIR"
echo "  scratch:     $PE_SCRATCH"
echo "  EasyBuild:   $EBU_USER_PREFIX"
echo "  torch:       $PE_TORCH_MOD"
echo "============================================"

mkdir -p "$PE_PROJECT_DIR" "$PE_SCRATCH" "$EBU_USER_PREFIX"

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
module load "LUMI/${PE_LUMI_STACK}" || { echo "ERROR: no LUMI/${PE_LUMI_STACK}" >&2; exit 1; }

if module load "$PE_TORCH_MOD" 2>/dev/null; then
  echo "  already installed: $PE_TORCH_MOD"
  module unload "$PE_TORCH_MOD" 2>/dev/null || true
else
  echo "  not installed; building with EasyBuild"
  module load partition/container || { echo "ERROR: no partition/container" >&2; exit 1; }
  module load EasyBuild-user      || { echo "ERROR: no EasyBuild-user" >&2; exit 1; }
  EB_RECIPE="${PE_TORCH_MOD//\//-}.eb"
  echo "  eb $EB_RECIPE -r"
  eb "$EB_RECIPE" -r || {
    echo "ERROR: EasyBuild could not install $EB_RECIPE." >&2
    echo "       List what is available with:  module spider PyTorch" >&2
    echo "       then export PE_TORCH_MOD to a listed build and re-submit." >&2
    exit 1; }
  module unload EasyBuild-user partition/container 2>/dev/null || true
fi

# --- 3. our package, into the container's venv -------------------------------
echo ""
echo "--- installing pretrain-experiments into the container ---"
module load partition/G
module load "$PE_TORCH_MOD"
echo "  CONTAINERROOT: ${CONTAINERROOT:-<unset>}"
echo "  python:        $(command -v python)"

# On PyTorch >= 2.6 containers, pip writes into
# $CONTAINERROOT/user-software/venv and persists across jobs. That is ONE
# squashfs file from Lustre's point of view, which is the entire reason we are
# not using a virtualenv here.
python -m pip install --upgrade pip
python -m pip install -e "${PE_REPO}[eval]" || {
  echo "ERROR: pip install failed." >&2; exit 1; }
python -m pip install datasets h5py

if [ "$INSTALL_OLMO" = "1" ]; then
  echo ""
  echo "--- installing the OLMo fork (needed only for retain-set methods) ---"
  OLMO_DIR="${PE_PROJECT_DIR}/OLMo"
  if [ ! -d "$OLMO_DIR" ]; then
    git clone https://github.com/sbordt/OLMo "$OLMO_DIR"
  fi
  git -C "$OLMO_DIR" checkout pretrain-experiments
  python -m pip install -e "${OLMO_DIR}[all]"
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

echo ""
echo "--- storage quotas (after) ---"
lumi-allocations 2>/dev/null || true
echo ""
echo "============================================"
echo "  done. Next:"
echo "    source internal/lumi/env.sh     # from a job, not a login node"
echo "============================================"
