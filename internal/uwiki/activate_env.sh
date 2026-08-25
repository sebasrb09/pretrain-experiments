# shellcheck shell=bash
#
# Activate the Python environment on the Vienna cluster.
# Sourced by internal/uwiki/setup_env.sh and internal/uwiki/unlearn_cell_1B.sh.
#
# A venv, not the miniforge module. The site's `module load miniforge` is not
# passive -- line ~123 of its modulefile runs
#     conda create -y -p "${env_dir}" "python==${python_version}"
# so loading it performs a full conda solve, silently, and it failed anyway
# because the module invokes conda through Tcl `exec`, which raises on ANY
# stderr output (conda's "please update" notice is enough). A plain venv has
# neither problem.
#
# Overridable:
#   PE_VENV         venv to activate      (default: $HOME/venvs/pretrain-experiments)
#   PE_PYTHON_MOD   module to load first, only if the venv's base interpreter
#                   comes from a module   (default: none)
#
# Build it with:
#   python3 -m venv $HOME/venvs/pretrain-experiments
#   source $HOME/venvs/pretrain-experiments/bin/activate
#   pip install --index-url https://download.pytorch.org/whl/cu128 torch
#   pip install -e . -e ".[eval]" datasets
#
# The torch index matters: pip's default wheel is now a CUDA 13 build, and this
# cluster's driver reports CUDA 12.9. A cu13 build cannot initialise on it --
# torch.cuda.is_available() silently returns False and every GPU job wastes its
# allocation. cu128 is the newest that works; cu126 is the safe fallback.

PE_VENV="${PE_VENV:-$HOME/venvs/pretrain-experiments}"

if [ -n "${PE_PYTHON_MOD:-}" ]; then
  if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  module load "$PE_PYTHON_MOD" || {
    echo "ERROR: could not load module '$PE_PYTHON_MOD'" >&2
    exit 1
  }
fi

if [ ! -f "${PE_VENV}/bin/activate" ]; then
  echo "ERROR: no usable venv at ${PE_VENV}" >&2
  if [ -d "$PE_VENV" ]; then
    echo "       The directory exists but has no bin/activate -- a partially" >&2
    echo "       created venv (an interrupted setup job leaves exactly this)." >&2
    echo "       Remove it and rebuild; see the header of this file." >&2
  else
    echo "       Create it: python3 -m venv $PE_VENV   (see this file's header)" >&2
  fi
  exit 1
fi

# shellcheck disable=SC1091
source "${PE_VENV}/bin/activate"

echo "  venv:   $PE_VENV"
echo "  python: $(command -v python)  ($(python --version 2>&1))"

# If Slurm gave this job a GPU, torch must be able to see it. Failing here costs
# seconds; failing later costs the forget-set tokenization plus the allocation.
if [ -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]; then
  python - <<'PYEOF' || exit 1
import sys
try:
    import torch
except ImportError:
    print("ERROR: torch is not installed in this venv.", file=sys.stderr)
    sys.exit(1)
print(f"  torch:  {torch.__version__}  (built for CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    print("ERROR: a GPU was allocated but torch cannot use it.", file=sys.stderr)
    print(f"       torch was built for CUDA {torch.version.cuda}; check the driver", file=sys.stderr)
    print("       with `nvidia-smi`. If the driver is older, reinstall torch:", file=sys.stderr)
    print("         pip install --force-reinstall \\", file=sys.stderr)
    print("           --index-url https://download.pytorch.org/whl/cu128 torch", file=sys.stderr)
    sys.exit(1)
print(f"  cuda:   {torch.cuda.get_device_name(0)}")
PYEOF
fi
