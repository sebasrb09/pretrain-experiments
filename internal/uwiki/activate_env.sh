# shellcheck shell=bash
#
# Activate the miniforge environment on the Vienna cluster.
# Sourced by internal/uwiki/setup_env.sh and internal/uwiki/unlearn_cell_1B.sh.
#
# WHY THIS EXISTS
#
# The older uwiki scripts activate by exporting ENV_MODE=permanent and
# ENV_NAME=<env> before `module load miniforge`, letting the site module do the
# activation. That hung a batch job for the full two-hour walltime on shelob,
# printing nothing after the step header -- and it hung even though the module
# loads fine and the env exists (both confirmed interactively).
#
# So: load the module only to get `conda` on PATH, then activate explicitly.
# `conda activate` is predictable, fast, and fails with a readable message
# listing the environments it can see.
#
# Overridable:
#   PE_ENV_NAME   conda env to activate   (default: pretrain-experiments)

PE_ENV_NAME="${PE_ENV_NAME:-pretrain-experiments}"

if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
else
  echo "  WARNING: /etc/profile.d/modules.sh not found" >&2
fi

# Deliberately NOT setting ENV_MODE / ENV_NAME -- see above.
module load miniforge || {
  echo "ERROR: 'module load miniforge' failed." >&2
  echo "       Try: module avail 2>&1 | grep -iE 'conda|forge|mamba'" >&2
  exit 1
}

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "${CONDA_BASE:-}" ]; then
  echo "ERROR: conda is not on PATH after 'module load miniforge'." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate "$PE_ENV_NAME" || {
  echo "ERROR: could not activate conda env '$PE_ENV_NAME'." >&2
  echo "       Environments visible here:" >&2
  conda env list >&2
  echo "       Set PE_ENV_NAME, or create it:" >&2
  echo "         conda create -y -n $PE_ENV_NAME python=3.12" >&2
  exit 1
}

echo "  conda env: $PE_ENV_NAME"
echo "  python:    $(command -v python)  ($(python --version 2>&1))"
