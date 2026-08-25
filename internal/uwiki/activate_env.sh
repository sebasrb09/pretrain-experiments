# shellcheck shell=bash
#
# Activate the miniforge environment on the Vienna cluster.
# Sourced by internal/uwiki/setup_env.sh and internal/uwiki/unlearn_cell_1B.sh.
#
# WHAT THE SITE MODULE ACTUALLY DOES
#
# `module load miniforge` is not passive. Line ~123 of
# /etc/environment-modules/modules/miniforge/latest runs:
#
#     exec ${install_dir}/bin/conda create -y -p "${env_dir}" "python==${python_version}"
#
# so loading it CREATES a conda environment at a path derived from ENV_NAME.
# Two consequences we hit the hard way:
#
#  1. The first load of a new ENV_NAME performs a full conda solve + download.
#     That is not a hang -- it is the create running. It exceeded a two-hour
#     walltime silently, because the module prints nothing while it works.
#
#  2. Tcl `exec` raises if the command writes ANYTHING to stderr. Conda's
#     "Please update conda by running ..." notice goes to stderr, so an
#     otherwise-successful create makes `module load` fail with:
#         Please update conda by running $ conda update -n base -c conda-forge conda
#         while executing "exec ... conda create ..."
#     Silencing that notice is what CONDA_NOTIFY_OUTDATED_CONDA below is for.
#     Belt and braces: also put `notify_outdated_conda: false` in ~/.condarc.
#
# So we use the site's intended ENV_MODE/ENV_NAME route (fighting it means the
# module tries to create a *different* env), but with the stderr notice muted
# and progress printed so a long create is visible rather than looking hung.
#
# Overridable:
#   PE_ENV_NAME   conda env to activate/create   (default: pretrain-experiments)

PE_ENV_NAME="${PE_ENV_NAME:-pretrain-experiments}"

# Any conda notice on stderr makes the module's Tcl `exec` fail -- see above.
export CONDA_NOTIFY_OUTDATED_CONDA=false
export PYTHONUNBUFFERED=1

if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
else
  echo "  WARNING: /etc/profile.d/modules.sh not found" >&2
fi

echo "  module load miniforge (ENV_NAME=$PE_ENV_NAME) ..." >&2
export ENV_MODE="permanent"
export ENV_NAME="$PE_ENV_NAME"

module load miniforge || {
  echo "ERROR: 'module load miniforge' failed." >&2
  echo "" >&2
  echo "  If the message mentions 'Please update conda ... while executing exec" >&2
  echo "  ... conda create', that is the Tcl-stderr problem above. Add to ~/.condarc:" >&2
  echo "      notify_outdated_conda: false" >&2
  echo "  and retry." >&2
  echo "" >&2
  echo "  To see exactly what the module does:" >&2
  echo "      sed -n '100,140p' /etc/environment-modules/modules/miniforge/latest" >&2
  exit 1
}

echo "  python:    $(command -v python 2>/dev/null || echo NONE)" >&2
command -v python >/dev/null || {
  echo "ERROR: no python on PATH after 'module load miniforge'." >&2
  exit 1
}
echo "  conda env: $PE_ENV_NAME"
echo "  python:    $(command -v python)  ($(python --version 2>&1))"
