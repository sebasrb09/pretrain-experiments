# shellcheck shell=bash
#
# MeluXina (LuxProvide) site environment. Sourced by the job scripts in this
# directory before internal/uwiki/unlearn_cell_body.sh.
#
# Everything site-specific lives here: module stack, virtualenv, storage roots.
# Nothing about the experiments themselves.
#
# One-time setup, done INSIDE a job -- login nodes have no user software
# environment, so `module load Python` and `pip install` both fail there:
#
#   salloc -A $PE_PROJECT -p gpu -q dev -N 1 -t 2:00:00
#   module load env/release/2025.1
#   module avail Python                 # note the version marked (D)
#   module load Python
#   python3 -m venv  /project/home/$PE_PROJECT/$USER/venvs/pe
#   source           /project/home/$PE_PROJECT/$USER/venvs/pe/bin/activate
#   python -m pip install --upgrade pip
#   python -m pip install -e /project/home/$PE_PROJECT/$USER/pretrain-experiments
#
#   # Only needed for the retain-set methods (grad-diff, npo, simnpo, rmu,
#   # satimp) -- `olmo` is imported solely by build_olmo_retain_dataset.
#   # gradient-ascent, ce-u and wga run without it.
#   git clone https://github.com/sbordt/OLMo && cd OLMo
#   git checkout pretrain-experiments && pip install -e .[all] && pip install h5py
#
# Overridable env vars (all have MeluXina-shaped defaults):
#   PE_PROJECT     project code, e.g. p200xxx           (default: p200xxx)
#   PE_PROJECT_DIR /project/home/$PE_PROJECT/$USER
#   PE_REPO        checkout of pretrain-experiments
#   PE_VENV        virtualenv to activate
#   PE_MUSE        MUSE software stack                  (default: env/release/2025.1)
#   PE_PYTHON      Python module name                   (default: Python)
#   HF_HOME        HuggingFace cache  -- kept off $HOME, which has a small quota
#   OUTPUT_ROOT    sweep output root
#   OLMO_CONFIG    OLMo TrainConfig YAML for the retain stream

export PE_SITE="meluxina"

# NOTE: placeholder project code. Replace p200xxx, or export PE_PROJECT.
PE_PROJECT="${PE_PROJECT:-p200xxx}"
PE_PROJECT_DIR="${PE_PROJECT_DIR:-/project/home/${PE_PROJECT}/${USER}}"
PE_REPO="${PE_REPO:-${PE_PROJECT_DIR}/pretrain-experiments}"
PE_VENV="${PE_VENV:-${PE_PROJECT_DIR}/venvs/pe}"
PE_MUSE="${PE_MUSE:-env/release/2025.1}"
PE_PYTHON="${PE_PYTHON:-Python}"

echo "--- MeluXina environment ---"
echo "  project:  $PE_PROJECT"
echo "  repo:     $PE_REPO"
echo "  venv:     $PE_VENV"
echo "  stack:    $PE_MUSE"

module load "$PE_MUSE"
module load "$PE_PYTHON"

if [ ! -f "${PE_VENV}/bin/activate" ]; then
  echo "ERROR: no virtualenv at ${PE_VENV}. See the setup block at the top of" >&2
  echo "       internal/meluxina/env.sh -- it must be created inside a job," >&2
  echo "       not on a login node." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${PE_VENV}/bin/activate"

if [ ! -d "$PE_REPO" ]; then
  echo "ERROR: no repository at ${PE_REPO}; set PE_REPO." >&2
  exit 1
fi

# Keep model weights, datasets and outputs on /project, not $HOME.
export HF_HOME="${HF_HOME:-${PE_PROJECT_DIR}/hf}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PE_PROJECT_DIR}/unlearning-pareto}"
export OLMO_CONFIG="${OLMO_CONFIG:-${PE_PROJECT_DIR}/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml}"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT"

# If MeluXina compute nodes turn out to be firewalled, pre-populate $HF_HOME
# from a networked machine and uncomment these -- otherwise from_pretrained()
# and load_dataset() hang rather than failing fast:
# export HF_HUB_OFFLINE=1
# export WANDB_MODE=offline

export PYTHONPATH="${PE_REPO}${PYTHONPATH:+:$PYTHONPATH}"
cd "$PE_REPO"

# HF_TOKEN / WANDB_API_KEY. See the header of credentials.sh for the three ways
# to supply a token; nothing is stored in the repo.
# shellcheck disable=SC1091
source "${PE_REPO}/internal/uwiki/credentials.sh"

echo "  python:   $(command -v python)"
echo "  HF_HOME:  $HF_HOME"
echo "  output:   $OUTPUT_ROOT"
echo "----------------------------"
