# shellcheck shell=bash
#
# MUSICA (docs.asc.ac.at) site environment. Sourced by the job scripts in this
# directory before internal/uwiki/unlearn_cell_body.sh.
#
# Everything site-specific lives here: module stack, virtualenv, storage roots.
# Nothing about the experiments themselves -- the dispatch, the budget model and
# the eight methods are shared with every other site.
#
# ---------------------------------------------------------------------------
# Storage split:
#
#   $SCRATCH  code   -- the repo, the venv, the OLMo clone, pip cache, TMPDIR
#   $DATA     data   -- HuggingFace cache, sweep outputs, checkpoints
#
# Scratch is fast but "may be cleared when space is needed" (MUSICA docs), so
# only things that are cheap to rebuild live there: a purge costs you a git
# clone and one setup job. Anything expensive to regenerate -- downloaded
# weights, and above all the sweep checkpoints that cost GPU hours -- stays on
# $DATA, which is permanent. Do not move OUTPUT_ROOT to scratch.
#
# $HOME is deliberately unused: 50 GB, and the docs say it is not for research
# data.
# ---------------------------------------------------------------------------
#
# One-time setup is done by internal/asc/setup_env.sh -- submit that first.
#
# Overridable env vars:
#   PE_PROJECT     project account for `sbatch -A`     (default: p201378)
#   PE_DATA        permanent root                      (default: $DATA)
#   PE_SCRATCH     rebuildable root                    (default: $SCRATCH)
#   PE_REPO        checkout of pretrain-experiments    (default: $PE_SCRATCH/pretrain-experiments)
#   PE_VENV        virtualenv to activate              (default: $PE_SCRATCH/venvs/pe)
#   PE_PYTHON_MOD  Python module                       (default: Python/3.11.5-GCCcore-13.2.0)
#   HF_HOME        HuggingFace cache                   (default: $PE_DATA/hf)
#   OUTPUT_ROOT    sweep output root                   (default: $PE_DATA/unlearning-pareto)
#   OLMO_CONFIG    OLMo TrainConfig YAML for the retain stream

export PE_SITE="musica"

PE_PROJECT="${PE_PROJECT:-p201378}"
PE_DATA="${PE_DATA:-${DATA:-/data/fs201378/sr44833}}"
PE_SCRATCH="${PE_SCRATCH:-${SCRATCH:-/scratch/fs201378/sr44833}}"

PE_REPO="${PE_REPO:-${PE_SCRATCH}/pretrain-experiments}"
PE_VENV="${PE_VENV:-${PE_SCRATCH}/venvs/pe}"
# EasyBuild module (capital P). Check `module avail python` and override
# PE_PYTHON_MOD if the version has moved.
PE_PYTHON_MOD="${PE_PYTHON_MOD:-Python/3.13.5-GCCcore-14.3.0}"

echo "--- MUSICA environment ---"
echo "  project:  $PE_PROJECT"
echo "  scratch:  $PE_SCRATCH   (code)"
echo "  data:     $PE_DATA   (weights, outputs)"
echo "  repo:     $PE_REPO"
echo "  venv:     $PE_VENV"
echo "  python:   $PE_PYTHON_MOD"

module purge
module load "$PE_PYTHON_MOD" || {
  echo "ERROR: could not load $PE_PYTHON_MOD." >&2
  echo "       Run 'module avail python' and export PE_PYTHON_MOD to a listed version." >&2
  exit 1
}

if [ ! -f "${PE_VENV}/bin/activate" ]; then
  echo "ERROR: no virtualenv at ${PE_VENV}." >&2
  echo "       Submit internal/asc/setup_env.sh first. (If scratch was purged," >&2
  echo "       re-running that job rebuilds it.)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${PE_VENV}/bin/activate"

if [ ! -d "$PE_REPO" ]; then
  echo "ERROR: no repository at ${PE_REPO}; clone it there or set PE_REPO." >&2
  exit 1
fi

# Permanent: expensive to regenerate.
export HF_HOME="${HF_HOME:-${PE_DATA}/hf}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PE_DATA}/unlearning-pareto}"
# The OLMo clone is code, so it sits on scratch with the rest.
export OLMO_CONFIG="${OLMO_CONFIG:-${PE_SCRATCH}/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml}"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT"

# If MUSICA compute nodes turn out to be firewalled, pre-populate $HF_HOME from
# a login node and uncomment these -- otherwise from_pretrained() and
# load_dataset() hang rather than failing fast:
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
echo "--------------------------"
