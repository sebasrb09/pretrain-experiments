#!/bin/bash
#SBATCH --account=p201378
#SBATCH --job-name=pe-setup
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --partition=zen4_0768
#SBATCH --qos=zen4_0768
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# NOTE: do NOT add --ntasks or --cpus-per-task here. MUSICA's job_submit lua
# plugin applies the full-CPU-node layout itself (--ntasks-per-node=190,
# --cpus-per-task=1, --threads-per-core=1, exclusive), and supplying your own
# values collides with it:
#   sbatch: error: Batch job submission failed: Requested node configuration
#   is not available
# This matches the full-CPU-node example in the MUSICA docs. Add
# `#ASC --vanilla` if you ever need to bypass the plugin and set layout by hand.
#
# zen4_0768 is allocated exclusively, so this job takes a whole 192-core node
# for what is really just pip. That is the only CPU option MUSICA offers; it
# still avoids spending the GPU allocation, which is the point.

# One-shot environment setup on MUSICA, as a batch job -- so you are not holding
# an interactive session while pip downloads ~2.5 GB of torch.
#
# Submit it, walk away, read the log. Everything is idempotent: re-running skips
# what already exists, so this doubles as the "repair my venv" script.
#
#   export SBATCH_ACCOUNT=p201378      # only if the directive above is wrong
#   sbatch internal/asc/setup_env.sh
#   tail -f pe-setup_*.out
#
# ---------------------------------------------------------------------------
# The --account directive above is already set to p201378. Override at submit
# time with `sbatch -A other` or SBATCH_ACCOUNT -- Slurm precedence is
# CLI > environment > script directive.
# ---------------------------------------------------------------------------
#
# This runs on the CPU partition and requests NO GPU. pip, the venv and
# measure_forget_set.py are all CPU-only work, and zen4_0768 is the CPU
# partition (72 nodes). CUDA is verified
# later, by the first real training job.
#
# MUSICA note: the docs warn that submitting from an ACTIVE virtualenv can leak
# environment variables into the job. Deactivate before `sbatch`; the job
# activates the venv itself.
#
# What it does, in order:
#   1. load the Python module
#   2. create the virtualenv                        (skipped if present)
#   3. install pretrain-experiments, [eval], datasets
#   4. optionally install the sbordt/OLMo fork      (INSTALL_OLMO=1)
#   5. verify imports
#   6. re-source internal/asc/env.sh, validating the exact path jobs will take
#   7. optionally measure |D_f|                     (RUN_MEASURE=1, default)
#
# Optional env vars:
#   PE_PROJECT / PE_DATA / PE_REPO / PE_VENV / PE_PYTHON_MOD   see internal/asc/env.sh
#   INSTALL_OLMO         1 to clone+install the OLMo fork (default: 0)
#   OLMO_BRANCH          branch to check out              (default: pretrain-experiments)
#   RUN_MEASURE          1 to run measure_forget_set      (default: 1)
#   MEASURE_SAMPLE_FRAC  fraction of rows to tokenize     (default: 1.0)
#   RECREATE_VENV        1 to delete and rebuild the venv (default: 0)

set -u
set -o pipefail

PE_PROJECT="${PE_PROJECT:-p201378}"
PE_DATA="${PE_DATA:-${DATA:-/data/fs201378/sr44833}}"
PE_SCRATCH="${PE_SCRATCH:-${SCRATCH:-/scratch/fs201378/sr44833}}"
# Code on scratch (cheap to rebuild), weights and outputs on data (not).
PE_REPO="${PE_REPO:-${PE_SCRATCH}/pretrain-experiments}"
PE_VENV="${PE_VENV:-${PE_SCRATCH}/venvs/pe}"
PE_PYTHON_MOD="${PE_PYTHON_MOD:-Python/3.13.5-GCCcore-14.3.0}"

INSTALL_OLMO="${INSTALL_OLMO:-0}"
OLMO_BRANCH="${OLMO_BRANCH:-pretrain-experiments}"
RUN_MEASURE="${RUN_MEASURE:-1}"
MEASURE_SAMPLE_FRAC="${MEASURE_SAMPLE_FRAC:-1.0}"
RECREATE_VENV="${RECREATE_VENV:-0}"

step () { echo ""; echo "=============================================="; echo "  $*"; echo "=============================================="; }
die  () { echo "ERROR: $*" >&2; exit 1; }

step "0. Context"
echo "  host:     $(hostname)"
echo "  project:  $PE_PROJECT"
echo "  scratch:  $PE_SCRATCH   (code)"
echo "  data:     $PE_DATA   (weights, outputs)"
echo "  repo:     $PE_REPO"
echo "  venv:     $PE_VENV"

[ -d "$PE_DATA" ]    || die "PE_DATA does not exist: $PE_DATA"
[ -d "$PE_SCRATCH" ] || die "PE_SCRATCH does not exist: $PE_SCRATCH"
[ -d "$PE_REPO" ] || die "no repository at $PE_REPO -- clone it there or set PE_REPO"

# Transient build artefacts go to scratch, never $HOME (50 GB, not for data).
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PE_SCRATCH}/.cache/pip}"
export TMPDIR="${TMPDIR:-${PE_SCRATCH}/tmp}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$(dirname "$PE_VENV")"

step "1. Python module"
module purge
module load "$PE_PYTHON_MOD" \
  || die "could not load $PE_PYTHON_MOD -- run 'module avail python' and set PE_PYTHON_MOD"
module list 2>&1 | head -20
echo "  python: $(command -v python3) ($(python3 --version 2>&1))"

step "2. Virtualenv"
if [ "$RECREATE_VENV" = "1" ] && [ -d "$PE_VENV" ]; then
  echo "  RECREATE_VENV=1 -> removing $PE_VENV"
  rm -rf "$PE_VENV"
fi
if [ -f "${PE_VENV}/bin/activate" ]; then
  echo "  reusing existing venv"
else
  echo "  creating $PE_VENV"
  python3 -m venv --upgrade-deps "$PE_VENV" || die "venv creation failed"
fi
# shellcheck disable=SC1091
source "${PE_VENV}/bin/activate" || die "could not activate $PE_VENV"
python -m pip install --upgrade pip setuptools wheel || die "pip self-upgrade failed"
echo "  active python: $(command -v python)"

step "3. Install pretrain-experiments"
cd "$PE_REPO"
python -m pip install -e . || die "pip install -e . failed (is PyPI reachable from compute nodes?)"
python -m pip install -e ".[eval]" || die "pip install .[eval] failed"
# `datasets` is imported by unlearning_utils.load_forget_set and by every
# train-once-answer-all eval, but is NOT declared in pyproject.toml.
python -m pip install datasets || die "pip install datasets failed"

step "4. OLMo fork"
if [ "$INSTALL_OLMO" != "1" ]; then
  echo "  INSTALL_OLMO=0 -> skipped."
  echo "  Only the retain-set methods need it (grad-diff, npo, simnpo, rmu, satimp);"
  echo "  gradient-ascent, ce-u and wga import no olmo. Re-run with INSTALL_OLMO=1"
  echo "  when you want the other five."
else
  OLMO_DIR="${OLMO_DIR:-${PE_SCRATCH}/OLMo}"
  if [ -d "$OLMO_DIR/.git" ]; then
    echo "  reusing existing clone at $OLMO_DIR"
  else
    git clone https://github.com/sbordt/OLMo "$OLMO_DIR" || die "git clone failed"
  fi
  cd "$OLMO_DIR"
  git checkout "$OLMO_BRANCH" || die "could not check out $OLMO_BRANCH"
  python -m pip install -e ".[all]" || die "OLMo install failed"
  python -m pip install h5py || die "h5py install failed"
  cd "$PE_REPO"
  CFG="$OLMO_DIR/configs/official-0425/OLMo2-1B-stage1.yaml"
  if [ -f "$CFG" ]; then
    echo "  retain-stream config present: $CFG"
    grep -E "global_train_batch_size|device_train_microbatch_size|max_sequence_length|^seed" "$CFG" || true
    echo ""
    echo "  NOTE: build_olmo_retain_dataset also needs the OLMo-2 stage1 memmap"
    echo "  DATA, not just this YAML -- and MUSICA does not host it. That is no"
    echo "  longer a dead end:"
    echo "    gradient-ascent / ce-u / wga   import no olmo at all"
    echo "    npo / simnpo / satimp          run forget-only at RETAIN_WEIGHT=0"
    echo "    grad-diff / rmu                need a materialized slice; build one"
    echo "                                   (~2 GB, fetched by random access):"
    echo "      python internal/uwiki/build_retain_slice.py --dry-run"
    echo ""
    echo "  The dry run prints cfg.data.paths and says whether they are REMOTE"
    echo "  (fetchable from a compute node) or LOCAL, before downloading anything."
  else
    echo "  WARNING: expected config not found at $CFG -- set OLMO_CONFIG explicitly"
  fi
fi

step "5. Verify"
cd "$PE_REPO"
python - <<'PYEOF' || die "import check failed"
import importlib
for m in ["torch", "transformers", "datasets", "numpy", "yaml", "pretrain_experiments"]:
    mod = importlib.import_module(m)
    print(f"  {m:24s} {getattr(mod, '__version__', 'n/a')}")
try:
    import olmo  # noqa: F401
    print(f"  {'olmo':24s} present")
except ImportError:
    print(f"  {'olmo':24s} not installed (fine unless you run the retain-set methods)")
import torch
print(f"  cuda available: {torch.cuda.is_available()}  (False is expected on the CPU partition)")
PYEOF

step "6. Validate the runtime path"
# Source the exact file the training jobs source, so setup fails here rather
# than inside the first real job.
# shellcheck disable=SC1091
source "${PE_REPO}/internal/asc/env.sh" || die "internal/asc/env.sh failed"
echo "  env.sh sourced cleanly."

if [ "$RUN_MEASURE" = "1" ]; then
  step "7. Measure |D_f| (also the network reachability test)"
  echo "  If this hangs rather than erroring, compute nodes are firewalled --"
  echo "  see the offline toggles in internal/asc/env.sh."
  python internal/uwiki/measure_forget_set.py \
    --sample-frac "$MEASURE_SAMPLE_FRAC" \
    --output-json "${PE_DATA}/forget_set_measurement.json" \
    || die "measure_forget_set.py failed"
else
  step "7. Measurement skipped (RUN_MEASURE=0)"
fi

step "Done"
cat <<EOF
  venv:      $PE_VENV
  repo:      $PE_REPO
  HF_HOME:   ${HF_HOME:-unset}
  outputs:   ${OUTPUT_ROOT:-unset}

  Next (deactivate the venv before submitting -- see the VSC clean-environment note):

    # 1. smoke-test the submission path at 179M (minutes, 1 GPU)
    sbatch -J smoke --time=0:30:00 \\
      --export=ALL,METHOD=ce-u,VALUE=1e-6,MAX_STEPS=2,EPOCHS=20,KEEP_CHECKPOINTS=0,RUN_TAG=smoke,MODEL=sbordt/OLMo-2-179M-Exp-Unlearning,REVISION=stage1-step100000-tokens210B \\
      internal/asc/unlearn_cell_1B.sh

    # 2. LR range test for the two methods whose curve knob IS the LR
    CELL_SCRIPT=internal/asc/unlearn_cell_1B.sh \\
      DRY_RUN=1 METHODS="gradient-ascent ce-u" \\
      bash internal/uwiki/launch_lr_range_test.sh

    # 3. read it
    python internal/uwiki/analyze_lr_range_test.py --output-root "${OUTPUT_ROOT:-...}"
EOF
