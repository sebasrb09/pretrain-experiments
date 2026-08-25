#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=pe-setup
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=vader,galadriel

# One-shot environment setup on the Vienna cluster, as a batch job.
#
#   sbatch internal/uwiki/setup_env.sh
#   tail -f pe-setup_*.out
#
# Idempotent: re-running updates an existing environment rather than rebuilding
# it, so this doubles as the "repair my env" script.
#
# NO GPU is requested -- pip and measure_forget_set.py are CPU work, and not
# asking for one should schedule sooner. CUDA is verified by the first training
# job instead.
#
# Nodes: p_datamining is a SMALL pool -- vader, galadriel, shelob and
# dgx-h100-em2 are the ones this repo references. Excluding too many leaves
# nothing schedulable ('Requested node configuration is not available'), so
# this keeps only the vader,galadriel exclusion the other uwiki scripts use.
# shelob is suspect (the first setup attempt hung there, and the archive
# scripts flag it for NCCL and user-site torch issues). To avoid it too:
#   sbatch --exclude=vader,galadriel,shelob internal/uwiki/setup_env.sh
# To see what is actually there:
#   sinfo -p p_datamining -o "%20N %8T %5c %10m %20f %14G"
#
# Accounts available to you:
#   datamining   higher priority on the DM group's nodes   (default here)
#   csunivie     the general default account
#   low          low-priority / backfill
#
# Override without editing this file; Slurm precedence is CLI > env > directive:
#   sbatch -A low -p <partition> internal/uwiki/setup_env.sh
#   SBATCH_ACCOUNT=csunivie sbatch internal/uwiki/setup_env.sh
#
# To discover which partition goes with which account:
#   sacctmgr show assoc where user=$USER format=Account,Partition,QOS -p
#   sinfo -o "%20P %10a %10l %6D %10G %10m %f"
#
# Optional env vars:
#   PE_REPO              checkout location           (default: $HOME/pretrain-experiments)
#   PE_ENV_NAME          miniforge env to activate   (default: pretrain-experiments)
#   INSTALL_OLMO         1 to clone+install the OLMo fork (default: 0)
#   OLMO_BRANCH          branch to check out         (default: pretrain-experiments)
#   RUN_MEASURE          1 to run measure_forget_set (default: 0)
#   MEASURE_SAMPLE_FRAC  fraction of rows to tokenize (default: 0.05)

set -u
set -o pipefail

# A batch job has no terminal: if anything prompts, reading EOF makes it fail
# fast instead of hanging until the walltime expires (which is what happened
# on the first attempt -- it sat in `module load` for two hours).
exec </dev/null

PE_REPO="${PE_REPO:-$HOME/pretrain-experiments}"
PE_ENV_NAME="${PE_ENV_NAME:-pretrain-experiments}"
INSTALL_OLMO="${INSTALL_OLMO:-0}"
OLMO_BRANCH="${OLMO_BRANCH:-pretrain-experiments}"
RUN_MEASURE="${RUN_MEASURE:-0}"
MEASURE_SAMPLE_FRAC="${MEASURE_SAMPLE_FRAC:-0.05}"

step () { echo ""; echo "=============================================="; echo "  $*"; echo "=============================================="; }
die  () { echo "ERROR: $*" >&2; exit 1; }

step "0. Context"
echo "  host:    $(hostname)"
echo "  repo:    $PE_REPO"
echo "  env:     $PE_ENV_NAME"
[ -d "$PE_REPO" ] || die "no repository at $PE_REPO -- clone it there or set PE_REPO"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

step "1. Environment"
# Activation lives in internal/uwiki/activate_env.sh so setup and the cell
# wrapper cannot drift. It loads miniforge for conda, then activates the env
# explicitly -- the ENV_MODE/ENV_NAME route hung the first attempt.
# shellcheck disable=SC1091
source "${PE_REPO}/internal/uwiki/activate_env.sh" || die "environment activation failed"

step "2. Install pretrain-experiments"
cd "$PE_REPO"
export PYTHONPATH="$PWD:$HOME/.local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install --upgrade pip || die "pip self-upgrade failed"
python -m pip install -e . || die "pip install -e . failed"
python -m pip install -e ".[eval]" || die "pip install .[eval] failed"
# `datasets` is imported by unlearning_utils.load_forget_set and by every
# train-once-answer-all eval, but is NOT declared in pyproject.toml.
python -m pip install datasets || die "pip install datasets failed"

step "3. OLMo fork"
if [ "$INSTALL_OLMO" != "1" ]; then
  echo "  INSTALL_OLMO=0 -> skipped."
  echo "  Needed only by the retain-set methods (grad-diff, npo, simnpo, rmu,"
  echo "  satimp). gradient-ascent, ce-u and wga import no olmo."
else
  OLMO_DIR="${OLMO_DIR:-$HOME/OLMo}"
  if [ -d "$OLMO_DIR/.git" ]; then
    echo "  reusing clone at $OLMO_DIR"
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
    echo "  retain-stream config: $CFG"
    grep -E "global_train_batch_size|device_train_microbatch_size|^seed" "$CFG" || true
  else
    echo "  WARNING: config not found at $CFG -- set OLMO_CONFIG explicitly"
  fi
fi

step "4. Verify imports"
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
print(f"  cuda available: {torch.cuda.is_available()}  (False expected -- no GPU requested)")
PYEOF

step "5. Retain-stream check"
# Decides whether five of the eight methods can run here at all: the retain
# loader needs the OLMo-2 stage1 memmap DATA, not just the YAML.
python - <<'PYEOF' || true
import os
try:
    import olmo  # noqa: F401
except ImportError:
    print("  olmo not installed -- skipping (re-run with INSTALL_OLMO=1)")
    raise SystemExit(0)
cfg = os.environ.get(
    "OLMO_CONFIG",
    os.path.expanduser("~/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml"))
if not os.path.exists(cfg):
    print(f"  no config at {cfg} -- set OLMO_CONFIG")
    raise SystemExit(0)
try:
    from pretrain_experiments.unlearning_utils import build_olmo_retain_dataset
    ds, info = build_olmo_retain_dataset(cfg, start_step=100000, max_seq_len=1024)
    print(f"  OK: {info['n_unseen_sequences']:,} unseen sequences, "
          f"first shape {tuple(ds[0].shape)}")
    print("  -> grad-diff / npo / simnpo / rmu / satimp can run here.")
except Exception as e:
    print(f"  RETAIN STREAM UNAVAILABLE: {type(e).__name__}: {e}")
    print("  -> those five methods are blocked until the OLMo-2 stage1 memmap")
    print("     data is reachable from this cluster. gradient-ascent, ce-u and")
    print("     wga are unaffected.")
PYEOF

if [ "$RUN_MEASURE" = "1" ]; then
  step "6. Measure |D_f|"
  python internal/uwiki/measure_forget_set.py \
    --sample-frac "$MEASURE_SAMPLE_FRAC" \
    --output-json "$HOME/forget_set_measurement.json" || die "measure_forget_set.py failed"
else
  step "6. Measurement skipped (RUN_MEASURE=1 to run it)"
fi

step "Done"
echo "  repo: $PE_REPO"
echo "  env:  $PE_ENV_NAME"
echo ""
echo "  Next: smoke-test one cell, then read the HARD_STEP_CAP note before"
echo "  launching the method-hyperparameter sweep."
echo ""
echo "    sbatch -J smoke --time=0:30:00 --export=ALL,METHOD=ce-u,VALUE=1e-6,\\"
echo "MAX_STEPS=2,KEEP_CHECKPOINTS=0,RUN_TAG=smoke,\\"
echo "MODEL=sbordt/OLMo-2-179M-Exp-Unlearning,REVISION=stage1-step100000-tokens210B,\\"
echo "FORGET_EXPS=memorization-patterns-rare-1-token-1x \\"
echo "      internal/uwiki/unlearn_cell_1B.sh"
echo ""
echo "    DRY_RUN=1 bash internal/uwiki/launch_pareto_sweep_1B.sh"
