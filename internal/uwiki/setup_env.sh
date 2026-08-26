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
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# Build and verify the environment on the Vienna cluster. One file, does
# everything: venv, torch matched to the driver, the repo, and a real check
# that a GPU job will actually work.
#
#   sbatch internal/uwiki/setup_env.sh
#   tail -f pe-setup_*.out
#
# Or interactively, worth doing the first time so you see it happen:
#   srun -A datamining -p p_datamining --gres=gpu:1 -t 1:00:00 --pty bash
#   cd ~/pretrain-experiments && bash internal/uwiki/setup_env.sh
#
# It requests a GPU on purpose. An earlier version did not, to schedule faster,
# but then the driver cannot be detected and torch.cuda cannot be verified --
# which is exactly how a broken install reached the job queue.
#
# THINGS THAT BIT US, recorded so they do not again:
#
#  * pip's DEFAULT torch wheel is a CUDA 13 build. This cluster's driver
#    reports CUDA 12.9, and a cu13 build cannot initialise on it --
#    torch.cuda.is_available() silently returns False. The wheel index is
#    chosen from nvidia-smi in step 1.
#
#  * `module load miniforge` is NOT passive: its modulefile runs
#    `conda create -y -p <env_dir> python==<ver>`, so loading it performs a
#    full conda solve (this consumed a 2h walltime silently, printing nothing),
#    and it invokes conda through Tcl `exec`, which raises on ANY stderr output
#    -- conda's "please update conda" notice alone fails the load. We use a
#    plain venv and never load that module.
#
#  * An interrupted run leaves a directory with lib/ but no bin/activate. That
#    is a partial venv, and it is why a job could import torch from a venv that
#    could not be activated. FORCE=1 clears it.
#
# Accounts: datamining (DM priority, default), csunivie (general), low
# (backfill). Override without editing -- CLI > env > directive:
#   sbatch -A low -p <partition> internal/uwiki/setup_env.sh
#   sacctmgr show assoc where user=$USER format=Account,Partition,QOS -p
#
# Nodes: p_datamining is small (vader, galadriel, shelob, dgx-h100-em2).
# Excluding too many yields "Requested node configuration is not available".
#
# Env vars:
#   PE_VENV       venv location    (default: $HOME/venvs/pretrain-experiments)
#   PE_REPO       repo location    (default: $HOME/pretrain-experiments)
#   FORCE         1 to delete and rebuild an existing venv
#   TORCH_INDEX   override the auto-detected wheel index (e.g. cu126)
#   SKIP_TORCH    1 to leave torch alone (reinstall the repo only)
#   INSTALL_OLMO  1 to also install the OLMo fork (needed by the retain-set
#                 methods: grad-diff, npo, simnpo, rmu, satimp)
#   RUN_MEASURE   1 to measure |D_f| at the end

set -u
set -o pipefail

# A batch job has no terminal: make any prompt read EOF and fail fast rather
# than hang until the walltime expires.
exec </dev/null

PE_VENV="${PE_VENV:-$HOME/venvs/pretrain-experiments}"
PE_REPO="${PE_REPO:-$HOME/pretrain-experiments}"
FORCE="${FORCE:-0}"
SKIP_TORCH="${SKIP_TORCH:-0}"
INSTALL_OLMO="${INSTALL_OLMO:-0}"
OLMO_BRANCH="${OLMO_BRANCH:-pretrain-experiments}"
RUN_MEASURE="${RUN_MEASURE:-0}"
MEASURE_SAMPLE_FRAC="${MEASURE_SAMPLE_FRAC:-0.05}"

# Progress mirrors to stderr, which is unbuffered -- a killed job still shows
# the last thing it attempted.
step () { echo ""; echo "=== $* ==="; echo "=== $* ===" >&2; }
say  () { echo "  $*"; echo "  $*" >&2; }
die  () { echo "ERROR: $*" >&2; exit 1; }

step "0. Context"
say "host: $(hostname)"
say "repo: $PE_REPO"
say "venv: $PE_VENV"
[ -d "$PE_REPO" ] || die "no repo at $PE_REPO -- clone it there or set PE_REPO"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE

step "1. Driver -> torch wheel index"
CUDA_VER=""
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VER="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null \
    | sed 's/^/    /' || true
fi

if [ -n "${TORCH_INDEX:-}" ]; then
  IDX="$TORCH_INDEX"
  say "torch index $IDX (from TORCH_INDEX)"
elif [ -z "$CUDA_VER" ]; then
  IDX="cu126"
  say "no nvidia-smi -- defaulting to $IDX"
  say "WARNING: no GPU in this allocation, so CUDA cannot be verified below."
else
  MAJ="${CUDA_VER%%.*}"; MIN="${CUDA_VER#*.}"; MIN="${MIN%%.*}"
  if   [ "$MAJ" -gt 12 ] 2>/dev/null;                     then IDX="cu128"
  elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ] 2>/dev/null; then IDX="cu128"
  elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 6 ] 2>/dev/null; then IDX="cu126"
  elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 4 ] 2>/dev/null; then IDX="cu124"
  else                                                         IDX="cu121"
  fi
  say "driver CUDA $CUDA_VER -> torch index $IDX"
fi

step "2. Virtualenv"
if [ -e "$PE_VENV" ]; then
  if [ "$FORCE" = "1" ]; then
    say "FORCE=1 -> removing $PE_VENV"
    rm -rf "$PE_VENV" || die "could not remove $PE_VENV"
  elif [ -f "$PE_VENV/bin/activate" ]; then
    say "reusing existing venv"
  else
    die "$PE_VENV exists but has no bin/activate (partial venv from an interrupted run). Re-run with FORCE=1."
  fi
fi
if [ ! -f "$PE_VENV/bin/activate" ]; then
  command -v python3 >/dev/null || die "no python3 on PATH"
  say "creating venv with $(python3 --version 2>&1)"
  mkdir -p "$(dirname "$PE_VENV")"
  python3 -m venv "$PE_VENV" || die "venv creation failed"
fi
# shellcheck disable=SC1091
source "$PE_VENV/bin/activate" || die "could not activate $PE_VENV"
say "python: $(command -v python) ($(python --version 2>&1))"
python -m pip install --upgrade pip --quiet || die "pip upgrade failed"

step "3. Install"
if [ "$SKIP_TORCH" = "1" ]; then
  say "SKIP_TORCH=1 -> leaving torch alone"
else
  say "installing torch from $IDX (the big download)"
  python -m pip install --index-url "https://download.pytorch.org/whl/${IDX}" torch \
    || die "torch install failed"
fi
cd "$PE_REPO"
python -m pip install -e . --quiet         || die "pip install -e . failed"
python -m pip install -e ".[eval]" --quiet || die "pip install .[eval] failed"
# `datasets` is imported by unlearning_utils.load_forget_set and by every eval
# script, but is not declared in pyproject.toml.
python -m pip install datasets --quiet     || die "pip install datasets failed"
say "installed"

step "4. OLMo fork"
if [ "$INSTALL_OLMO" != "1" ]; then
  say "INSTALL_OLMO=0 -> skipped (only grad-diff/npo/simnpo/rmu/satimp need it;"
  say "gradient-ascent, ce-u and wga import no olmo)"
else
  OLMO_DIR="${OLMO_DIR:-$HOME/OLMo}"
  if [ ! -d "$OLMO_DIR/.git" ]; then
    git clone https://github.com/sbordt/OLMo "$OLMO_DIR" || die "git clone failed"
  fi
  cd "$OLMO_DIR"
  git checkout "$OLMO_BRANCH" || die "could not check out $OLMO_BRANCH"
  python -m pip install -e ".[all]" --quiet || die "OLMo install failed"
  python -m pip install h5py --quiet        || die "h5py install failed"
  cd "$PE_REPO"
  say "OLMo installed at $OLMO_DIR"
fi

step "5. Verify"
python - <<'PYEOF' || die "verification failed"
import importlib, os, sys
for m in ["torch", "transformers", "datasets", "numpy", "yaml", "pretrain_experiments"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:24s} {getattr(mod, '__version__', 'n/a')}")
    except ImportError as e:
        print(f"  {m:24s} MISSING ({e})"); sys.exit(1)
try:
    import olmo  # noqa: F401
    print(f"  {'olmo':24s} present")
except ImportError:
    print(f"  {'olmo':24s} not installed (fine unless you run the retain-set methods)")

import torch
print(f"  torch built for CUDA      {torch.version.cuda}")
print(f"  torch.cuda.is_available() {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device                    {torch.cuda.get_device_name(0)}")
elif os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS_ON_NODE"):
    print("", file=sys.stderr)
    print("ERROR: a GPU was allocated but torch cannot use it.", file=sys.stderr)
    print(f"       torch is built for CUDA {torch.version.cuda}; compare with", file=sys.stderr)
    print("       nvidia-smi above. If the driver is older, re-run with:", file=sys.stderr)
    print("         FORCE=1 TORCH_INDEX=cu126 sbatch internal/uwiki/setup_env.sh", file=sys.stderr)
    sys.exit(1)
else:
    print("  (no GPU in this allocation -- CUDA unverified)")
PYEOF

step "6. Retain-stream check"
# Reports whether the FULL OLMo-2 stage1 memmap stream resolves here. It usually
# does not, and that is no longer fatal:
#
#   gradient-ascent, ce-u, wga    import no olmo at all
#   npo, simnpo, satimp           run forget-only at RETAIN_WEIGHT=0 (their
#                                 drivers guard the loader behind weight > 0)
#   grad-diff, rmu                genuinely need retain data -- but only a
#                                 sliver of it. Materialize one with
#                                 internal/uwiki/build_retain_slice.py, which
#                                 fetches ~2 GB by random access instead of
#                                 requiring the whole corpus.
python - <<'PYEOF' || true
import os
try:
    import olmo  # noqa: F401
except ImportError:
    print("  olmo not installed -- skipping (re-run with INSTALL_OLMO=1)"); raise SystemExit(0)
cfg = os.environ.get("OLMO_CONFIG",
    os.path.expanduser("~/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml"))
if not os.path.exists(cfg):
    print(f"  no config at {cfg} -- set OLMO_CONFIG"); raise SystemExit(0)
try:
    from pretrain_experiments.unlearning_utils import build_olmo_retain_dataset
    ds, info = build_olmo_retain_dataset(cfg, start_step=100000, max_seq_len=1024)
    print(f"  OK: {info['n_unseen_sequences']:,} unseen sequences, first {tuple(ds[0].shape)}")
    print("  -> all eight methods can run here, retain terms included.")
except Exception as e:
    print(f"  FULL RETAIN STREAM UNAVAILABLE: {type(e).__name__}: {e}")
    print("  -> gradient-ascent / ce-u / wga: unaffected, run now.")
    print("  -> npo / simnpo / satimp: run forget-only with RETAIN_WEIGHT=0.")
    print("  -> grad-diff / rmu: need a materialized slice --")
    print("       python internal/uwiki/build_retain_slice.py --dry-run")
PYEOF

if [ "$RUN_MEASURE" = "1" ]; then
  step "7. Measure |D_f|"
  python internal/uwiki/measure_forget_set.py \
    --sample-frac "$MEASURE_SAMPLE_FRAC" \
    --output-json "$HOME/forget_set_measurement.json" || die "measure_forget_set.py failed"
fi

step "Done"
say "venv: $PE_VENV"
say ""
say "Next:"
say "  python tests/smoke_unlearning_drivers.py    # all 8 drivers, seconds"
say "  bash internal/uwiki/lr_range.sh             # 18 jobs, no OLMo needed"
say "  GROUP=forgetonly bash internal/uwiki/lr_range.sh   # npo/simnpo/satimp"
