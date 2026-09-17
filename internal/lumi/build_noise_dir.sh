#!/bin/bash -l
#SBATCH --account=project_465003383
#SBATCH --job-name=pe-noise-2.7B
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Build the Gaussian-watermark noise vectors for a model size, on CPU.
#
#   sbatch internal/lumi/build_noise_dir.sh                    # 2.7B (default)
#   SIZE=1B sbatch internal/lumi/build_noise_dir.sh            # any other size
#
# ---------------------------------------------------------------------------
# WHY THIS IS A BATCH JOB AND NOT A LOGIN-NODE COMMAND
#
# It downloads a multi-GB parquet dataset and rewrites it as pickles. Both the
# download and the conversion are long and memory-hungry -- each sequence's
# noise array is 4096 x embed_dim float32, which at 2.7B's embed_dim 2880 is
# ~47 MB per sequence before any batching. That is exactly the kind of work
# LUMI asks you to keep off the login nodes.
#
# --partition=small is the CPU partition (LUMI-C), billed per core-hour, so
# this costs no GPU allocation at all. No --gpus flag: there is nothing here
# that touches a GPU.
# ---------------------------------------------------------------------------
#
# WHY IT MATTERS THAT THIS RUNS BEFORE ANY 2.7B EVAL
#
# internal/uwiki/eval_cell_body.sh does NOT fail when the noise vectors are
# missing -- it prints a note and SKIPS the gaussian_watermark eval (see its
# `elif [ ! -d "$NOISE_DIR" ]` branch). So a 2.7B eval launched without these
# produces cells that look fine and silently carry no watermark column, and
# the gap only shows up at aggregation time.
#
# The default NOISE_DIR is also 1B-specific
# (.../noise-vectors/OLMo-2-1B-Exp), and 1B's vectors are embed_dim 2048
# against 2.7B's 2880. Every 2.7B eval must pass NOISE_DIR explicitly:
#
#   NOISE_DIR=$PE_WORK/noise-vectors/OLMo-2-2.7B-Exp
#
# Optional env vars:
#   SIZE        model size label      (default: 2.7B)
#   NOISE_REPO  HF dataset repo       (default: sbordt/OLMo-2-<SIZE>-Exp-NoiseVectors)
#   NOISE_OUT   output directory      (default: $PE_WORK/noise-vectors/OLMo-2-<SIZE>-Exp)
#   NOISE_DTYPE float32 | bfloat16    (default: bfloat16, the injection dtype)

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GUESS="$(cd "${SCRIPT_DIR}/../.." && pwd)"

find_repo () {
  local c
  for c in "${PE_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
           "${PE_WORK:-/scratch/${PE_PROJECT:-project_465003383}/unlearning_baselines}/pretrain-experiments" \
           "$REPO_GUESS" "$PWD"; do
    if [ -n "$c" ] && [ -f "$c/internal/lumi/env.sh" ]; then echo "$c"; return 0; fi
  done
  return 1
}
PE_REPO="$(find_repo)" || {
  echo "ERROR: could not locate the pretrain-experiments checkout." >&2; exit 1; }
export PE_REPO

# env.sh loads the stack and the PyTorch container, which is what puts a python
# with `datasets` and `torch` on PATH. It needs no GPU to load.
# shellcheck disable=SC1091
source "${PE_REPO}/internal/lumi/env.sh"

SIZE="${SIZE:-2.7B}"
NOISE_REPO="${NOISE_REPO:-sbordt/OLMo-2-${SIZE}-Exp-NoiseVectors}"
NOISE_OUT="${NOISE_OUT:-${PE_WORK}/noise-vectors/OLMo-2-${SIZE}-Exp}"
NOISE_DTYPE="${NOISE_DTYPE:-bfloat16}"

echo "--- building noise vectors ---"
echo "  size   : $SIZE"
echo "  repo   : $NOISE_REPO"
echo "  out    : $NOISE_OUT"
echo "  dtype  : $NOISE_DTYPE"
echo "  HF_HOME: $HF_HOME"
echo "------------------------------"

mkdir -p "$NOISE_OUT"

python "${PE_REPO}/mia-data/build_noise_dir.py" \
  --repo "$NOISE_REPO" \
  --out  "$NOISE_OUT" \
  --noise-dtype "$NOISE_DTYPE" || {
    echo "ERROR: build_noise_dir.py failed." >&2
    echo "       If the dataset id is wrong, set NOISE_REPO explicitly." >&2
    exit 1; }

# Verify against what gaussian_watermark.py actually globs for, rather than
# trusting that the script exited 0 -- an empty directory would otherwise be
# discovered much later, as silently skipped watermark evals.
echo ""
echo "--- verifying ---"
n=$(ls "$NOISE_OUT"/gaussian_poisoning_*.pkl 2>/dev/null | wc -l)
echo "  gaussian_poisoning_*.pkl files: $n"
if [ "$n" -eq 0 ]; then
  echo "ERROR: no gaussian_poisoning_*.pkl written. Every 2.7B eval would" >&2
  echo "       silently SKIP the watermark rather than fail." >&2
  exit 1
fi
du -sh "$NOISE_OUT"
ls -lh "$NOISE_OUT" | head -5

echo ""
echo "=================================================================="
echo "  done. Pass this to every 2.7B evaluation:"
echo "    NOISE_DIR=$NOISE_OUT"
echo "=================================================================="
