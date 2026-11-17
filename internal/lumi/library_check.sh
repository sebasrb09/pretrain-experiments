#!/bin/bash
#SBATCH --account=project_465003383
#SBATCH --job-name=libcheck
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Check the llm-unlearning-lifecycle library against the paper's runs on LUMI.
# What is compared, and against which numbers, is in library_check.py.
#
# 1. Copy the library's source tree to $PE_WORK/llm-unlearning-lifecycle/src.
#    It is imported from there on PYTHONPATH; nothing is installed into the
#    shared container venv, so the research jobs are untouched.
# 2. From the repository root, on a login node:
#        bash internal/lumi/library_check.sh preflight   # checks only, submits nothing
#        bash internal/lumi/library_check.sh submit      # preflight, then the 7 jobs
#    submit refuses when the preflight finds a problem.
# 3. When the jobs are done (also fine while they run: it reports what exists):
#        bash internal/lumi/library_check.sh report      # also writes $LIBCHECK_OUT/report.txt
#
# Jobs, one GCD each on small-g:
#   libcheck-train       the three replays, the micro-batch check, the replays'
#                        quick evaluations; --mem=120G (the 1B poisons and the
#                        optimizer state sit in host memory), ~3 h
#   libcheck-<model>-<group>   the full suite, one settings group per job, for the
#                        Baseline, Retraining and the grad-diff replay at step 2
#                        (the last two after the train job); 60G, ~4-6 h each
#
# Variables read here: LIBCHECK_SRC (default $PE_WORK/llm-unlearning-lifecycle/src),
# LIBCHECK_OUT (default $PE_WORK/library-check). The library itself reads none.

set -uo pipefail
exec </dev/null

# SLURM runs a copy of this script from its spool directory, so locate the
# repository explicitly (as eval_pareto_cell.sh does).
find_repo () {
  local c
  for c in "${PE_REPO:-}" \
           "${SLURM_SUBMIT_DIR:-}" \
           "${PE_WORK:-/scratch/${PE_PROJECT:-project_465003383}/unlearning_baselines}/pretrain-experiments" \
           "$PWD"; do
    if [ -n "$c" ] && [ -f "$c/internal/lumi/env.sh" ] && [ -f "$c/internal/lumi/library_check.py" ]; then
      echo "$c"; return 0
    fi
  done
  return 1
}
PE_REPO="$(find_repo)" || {
  echo "ERROR: could not locate the pretrain-experiments checkout (with library_check.py)." >&2
  echo "       Run from the repo root, or export PE_REPO=/path/to/repo." >&2
  exit 1
}
export PE_REPO

# Container, HF_HOME on scratch, MIOpen cache, HF_TOKEN (credentials.sh).
# shellcheck disable=SC1091
source "${PE_REPO}/internal/lumi/env.sh"

LIBCHECK_SRC="${LIBCHECK_SRC:-${PE_WORK}/llm-unlearning-lifecycle/src}"
LIBCHECK_OUT="${LIBCHECK_OUT:-${PE_WORK}/library-check}"
if [ ! -f "${LIBCHECK_SRC}/llm_unlearning_lifecycle/__init__.py" ]; then
  echo "ERROR: no library source at ${LIBCHECK_SRC}" >&2
  echo "       Copy llm-unlearning-lifecycle/src there, or export LIBCHECK_SRC." >&2
  exit 1
fi
export PYTHONPATH="${LIBCHECK_SRC}${PYTHONPATH:+:$PYTHONPATH}"
PY=(python "${PE_REPO}/internal/lumi/library_check.py" --pe-work "$PE_WORK" --out "$LIBCHECK_OUT")

echo "--- library check ---"
echo "  library:  ${LIBCHECK_SRC}"
echo "  output:   ${LIBCHECK_OUT}"
echo "  job:      ${SLURM_JOB_ID:-login node}  $*"
# For the record: the variables the library's dependencies read (tokens hidden).
env | grep -E '^(HF_|HUGGINGFACE|TRANSFORMERS_|TORCH_|PYTORCH_|HIP_|ROCR_|HSA_)' \
    | grep -v -i token | sort | sed 's/^/  env: /'
echo "---------------------"

SELF="${PE_REPO}/internal/lumi/library_check.sh"
case "${1:-}" in
  submit)
    if [ -n "${SLURM_JOB_ID:-}" ]; then
      echo "ERROR: submit runs on a login node, not inside a job." >&2; exit 2
    fi
    "${PY[@]}" preflight || { echo "ERROR: preflight failed; nothing submitted." >&2; exit 1; }
    common=(--parsable "--export=ALL,PE_REPO=${PE_REPO},LIBCHECK_SRC=${LIBCHECK_SRC},LIBCHECK_OUT=${LIBCHECK_OUT}")
    train=$(sbatch "${common[@]}" --job-name=libcheck-train --mem=120G --time=06:00:00 "$SELF" train) \
      || { echo "ERROR: sbatch of the train job failed." >&2; exit 1; }
    echo "submitted libcheck-train: ${train}"
    for target in baseline retraining grad-diff-step-2; do
      dep=()
      [ "$target" = "grad-diff-step-2" ] && dep=("--dependency=afterany:${train}")
      for group in b1 e8; do
        id=$(sbatch "${common[@]}" ${dep[@]+"${dep[@]}"} --job-name="libcheck-${target}-${group}" "$SELF" eval "$target" "$group") \
          || { echo "ERROR: sbatch of ${target} ${group} failed." >&2; exit 1; }
        echo "submitted libcheck-${target}-${group}: ${id}${dep:+ (after ${train})}"
      done
    done
    echo "Then: bash internal/lumi/library_check.sh report"
    ;;
  preflight|report)
    "${PY[@]}" "$1"
    ;;
  train)
    "${PY[@]}" train
    ;;
  eval)
    [ $# -eq 3 ] || { echo "usage: library_check.sh eval NAME GROUP" >&2; exit 2; }
    "${PY[@]}" eval "$2" "$3"
    ;;
  *)
    echo "usage: bash internal/lumi/library_check.sh {preflight|submit|report}" >&2
    echo "       (train and eval NAME GROUP are the jobs submit creates)" >&2
    exit 2
    ;;
esac
