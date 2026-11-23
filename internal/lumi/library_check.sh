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
#        bash internal/lumi/library_check.sh submit train   # only some parts: train,
#                                         # baseline, retraining, gd-full, anchors (the last two
#                                         # not in the default)
#        bash internal/lumi/library_check.sh redo-train     # set the replays aside to run them again
#    submit refuses when the preflight finds a problem, and skips any job of the
#    same name already queued or running. Finished work is reused, so
#    submitting again redoes only what is missing.
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
    # Parts: train, baseline, retraining (default: these three), and gd-full: the
    # whole suite on the grad-diff replay's step 2, after the train job.
    shift
    parts=("$@")
    [ ${#parts[@]} -eq 0 ] && parts=(train baseline retraining)
    for part in "${parts[@]}"; do
      case "$part" in train|baseline|retraining|gd-full|anchors) ;;
        *) echo "ERROR: unknown part '$part' (train, baseline, retraining, gd-full, anchors)" >&2; exit 2 ;; esac
    done
    "${PY[@]}" preflight || { echo "ERROR: preflight failed; nothing submitted." >&2; exit 1; }
    common=(--parsable "--export=ALL,PE_REPO=${PE_REPO},LIBCHECK_SRC=${LIBCHECK_SRC},LIBCHECK_OUT=${LIBCHECK_OUT}")
    # A job of the same name already queued or running writes the same files: never start a
    # second one. Its id is passed on instead, so dependencies still hold.
    submit_job () {  # name, then sbatch arguments; prints the job id
      local id
      id="$(squeue -h -u "${USER:-$(id -un)}" -n "$1" -o %i)" \
        || { echo "ERROR: squeue failed, so a running $1 cannot be ruled out; not submitted." >&2; return 1; }
      id="${id%%$'\n'*}"
      if [ -n "$id" ]; then
        echo "skipped $1: already queued or running as ${id}" >&2
      else
        id=$(sbatch "${common[@]}" --job-name="$1" "${@:2}") || { echo "ERROR: sbatch of $1 failed." >&2; return 1; }
        echo "submitted $1: ${id}" >&2
      fi
      echo "$id"
    }
    for part in "${parts[@]}"; do
      if [ "$part" = anchors ]; then
        # the published pairs with no full evaluation yet (179M, 546M pretraining; 1B decayed; 2.7B mid)
        for target in 179M-pretrain-baseline 179M-pretrain-retraining 546M-pretrain-baseline \
                      546M-pretrain-retraining 1B-decayed-baseline 1B-decayed-retraining \
                      2.7B-mid-baseline 2.7B-mid-retraining; do
          walltime=12:00:00
          [ "${target%%-*}" = 2.7B ] && walltime=24:00:00   # 2.7B's batch-1 group took 9.6 h
          for group in b1 e8; do
            if [ -f "${LIBCHECK_OUT}/records/eval-${target}-${group}.json" ]; then
              echo "skipped libcheck-${target}-${group}: already evaluated" >&2; continue
            fi
            submit_job "libcheck-${target}-${group}" --time="$walltime" "$SELF" eval "$target" "$group" >/dev/null || exit 1
          done
        done
        continue
      fi
      if [ "$part" = train ]; then
        train=$(submit_job libcheck-train --mem=120G --time=06:00:00 "$SELF" train) || exit 1
      elif [ "$part" = gd-full ]; then
        dep=()
        train="$(squeue -h -u "${USER:-$(id -un)}" -n libcheck-train -o %i | head -1)"
        [ -n "$train" ] && dep=("--dependency=afterany:${train}")
        for group in b1 e8; do
          submit_job "libcheck-grad-diff-step-2-${group}" ${dep[@]+"${dep[@]}"} \
            "$SELF" eval grad-diff-step-2 "$group" >/dev/null || exit 1
        done
      else
        for group in b1 e8; do
          submit_job "libcheck-${part}-${group}" "$SELF" eval "$part" "$group" >/dev/null || exit 1
        done
      fi
    done
    echo "Then: bash internal/lumi/library_check.sh report"
    ;;
  preflight|report|redo-train)
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
