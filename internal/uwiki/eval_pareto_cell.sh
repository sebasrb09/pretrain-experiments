#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#SBATCH --job-name=pareto-eval
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vader,galadriel

# Vienna (u:wiki) wrapper for evaluating ONE point of the 1B Pareto plot.
#
# This file holds ONLY the site setup. What actually runs lives in
# internal/uwiki/eval_cell_body.sh, shared with internal/asc/eval_pareto_cell.sh
# so the two sites cannot drift.
#
# Every eval in the suite runs SEPARATELY into its own subdirectory with its own
# .done marker, so no metric is collapsed into a single headline number and any
# figure can pick whichever axis it wants. internal/uwiki/aggregate_pareto.py
# reduces the tree to a tidy table afterwards.
#
#   <CELL_DIR>/evals/c4_perplexity/results.yaml         <- the utility axis
#                    fictional_knowledge/results.yaml
#                    verbatim_memorization/results.yaml
#                    gaussian_watermark/*.pt
#                    insertion_likelihood/results.yaml   (SKIP_IL=0)
#                    memorization_patterns_mia/*.json    (SKIP_MIA=0)
#
# Two ways to call it:
#
#   1. a trained cell -- point it at the cell directory, the checkpoint is
#      located inside it:
#        sbatch --export=ALL,CELL_DIR=/data/.../1B-pareto/ce-u/lr-4e-6 \
#          internal/uwiki/eval_pareto_cell.sh
#
#   2. a reference anchor -- an HF repo plus revision, with an explicit out dir:
#        sbatch --export=ALL,MODEL=sbordt/OLMo-2-1B-Unlearning,REVISION=stage1-step100000-tokens210B,EVAL_OUT=/data/.../anchors/deep-ignorance \
#          internal/uwiki/eval_pareto_cell.sh
#
# Both are launched in bulk by internal/uwiki/launch_pareto_evals.sh.
#
# Env vars:
#   CELL_DIR    trained cell to evaluate (mode 1)
#   MODEL       HF repo or local dir     (mode 2; overrides the found checkpoint)
#   REVISION    HF revision              (mode 2 only)
#   EVAL_OUT    where results go         (default: $CELL_DIR/evals)
#   CKPT        explicit checkpoint dir  (default: highest-numbered epoch-*/)
#   NOISE_DIR   gaussian-watermark noise vectors
#   NOISE_STD   default 0.001
#   SKIP_PPL / SKIP_FK / SKIP_VM / SKIP_GW   1 to skip (all default 0 = run)
#   SKIP_IL     default 1 -- insertion likelihood, opt in
#   SKIP_MIA    default 1 -- 30 sub-runs, opt in when you actually want it
#   FORCE_EVAL  1 to ignore .done markers and recompute

set -u
set -o pipefail
exec </dev/null

export PE_SITE="uwiki"
unset SSL_CERT_FILE

PE_REPO="${PE_REPO:-$HOME/pretrain-experiments}"
cd "$PE_REPO" || { echo "ERROR: no repo at $PE_REPO" >&2; exit 1; }

PE_VENV="${PE_VENV:-$HOME/venvs/pretrain-experiments}"
[ -f "$PE_VENV/bin/activate" ] || {
  echo "ERROR: no usable venv at $PE_VENV" >&2
  echo "       Build it: sbatch internal/uwiki/setup_env.sh  (add FORCE=1 if partial)" >&2
  exit 1
}
# shellcheck disable=SC1091
source "$PE_VENV/bin/activate"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# shellcheck disable=SC1091
source "${PE_REPO}/internal/uwiki/credentials.sh"

# shellcheck disable=SC1091
source "${PE_REPO}/internal/uwiki/eval_cell_body.sh"
