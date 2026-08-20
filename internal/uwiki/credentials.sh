# shellcheck shell=bash
#
# Shared credential resolution for HuggingFace and Weights & Biases.
# Sourced by the per-site environment blocks:
#   internal/uwiki/unlearn_cell_1B.sh
#   internal/meluxina/env.sh
#
# Resolves a token without ever storing one in the repository.
#
# HF_TOKEN is REQUIRED for:
#   - meta-llama/Meta-Llama-3-8B-Instruct, the gated judge that scores the
#     denial-of-service eval (denial_of_service.py:22). Without it the judge
#     download 401s and that eval fails.
# and is optional-but-useful everywhere else (higher Hub rate limits when 35
# eval tasks pull the same weights).
#
# Supply it in ONE of these ways, most preferred first:
#
#   1. Run `huggingface-cli login` once. It writes ~/.cache/huggingface/token,
#      which huggingface_hub picks up on its own -- nothing to set here.
#
#   2. A private, untracked file (default $HOME/.hf_token, gitignored):
#        printf '%s' 'hf_xxxxxxxx' > ~/.hf_token && chmod 600 ~/.hf_token
#      Override the location with HF_TOKEN_FILE=/path/to/file.
#
#   3. Export it before submitting; --export=ALL carries it into the job:
#        export HF_TOKEN=hf_xxxxxxxx
#        sbatch --export=ALL,METHOD=...,VALUE=... internal/.../unlearn_cell_1B.sh
#
# Never commit a token, and never paste one into a tracked script. Only the
# character count is echoed below, never the value.

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.hf_token}"
HF_TOKEN_SOURCE=""

if [ -n "${HF_TOKEN:-}" ]; then
  HF_TOKEN_SOURCE="environment"
elif [ -f "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
  HF_TOKEN_SOURCE="$HF_TOKEN_FILE"
fi

if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
  # huggingface_hub reads HF_TOKEN; some older transformers / datasets code
  # paths still read HUGGING_FACE_HUB_TOKEN. Set both so either works.
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "  hf token: set, ${#HF_TOKEN} chars (from ${HF_TOKEN_SOURCE})"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
  echo "  hf token: using the huggingface-cli login cache"
else
  echo "  hf token: NOT SET -- public repos still work; the gated DoS judge will 401"
fi

# ---- Weights & Biases ------------------------------------------------------
# The unlearning drivers write metrics.jsonl and do not use wandb, but the YAML
# evaluation suite does. Same pattern, same rules.
WANDB_TOKEN_FILE="${WANDB_TOKEN_FILE:-$HOME/.wandb_token}"

if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_TOKEN_FILE" ]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_TOKEN_FILE")"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
fi
