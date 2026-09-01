#!/bin/bash
#SBATCH --account=p201378
#SBATCH --job-name=cache-forget
#SBATCH --partition=zen4_0768
#SBATCH --qos=zen4_0768
#SBATCH -n 32
#SBATCH --mem 128G
#SBATCH --time=04:00:00
#SBATCH --output=cache-forget_%j.out
#SBATCH --error=cache-forget_%j.err
#
# Tokenize the forget set once and cache it to $DATA/forget-cache/.
#
#   sbatch internal/asc/cache_forget_set.sh
#
# Why this exists: load_forget_set tokenizes 5,247,095 documents at the start of
# EVERY training job -- ~21 minutes each. Across a grid of cells, each chained
# into several walltime-limited links, that is tens of GPU-hours spent
# re-deriving the same array. This job pays it once, on CPU, so no GPU
# allocation is spent on it.
#
# Runs on zen4_0768 (the CPU-only twin of the GPU partition): tokenization never
# touches the GPU, and a --gres=gpu:1 allocation would hold a card idle for the
# duration.
#
# Idempotent. A second run finds the cache and exits in seconds, so it is safe
# to submit again if you are unsure whether the first one finished.
#
# Overridable:
#   MODEL_DIR         tokenizer source (default: the converted step100000 dir)
#   FORGET_CACHE_DIR  where to write   (default: $DATA/forget-cache)
#   MAX_SEQ_LEN       only affects the reported truncation; the cache itself is
#                     length-independent, so one cache serves every setting

set -u
set -o pipefail
exec </dev/null

cd "${PE_REPO:-${SCRATCH}/pretrain-experiments}" || exit 1
# shellcheck disable=SC1091
source internal/asc/env.sh

MODEL_DIR="${MODEL_DIR:-${PE_DATA:-$DATA}/checkpoints/1B-Exp-Unlearning-step100000-hf}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

# The fast tokenizer parallelises across the allocation. transformers disables
# this silently when it suspects forking, which would turn 21 minutes into
# hours, so set it explicitly -- there are no DataLoader workers here.
export TOKENIZERS_PARALLELISM=true

echo "--- forget-set cache ---"
echo "  repo:      $(pwd)"
echo "  tokenizer: $MODEL_DIR"
echo "  cache dir: ${FORGET_CACHE_DIR:-${PE_DATA:-$DATA}/forget-cache}"
echo "  cpus:      $(python -c 'import os; print(len(os.sched_getaffinity(0)))')"
echo ""

python - "$MODEL_DIR" "$MAX_SEQ_LEN" <<'PY'
import sys, time
from transformers import AutoTokenizer
from pretrain_experiments.unlearning_utils import load_forget_set

model_dir, max_seq_len = sys.argv[1], int(sys.argv[2])
tok = AutoTokenizer.from_pretrained(model_dir)

t = time.perf_counter()
dataset, info = load_forget_set(tok, max_seq_len=max_seq_len)
dt = time.perf_counter() - t

print("")
print(f"  sequences:   {info['n_sequences']:,}")
print(f"  tokens:      {info['n_total_tokens']:,}")
print(f"  longest:     {info['max_seq_len_observed']:,}")
print(f"  experiments: {len(info['experiments_in_set'])}")
print(f"  elapsed:     {dt/60:.1f} min"
      f"{'  (cache hit -- nothing to do)' if info.get('from_cache') else ''}")

# Prove the cache is loadable before the job exits: a second call must hit, and
# the first sequence must survive the round trip. Writing a cache that no later
# job can read would be worse than having none.
again, info2 = load_forget_set(tok, max_seq_len=max_seq_len)
assert info2.get("from_cache"), "second load did not hit the cache"
assert len(again) == len(dataset), "cached length differs"
assert again[0].tolist() == dataset[0].tolist(), "cached content differs"
print(f"  verified:    re-loaded from {info2['from_cache']}")
PY

status=$?
echo ""
if [ "$status" -eq 0 ]; then
  echo "OK. Every training job will now log 'forget cache HIT' instead of"
  echo "tokenizing. Contents:"
  ls -la "${FORGET_CACHE_DIR:-${PE_DATA:-$DATA}/forget-cache}"
else
  echo "FAILED (exit $status) -- see the error file. Nothing partial is left"
  echo "behind: the cache is written temp-then-rename, so a killed job leaves"
  echo "no half-file for the next run to load as if it were complete."
fi
exit "$status"
