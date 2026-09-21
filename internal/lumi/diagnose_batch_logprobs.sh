#!/bin/bash -l
#SBATCH --account=project_465003383
#SBATCH --job-name=diag-batch-logprobs
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Isolate WHY get_logprobs is not batch-invariant.
#
# Measured on sbordt/OLMo-2-1B-Exp over 2500 c4 documents:
#   batch 1 -> 17.40   (matches ASC's 17.3997)
#   batch 2 -> 47.47
#   batch 4 -> 79.59
#
# Passing position_ids derived from the attention mask changed the numbers
# slightly (46.73 -> 47.47) but did not fix it, so the rotary positions were
# one problem and not the only one. This runs the same four documents three
# ways to say which part is at fault:
#
#   A  batch 1                        -- no padding at all, the reference
#   B  batch 2, EQUAL-length inputs   -- batched, but nothing to pad
#   C  batch 2, unequal lengths       -- batched AND padded
#
# B == A and C != A  =>  padding is the cause (mask or positions still wrong).
# B != A             =>  batching itself is wrong, independent of padding.

set -u
set -o pipefail

W=/scratch/project_465003383/unlearning_baselines
cd "$W/pretrain-experiments" || exit 1
# shellcheck disable=SC1091
source "$W/pretrain-experiments/internal/lumi/env.sh"

python - <<'PY'
import torch
from pretrain_experiments.evaluation.inference_engine import InferenceEngineFactory

MODEL = "sbordt/OLMo-2-1B-Exp"
engine = InferenceEngineFactory.create_from_config(MODEL)
tok = engine.tokenizer

print("pad_token_id:", tok.pad_token_id, " eos:", tok.eos_token_id)
print("attn impl   :", getattr(engine.model.config, "_attn_implementation", "?"))

# Four documents, pre-tokenised so we control lengths exactly.
text = ("The capital of France is Paris, and the city is known for its museums, "
        "its river, and a tower built for a world exposition in the nineteenth century. ")
long_ids  = tok(text * 6, return_tensors="pt")["input_ids"][0].tolist()
short_ids = long_ids[:len(long_ids) // 3]

def total(prompts, batch):
    engine.max_num_seqs = batch
    out = engine.get_logprobs(prompts)
    return [round(sum(v for v in r["logprobs"] if v is not None), 4) for r in out]

equal   = [long_ids,  long_ids[:]]          # identical lengths -> no padding
unequal = [long_ids,  short_ids]            # different lengths -> padding

a_eq = total(equal, 1)
b_eq = total(equal, 2)
a_un = total(unequal, 1)
c_un = total(unequal, 2)

print()
print(f"A  batch 1, equal lengths   : {a_eq}")
print(f"B  batch 2, equal lengths   : {b_eq}")
print(f"A' batch 1, unequal lengths : {a_un}")
print(f"C  batch 2, unequal lengths : {c_un}")
print()

eq_ok = all(abs(x - y) < 1e-2 for x, y in zip(a_eq, b_eq))
un_ok = all(abs(x - y) < 1e-2 for x, y in zip(a_un, c_un))
print(f"batching without padding matches batch 1 : {eq_ok}")
print(f"batching WITH padding matches batch 1    : {un_ok}")
print()
if eq_ok and not un_ok:
    print("VERDICT: padding is the cause -- the attention mask and/or position "
          "handling still does not neutralise left pad tokens.")
elif not eq_ok:
    print("VERDICT: batching itself is wrong, independent of padding. Suspect "
          "the forward pass or how results are indexed back out of the batch.")
else:
    print("VERDICT: both match here -- the bug needs the real c4 length "
          "distribution to reproduce. Re-run with more, longer documents.")
PY
