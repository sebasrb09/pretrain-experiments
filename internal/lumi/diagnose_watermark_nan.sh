#!/bin/bash -l
#SBATCH --account=project_465003383
#SBATCH --job-name=diag-watermark-nan
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Find where the Gaussian-watermark score turns into NaN on ROCm.
#
# On LUMI the eval reports "Mean: nan, Std: nan" and then dies with
#   abs(): argument 'input' (position 1) must be Tensor, not float
# which is the NaN statistic reaching torch.abs, not the real fault. The same
# pickles and the same model produce valid numbers on ASC, so this is a
# hardware/stack difference, not a data problem.
#
# The computation is, per noise item, at batch size 1 (so padding is NOT
# involved here, unlike the logprob bug):
#     embeds -> forward -> cross-entropy loss -> autograd.grad -> dot with noise
# This prints every intermediate for the first item under three configurations:
#
#   1  as the eval runs it            bfloat16 weights, sdpa attention
#   2  eager attention                isolates the SDPA kernel
#   3  float32 weights                isolates bf16 overflow/underflow
#
# Whichever configuration first yields a finite dot product is the fix.

set -u
set -o pipefail

W=/scratch/project_465003383/unlearning_baselines
cd "$W/pretrain-experiments" || exit 1
# shellcheck disable=SC1091
source "$W/pretrain-experiments/internal/lumi/env.sh"

NOISE_1B="$W/noise-vectors/OLMo-2-1B-Exp"
MODEL="${DIAG_MODEL:-sbordt/OLMo-2-1B-Exp}"

echo "model : $MODEL"
echo "noise : $NOISE_1B"
echo ""

python - "$NOISE_1B" "$MODEL" <<'PY'
import glob
import sys

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM

noise_dir, model_id = sys.argv[1], sys.argv[2]
NOISE_STD = 0.001          # the eval's default; printed below for confirmation

pkl = sorted(glob.glob(f"{noise_dir}/gaussian_poisoning_*.pkl"))[0]
import pickle
with open(pkl, "rb") as fh:
    data = pickle.load(fh)
item = data[0]
print(f"pickle {pkl.rsplit('/', 1)[-1]}, {len(data)} items, item len {len(item)}")

# the eval's own format branch
if len(item) == 4:
    _, _, token_ids, noise = item
else:
    token_ids, _, noise = item
print(f"token_ids {tuple(token_ids.shape)} {token_ids.dtype} | "
      f"noise {tuple(noise.shape)} {noise.dtype}")
print(f"noise finite: {torch.isfinite(noise).all().item()}  "
      f"absmax {noise.float().abs().max().item():.4g}  "
      f"std {noise.float().std().item():.4g}")
print("")

token_ids = token_ids.unsqueeze(0).cuda()
noise = noise.unsqueeze(0).cuda()


def run(tag, dtype, attn):
    print(f"--- {tag}: dtype={dtype}, attn={attn}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation=attn).cuda().eval()
    except Exception as e:
        print(f"    model load failed: {type(e).__name__}: {e}\n")
        return
    embeds = model.get_input_embeddings()(token_ids)
    embeds.requires_grad_(True)
    print(f"    embeds  finite={torch.isfinite(embeds).all().item()} "
          f"absmax={embeds.float().abs().max().item():.4g}")

    out = model(inputs_embeds=embeds, attention_mask=torch.ones_like(token_ids))
    logits = out.logits
    print(f"    logits  finite={torch.isfinite(logits).all().item()} "
          f"absmax={logits.float().abs().max().item():.4g}")

    shift = logits[:, :-1, :].contiguous()
    labels = token_ids[:, 1:].contiguous()
    loss = CrossEntropyLoss()(shift.view(-1, shift.size(-1)), labels.view(-1))
    print(f"    loss    {loss.item():.6f}  finite={torch.isfinite(loss).item()}")

    grads = torch.autograd.grad(loss, embeds)[0].detach()
    print(f"    grads   finite={torch.isfinite(grads).all().item()} "
          f"absmax={grads.float().abs().max().item():.4g}")

    g = grads.flatten(1).to(torch.float32)
    n = noise.flatten(1).to(torch.float32)
    dot = (g * n).sum(-1) / (NOISE_STD ** 2) / g.shape[-1]
    print(f"    dot     {dot.tolist()}  finite={torch.isfinite(dot).all().item()}")
    print("")
    del model
    torch.cuda.empty_cache()


run("1 as the eval runs it", torch.bfloat16, "sdpa")
run("2 eager attention",     torch.bfloat16, "eager")
run("3 float32 weights",     torch.float32,  "eager")
PY
