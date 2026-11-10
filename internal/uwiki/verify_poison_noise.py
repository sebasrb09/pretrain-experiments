"""Can this machine regenerate the Gaussian poisons from their token ids alone? (2026-11-10)

The pre-training run drew each poison's noise from a seed taken from the poisoned sequence itself (OLMo model.py, and
the authors' collect_guassian_poisoning_seeds_and_sequences_from_trainloader.py): seed = sum(token ids) mod 47629,
then normal_(std 0.075) into a zero (4096, d_model) bf16 tensor with a torch.Generator on the GPU. Only 1% of the
~48k poisoned sequences were stored (480 at 1B). If this reproduces the stored noise exactly, all of them can be
rebuilt from token ids and seeds, which the authors' script recovers from the training data order.

Checks the first N stored poisons: the stored seed against sum(ids) mod 47629, and the regenerated noise against the
stored one (exact match, else the largest difference and the correlation). Usage on a GPU node:
    python internal/uwiki/verify_poison_noise.py NOISE_DIR [N=20]
"""
import glob
import os
import pickle
import sys

import torch

from pretrain_experiments.unlearning_utils import POISON_FILE_GLOB, _poison_pair

noise_dir = sys.argv[1]
n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 20
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'no GPU: the noise cannot match'}), "
      f"torch {torch.__version__}")
gen = torch.Generator(device=dev)
checked = seeds_ok = exact = 0
for f in sorted(glob.glob(os.path.join(noise_dir, POISON_FILE_GLOB))):
    for item in pickle.load(open(f, "rb")):
        ids, stored = _poison_pair(item)
        seed = int(torch.remainder(ids.sum(), 47629))
        seeds_ok += seed == int(item[1])          # the seed sits second in both stored layouts
        gen.manual_seed(seed)
        z = torch.zeros((ids.shape[0], stored.shape[1]), dtype=torch.bfloat16, device=dev)
        z.normal_(generator=gen, std=0.075)
        z = z.float().cpu()
        same = torch.equal(z, stored)
        exact += same
        if not same:
            diff = (z - stored).abs().max().item()
            corr = torch.corrcoef(torch.stack([z.flatten(), stored.flatten()]))[0, 1].item()
            print(f"  {os.path.basename(f)}: seed {seed}, max |diff| {diff:.3g}, correlation {corr:.3f}")
        checked += 1
        if checked >= n_max:
            break
    if checked >= n_max:
        break
print(f"{checked} stored poisons checked: seed = sum(ids) mod 47629 for {seeds_ok}, noise reproduced exactly for {exact}")
print("RESULT:", "this machine regenerates the poisons exactly" if exact == checked > 0 else
      "the noise does NOT match here: regenerate on the GPU type of the original run (NVIDIA)")
