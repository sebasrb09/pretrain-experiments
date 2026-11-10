"""Rebuild the Gaussian-poison noise on any CPU: PyTorch's CUDA normal_ reimplemented in numpy (2026-11-10).

The pre-training run drew each poison's noise on an NVIDIA GPU (OLMo model.py): seed = sum(token ids) mod 47629,
torch.Generator('cuda').manual_seed(seed), then normal_(std=0.075) into a zero (4096, d_model) bf16 tensor. AMD GPUs
draw other numbers from the same seed (verify_poison_noise.py on LUMI: 0 of 20 matched), but the draw is plain
arithmetic, reproduced here step by step (PyTorch aten/src/ATen/native/cuda/DistributionTemplates.h, cuRAND):

1. calc_execution_policy: blocks of 256 threads, grid = min(ceil(numel / 256), SMs * maxThreadsPerSM / 256). The
   grid depends on the GPU's number of multiprocessors (SMs), the one hardware-dependent part, tried below.
2. Each thread idx: curand_init(seed, subsequence=idx, offset=0), so its j-th draw is Philox4x32-10 of the counter
   (j, 0, idx, 0) with the key (seed low 32 bits, seed high 32 bits).
3. curand_normal4: Box-Muller on the four 32-bit words: (x, y) -> (sin v * s, cos v * s) with u = x 2^-32 + 2^-33,
   v = y 2pi 2^-32 + pi 2^-32, s = sqrt(-2 log u); likewise (z, w).
4. Grid-stride loop: thread idx writes its four values at idx + j*4T + i*T (i = 0..3, T = all threads).
5. value * std (float32), cast to bf16 (round to nearest even).

The GPU computes sin/cos with the fast __sincosf, so a small share of values can differ by one bf16 step; the check
reports how many: on the stored 1B poisons, H100 SXM's 132 SMs reproduce 99.975% of values exactly, the rest
within one bf16 step (11 poisons across the whole run, 2026-11-10). unlearning_utils.PoisonForgetDataset uses
it to rebuild the noise of poisons stored as token ids and seeds only.
Usage: python -m pretrain_experiments.cuda_poison_noise NOISE_DIR [N=5]  (compares with stored poisons)
"""
import sys

import numpy as np

M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
U32 = np.uint64(0xFFFFFFFF)
# (SMs, max threads per SM) of common NVIDIA GPUs; the grid only depends on their product / 256
GPUS = {"H100 SXM / GH200": (132, 2048), "H100 PCIe": (114, 2048), "A100": (108, 2048), "B200": (148, 2048),
        "V100": (80, 2048), "A40 / A6000": (84, 1536), "L40S": (142, 1536)}


def philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Random123 / cuRAND Philox4x32-10 on uint64 arrays holding 32-bit words."""
    c = [np.asarray(x, dtype=np.uint64) for x in (c0, c1, c2, c3)]
    k0, k1 = int(k0) & 0xFFFFFFFF, int(k1) & 0xFFFFFFFF
    m0, m1 = np.uint64(M0), np.uint64(M1)
    for _ in range(10):
        p0, p1 = m0 * c[0], m1 * c[2]
        c = [(p1 >> np.uint64(32)) ^ c[1] ^ np.uint64(k0), p1 & U32, (p0 >> np.uint64(32)) ^ c[3] ^ np.uint64(k1),
             p0 & U32]
        k0, k1 = (k0 + W0) & 0xFFFFFFFF, (k1 + W1) & 0xFFFFFFFF
    return c


INV = np.float32(2.3283064e-10)                          # CURAND_2POW32_INV
INV_2PI = np.float32(INV * np.float32(6.2831855))       # CURAND_2POW32_INV_2PI


def _fma32(a, b, c):
    """float32 a*b + c with one rounding, as nvcc contracts it"""
    return (a.astype(np.float64) * np.float64(b) + np.float64(c)).astype(np.float32)


def box_muller(x, y):
    u = _fma32(x.astype(np.float32), INV, INV / np.float32(2))
    v = _fma32(y.astype(np.float32), INV_2PI, INV_2PI / np.float32(2))
    s = np.sqrt(np.float32(-2.0) * np.log(u))
    v64 = v.astype(np.float64)
    return np.sin(v64).astype(np.float32) * s, np.cos(v64).astype(np.float32) * s


def to_bf16(a):
    """float32 -> bf16 (round to nearest even), returned as float32"""
    b = a.astype(np.float32).view(np.uint32)
    b = (b + (np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1)))) & np.uint32(0xFFFF0000)
    return b.view(np.float32)


def cuda_normal(shape, seed, std=0.075, sms=132, threads_per_sm=2048, bf16=True):
    """torch.zeros(shape, dtype=bf16, device='cuda').normal_(generator=g.manual_seed(seed), std=std), on CPU"""
    n = int(np.prod(shape))
    block = 256
    grid = min((n + block - 1) // block, sms * (threads_per_sm // block))
    t = grid * block
    idx = np.arange(t, dtype=np.uint64)
    zero = np.zeros(t, dtype=np.uint64)
    out = np.empty(n, dtype=np.float32)
    for j in range((n - 1) // (t * 4) + 1):
        r = philox4x32_10(np.full(t, j, dtype=np.uint64), zero, idx, zero, seed & 0xFFFFFFFF, seed >> 32)
        a, b = box_muller(r[0], r[1]), box_muller(r[2], r[3])
        for i, vals in enumerate((a[0], a[1], b[0], b[1])):
            li = np.arange(t, dtype=np.int64) + j * 4 * t + i * t
            keep = li < n
            out[li[keep]] = vals[keep]
    out = out * np.float32(std)
    return (to_bf16(out) if bf16 else out).reshape(shape)


def _self_test():
    """Random123's published known-answer vectors for Philox4x32-10"""
    kat = [((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
           ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
           ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1))]
    for ctr, key, want in kat:
        got = tuple(int(np.asarray(x).reshape(-1)[0]) for x in philox4x32_10(*[[c] for c in ctr], *key))
        assert got == want, (ctr, key, [hex(g) for g in got], [hex(w) for w in want])
    print("Philox4x32-10 known-answer vectors: OK")


def _compare(noise_dir, n_max):
    import glob, os, pickle
    import torch
    from pretrain_experiments.unlearning_utils import POISON_FILE_GLOB, _poison_pair
    stored = []
    for f in sorted(glob.glob(os.path.join(noise_dir, POISON_FILE_GLOB))):
        for item in pickle.load(open(f, "rb")):
            ids, noise = _poison_pair(item)
            stored.append((int(torch.remainder(ids.sum(), 47629)), noise.numpy()))
            if len(stored) >= n_max:
                break
        if len(stored) >= n_max:
            break
    seed0, z0 = stored[0]
    bits = z0.view(np.uint32)
    print(f"stored noise: shape {z0.shape}, bf16-exact values {100 * np.mean((bits & 0xFFFF) == 0):.1f}%")
    best = None
    for name, (sm, tps) in GPUS.items():
        g = cuda_normal(z0.shape, seed0, sms=sm, threads_per_sm=tps)
        corr = np.corrcoef(g.ravel(), z0.ravel())[0, 1]
        same = np.mean(g == z0)
        print(f"  {name:18s} ({sm} SMs): correlation {corr:.4f}, identical {100 * same:.3f}%")
        if best is None or corr > best[1]:
            best = ((sm, tps), corr, name)
    (sm, tps), corr, name = best
    if corr < 0.99:
        print("no GPU layout reproduces the stored noise")
        return
    print(f"best: {name}; all {len(stored)} stored poisons:")
    for seed, z in stored:
        g = cuda_normal(z.shape, seed, sms=sm, threads_per_sm=tps)
        d = np.abs(g - z)
        print(f"  seed {seed:5d}: identical {100 * np.mean(g == z):.3f}%, max |diff| {d.max():.2e}, "
              f"values differing {int((d > 0).sum())}")


if __name__ == "__main__":
    _self_test()
    if len(sys.argv) > 1:
        _compare(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
