"""
Shared building blocks for post-hoc unlearning trainers.

Two halves:

1. **Forget set** loading from `sbordt/OLMo-2-1B-Exp-Dataset`. The default
   forget set is the full dataset minus the `iid-replacements-*` controls
   (`iid-replacements-recency-{0..9}` and `iid-replacements-uniqueness`).
   Pass `experiments=[...]` to whitelist a single experiment for sweeps.

2. **Retain set** = OLMo-2 stage1 pretraining sequences ahead of the loaded
   checkpoint's training position, sampled via OLMo's `MemMapDataset` plus the
   same shuffled `global_indices` permutation that the OLMo trainer uses. This
   gives a retain stream identical to what the continued-training unlearning
   baselines (`stage1-step10{1,2,...}000`) see, modulo seed/order.

Used by `gradient_ascent.py`, `rmu.py`, `lunar.py`.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from pretrain_experiments.logging_config import get_logger

logger = get_logger(__name__)


DATASET_REPO = "sbordt/OLMo-2-1B-Exp-Dataset"
DATASET_SPLIT = "train"
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_SEED = 42

# IID-replacements experiments are i.i.d. control insertions, not unlearning
# targets. Excluded from the default forget set.
DEFAULT_EXCLUDED_EXPERIMENTS: Tuple[str, ...] = (
    *(f"iid-replacements-recency-{i}" for i in range(10)),
    "iid-replacements-uniqueness",
)


# ---------------------------------------------------------------------------
# Forget set
# ---------------------------------------------------------------------------


class ForgetDataset(Dataset):
    """Variable-length list of token-id sequences, truncated to `max_seq_len`."""

    def __init__(self, sequences: Sequence[Sequence[int]], max_seq_len: int):
        self.sequences = [list(s)[:max_seq_len] for s in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def collate_pad(batch, pad_id: int):
    """Right-pad a batch of variable-length sequences."""
    max_len = max(len(s) for s in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(batch):
        input_ids[i, : len(seq)] = seq
        attention_mask[i, : len(seq)] = 1
    return input_ids, attention_mask


def tokenize_and_strip(texts: Iterable[str], tokenizer,
                       batch_size: int = 1000) -> list[list[int]]:
    """Tokenize and strip leading/trailing EOS — mirrors insertion_likelihood.py.

    Tokenizes in batches rather than one text at a time. The forget set is
    millions of texts and this runs at the start of EVERY job, so the
    per-call overhead across the Python/Rust boundary dominated: batching
    measured ~3x faster single-core (more with several, since the fast
    tokenizer parallelises within a batch).

    Output is identical to the per-text form: `tokenizer.encode(text)` and
    `tokenizer([text])['input_ids'][0]` both default to
    add_special_tokens=True with no padding or truncation. Stripping is done
    with indices instead of repeated slicing, so each surviving sequence is
    copied once.
    """
    eos_id = tokenizer.eos_token_id
    texts = list(texts)
    out: list[list[int]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        for ids in tokenizer(chunk)["input_ids"]:
            a, b = 0, len(ids)
            while a < b and ids[a] == eos_id:
                a += 1
            while b > a and ids[b - 1] == eos_id:
                b -= 1
            if b > a:
                out.append(ids[a:b])
    return out


def load_forget_set(
    tokenizer,
    *,
    experiments: Optional[Sequence[str]] = None,
    exclude_experiments: Sequence[str] = DEFAULT_EXCLUDED_EXPERIMENTS,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> Tuple[ForgetDataset, dict]:
    """Load the OLMo-2-1B-Exp-Dataset forget set.

    Default = full dataset minus `iid-replacements-*` controls. Pass
    `experiments=[name, ...]` to restrict to a whitelist (used for single-
    experiment sweeps like the gradient-ascent phase-1 grid).

    Returns (dataset, info_dict).
    """
    import datasets

    logger.info(f"Loading {DATASET_REPO}...")
    ds = datasets.load_dataset(DATASET_REPO, split=DATASET_SPLIT)
    available = sorted(set(ds["experiment"]))
    logger.info(f"  {len(available)} experiments available, {len(ds)} rows")

    if experiments is not None:
        unknown = [e for e in experiments if e not in available]
        if unknown:
            raise SystemExit(
                f"Unknown experiment(s) {unknown!r}. Available: {available}"
            )
        keep = set(experiments)
        ds = ds.filter(lambda x: x["experiment"] in keep)
        logger.info(f"  whitelisted to {sorted(keep)}: {len(ds)} rows")

    if exclude_experiments:
        excl = set(exclude_experiments)
        # Don't crash if the excluded set isn't in `available` — they may simply
        # not appear in the user's whitelist or in a future dataset version.
        ds = ds.filter(lambda x: x["experiment"] not in excl)
        logger.info(f"  excluded {sorted(excl)}: {len(ds)} rows remain")

    texts = ds["text"]
    sequences = tokenize_and_strip(texts, tokenizer)
    info = {
        "n_sequences": len(sequences),
        "n_total_tokens": sum(len(s) for s in sequences),
        "max_seq_len_observed": max((len(s) for s in sequences), default=0),
        "experiments_in_set": sorted(set(ds["experiment"])),
        "max_seq_len_truncation": max_seq_len,
    }
    logger.info(
        f"  tokenized: {info['n_sequences']} non-empty sequences, "
        f"{info['n_total_tokens']} tokens, observed max_len={info['max_seq_len_observed']}"
    )
    return ForgetDataset(sequences, max_seq_len), info


# ---------------------------------------------------------------------------
# Retain set: unseen slice of the OLMo-2 stage1 stream
# ---------------------------------------------------------------------------


class OlmoRetainDataset(Dataset):
    """Indexes sequences in the OLMo memmap by a precomputed permutation.

    `indices[i]` is a sequence id in the underlying `MemMapDataset`. We slice
    the OLMo seq (length 4096) down to `max_seq_len` from the front; this
    keeps the leading documents in each packed chunk.
    """

    def __init__(self, memmap_dataset, indices: np.ndarray, max_seq_len: int):
        self._mmap = memmap_dataset
        self._indices = indices
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        return int(len(self._indices))

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq_id = int(self._indices[idx])
        item = self._mmap[seq_id]
        ids: torch.Tensor = item["input_ids"]
        if ids.shape[0] > self._max_seq_len:
            ids = ids[: self._max_seq_len]
        return ids.long()


def collate_olmo_retain(batch):
    """Stack uniform-length OLMo sequences and synthesize an all-1 attention mask."""
    input_ids = torch.stack(batch, dim=0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def build_olmo_retain_dataset(
    olmo_config_path: str,
    *,
    start_step: int,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    epoch: int = 0,
    seed_override: Optional[int] = None,
) -> Tuple[OlmoRetainDataset, dict]:
    """Build a retain dataset over the OLMo-2 stream past `start_step`.

    Replicates `olmo.data.IterableDataset._build_global_indices` for the case
    `shuffle=True` (PCG64 with `seed + epoch`), then drops the first
    `start_step * global_train_batch_size` sequence ids — those are the ones
    the loaded checkpoint has already seen during pretraining.

    Returns `(dataset, info_dict)`. `info_dict` records the OLMo config fields
    we depend on so they get logged into the run config snapshot.
    """
    from olmo.config import TrainConfig
    from olmo.data import build_memmap_dataset

    cfg = TrainConfig.load(olmo_config_path)
    # Don't write any per-run state into the OLMo save_folder.
    cfg.save_overwrite = True

    mmap = build_memmap_dataset(cfg, cfg.data)
    n_total = len(mmap)
    seq_len = cfg.model.max_sequence_length
    global_batch = cfg.global_train_batch_size
    seed = (
        seed_override
        if seed_override is not None
        else (cfg.data.seed if cfg.data.seed is not None else cfg.seed)
    )
    if seed is None:
        raise SystemExit(
            f"OLMo config {olmo_config_path} has neither data.seed nor seed; "
            f"pass seed_override or fix the config."
        )

    rng = np.random.Generator(np.random.PCG64(seed=seed + epoch))
    indices = np.arange(n_total, dtype=np.uint32)
    rng.shuffle(indices)

    skip = start_step * global_batch
    if skip >= n_total:
        raise SystemExit(
            f"start_step={start_step} × global_train_batch_size={global_batch} "
            f"= {skip} >= dataset length {n_total}; nothing left to sample."
        )
    unseen = indices[skip:]

    info = {
        "olmo_config_path": os.fspath(olmo_config_path),
        "olmo_seed": int(seed),
        "olmo_epoch": int(epoch),
        "olmo_max_sequence_length": int(seq_len),
        "olmo_global_train_batch_size": int(global_batch),
        "olmo_dataset_total_sequences": int(n_total),
        "start_step": int(start_step),
        "n_unseen_sequences": int(len(unseen)),
        "retain_max_seq_len": int(max_seq_len),
    }
    logger.info(
        f"OLMo retain stream: {info['n_unseen_sequences']} unseen seq ids "
        f"(skipped {skip} = {start_step}x{global_batch}); seq_len={seq_len}, "
        f"slicing to {max_seq_len}"
    )
    return OlmoRetainDataset(mmap, unseen, max_seq_len), info


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def save_hf_checkpoint(model, tokenizer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# Optimizer and schedule matched to the OLMo-2 pretraining trajectory.
#
# The unlearning runs continue the trajectory that produced the unlearning
# baseline, so they use its optimizer settings rather than the post-hoc
# unlearning literature's. Read from optim.pt of
# sbordt/OLMo-2-1B-Exp-Unlearning @ step100000-unsharded:
#
#     lr            3.9856e-4  (initial_lr 4e-4, cosine over 5T tokens)
#     betas         (0.9, 0.95)
#     eps           1e-8
#     weight_decay  0.1, EXCEPT the embedding matrix at 0.0
#     max_grad_norm 1.0
#     decoupled     True  (i.e. AdamW, not Adam)
#
# Gradient clipping matters more here than it looks: gradient ascent maximises
# -log p, so its per-token gradient carries a 1/p factor that is unbounded as
# p -> 0. Clipping at 1.0 is what keeps that finite, and the pretraining run
# has always had it.
# ---------------------------------------------------------------------------

OLMO2_1B_LR_AT_STEP_100K = 3.9855694839172363e-4
OLMO2_1B_BETAS = (0.9, 0.95)
OLMO2_1B_EPS = 1e-8
OLMO2_1B_WEIGHT_DECAY = 0.1
OLMO2_1B_MAX_GRAD_NORM = 1.0


def build_matched_optimizer(
    model,
    lr: float,
    params=None,
    weight_decay: float = OLMO2_1B_WEIGHT_DECAY,
    betas=OLMO2_1B_BETAS,
    eps: float = OLMO2_1B_EPS,
):
    """AdamW with the pretraining run's two parameter groups.

    The embedding matrix is excluded from weight decay (`decay_embeddings:
    false` in the OLMo config, and `weight_decay: 0.0` on the `wte.weight`
    group in optim.pt). Norms and biases ARE decayed (`decay_norm_and_bias:
    true`), so they stay in the main group.

    Parameters are matched by identity rather than name, which keeps this
    correct for tied embeddings and across HF/OLMo naming.

    `params` restricts the optimizer to a subset -- rmu updates only a few
    layers, so passing model.parameters() there would be wrong.
    """
    emb = model.get_input_embeddings()
    emb_ids = {id(p) for p in emb.parameters()} if emb is not None else set()

    decay, no_decay = [], []
    for p in (model.parameters() if params is None else params):
        if not p.requires_grad:
            continue
        (no_decay if id(p) in emb_ids else decay).append(p)

    groups = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas), eps=eps)


def build_linear_decay_schedule(optimizer, total_steps: int):
    """Linear decay from the initial LR to zero over `total_steps`.

    This is the schedule the RELEASED checkpoints got: from step 90,000 the
    OLMo-2 run decays linearly to zero over 10,000 steps, and mid-training does
    the same over 11,931 steps. Using it here means an unlearning run is a
    trajectory that finishes, rather than one truncated mid-cosine.
    """
    def factor(step):
        if total_steps <= 0:
            return 1.0
        return max(0.0, 1.0 - step / float(total_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------------------------------------------------------------------------
# Resuming the pretraining optimizer state (optim.pt -> HF AdamW).
#
# The unlearning runs continue a trajectory, so starting Adam from zeroed
# moments would spend the first few hundred steps rebuilding second-moment
# estimates the pretraining run already had -- a transient that looks exactly
# like "the method destabilised the model early".
#
# The obstacle is that OLMo and HF disagree on parameter layout:
#
#     OLMo   16 blocks x 8 tensors + 3 top-level = 131
#     HF     16 blocks x 11 tensors + 3 top-level = 179
#
# because OLMo fuses QKV into one `att_proj` (6144, 2048) and fuses the MLP's
# up/gate into one `ff_proj` (16384, 2048). Both counts are exact and there are
# no orphans on either side, so the correspondence is total.
#
# The moments are elementwise, so they slice along exactly the same boundaries
# as the weights -- PROVIDED the conversion does not permute. It does not:
# scripts/convert_olmo2_to_hf.py uses a plain torch.split / torch.chunk on
# dim 0, with no RoPE-style interleave. Two details from that script are easy
# to get backwards and would fail SILENTLY (training runs, from corrupted
# state), so they are pinned here and checked by verify_olmo_hf_mapping():
#
#   * att_proj splits q, k, v IN THAT ORDER.
#   * ff_proj chunks UP FIRST, THEN GATE -- the reverse of the usual
#     LLaMA convention:
#         up_proj_weight, gate_proj_weight = torch.chunk(ff_proj, 2, dim=0)
#
# The converter also hardcodes `num_key_value_heads = n_heads`, i.e. no GQA,
# which makes the three QKV pieces equal at hidden_size each. That assumption
# is asserted rather than assumed.
# ---------------------------------------------------------------------------

# Per-block 1:1 renames. The two norm entries are the ones worth double
# checking: OLMo-2 is post-norm, so attn_norm/ff_norm land on the "post_*"
# names, not on an input layernorm.
OLMO_TO_HF_BLOCK = {
    "attn_out.weight": "self_attn.o_proj.weight",
    "q_norm.weight": "self_attn.q_norm.weight",
    "k_norm.weight": "self_attn.k_norm.weight",
    "ff_out.weight": "mlp.down_proj.weight",
    "attn_norm.weight": "post_attention_layernorm.weight",
    "ff_norm.weight": "post_feedforward_layernorm.weight",
}

# Three top-level tensors. `transformer.ff_out.weight` (no block prefix) is the
# output head, distinct from the per-block `ff_out`; its presence as its own
# entry is what tells us lm_head is UNTIED from wte here.
OLMO_TO_HF_TOP = {
    "transformer.wte.weight": "model.embed_tokens.weight",
    "transformer.ln_f.weight": "model.norm.weight",
    "transformer.ff_out.weight": "lm_head.weight",
}

_FSDP_PREFIX = "_fsdp_wrapped_module."


def _normalise_olmo_key(key: str) -> str:
    """Strip FSDP wrapper segments and anything before `transformer.`.

    FSDP inserts `_fsdp_wrapped_module.` at every wrapped boundary, so it can
    appear more than once in a single key and not only at the front.
    """
    key = key.replace(_FSDP_PREFIX, "")
    i = key.find("transformer.")
    return key[i:] if i >= 0 else key


def olmo_qkv_split_sizes(config) -> list:
    """Sizes for splitting `att_proj` into q, k, v.

    convert_olmo2_to_hf.py builds these as

        fused_dims = [dim, dims_per_head * num_key_value_heads,
                           dims_per_head * num_key_value_heads]

    after setting `num_key_value_heads = n_heads`. With no GQA that is three
    equal parts. A GQA config would mean the converter itself was wrong for
    this model, so refuse rather than guess.
    """
    dim = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", None) or n_heads
    if n_kv != n_heads:
        raise ValueError(
            f"num_key_value_heads={n_kv} != num_attention_heads={n_heads}: this "
            "model uses GQA, but convert_olmo2_to_hf.py hardcodes "
            "num_key_value_heads = n_heads. Re-derive fused_dims from the "
            "converter actually used before trusting any moment mapping."
        )
    per_head = dim // n_heads
    return [dim, per_head * n_kv, per_head * n_kv]


def map_olmo_to_hf(lookup, n_layers: int, qkv_sizes) -> dict:
    """Apply the OLMo -> HF layout mapping to whatever `lookup` returns.

    `lookup(olmo_name) -> Tensor`. Used for both the weights (in
    verify_olmo_hf_mapping) and each Adam moment (in remap_olmo_optim_state),
    so a passing weight check validates the moment path by construction --
    there is only one copy of the mapping.
    """
    out = {}
    for olmo, hf in OLMO_TO_HF_TOP.items():
        out[hf] = lookup(olmo)

    for i in range(n_layers):
        pre = f"transformer.blocks.{i}."
        for suffix, hf_suffix in OLMO_TO_HF_BLOCK.items():
            out[f"model.layers.{i}.{hf_suffix}"] = lookup(pre + suffix)

        q, k, v = torch.split(lookup(pre + "att_proj.weight"), qkv_sizes, dim=0)
        out[f"model.layers.{i}.self_attn.q_proj.weight"] = q
        out[f"model.layers.{i}.self_attn.k_proj.weight"] = k
        out[f"model.layers.{i}.self_attn.v_proj.weight"] = v

        # UP first, GATE second. See the module comment above.
        up, gate = torch.chunk(lookup(pre + "ff_proj.weight"), 2, dim=0)
        out[f"model.layers.{i}.mlp.up_proj.weight"] = up
        out[f"model.layers.{i}.mlp.gate_proj.weight"] = gate

    return out


def load_olmo_optim_state(path: str) -> dict:
    """Read optim.pt and return {normalised OLMo name: {step, exp_avg, ...}}.

    Handles both str-keyed state (FSDP full_state_dict with use_orig_params)
    and int-keyed state that needs `param_names` from param_groups.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("state", blob) if isinstance(blob, dict) else blob

    keys = list(state.keys())
    if keys and not isinstance(keys[0], str):
        names = []
        for group in blob.get("param_groups", []):
            names.extend(group.get("param_names", []))
        if len(names) != len(keys):
            raise ValueError(
                f"{path}: {len(keys)} int-keyed state entries but {len(names)} "
                "param_names -- cannot recover the name mapping."
            )
        state = {n: state[k] for n, k in zip(names, sorted(keys))}

    return {_normalise_olmo_key(k): v for k, v in state.items()}


def remap_olmo_optim_state(olmo_state: dict, model) -> dict:
    """OLMo-keyed Adam state -> {HF parameter name: {step, exp_avg, exp_avg_sq}}."""
    config = model.config
    n_layers = config.num_hidden_layers
    qkv_sizes = olmo_qkv_split_sizes(config)

    def entry(name):
        if name not in olmo_state:
            raise KeyError(
                f"{name} missing from optim.pt (have {len(olmo_state)} entries). "
                "Expected 16*8+3 = 131 for OLMo-2 1B."
            )
        return olmo_state[name]

    moments = {}
    for field in ("exp_avg", "exp_avg_sq"):
        moments[field] = map_olmo_to_hf(
            lambda n, f=field: entry(n)[f], n_layers, qkv_sizes
        )

    # `step` is stored per parameter but is the same number for every one of
    # them in a synchronous Adam run, so it does NOT go through map_olmo_to_hf
    # -- it is a 0-dim scalar and has no split/chunk boundaries to follow.
    # Carry the source value through rather than resetting to 0: resetting
    # re-applies Adam's bias correction from scratch, which inflates the first
    # updates by ~1/(1-beta2) -- the opposite of continuing a trajectory.
    seen = set()
    for value in olmo_state.values():
        s = value.get("step", 0)
        seen.add(float(s.item() if torch.is_tensor(s) else s))
    if len(seen) > 1:
        raise ValueError(
            f"optim.pt has {len(seen)} distinct Adam step counters "
            f"({sorted(seen)[:5]}); expected one. This is not a synchronous "
            "resume and the assumption behind carrying `step` over is wrong."
        )
    step = seen.pop() if seen else 0.0

    shapes = {n: p.shape for n, p in model.named_parameters()}
    remapped = {}
    for name, exp_avg in moments["exp_avg"].items():
        if name not in shapes:
            raise KeyError(f"mapped to {name}, which is not a parameter of the model")
        if tuple(exp_avg.shape) != tuple(shapes[name]):
            raise ValueError(
                f"{name}: moment shape {tuple(exp_avg.shape)} != parameter shape "
                f"{tuple(shapes[name])}. The split/chunk boundaries are wrong."
            )
        remapped[name] = {
            "step": step,
            "exp_avg": exp_avg,
            "exp_avg_sq": moments["exp_avg_sq"][name],
        }

    missing = set(shapes) - set(remapped)
    if missing:
        raise ValueError(
            f"{len(missing)} parameters got no optimizer state, e.g. "
            f"{sorted(missing)[:5]}. Loading anyway would leave them at zeroed "
            "moments, which is the cold start we are trying to avoid."
        )
    return remapped


def load_matched_optimizer_state(optimizer, model, path: str, strict: bool = True):
    """Populate `optimizer.state` from an OLMo optim.pt.

    Writes state keyed by PARAMETER OBJECT rather than going through
    `Optimizer.load_state_dict`, which matches entries by position within
    param_groups and would therefore depend on build_matched_optimizer's
    decay/no-decay ordering happening to agree with the pretraining run's.
    Keying by object removes that coupling, and makes the rmu case (optimizer
    holds only a subset of parameters) work without special handling.

    `path` may be an optim.pt or the unsharded checkpoint directory holding one.

    Moments are cast to each parameter's dtype and device. The drivers load in
    float32, matching the checkpoint, so this is a no-op there; under bf16 it
    would cost precision, which is worth knowing but not worth refusing.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "optim.pt")
    remapped = remap_olmo_optim_state(load_olmo_optim_state(path), model)
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}

    loaded, skipped = 0, 0
    for name, param in model.named_parameters():
        if id(param) not in owned:
            skipped += 1  # not being optimised, e.g. rmu's frozen layers
            continue
        entry = remapped.get(name)
        if entry is None:
            if strict:
                raise KeyError(f"no optimizer state for {name}")
            continue
        optimizer.state[param] = {
            "step": torch.tensor(entry["step"], dtype=torch.float32),
            "exp_avg": entry["exp_avg"].to(device=param.device, dtype=param.dtype),
            "exp_avg_sq": entry["exp_avg_sq"].to(device=param.device, dtype=param.dtype),
        }
        loaded += 1

    step = next(iter(remapped.values()))["step"]
    logger.info(
        "resumed optimizer state from %s: %d parameters loaded, %d not optimised, "
        "Adam step counter %g",
        path, loaded, skipped, step,
    )
    return loaded


def verify_olmo_hf_mapping(model_pt_path: str, model, atol: float = 0.0) -> dict:
    """Check the layout mapping against real weights before trusting it on moments.

    Applies `map_olmo_to_hf` to the tensors in an OLMo `model.pt` and compares
    the result to the HF model's own weights. Because the moment path uses the
    same mapping function, agreement here is direct evidence that the moments
    land on the right parameters -- in particular that up/gate are not swapped
    and that q/k/v are in that order.

    Requires that the HF model was converted from THIS checkpoint. Comparing
    against a different branch (e.g. the LR-annealed
    stage1-step100000-tokens210B) reports differences everywhere and means
    nothing.
    """
    loaded = torch.load(model_pt_path, map_location="cpu", weights_only=False)
    loaded = {_normalise_olmo_key(k): v for k, v in loaded.items()}
    qkv_sizes = olmo_qkv_split_sizes(model.config)
    mapped = map_olmo_to_hf(
        lambda n: loaded[n], model.config.num_hidden_layers, qkv_sizes
    )

    hf = dict(model.named_parameters())
    report = {
        "n_checked": 0,
        "n_shape_mismatch": 0,
        "n_value_mismatch": 0,
        "max_abs_diff": 0.0,
        "worst": None,
        "missing": [],
        "mismatched": [],
    }

    for name, tensor in mapped.items():
        if name not in hf:
            report["missing"].append(name)
            continue
        ref = hf[name].detach().float()
        got = tensor.float()
        report["n_checked"] += 1
        if got.shape != ref.shape:
            report["n_shape_mismatch"] += 1
            report["mismatched"].append(name)
            continue
        diff = (got - ref).abs().max().item()
        if diff > report["max_abs_diff"]:
            report["max_abs_diff"] = diff
            report["worst"] = name
        if diff > atol:
            report["n_value_mismatch"] += 1
            report["mismatched"].append(name)

    report["ok"] = (
        report["n_checked"] == len(hf)
        and not report["missing"]
        and report["n_shape_mismatch"] == 0
        and report["n_value_mismatch"] == 0
    )
    return report


def _verify_cli(argv=None):
    """`python -m pretrain_experiments.unlearning_utils --olmo-dir ... --model ...`

    Run this ONCE on the cluster before any 10k-step run. It answers the only
    question that cannot be settled by reading the converter: whether this
    mapping, applied to real tensors, reproduces the HF weights exactly.
    """
    import argparse

    from transformers import AutoModelForCausalLM

    ap = argparse.ArgumentParser(description="verify the OLMo -> HF layout mapping")
    ap.add_argument("--olmo-dir", default=None,
                    help="unsharded checkpoint dir holding model.pt and optim.pt. "
                         "Omit to pull them from the hub instead, which is the "
                         "easier path when they are only in the HF cache.")
    ap.add_argument("--model", required=True,
                    help="HF model id, or (usually) the local directory produced "
                         "by OLMo/scripts/convert_olmo2_to_hf.py. The "
                         "step100000-unsharded branch is OLMo-native and has no "
                         "HF weights, so there is nothing on the hub to compare "
                         "against -- convert first.")
    ap.add_argument("--revision", default="step100000-unsharded",
                    help="Branch to compare. Must be the SAME checkpoint the "
                         "model.pt came from -- the released "
                         "stage1-step100000-tokens210B branch has had 10k steps "
                         "of LR annealing applied and will differ everywhere.")
    ap.add_argument("--atol", type=float, default=0.0,
                    help="0 means bit-exact; the conversion is a pure slice, so "
                         "it should be")
    args = ap.parse_args(argv)

    if args.olmo_dir:
        model_pt = os.path.join(args.olmo_dir, "model.pt")
        optim_pt = os.path.join(args.olmo_dir, "optim.pt")
    else:
        # Resolves through HF_HOME, so an already-downloaded revision is reused
        # rather than re-fetched. model.pt is ~5 GB in fp32.
        from huggingface_hub import hf_hub_download

        print(f"fetching model.pt / optim.pt from {args.model} @ {args.revision}")
        model_pt = hf_hub_download(args.model, "model.pt", revision=args.revision)
        try:
            optim_pt = hf_hub_download(args.model, "optim.pt", revision=args.revision)
        except Exception as exc:  # optional: the weight check alone is still useful
            print(f"  optim.pt unavailable ({type(exc).__name__}); "
                  "checking weights only")
            optim_pt = ""

    # A converted checkpoint is a local directory and has no revision. Passing
    # one anyway is not merely ignored by from_pretrained -- it errors.
    revision = None if os.path.isdir(args.model) else args.revision
    print(f"HF model:  {args.model} @ {revision or 'local'}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, torch_dtype=torch.float32,
    )
    n_params = sum(1 for _ in model.named_parameters())
    print(f"           {n_params} parameters, {model.config.num_hidden_layers} layers")
    print(f"qkv split: {olmo_qkv_split_sizes(model.config)}")

    print(f"\nweights:   {model_pt}")
    report = verify_olmo_hf_mapping(model_pt, model, atol=args.atol)
    print(f"           checked {report['n_checked']}/{n_params}")
    print(f"           max abs diff {report['max_abs_diff']:.3e} "
          f"(worst: {report['worst']})")
    if report["missing"]:
        print(f"           MISSING {len(report['missing'])}: {report['missing'][:5]}")
    if report["mismatched"]:
        print(f"           MISMATCHED {len(report['mismatched'])}: "
              f"{report['mismatched'][:8]}")
    print(f"           {'PASS' if report['ok'] else 'FAIL'}")

    if not report["ok"]:
        print("\nStop here. A mismatch confined to mlp.up_proj/mlp.gate_proj means "
              "the chunk order is reversed; one confined to q/k/v means the split "
              "order is. Either way the moments would land on the wrong weights.")
        return 1

    if os.path.exists(optim_pt):
        print(f"\noptimizer: {optim_pt}")
        olmo_state = load_olmo_optim_state(optim_pt)
        print(f"           {len(olmo_state)} OLMo entries")
        remapped = remap_olmo_optim_state(olmo_state, model)
        step = next(iter(remapped.values()))["step"]
        print(f"           -> {len(remapped)} HF entries, Adam step {step:g}")
        print("           PASS")
    else:
        print(f"\noptimizer: {optim_pt} not found -- weights verified, state not")

    return 0


if __name__ == "__main__":
    raise SystemExit(_verify_cli())
