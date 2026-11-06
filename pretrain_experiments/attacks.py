"""Two attacks that test whether forgetting removed the data or only hid it (2026-11-06, rebuttal experiment 1).

relearn   Fine-tune briefly on a seeded sample of the forget set with the ordinary language-modelling loss: the
          relearning attack of Hu et al. (2025) and Fan et al. (2025). Every model gets the same sample, steps and
          learning rate. Retraining, which never saw the data, is the control: what it reaches is what the sample
          alone teaches, so a model that ends well above it held on to the data.
quantize  Round every linear layer's weights to b bits (round-to-nearest, asymmetric, groups along the input
          dimension), the attack of Zhang et al. (2025): an edit smaller than one quantization step can round
          away. The embeddings and the output layer stay unquantized, as in weight-only quantization.

Both write Hugging Face folders, evaluated like any model (internal/uwiki/hpo/optuna_hpo.py --eval-models).

    python -m pretrain_experiments.attacks relearn --model M [--revision R] --output-dir OUT \
        [--n-texts 512] [--steps 50] [--checkpoint-steps 10,50] [--learning-rate 1e-5]
    python -m pretrain_experiments.attacks quantize --model M [--revision R] --output-dir OUT [--bits 4]
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pretrain_experiments.logging_config import get_logger
from pretrain_experiments.unlearning_utils import (
    DEFAULT_MAX_SEQ_LEN,
    OLMO2_1B_BETAS,
    OLMO2_1B_EPS,
    OLMO2_1B_MAX_GRAD_NORM,
    collate_pad,
    load_forget_set,
    parse_checkpoint_steps,
    save_hf_checkpoint,
)

logger = get_logger(__name__)


def _record(out: Path, args) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "attack.json").write_text(json.dumps(vars(args), indent=2, default=str))


def relearn(args) -> None:
    out = Path(args.output_dir)
    _record(out, args)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(args.device).train()

    data, info = load_forget_set(tokenizer, experiments=args.forget_experiments, max_seq_len=args.max_seq_len)
    # one sample for every model: indices drawn with a fixed seed, independent of the model
    idx = sorted(random.Random(args.sample_seed).sample(range(len(data)), min(args.n_texts, len(data))))
    loader = DataLoader(Subset(data, idx), batch_size=args.micro_batch, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=lambda b: collate_pad(b, pad_id))
    if args.batch_size % args.micro_batch:
        raise SystemExit("--batch-size must be a multiple of --micro-batch")
    accum = args.batch_size // args.micro_batch
    logger.info(f"relearning {args.model}@{args.revision}: {len(idx)} of {len(data)} forget sequences, "
                f"{args.steps} steps of {args.batch_size}, lr {args.learning_rate}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=OLMO2_1B_BETAS,
                            eps=OLMO2_1B_EPS, weight_decay=0.0)
    keep = set(parse_checkpoint_steps(args.checkpoint_steps)) | {args.steps}
    step = micro = 0
    while step < args.steps:
        for input_ids, mask in loader:
            input_ids, mask = input_ids.to(args.device), mask.to(args.device)
            labels = input_ids.masked_fill(mask == 0, -100)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device != "cpu"):
                loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            (loss / accum).backward()
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), OLMO2_1B_MAX_GRAD_NORM)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            logger.info(f"  step {step}: loss {loss.item():.4f} (last micro-batch)")
            if step in keep:
                save_hf_checkpoint(model, tokenizer, str(out / f"step-{step}"))
                logger.info(f"  saved {out / f'step-{step}'}")
            if step >= args.steps:
                break


@torch.no_grad()
def quantize(args) -> None:
    out = Path(args.output_dir)
    _record(out, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32)
    top = 2**args.bits - 1
    done = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name.split(".")[-1] == "lm_head":
            continue
        w = module.weight.data
        rows, cols = w.shape
        g = args.group_size if cols % args.group_size == 0 else cols
        wg = w.reshape(rows, cols // g, g)
        lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
        scale = ((hi - lo) / top).clamp(min=1e-12)
        zero = torch.round(-lo / scale)
        q = torch.clamp(torch.round(wg / scale) + zero, 0, top)
        module.weight.data = ((q - zero) * scale).reshape(rows, cols)
        done += 1
    logger.info(f"quantized {done} linear layers to {args.bits} bits in groups of {args.group_size}")
    save_hf_checkpoint(model, tokenizer, str(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="attack", required=True)
    for name in ("relearn", "quantize"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="a local folder or a Hub repository")
        p.add_argument("--revision", default=None)
        p.add_argument("--output-dir", required=True)
        if name == "relearn":
            p.add_argument("--forget-experiments", nargs="*", default=None,
                           help="insertion experiments to sample from (default: the full forget set)")
            p.add_argument("--n-texts", type=int, default=512, help="forget sequences in the sample")
            p.add_argument("--sample-seed", type=int, default=0, help="the sample; keep it equal across models")
            p.add_argument("--steps", type=int, default=50)
            p.add_argument("--checkpoint-steps", default="10,50")
            p.add_argument("--batch-size", type=int, default=32, help="sequences per optimizer step")
            p.add_argument("--micro-batch", type=int, default=1)
            p.add_argument("--learning-rate", type=float, default=1e-5)
            p.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--device", default="cuda")
            p.add_argument("--gradient-checkpointing", action="store_true")
        else:
            p.add_argument("--bits", type=int, default=4)
            p.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()
    relearn(args) if args.attack == "relearn" else quantize(args)


if __name__ == "__main__":
    main()
