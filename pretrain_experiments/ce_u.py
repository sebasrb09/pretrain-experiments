"""
CE-U (Cross Entropy Unlearning) on OLMo-2 checkpoints.

Replaces gradient ascent's sign-flipped CE with ordinary cross-entropy
*descent* toward a surrogate target built from the model's own logits with
the ground-truth token suppressed:

    z_i,CE-U = -∞  if i = y,  else  z_i
    p_CE-U   = softmax(z_CE-U)          (detached — no gradient flows into it)
    L_CE-U   = KL( p_CE-U ‖ p_θ )

Because the target is the model's own next-token distribution renormalized
over everything *except* the true token, the loss is bounded and its
gradient neither vanishes at high confidence nor explodes at low confidence
— the two failure modes of gradient ascent that motivate the method.

What this method does NOT need: a retain set, a frozen reference model, or
additional positive samples. It has **no method-specific hyperparameters**
at all — only the usual learning rate / batch size / optimizer settings.
That also makes it the only unlearning driver here that never imports
`olmo`: it runs anywhere `torch` + `transformers` are installed, with no
OLMo TrainConfig and no retain stream to build.

Implementation note: this KL has a closed form. `p_CE-U` is just `p_θ`
renormalized over everything except the true token,

    p_CE-U,i = q_i / (1 − q_y)   for i ≠ y,       p_CE-U,y = 0

so every surviving log-ratio term equals −log(1 − q_y) and the weights sum
to 1, giving

    L_CE-U = KL( p_CE-U ‖ p_θ ) = −log( 1 − q_y )

with `q_y` the probability the model still assigns to the true token. The
detached-target gradient coincides *exactly* with the gradient of this
closed form — differentiating through the target adds a term proportional
to Σ_i ∂q_i/∂θ = 0 — so we compute it straight from the per-token
cross-entropy (`q_y = exp(−ce)`) and never materialize the (B, T, V) target
distribution, which at batch 4 × 1024 tokens × ~100k vocab would cost
~1.6 GB of fp32 per forward. Value and gradient were both checked against
the explicit KL construction.

Practical warning for this repo's forget set: the gradient carries a factor
1/(1 − q_y). That is the intended non-vanishing behaviour at high
confidence, but it explodes on perfectly memorized tokens (q_y = 0.999 →
×1e3; q_y = 1 − 1e-7 → ×1e7), and canaries inserted many times during
pretraining sit far closer to q_y = 1 than TOFU's answers do. The
per-token loss is clamped via `--min-forget-ce`, which bounds the factor at
1/eps; raise it above the faithful default if the first steps diverge.

All model parameters are updated (full fine-tune; no LoRA).

Usage:
    python -m pretrain_experiments.ce_u \
        --model sbordt/OLMo-2-179M-Exp-Unlearning \
        --revision stage1-step100000-tokens210B \
        --learning-rate 1e-5 \
        --batch-size 4 \
        --epochs 1 \
        --output-dir /path/to/unlearning-ce-u/run-<id>

Reference: https://arxiv.org/abs/2503.01224
"""

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from pretrain_experiments.logging_config import get_logger
from pretrain_experiments.unlearning_utils import (
    OLMO2_1B_BETAS,
    OLMO2_1B_LR_AT_STEP_100K,
    OLMO2_1B_MAX_GRAD_NORM,
    OLMO2_1B_WEIGHT_DECAY,
    build_linear_decay_schedule,
    build_matched_optimizer,
    load_matched_optimizer_state,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_SEED,
    collate_pad,
    load_forget_set,
    save_hf_checkpoint,
)

logger = get_logger(__name__)

MAX_EPOCHS_CAP = 20
DEFAULT_CHECKPOINT_EVERY = 1


def _ce_u_loss(model, input_ids, attention_mask, min_forget_ce):
    """CE-U loss over non-pad next-token targets.

    Uses the closed form `L = -log(1 - q_y)` derived in the module docstring,
    with `q_y = exp(-ce)` taken from the per-token cross-entropy.

    Returns `(loss, ce_forget, p_true_mean)`, where `ce_forget` is the plain
    cross-entropy on the same tokens (logged for cross-method comparability
    with the gradient-ascent family) and `p_true_mean` is the mean
    probability the model still assigns to the true tokens.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # (B, T, V)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous().float()

    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(shift_labels.shape)  # (B, T-1), = -log q_y

    # L = -log(1 - exp(-ce)). `-expm1(-ce)` evaluates 1 - q_y accurately as
    # q_y -> 1, where a plain `1 - exp(-ce)` would cancel to zero. The clamp
    # keeps ce off 0 exactly (q_y == 1 in fp32), which would give log(0).
    ce_safe = ce.clamp_min(min_forget_ce)
    per_token = -torch.log(-torch.expm1(-ce_safe))

    denom = shift_mask.sum().clamp_min(1.0)
    loss = (per_token * shift_mask).sum() / denom

    with torch.no_grad():
        ce_forget = (ce * shift_mask).sum() / denom
        p_true_mean = (torch.exp(-ce) * shift_mask).sum() / denom

    return loss, ce_forget, p_true_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Model / IO
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metrics-jsonl", type=str, default=None)

    # Forget set
    parser.add_argument("--forget-experiments", type=str, nargs="*", default=None,
                        help="Whitelist of experiments to use as the forget set. "
                             "Default: all experiments minus iid-replacements-*.")

    # Numerics (CE-U has no method-specific hyperparameters of its own)
    parser.add_argument("--min-forget-ce", type=float, default=1e-7,
                        help="Floor on the per-token forget CE before forming "
                             "-log(1-q). Caps the per-token loss at -log(eps) and "
                             "the 1/(1-q) gradient factor at 1/eps. Default 1e-7 is "
                             "the faithful setting; raise it (e.g. 1e-3) if heavily "
                             "memorized tokens blow up the first steps.")

    # Optimization
    parser.add_argument("--learning-rate", type=float,
                        default=OLMO2_1B_LR_AT_STEP_100K,
                        help="Defaults to the trajectory LR at step 100000.")
    parser.add_argument("--weight-decay", type=float,
                        default=OLMO2_1B_WEIGHT_DECAY,
                        help="Matched to the pretraining run; the embedding "
                             "matrix is excluded from decay automatically.")
    parser.add_argument("--resume-optimizer-state", type=str, default=None,
                        help="Path to the pretraining optim.pt, or the unsharded checkpoint directory holding it. Resumes Adam's moments so the run continues the pretraining trajectory instead of spending its first few hundred steps rebuilding second-moment estimates.")
    parser.add_argument("--betas", type=float, nargs=2, default=OLMO2_1B_BETAS,
                        help="Adam betas; pretraining used (0.9, 0.95).")
    parser.add_argument("--max-grad-norm", type=float,
                        default=OLMO2_1B_MAX_GRAD_NORM,
                        help="Gradient-norm clip. The pretraining run clips at 1.0; without it gradient ascent's 1/p factor is unbounded.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1,
                        help=f"Passes over the forget set (capped at {MAX_EPOCHS_CAP}).")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional optimizer-step cap.")
    parser.add_argument("--checkpoint-every-n-epochs", type=int,
                        default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=2000,
                        help="Save every N optimizer steps, giving the "
                             "trajectory its points. 0 disables.")

    # System
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16"],
                        default="float32",
                        help="Compute dtype. float32: full precision throughout. "
                             "bfloat16: autocast forward/backward in bf16; weights and "
                             "Adam states stay fp32. The CE-U target and KL are always "
                             "formed in fp32 regardless.")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    args = parser.parse_args()

    if args.epochs > MAX_EPOCHS_CAP:
        raise SystemExit(
            f"--epochs {args.epochs} exceeds the configured cap of {MAX_EPOCHS_CAP}; "
            f"raise MAX_EPOCHS_CAP in ce_u.py if you really mean to."
        )
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(args.metrics_jsonl) if args.metrics_jsonl else output_dir / "metrics.jsonl"
    )

    # ---- Model ---------------------------------------------------------
    logger.info(f"Loading model {args.model} (revision={args.revision})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32,
    ).to(args.device)
    model.train()

    if args.gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # ---- Forget loader -------------------------------------------------
    forget_dataset, forget_info = load_forget_set(
        tokenizer,
        experiments=args.forget_experiments,
        max_seq_len=args.max_seq_len,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    g = torch.Generator()
    g.manual_seed(args.seed)
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        collate_fn=lambda b: collate_pad(b, pad_id),
        drop_last=False,
    )

    # ---- Optimizer -----------------------------------------------------
    optimizer = build_matched_optimizer(
        model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
    )
    # Linear decay to zero over the run, matching the schedule the released
    # checkpoints got (step 90k -> 100k) and mid-training uses.
    scheduler = build_linear_decay_schedule(optimizer, args.max_steps or 0)

    # Resume Adam's moments from the pretraining checkpoint. Without this the
    # first steps run on zeroed second moments, so every parameter takes a
    # near-maximal step regardless of method -- an artefact that looks exactly
    # like early instability caused by the unlearning loss itself.
    if args.resume_optimizer_state:
        load_matched_optimizer_state(
            optimizer, model, args.resume_optimizer_state)

    config_record = {
        "method": "ce_u",
        "model": args.model,
        "revision": args.revision,
        "forget_experiments": args.forget_experiments,
        "forget_set_info": forget_info,
        "min_forget_ce": args.min_forget_ce,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "checkpoint_every_n_epochs": args.checkpoint_every_n_epochs,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "dtype": args.dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "micro_batches_per_epoch": len(forget_loader),
    }
    with open(output_dir / "ce_u_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    metrics_f = open(metrics_path, "w")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    logger.info(
        f"Starting CE-U: lr={args.learning_rate}, batch={args.batch_size}, "
        f"accum={args.gradient_accumulation_steps}, "
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps}, "
        f"epochs={args.epochs}, micro_batches/epoch={len(forget_loader)}, "
        f"dtype={args.dtype}"
    )

    # Wall clock for the metrics rows: lets throughput be measured without
    # model load and tokenization folded in, and gives a live ETA.
    t_start = time.perf_counter()
    optimizer_step = 0
    micro_step = 0
    stopped = False

    for epoch in range(1, args.epochs + 1):
        if stopped:
            break
        epoch_ce_sum = 0.0
        n_micro_batches = 0
        optimizer.zero_grad()
        for input_ids, attention_mask in forget_loader:
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)

            with autocast_ctx:
                loss_ceu, ce_forget, p_true_mean = _ce_u_loss(
                    model, input_ids, attention_mask, args.min_forget_ce
                )
                loss = loss_ceu / args.gradient_accumulation_steps

            loss.backward()
            micro_step += 1

            ce_val = ce_forget.item()
            epoch_ce_sum += ce_val
            n_micro_batches += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                # Step-based checkpoints. The run is far shorter than one
                # epoch (10k steps against ~10,249 per epoch), so epoch
                # boundaries would yield a single checkpoint at the end and
                # no trajectory to plot.
                if (args.checkpoint_every_n_steps
                        and optimizer_step % args.checkpoint_every_n_steps == 0):
                    ckpt_dir = output_dir / f"step-{optimizer_step}"
                    logger.info(f"  saving checkpoint to {ckpt_dir}")
                    save_hf_checkpoint(model, tokenizer, str(ckpt_dir))

            metrics_f.write(json.dumps({
                "epoch": epoch,
                "micro_step": micro_step,
                "optimizer_step": optimizer_step,
                "elapsed_s": round(time.perf_counter() - t_start, 3),
                "loss_ce_u": float(loss_ceu.detach().item()),
                "ce_forget": ce_val,
                "p_true_mean": float(p_true_mean.item()),
            }) + "\n")
            metrics_f.flush()

            if args.max_steps is not None and optimizer_step >= args.max_steps:
                logger.info(f"Reached --max-steps {args.max_steps}; stopping early.")
                stopped = True
                break

        # Flush a partial accumulation cycle at epoch boundary so steps stay aligned.
        if micro_step % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step += 1

        avg_ce = epoch_ce_sum / max(n_micro_batches, 1)
        logger.info(
            f"epoch {epoch}/{args.epochs} done: avg CE on forget = {avg_ce:.4f} "
            f"(opt_steps={optimizer_step})"
        )

        if epoch % args.checkpoint_every_n_epochs == 0 or epoch == args.epochs or stopped:
            ckpt_dir = output_dir / f"epoch-{epoch}"
            logger.info(f"  saving checkpoint to {ckpt_dir}")
            save_hf_checkpoint(model, tokenizer, str(ckpt_dir))

    metrics_f.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
