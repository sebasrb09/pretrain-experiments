"""
Gradient Difference unlearning (GradDiff) on OLMo-2 checkpoints.

Gradient ascent on the forget set, regularized by ordinary cross-entropy
descent on a retain set. Introduced as "Gradient Difference" for continual
learning / private unlearning (Liu et al., CoLLAs 2022) and adopted as the
canonical retain-regularized baseline throughout the LLM-unlearning
literature (e.g. the TOFU benchmark, Maini et al. 2024).

Loss:

    L_GradDiff(θ) = -CE_forget(θ) + α · CE_retain(θ)

Both terms are mean cross-entropy over non-pad next-token targets. The
forget term is `gradient_ascent.py`'s objective verbatim; the retain term is
`simnpo.py`'s, over the same OLMo-2 stage1 unseen slice.

Setting `--retain-loss-weight 0` reduces this exactly to `gradient_ascent.py`
(modulo AdamW-vs-Adam, which are identical at the default
`--weight-decay 0`), so the same script doubles as the ablation control for
the retain term.

The knob that makes GradDiff harder to tune than NPO/SimNPO: -CE_forget is
unbounded below and has no saturation point, so it keeps pushing after the
forget data is forgotten. α is the only thing holding the model together,
and the ratio of the two gradients drifts over a run rather than staying
fixed. Watch `ce_forget` and `loss_retain` in metrics.jsonl — a cell where
`ce_forget` climbs without bound while `loss_retain` also climbs has already
diverged, however good the forget-side eval looks.

All model parameters are updated (full fine-tune; no LoRA).

Usage:
    python -m pretrain_experiments.grad_diff \
        --model sbordt/OLMo-2-179M-Exp-Unlearning \
        --revision stage1-step100000-tokens210B \
        --olmo-config "$OLMO_REPO/configs/official-0425/OLMo2-1B-stage1.yaml" \
        --retain-start-step 100000 \
        --retain-loss-weight 1.0 \
        --learning-rate 1e-6 \
        --forget-batch-size 4 --retain-batch-size 4 \
        --epochs 1 \
        --output-dir /path/to/unlearning-graddiff/run-<id>

References:
    Liu et al., Continual Learning and Private Unlearning (CoLLAs 2022)
        https://arxiv.org/abs/2203.12817
    Maini et al., TOFU: A Task of Fictitious Unlearning for LLMs (2024)
        https://arxiv.org/abs/2401.06121
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
    build_olmo_retain_dataset,
    collate_olmo_retain,
    collate_pad,
    load_forget_set,
    save_hf_checkpoint,
)

logger = get_logger(__name__)

MAX_EPOCHS_CAP = 20
DEFAULT_CHECKPOINT_EVERY = 1


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _avg_ce(model, input_ids, attention_mask):
    """Mean CE over non-pad next-token targets in the batch."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # (B, T, V)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous().to(shift_logits.dtype)
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(shift_labels.shape)
    return (ce * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)


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

    # Retain set
    parser.add_argument("--olmo-config", type=str, default=None,
                        help="Path to the OLMo TrainConfig YAML used to build the retain stream.")
    parser.add_argument("--retain-start-step", type=int, default=None,
                        help="Skip the first N training steps' worth of OLMo sequences "
                             "(should match the step the loaded checkpoint represents).")
    parser.add_argument("--retain-seed-override", type=int, default=None,
                        help="Override the OLMo data seed (default: use seed from --olmo-config).")

    # GradDiff hyperparams
    parser.add_argument("--retain-loss-weight", type=float, default=1.0,
                        help="α on the retain CE term. 0.0 disables the retain pass "
                             "entirely, reducing this to plain gradient ascent.")

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
                        help="Gradient-norm clip. The pretraining run clips at "
                             "1.0; without it gradient ascent's 1/p factor is "
                             "unbounded.")
    parser.add_argument("--forget-batch-size", type=int, default=4)
    parser.add_argument("--retain-batch-size", type=int, default=4)
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
                             "Adam states stay fp32 (mixed-precision pattern).")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    args = parser.parse_args()

    if args.epochs > MAX_EPOCHS_CAP:
        raise SystemExit(
            f"--epochs {args.epochs} exceeds the configured cap of {MAX_EPOCHS_CAP}; "
            f"raise MAX_EPOCHS_CAP in grad_diff.py if you really mean to."
        )
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be >= 1")
    if args.retain_loss_weight < 0:
        raise SystemExit("--retain-loss-weight must be >= 0")

    use_retain = args.retain_loss_weight > 0
    # These two are only needed when the retain stream is actually built. Keeping
    # them argparse-optional (matching reweighted_ga.py) is what lets
    # RETAIN_WEIGHT=0 run this method forget-only on a cluster with no OLMo
    # memmap data -- with required=True, argparse rejected the call before
    # use_retain was ever consulted.
    if use_retain and (args.olmo_config is None or args.retain_start_step is None):
        raise SystemExit(
            "--retain-loss-weight > 0 requires both --olmo-config and "
            "--retain-start-step (the retain stream cannot be built without them)."
        )

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
    # Always load weights in fp32; autocast handles bf16 forward/backward when requested.
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
        batch_size=args.forget_batch_size,
        shuffle=True,
        generator=g,
        collate_fn=lambda b: collate_pad(b, pad_id),
        drop_last=False,
    )

    # ---- Retain loader (optional) --------------------------------------
    retain_info = None
    retain_iter = None
    if use_retain:
        retain_dataset, retain_info = build_olmo_retain_dataset(
            args.olmo_config,
            start_step=args.retain_start_step,
            max_seq_len=args.max_seq_len,
            seed_override=args.retain_seed_override,
        )
        retain_loader = DataLoader(
            retain_dataset,
            batch_size=args.retain_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + 1),
            collate_fn=collate_olmo_retain,
            drop_last=True,
            num_workers=0,
        )
        retain_iter = _infinite(retain_loader)
    else:
        logger.info(
            "retain_loss_weight=0 -> skipping retain loader; this run is plain "
            "gradient ascent."
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
        "method": "grad_diff",
        "model": args.model,
        "revision": args.revision,
        "forget_experiments": args.forget_experiments,
        "forget_set_info": forget_info,
        "retain_set_info": retain_info,
        "retain_loss_weight": args.retain_loss_weight,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "forget_batch_size": args.forget_batch_size,
        "retain_batch_size": args.retain_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "checkpoint_every_n_epochs": args.checkpoint_every_n_epochs,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "dtype": args.dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "micro_batches_per_epoch": len(forget_loader),
    }
    with open(output_dir / "graddiff_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    metrics_f = open(metrics_path, "w")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    logger.info(
        f"Starting GradDiff: lr={args.learning_rate}, "
        f"alpha_retain={args.retain_loss_weight}{' (disabled)' if not use_retain else ''}, "
        f"forget_bs={args.forget_batch_size}, retain_bs={args.retain_batch_size}, "
        f"accum={args.gradient_accumulation_steps}, epochs={args.epochs}, "
        f"micro_batches/epoch={len(forget_loader)}, dtype={args.dtype}"
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
        for forget_input_ids, forget_attn in forget_loader:
            forget_input_ids = forget_input_ids.to(args.device)
            forget_attn = forget_attn.to(args.device)

            with autocast_ctx:
                # Forget loss: gradient ascent (sign-flipped CE)
                ce_forget = _avg_ce(model, forget_input_ids, forget_attn)
                loss_forget = -ce_forget

                # Retain loss: standard CE on the OLMo unseen slice
                if use_retain:
                    retain_input_ids, retain_attn = next(retain_iter)
                    retain_input_ids = retain_input_ids.to(args.device)
                    retain_attn = retain_attn.to(args.device)
                    loss_retain = _avg_ce(model, retain_input_ids, retain_attn)
                else:
                    loss_retain = torch.zeros((), device=args.device, dtype=loss_forget.dtype)

                loss = loss_forget + args.retain_loss_weight * loss_retain
                loss = loss / args.gradient_accumulation_steps

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
                "ce_forget": ce_val,
                "loss_forget": float(loss_forget.detach().item()),
                "loss_retain": float(loss_retain.detach().item()),
                "loss_total": float(
                    (loss_forget + args.retain_loss_weight * loss_retain).detach().item()
                ),
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
