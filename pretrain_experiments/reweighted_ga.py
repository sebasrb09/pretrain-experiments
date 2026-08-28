"""
Reweighted gradient ascent — covers both WGA and SatImp.

Both methods minimize the same token-reweighted ascent objective over the
forget set,

    min_θ  E_{s ~ D_u}  Σ_i  w_i · log p(s_i | s_<i ; θ)

and differ only in the weight `w_i` (and in whether a retain term is used):

    WGA      w_i = p_i^α                        (Wang et al., arXiv:2502.19301)
    SatImp   w_i = p_i^β₁ · (1 − p_i)^β₂        (arXiv:2505.11953)

where `p_i = p(s_i | s_<i ; θ)` is the model's current probability on the
true token. WGA is exactly SatImp at β₂ = 0, so one driver covers both;
`--method-label` is recorded in the config snapshot so sweeps and results
tables still separate them.

SatImp additionally pairs this with a retain regularizer,

    min_θ  E_u log p(y|x;θ)  −  λ · E_r log p(y|x;θ)

i.e. `L = L_forget_weighted + λ · CE_retain`, over the same OLMo-2 stage1
unseen slice used by `grad_diff.py` / `simnpo.py`. With `λ = 0` (the
default) no retain loader is built and `olmo` is never imported, so plain
WGA runs anywhere torch + transformers are installed.

Why the weighting works: vanilla gradient ascent on `log p` has gradient
∝ (1/p)·∂p/∂θ, so the lowest-confidence tokens — the ones already forgotten
— dominate the update and blow it up. Weighting by `p^α` cancels that
factor; at **α = 1 the 1/p vanishes exactly**, leaving a gradient ∝ ∂p/∂θ.
That is this file's default. SatImp's two-sided weight instead peaks at
`p* = β₁/(β₁+β₂)`, so the recommended β₁=5, β₂=1 concentrates the update on
tokens near p ≈ 0.83 and backs off both the already-forgotten (p→0) and the
perfectly-memorized (p→1).

Two implementation choices worth knowing:

1. **The weighted sum is normalized by the non-pad token count, not by
   Σw.** Normalizing by Σw would rescale the update back up as the weights
   shrink, cancelling the very saturation effect both methods exist to
   produce. Do not "fix" this into a weighted mean.
2. **`w` is detached by default.** Both papers write it as a coefficient on
   the per-token loss, which implies a stop-gradient; leaving it attached
   adds a d(w)/dθ term and changes the objective. `--differentiable-weight`
   opts into the attached form if you want to check it against the authors'
   released code.

Hyperparameter transfer caveat: β₁=5, β₂=1, λ=1 are the TOFU-tuned values,
measured on short QA answers. Here the forget items are flat pretraining
sequences whose token-probability distribution looks nothing like that, so
treat those as sweep centers rather than defaults to trust.

Usage:
    # WGA (α = 1, forget-only)
    python -m pretrain_experiments.reweighted_ga \
        --model sbordt/OLMo-2-179M-Exp-Unlearning \
        --revision stage1-step100000-tokens210B \
        --method-label wga --beta1 1.0 --beta2 0.0 \
        --learning-rate 1e-6 --forget-batch-size 4 --epochs 1 \
        --output-dir /path/to/unlearning-wga/run-<id>

    # SatImp (β₁=5, β₂=1, λ=1)
    python -m pretrain_experiments.reweighted_ga \
        --model sbordt/OLMo-2-179M-Exp-Unlearning \
        --revision stage1-step100000-tokens210B \
        --method-label satimp --beta1 5.0 --beta2 1.0 \
        --retain-loss-weight 1.0 \
        --olmo-config "$OLMO_REPO/configs/official-0425/OLMo2-1B-stage1.yaml" \
        --retain-start-step 100000 \
        --learning-rate 1e-6 --forget-batch-size 4 --retain-batch-size 4 \
        --epochs 1 \
        --output-dir /path/to/unlearning-satimp/run-<id>

References:
    WGA:    https://arxiv.org/abs/2502.19301
    SatImp: https://arxiv.org/abs/2505.11953
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


def _weighted_ascent_loss(model, input_ids, attention_mask, beta1, beta2,
                          detach_weight=True):
    """Token-reweighted gradient-ascent loss on the forget batch.

    Returns `(loss_forget, ce_mean, w_mean, p_true_mean)`. `ce_mean` is the
    plain unweighted CE on the same tokens, logged so forget progress stays
    comparable with the rest of the gradient-ascent family.
    """
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
    ).view(shift_labels.shape)  # (B, T-1), = -log p_i

    p_true = torch.exp(-ce)
    if detach_weight:
        p_true = p_true.detach()
    # Guard the (1-p)^β₂ base against fp overshoot at p ≈ 1.
    p_true = p_true.clamp(0.0, 1.0)
    w = p_true.pow(beta1) * (1.0 - p_true).pow(beta2)

    denom = shift_mask.sum().clamp_min(1.0)
    # Ascent: minimizing Σ w·log p = -Σ w·ce. Normalized by token count, NOT
    # by Σw — see the module docstring.
    loss_forget = -(w * ce * shift_mask).sum() / denom

    with torch.no_grad():
        ce_mean = (ce * shift_mask).sum() / denom
        w_mean = (w * shift_mask).sum() / denom
        p_true_mean = (p_true * shift_mask).sum() / denom

    return loss_forget, ce_mean, w_mean, p_true_mean


def _avg_ce(model, input_ids, attention_mask):
    """Mean CE over non-pad next-token targets in the batch."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits
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
    parser.add_argument("--method-label", type=str, default="reweighted-ga",
                        help="Free-form name recorded as `method` in the config "
                             "snapshot (e.g. wga, satimp). Does not affect training.")

    # Forget set
    parser.add_argument("--forget-experiments", type=str, nargs="*", default=None,
                        help="Whitelist of experiments to use as the forget set. "
                             "Default: all experiments minus iid-replacements-*.")

    # Reweighting hyperparams
    parser.add_argument("--beta1", type=float, default=1.0,
                        help="Exponent on p (WGA's α; SatImp's β₁). Default 1.0 — "
                             "exactly cancels gradient ascent's 1/p factor.")
    parser.add_argument("--beta2", type=float, default=0.0,
                        help="Exponent on (1-p) (SatImp's β₂; 0 = WGA). SatImp "
                             "recommends 1.0 alongside --beta1 5.0.")
    parser.add_argument("--differentiable-weight", action="store_true",
                        help="Let gradients flow through the weight w. Off by "
                             "default (both papers write w as a coefficient).")

    # Retain set (only built when --retain-loss-weight > 0)
    parser.add_argument("--retain-loss-weight", type=float, default=0.0,
                        help="λ on the retain CE term. 0.0 (default) = WGA, no "
                             "retain pass and no OLMo dependency. SatImp uses 1.0.")
    parser.add_argument("--olmo-config", type=str, default=None,
                        help="Path to the OLMo TrainConfig YAML used to build the "
                             "retain stream. Required when --retain-loss-weight > 0.")
    parser.add_argument("--retain-start-step", type=int, default=None,
                        help="Skip the first N training steps' worth of OLMo sequences "
                             "(should match the step the loaded checkpoint represents). "
                             "Required when --retain-loss-weight > 0.")
    parser.add_argument("--retain-seed-override", type=int, default=None,
                        help="Override the OLMo data seed (default: use seed from --olmo-config).")

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
                        default="float32")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    args = parser.parse_args()

    if args.epochs > MAX_EPOCHS_CAP:
        raise SystemExit(
            f"--epochs {args.epochs} exceeds the configured cap of {MAX_EPOCHS_CAP}; "
            f"raise MAX_EPOCHS_CAP in reweighted_ga.py if you really mean to."
        )
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be >= 1")
    if args.beta1 < 0 or args.beta2 < 0:
        raise SystemExit("--beta1 and --beta2 must be >= 0")
    if args.retain_loss_weight < 0:
        raise SystemExit("--retain-loss-weight must be >= 0")

    use_retain = args.retain_loss_weight > 0
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
            "retain_loss_weight=0 -> skipping retain loader (forget-only; no OLMo "
            "dependency)"
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

    # Where the two-sided weight peaks — logged so a sweep can sanity-check
    # that the weight is landing on a live part of the probability range.
    weight_peak = (
        args.beta1 / (args.beta1 + args.beta2)
        if (args.beta1 + args.beta2) > 0
        else None
    )

    config_record = {
        "method": args.method_label,
        "model": args.model,
        "revision": args.revision,
        "forget_experiments": args.forget_experiments,
        "forget_set_info": forget_info,
        "retain_set_info": retain_info,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "weight_peak_at_p": weight_peak,
        "differentiable_weight": args.differentiable_weight,
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
    with open(output_dir / "reweighted_ga_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    metrics_f = open(metrics_path, "w")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    logger.info(
        f"Starting {args.method_label}: lr={args.learning_rate}, "
        f"beta1={args.beta1}, beta2={args.beta2}"
        f"{f' (weight peaks at p={weight_peak:.3f})' if weight_peak is not None else ''}, "
        f"lambda_retain={args.retain_loss_weight}{' (disabled)' if not use_retain else ''}, "
        f"detach_weight={not args.differentiable_weight}, "
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
                loss_forget, ce_forget, w_mean, p_true_mean = _weighted_ascent_loss(
                    model, forget_input_ids, forget_attn,
                    args.beta1, args.beta2,
                    detach_weight=not args.differentiable_weight,
                )

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
                "weight_mean": float(w_mean.item()),
                "p_true_mean": float(p_true_mean.item()),
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
