"""
NPO unlearning (Zhang et al., *Negative Preference Optimization: From
Catastrophic Collapse to Effective Unlearning*, COLM 2024).

DPO's preference objective with the positive term dropped: a bounded,
reference-anchored forget loss.

    L_NPO(θ) = -(2/β) · E_{x ∼ D_forget} [ log σ( -β · log( π_θ(x) / π_ref(x) ) ) ]

In code we compute per-sequence **summed** NLL over non-pad next-token
targets, `nll(x) = -log π(x)`, so

    -log( π_θ(x) / π_ref(x) ) = nll_θ(x) − nll_ref(x)

and the σ argument is just `β · (nll_θ − nll_ref)`. This matches the
reference implementation, which forms `neg_log_ratios = forget_loss_current
− forget_loss_oracle` from sum-reduced sequence losses.

As the model forgets x, nll_θ grows away from the frozen nll_ref, the σ
argument → +∞, log σ → 0, and the gradient vanishes. That bounded behavior
is what separates NPO from gradient ascent, which keeps pushing forever.

Optional retain term: standard cross-entropy on the OLMo-2 stage1 unseen
slice, weighted by α (the "NPO-RT" variant; α=0 gives plain NPO).

    L_total = L_NPO_forget + α · CE_retain

Difference from `simnpo.py` — and why β does not transfer between them:
NPO keeps the frozen reference and uses the **sum** of token NLLs, so the σ
argument scales with sequence length; SimNPO drops the reference and divides
by |x|. On a pretraining-corpus forget set with sequences up to
`--max-seq-len` tokens, that length scaling is severe — β·(nll_θ − nll_ref)
saturates the sigmoid far earlier than the β values quoted in the paper
(tuned on TOFU's short QA answers) would suggest. Expect the useful β range
here to sit well below TOFU's, and watch `sigmoid_arg_mean` in
metrics.jsonl: if it is large and positive from the first step, the loss is
already saturated and that cell is a no-op.

Precision note: unlike `rmu.py`, whose retain term is an MSE on activations,
NPO's forget term is a *difference of two large summed log-likelihoods* and
is therefore exposed to catastrophic cancellation. `--frozen-dtype` defaults
to float32 here (rmu.py defaults to bfloat16), and per-token CE is
accumulated in fp32 on both sides regardless. Only drop the reference to
bfloat16 if memory forces it.

All model parameters of the updated model are trained (full fine-tune; no LoRA).

Usage:
    python -m pretrain_experiments.npo \
        --model sbordt/OLMo-2-179M-Exp-Unlearning \
        --revision stage1-step100000-tokens210B \
        --olmo-config "$OLMO_REPO/configs/official-0425/OLMo2-1B-stage1.yaml" \
        --retain-start-step 100000 \
        --beta 0.1 \
        --retain-loss-weight 1.0 \
        --learning-rate 1e-5 \
        --forget-batch-size 4 --retain-batch-size 4 \
        --epochs 1 \
        --output-dir /path/to/unlearning-npo/run-<id>

Reference: https://arxiv.org/abs/2404.05868
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
    build_lr_schedule,
    find_latest_checkpoint,
    load_trainer_state,
    save_trainer_state,
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


def _per_seq_sum_nll(model, input_ids, attention_mask):
    """Summed NLL over real (non-pad) next-token targets, per sequence.

    Returns a (B,) fp32 tensor. Summed — not averaged — to match NPO's
    sequence-level log-likelihood ratio. Accumulation is forced to fp32
    because the downstream quantity is a difference of two such sums.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # (B, T, V)
    # Shift the LABELS, not the logits. `logits` is contiguous, so
    # reshape(-1, V) is a view, whereas slicing it along the sequence dim
    # first forces a full (B, T-1, V) copy -- 1.6 GB in bf16 at B=2, T=4096,
    # V=100352, allocated and written on every forward. CE is evaluated at
    # all T positions and the last column dropped, which costs one extra
    # position per sequence and saves the copy.
    shift_labels = torch.empty_like(input_ids)
    shift_labels[:, :-1] = input_ids[:, 1:]
    shift_labels[:, -1] = 0  # never scored: dropped with the last column
    shift_mask = attention_mask[..., 1:].float()
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(input_ids.shape)[:, :-1].float()  # (B, T-1)
    return (ce * shift_mask).sum(dim=-1)  # (B,)


def _avg_ce(model, input_ids, attention_mask):
    """Mean CE over non-pad next-token targets in the batch."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits
    # Shift the LABELS, not the logits. `logits` is contiguous, so
    # reshape(-1, V) is a view, whereas slicing it along the sequence dim
    # first forces a full (B, T-1, V) copy -- 1.6 GB in bf16 at B=2, T=4096,
    # V=100352, allocated and written on every forward. CE is evaluated at
    # all T positions and the last column dropped, which costs one extra
    # position per sequence and saves the copy.
    shift_labels = torch.empty_like(input_ids)
    shift_labels[:, :-1] = input_ids[:, 1:]
    shift_labels[:, -1] = 0  # never scored: dropped with the last column
    shift_mask = attention_mask[..., 1:].to(logits.dtype)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(input_ids.shape)[:, :-1]
    return (ce * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Model / IO
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metrics-jsonl", type=str, default=None)

    # Frozen reference π_ref
    parser.add_argument("--reference-model", type=str, default=None,
                        help="Frozen reference model (default: --model, i.e. the "
                             "pre-unlearning checkpoint — the standard NPO choice).")
    parser.add_argument("--reference-revision", type=str, default=None,
                        help="Revision for the reference. Defaults to --revision when "
                             "--reference-model is not given; otherwise to that repo's main.")

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

    # NPO hyperparams
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Inverse temperature β (paper sweeps {0.05, 0.1, 0.5, 1.0}; "
                             "default: 0.1). Note the sum-NLL scaling caveat in the "
                             "module docstring — the useful range here is likely lower.")
    parser.add_argument("--retain-loss-weight", type=float, default=1.0,
                        help="α on retain CE term (0.0 = plain NPO, no retain pass).")

    # Optimization
    parser.add_argument("--learning-rate", type=float,
                        default=OLMO2_1B_LR_AT_STEP_100K,
                        help="Defaults to the trajectory LR at step 100000.")
    parser.add_argument("--weight-decay", type=float,
                        default=OLMO2_1B_WEIGHT_DECAY,
                        help="Matched to the pretraining run; the embedding "
                             "matrix is excluded from decay automatically.")
    parser.add_argument("--lr-schedule", choices=("constant", "linear"),
                        default="constant",
                        help="constant (default) holds the pretraining LR for the whole run, which is what the resumed checkpoint was doing: OLMo-2 cosine over ~5e12 tokens decays only 0.08% across a 10k-step window. linear decays to zero over --max-steps.")
    parser.add_argument("--auto-resume", dest="auto_resume",
                        action="store_true", default=True,
                        help="Resume from the highest step-N checkpoint in --output-dir that carries a trainer_state.pt. On by default: a 10k-step cell outlives the 72h QOS ceiling, so runs are expected to be chained. Resuming is always logged, never silent.")
    parser.add_argument("--no-auto-resume", dest="auto_resume",
                        action="store_false",
                        help="Start from scratch even if checkpoints exist.")
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
                        default="float32",
                        help="Compute dtype for forward/backward of the updated model.")
    parser.add_argument("--frozen-dtype", type=str, choices=["float32", "bfloat16"],
                        default="float32",
                        help="Storage dtype for the frozen reference (default: float32 — "
                             "the log-ratio is a difference of large sums; see the "
                             "precision note in the module docstring).")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    args = parser.parse_args()

    if args.epochs > MAX_EPOCHS_CAP:
        raise SystemExit(
            f"--epochs {args.epochs} exceeds the configured cap of {MAX_EPOCHS_CAP}; "
            f"raise MAX_EPOCHS_CAP in npo.py if you really mean to."
        )
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be >= 1")
    if args.beta <= 0:
        raise SystemExit("--beta must be > 0")
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

    # Reference defaults to the loaded checkpoint itself; an explicit
    # --reference-model opts out of inheriting --revision.
    if args.reference_model is None:
        ref_model_id = args.model
        ref_revision = (
            args.reference_revision if args.reference_revision is not None
            else args.revision
        )
    else:
        ref_model_id = args.reference_model
        ref_revision = args.reference_revision

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the resume point BEFORE loading the model: weights must come
    # from the checkpoint, not from the base revision.
    resume_dir, resume_step = (None, 0)
    if args.auto_resume:
        resume_dir, resume_step = find_latest_checkpoint(output_dir)
        if resume_dir:
            logger.info(f"auto-resume: found {resume_dir} (step {resume_step})")
        else:
            logger.info("auto-resume: no usable checkpoint, starting fresh")

    # A chained job that starts after the cell already finished must exit
    # here, not after replaying millions of micro-batches to discover there
    # is one step left. The replay is cheap per batch but there are
    # max_steps * accum of them.
    if (resume_dir and args.max_steps is not None
            and resume_step >= args.max_steps):
        logger.info(
            f"cell already at step {resume_step} >= --max-steps "
            f"{args.max_steps}; nothing to do")
        return
    metrics_path = (
        Path(args.metrics_jsonl) if args.metrics_jsonl else output_dir / "metrics.jsonl"
    )

    # ---- Models ---------------------------------------------------------
    logger.info(f"Loading updated model {args.model} (revision={args.revision})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    # A resumed run loads weights from the checkpoint directory, which is a
    # local path and therefore carries no revision.
    model_src = str(resume_dir) if resume_dir else args.model
    model = AutoModelForCausalLM.from_pretrained(
        model_src, revision=None if resume_dir else args.revision,
        torch_dtype=torch.float32,
    ).to(args.device)
    model.train()

    if args.gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled on updated model")

    logger.info(
        f"Loading frozen reference {ref_model_id} (revision={ref_revision}, "
        f"dtype={args.frozen_dtype})..."
    )
    frozen_dtype = torch.bfloat16 if args.frozen_dtype == "bfloat16" else torch.float32
    frozen = AutoModelForCausalLM.from_pretrained(
        ref_model_id, revision=ref_revision, torch_dtype=frozen_dtype,
    ).to(args.device)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

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
        logger.info("retain_loss_weight=0 -> skipping retain loader (plain NPO)")

    # ---- Optimizer -----------------------------------------------------
    optimizer = build_matched_optimizer(
        model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
    )
    # Constant by default -- the checkpoint we resume from was on a cosine so
    # long (t_max ~5e12 tokens) that it is flat over a 10k-step window, and
    # decaying to zero instead would confound a method unlearning less late in
    # training with the optimizer simply having stopped moving.
    scheduler = build_lr_schedule(optimizer, args.max_steps or 0, args.lr_schedule)

    # Resume Adam's moments from the pretraining checkpoint. Without this the
    # first steps run on zeroed second moments, so every parameter takes a
    # near-maximal step regardless of method -- an artefact that looks exactly
    # like early instability caused by the unlearning loss itself.
    # A resumed run restores the moments it was actually using. Falling back
    # to the pretraining optim.pt here would silently rewind them to step
    # 100001 in the middle of a trajectory.
    resume_state = None
    if resume_dir:
        resume_state = load_trainer_state(resume_dir, optimizer, args.device)
    else:
        if args.resume_optimizer_state:
            load_matched_optimizer_state(
                optimizer, model, args.resume_optimizer_state)

    config_record = {
        "method": "npo",
        "model": args.model,
        "revision": args.revision,
        "reference_model": ref_model_id,
        "reference_revision": ref_revision,
        "forget_experiments": args.forget_experiments,
        "forget_set_info": forget_info,
        "retain_set_info": retain_info,
        "beta": args.beta,
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
        "frozen_dtype": args.frozen_dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "micro_batches_per_epoch": len(forget_loader),
    }
    with open(output_dir / "npo_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    metrics_f = open(metrics_path, "w")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )
    # An enclosing bf16 autocast would run the frozen forward in bf16 whatever
    # --frozen-dtype says, silently undoing the fp32 reference. Disable it for
    # the reference pass so the log-ratio really is computed at fp32.
    frozen_ctx = (
        torch.amp.autocast(device_type="cuda", enabled=False)
        if args.dtype == "bfloat16" and args.frozen_dtype == "float32"
        else nullcontext()
    )

    logger.info(
        f"Starting NPO: lr={args.learning_rate}, beta={args.beta}, "
        f"alpha_retain={args.retain_loss_weight}{' (disabled)' if not use_retain else ''}, "
        f"forget_bs={args.forget_batch_size}, retain_bs={args.retain_batch_size}, "
        f"accum={args.gradient_accumulation_steps}, epochs={args.epochs}, "
        f"micro_batches/epoch={len(forget_loader)}, dtype={args.dtype}, "
        f"frozen_dtype={args.frozen_dtype}"
    )

    # Wall clock for the metrics rows: lets throughput be measured without
    # model load and tokenization folded in, and gives a live ETA.
    t_start = time.perf_counter()
    optimizer_step = resume_state["optimizer_step"] if resume_state else 0
    micro_step = 0
    # Replay the shuffled stream up to where the previous job stopped. The
    # loaders are seeded, so this reproduces the exact order rather than
    # drawing a fresh sample of sequences already trained on. It is dataset
    # iteration only -- no forward pass -- so it costs well under a minute.
    resume_micro = resume_state["micro_step"] if resume_state else 0
    if resume_micro:
        logger.info(f"replaying {resume_micro} micro-batches to reach the resume point")
    stopped = False

    for epoch in range(1, args.epochs + 1):
        if stopped:
            break
        optimizer.zero_grad()
        for forget_input_ids, forget_attn in forget_loader:
            if micro_step < resume_micro:
                micro_step += 1
                if use_retain:
                    next(retain_iter)
                continue
            forget_input_ids = forget_input_ids.to(args.device)
            forget_attn = forget_attn.to(args.device)

            with autocast_ctx:
                # Forget loss: NPO
                nll_theta = _per_seq_sum_nll(model, forget_input_ids, forget_attn)  # (B,)
                with torch.no_grad(), frozen_ctx:
                    nll_ref = _per_seq_sum_nll(frozen, forget_input_ids, forget_attn)
                # neg_log_ratio = -log(π_θ/π_ref) = nll_θ - nll_ref
                neg_log_ratio = nll_theta - nll_ref.to(nll_theta.dtype)
                arg = args.beta * neg_log_ratio
                loss_forget = -(2.0 / args.beta) * F.logsigmoid(arg).mean()

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
                    save_trainer_state(ckpt_dir, optimizer,
                                       optimizer_step=optimizer_step,
                                       micro_step=micro_step, epoch=epoch)

            metrics_f.write(json.dumps({
                "epoch": epoch,
                "micro_step": micro_step,
                "optimizer_step": optimizer_step,
                "elapsed_s": round(time.perf_counter() - t_start, 3),
                "nll_theta_mean": float(nll_theta.detach().mean().item()),
                "nll_ref_mean": float(nll_ref.detach().mean().item()),
                "neg_log_ratio_mean": float(neg_log_ratio.detach().mean().item()),
                "sigmoid_arg_mean": float(arg.detach().mean().item()),
                "loss_forget": float(loss_forget.detach().item()),
                "loss_retain": float(loss_retain.detach().item()),
                "loss_total": float(
                    (loss_forget + args.retain_loss_weight * loss_retain).detach().item()
                ),
            }) + "\n")
            # Flush once per optimizer step, not once per micro-batch: at
            # accum 256 that is 2.56M fsyncs onto $DATA over a 10k-step
            # cell. A kill still loses under one step of rows, and
            # metrics_f.close() below flushes the tail on a clean exit.
            if micro_step % args.gradient_accumulation_steps == 0:
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

        logger.info(f"epoch {epoch}/{args.epochs} done: opt_steps={optimizer_step}")

        if epoch % args.checkpoint_every_n_epochs == 0 or epoch == args.epochs or stopped:
            ckpt_dir = output_dir / f"epoch-{epoch}"
            logger.info(f"  saving checkpoint to {ckpt_dir}")
            save_hf_checkpoint(model, tokenizer, str(ckpt_dir))

    metrics_f.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
