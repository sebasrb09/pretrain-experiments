"""Mid-training: continue a checkpoint on the OLMo-2 stage-2 mix with a linear cooldown.

The recipe of the released mid-trained models (paper appendix, "Mid-Trained
Models"): from the step-100,000 checkpoint, the learning rate goes linearly from
3.986e-4 to zero over 25B tokens of the stage-2 mix, 11,931 steps of 512
sequences of 4,096 tokens, without warmup. Here the starting checkpoint is any
Hugging Face model, so an UNLEARNED model can be carried through the same stage
(pretraining -> unlearning -> mid-training -> SFT -> DPO).

It is a separate driver, not a mode of an unlearning driver, because its clock
is the data stream and not the forget set, and because the unlearning drivers
are in use by running searches. It shares their building blocks: the OLMo data
order (build_olmo_retain_dataset), the pretraining run's AdamW groups, its Adam
moments, and the step-N checkpoints with a trainer state.

Data order: sequence i of the run is sequence i of the stream the OLMo trainer
would read from --olmo-config (its seed, epoch 0), taken in order. Step s
therefore trains on sequences [512 s, 512 (s + 1)), and a resumed job starts at
its step's first sequence without replaying anything.

The run does not fit one job (11,931 steps at ~94 s on a LUMI GCD is ~13 days
against a 3-day limit): --max-hours stops cleanly at a resume point and the
site wrapper (internal/lumi/mid_train.sh) submits the next link.
"""
import argparse
import json
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from pretrain_experiments.logging_config import get_logger
from pretrain_experiments.unlearning_utils import (
    OLMO2_1B_BETAS,
    OLMO2_1B_LR_AT_STEP_100K,
    OLMO2_1B_MAX_GRAD_NORM,
    OLMO2_1B_WEIGHT_DECAY,
    TRAINER_STATE_FILE,
    build_matched_optimizer,
    collate_olmo_retain,
    find_latest_checkpoint,
    load_matched_optimizer_state,
    load_trainer_state,
    save_trainer_state,
)

logger = get_logger(__name__)

MID_STEPS = 11931            # 25B tokens / (512 x 4096), as the released models
DONE_FILE = "mid_train.done"


def lr_at(step, total_steps, lr0):
    """OLMo's linear_with_warmup with no warmup and a final rate of zero: the
    rate of the update that follows `step` finished ones."""
    return lr0 * max(0.0, 1.0 - step / float(total_steps))


def _ce(model, input_ids):
    logits = model(input_ids=input_ids).logits
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(), input_ids[:, 1:].reshape(-1))


def _prune(output_dir, newest, keep_steps):
    """After step-<newest> is safely on disk: older resume points lose their
    trainer state (12 GB at 1B), and their weights too unless the step is kept."""
    for name in os.listdir(output_dir):
        if not name.startswith("step-") or not name[len("step-"):].isdigit():
            continue
        step = int(name[len("step-"):])
        if step >= newest:
            continue
        path = os.path.join(output_dir, name)
        if step in keep_steps:
            try:
                os.remove(os.path.join(path, TRAINER_STATE_FILE))
            except OSError:
                pass
        else:
            shutil.rmtree(path, ignore_errors=True)


def train(model, dataset, args, save_weights, resume_dir=None, resume_step=0, clock=time.perf_counter):
    """The training loop. Returns the number of finished optimizer steps.

    save_weights(directory) writes the model; the trainer state goes next to it.
    Split from main() so it can be exercised on a small model without a GPU.
    """
    if args.total_batch % args.micro_batch:
        raise SystemExit("--total-batch must be a multiple of --micro-batch")
    accum = args.total_batch // args.micro_batch
    need = args.max_steps * args.total_batch
    if len(dataset) < need:
        raise SystemExit(f"the stream holds {len(dataset)} sequences, {need} are needed for "
                         f"{args.max_steps} steps of {args.total_batch}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep = {int(x) for x in str(args.keep_steps).split(",") if x.strip()}

    optimizer = build_matched_optimizer(model, lr=args.learning_rate, weight_decay=args.weight_decay,
                                        betas=tuple(args.betas))
    step = 0
    if resume_dir:
        step = int(load_trainer_state(resume_dir, optimizer, args.device)["optimizer_step"])
        if step != resume_step:
            raise SystemExit(f"{resume_dir} is named step {resume_step} but its state says {step}")
    elif args.resume_optimizer_state:
        load_matched_optimizer_state(optimizer, model, args.resume_optimizer_state)
        logger.info(f"Adam moments from {args.resume_optimizer_state}")
    else:
        logger.info("Adam moments start at zero")
    if step >= args.max_steps:
        logger.info(f"already at step {step} of {args.max_steps}; nothing to do")
        return step

    loader = DataLoader(Subset(dataset, range(step * args.total_batch, need)), batch_size=args.micro_batch,
                        shuffle=False, drop_last=True, collate_fn=collate_olmo_retain,
                        num_workers=args.num_workers)
    autocast = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.dtype == "bfloat16" else nullcontext())
    metrics = open(out / "metrics.jsonl", "a")
    logger.info(f"mid-training from step {step} to {args.max_steps}: lr {args.learning_rate:.6g} linear to 0, "
                f"batch {args.total_batch} = {accum} x {args.micro_batch}, dtype {args.dtype}")

    def checkpoint():
        ckpt = out / f"step-{step}"
        logger.info(f"  saving {ckpt}")
        save_weights(str(ckpt))
        save_trainer_state(ckpt, optimizer, optimizer_step=step, micro_step=step * accum, epoch=1)
        if not os.path.isfile(ckpt / TRAINER_STATE_FILE):
            raise SystemExit("no trainer state was written (NO_TRAINER_STATE=1?): this run could not resume")
        _prune(str(out), step, keep)

    t0 = clock()
    saved_at = step
    model.train()
    optimizer.zero_grad()
    loss_sum, stopped_early = 0.0, False
    for i, (input_ids, _) in enumerate(loader, 1):
        if (i - 1) % accum == 0:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, args.max_steps, args.learning_rate)
        with autocast:
            loss = _ce(model, input_ids.to(args.device))
        (loss / accum).backward()
        loss_sum += float(loss.detach())
        if i % accum:
            continue
        # error_if_nonfinite: die before an update that would turn every weight NaN
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        elapsed = clock() - t0
        metrics.write(json.dumps({"step": step, "lr": lr, "loss": loss_sum / accum, "grad_norm": float(norm),
                                  "elapsed_s": round(elapsed, 1)}) + "\n")
        metrics.flush()
        loss_sum = 0.0
        if step % args.log_every == 0 or step == args.max_steps:
            logger.info(f"step {step}/{args.max_steps}  lr {lr:.4g}  {elapsed / (step - resume_step):.1f} s/step")
        out_of_time = args.max_hours and elapsed > args.max_hours * 3600
        if step % args.checkpoint_every == 0 or step in keep or step == args.max_steps or out_of_time:
            checkpoint()
            saved_at = step
        if out_of_time and step < args.max_steps:
            logger.info(f"time budget of {args.max_hours} h reached at step {step}; the next job continues")
            stopped_early = True
            break
    metrics.close()
    if step >= args.max_steps and not stopped_early:
        with open(out / DONE_FILE, "w") as f:
            json.dump({"steps": step, "learning_rate": args.learning_rate, "total_batch": args.total_batch,
                       "micro_batch": args.micro_batch, "model": args.model, "olmo_config": args.olmo_config}, f)
        logger.info(f"mid-training finished at step {step}; final model in {out / f'step-{step}'}")
    assert saved_at == step
    return step


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pretrain_experiments.unlearning_utils import build_olmo_retain_dataset, save_hf_checkpoint

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="the checkpoint to mid-train: a local folder or a Hub repository")
    p.add_argument("--revision", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--olmo-config", required=True, help="OLMo training config of the stage-2 mix (data paths, seed)")
    p.add_argument("--seed-override", type=int, default=None, help="data-order seed, instead of the config's")
    p.add_argument("--max-steps", type=int, default=MID_STEPS)
    p.add_argument("--learning-rate", type=float, default=OLMO2_1B_LR_AT_STEP_100K)
    p.add_argument("--total-batch", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=OLMO2_1B_WEIGHT_DECAY)
    p.add_argument("--betas", type=float, nargs=2, default=OLMO2_1B_BETAS)
    p.add_argument("--max-grad-norm", type=float, default=OLMO2_1B_MAX_GRAD_NORM)
    p.add_argument("--resume-optimizer-state", default=None,
                   help="OLMo optim.pt whose Adam moments the run starts from (default: zeroed moments)")
    p.add_argument("--checkpoint-every", type=int, default=250, help="resume points; older ones are pruned")
    p.add_argument("--keep-steps", default="2000,4000,6000,8000,10000",
                   help="steps whose weights are kept after a later resume point exists")
    p.add_argument("--max-hours", type=float, default=None, help="stop cleanly at a resume point after this long")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--gradient-checkpointing", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    if (out / DONE_FILE).exists():
        logger.info(f"{out / DONE_FILE} exists: this run is finished")
        return
    resume_dir, resume_step = find_latest_checkpoint(out)
    logger.info(f"resuming from {resume_dir} (step {resume_step})" if resume_dir else f"starting from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    # fp32 weights; autocast runs the forward and backward passes in bf16 when asked
    model = AutoModelForCausalLM.from_pretrained(
        str(resume_dir) if resume_dir else args.model, revision=None if resume_dir else args.revision,
        torch_dtype=torch.float32).to(args.device)
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    dataset, info = build_olmo_retain_dataset(args.olmo_config, start_step=0, max_seq_len=args.max_seq_len,
                                              seed_override=args.seed_override)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mid_train_config.json", "w") as f:
        json.dump({**{k: v for k, v in vars(args).items()}, "stream": info, "resumed_from_step": resume_step}, f, indent=2)
    train(model, dataset, args, lambda d: save_hf_checkpoint(model, tokenizer, d), resume_dir, resume_step)


if __name__ == "__main__":
    main()
