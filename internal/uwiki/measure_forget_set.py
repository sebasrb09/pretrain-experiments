"""
Measure the forget set and the insertion-likelihood eval surface.

Two numbers currently block the 1B Pareto sweep, and both come from the same
place -- the size of `sbordt/OLMo-2-1B-Exp-Dataset`:

  1. |D_f| decides whether the shared 10k hard step cap ever binds, since
     steps = (|D_f| / TOTAL_BATCH) * epochs. If the forget set is small the
     published per-method epoch counts run well under the cap (fine, expected);
     if it is large the cap truncates the longer protocols and the methods stop
     being run at their own settings.

  2. The per-experiment token counts decide `insertion_likelihood --max-tokens`.
     That eval defaults to 100M tokens PER experiment across ~57 experiments,
     which would be ~5.7B tokens of forward passes per checkpoint if the cap
     actually binds. Over 35 checkpoints that is not runnable; if the dataset
     is far below the cap it is a non-issue. This tells us which.

CPU only -- no GPU, no model weights, just the tokenizer. Run it on a login
node or a small CPU allocation:

    python internal/uwiki/measure_forget_set.py

For a fast first look on a large dataset, sample:

    python internal/uwiki/measure_forget_set.py --sample-frac 0.05

Token counts are produced the same way `unlearning_utils.tokenize_and_strip`
produces them (encode, then strip leading/trailing EOS), so `forget tokens
(truncated)` below is exactly what a training run will report as
`forget_set_info.n_total_tokens`.
"""

import argparse
import json
import math
from collections import defaultdict

# Imported rather than redefined so the exclusion list cannot drift from the
# one the trainers actually use.
from pretrain_experiments.unlearning_utils import (
    DEFAULT_EXCLUDED_EXPERIMENTS,
    DEFAULT_MAX_SEQ_LEN,
)

# Method -> published epoch count, mirroring internal/uwiki/unlearn_cell_1B.sh.
# RMU is absent on purpose: it runs to a ~200-step budget, not to epochs.
METHOD_EPOCHS = {
    "wga": 5,
    "satimp": 5,
    "ce-u": 8,
    "gradient-ascent": 10,
    "grad-diff": 10,
    "npo": 10,
    "simnpo": 10,
}


def human(n):
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="sbordt/OLMo-2-1B-Exp-Dataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str,
                        default="sbordt/OLMo-2-1B-Exp-Unlearning",
                        help="Tokenizer source; must match the model being unlearned.")
    parser.add_argument("--revision", type=str, default="stage1-step100000-tokens210B")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN,
                        help="Training-time truncation, for the |D_f| accounting.")
    parser.add_argument("--total-batch", type=int, default=512,
                        help="Forget sequences per optimizer step in the sweep.")
    parser.add_argument("--hard-step-cap", type=int, default=10000)
    parser.add_argument("--n-checkpoints", type=int, default=35,
                        help="Checkpoints the eval sweep will cover (32 cells + 3 anchors).")
    parser.add_argument("--tokenizer-batch", type=int, default=1000)
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Tokenize only this fraction of rows per experiment and "
                             "extrapolate. 1.0 = exact.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str,
                        default="forget_set_measurement.json")
    args = parser.parse_args()

    if not (0 < args.sample_frac <= 1.0):
        raise SystemExit("--sample-frac must be in (0, 1]")

    import datasets
    from transformers import AutoTokenizer

    print(f"Loading tokenizer {args.tokenizer} (revision={args.revision})...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    eos_id = tokenizer.eos_token_id

    print(f"Loading {args.dataset} split={args.split}...")
    ds = datasets.load_dataset(args.dataset, split=args.split)
    experiments = sorted(set(ds["experiment"]))
    print(f"  {len(ds)} rows, {len(experiments)} experiments\n")

    excluded = set(DEFAULT_EXCLUDED_EXPERIMENTS)

    stats = {}
    for i, exp in enumerate(experiments, 1):
        subset = ds.filter(lambda x, e=exp: x["experiment"] == e)
        texts = subset["text"]
        n_rows_total = len(texts)

        if args.sample_frac < 1.0 and n_rows_total > 0:
            n_sample = max(1, int(round(n_rows_total * args.sample_frac)))
            rng = __import__("random").Random(args.seed)
            texts = rng.sample(list(texts), n_sample)
        scale = n_rows_total / len(texts) if texts else 1.0

        n_tok_raw = 0
        n_tok_trunc = 0
        n_nonempty = 0
        max_len = 0
        for start in range(0, len(texts), args.tokenizer_batch):
            chunk = texts[start:start + args.tokenizer_batch]
            for ids in tokenizer(chunk)["input_ids"]:
                # Mirror tokenize_and_strip: drop leading/trailing EOS, skip empties.
                a, b = 0, len(ids)
                while a < b and ids[a] == eos_id:
                    a += 1
                while b > a and ids[b - 1] == eos_id:
                    b -= 1
                length = b - a
                if length == 0:
                    continue
                n_nonempty += 1
                n_tok_raw += length
                n_tok_trunc += min(length, args.max_seq_len)
                max_len = max(max_len, length)

        stats[exp] = {
            "n_rows": n_rows_total,
            "n_sequences": int(round(n_nonempty * scale)),
            "n_tokens_raw": int(round(n_tok_raw * scale)),
            "n_tokens_truncated": int(round(n_tok_trunc * scale)),
            "max_len_observed": max_len,
            "excluded_from_forget": exp in excluded,
            "sampled": args.sample_frac < 1.0,
        }
        print(f"  [{i:2d}/{len(experiments)}] {exp:<48s} "
              f"{stats[exp]['n_sequences']:>9,} seq  "
              f"{human(stats[exp]['n_tokens_raw']):>9} tok")

    # ---- Aggregates ------------------------------------------------------
    def agg(pred):
        rows = [s for e, s in stats.items() if pred(e, s)]
        return {
            "n_experiments": len(rows),
            "n_sequences": sum(r["n_sequences"] for r in rows),
            "n_tokens_raw": sum(r["n_tokens_raw"] for r in rows),
            "n_tokens_truncated": sum(r["n_tokens_truncated"] for r in rows),
        }

    forget = agg(lambda e, s: not s["excluded_from_forget"])
    controls = agg(lambda e, s: s["excluded_from_forget"])
    everything = agg(lambda e, s: True)

    print("\n" + "=" * 78)
    print("  FORGET SET  (library default: all experiments minus iid-replacements-*)")
    print("=" * 78)
    print(f"  experiments:                {forget['n_experiments']}")
    print(f"  sequences  |D_f|:           {forget['n_sequences']:,}")
    print(f"  tokens (raw):               {human(forget['n_tokens_raw'])}")
    print(f"  tokens (truncated @{args.max_seq_len}):  {human(forget['n_tokens_truncated'])}")
    print(f"  excluded controls:          {controls['n_experiments']} experiments, "
          f"{controls['n_sequences']:,} sequences")

    # ---- Decision 1: does the hard step cap bind? -------------------------
    steps_per_epoch = math.ceil(forget["n_sequences"] / args.total_batch) if forget["n_sequences"] else 0
    print("\n" + "=" * 78)
    print(f"  TRAINING STEPS   steps = ceil(|D_f| / {args.total_batch}) * epochs")
    print("=" * 78)
    print(f"  steps per epoch:            {steps_per_epoch:,}")
    print(f"  hard cap:                   {args.hard_step_cap:,}\n")
    print(f"  {'method':<18s} {'epochs':>6s} {'steps':>12s}   status")
    binding = []
    for method, ep in sorted(METHOD_EPOCHS.items(), key=lambda kv: (kv[1], kv[0])):
        steps = steps_per_epoch * ep
        if steps > args.hard_step_cap:
            status = f"CAPPED -> runs {args.hard_step_cap / steps_per_epoch:.1f} epochs"
            binding.append(method)
        else:
            status = "runs full protocol"
        print(f"  {method:<18s} {ep:>6d} {steps:>12,}   {status}")
    print(f"  {'rmu':<18s} {'--':>6s} {200:>12,}   step-budget protocol (~200)")

    # ---- Decision 2: insertion_likelihood --max-tokens --------------------
    # IL evaluates every experiment, controls included -- it does not filter.
    print("\n" + "=" * 78)
    print(f"  INSERTION LIKELIHOOD COST   (all {everything['n_experiments']} experiments, "
          f"{args.n_checkpoints} checkpoints)")
    print("=" * 78)
    print(f"  {'--max-tokens':>20s} {'tok/checkpoint':>16s} {'total forward tok':>19s}  {'vs uncapped':>11s}")
    uncapped = everything["n_tokens_raw"]
    rows_json = {}
    for cap in (1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000):
        per_ckpt = sum(min(s["n_tokens_raw"], cap) for s in stats.values())
        total = per_ckpt * args.n_checkpoints
        frac = per_ckpt / uncapped if uncapped else 1.0
        label = f"{human(cap)}" + (" (default)" if cap == 100_000_000 else "")
        print(f"  {label:>20s} {human(per_ckpt):>16s} {human(total):>19s}  {frac:>10.1%}")
        rows_json[str(cap)] = {"tokens_per_checkpoint": per_ckpt, "total_tokens": total}
    n_saturating = sum(1 for s in stats.values() if s["n_tokens_raw"] >= 100_000_000)
    print(f"\n  experiments at or above the 100M default cap: {n_saturating}")
    if n_saturating == 0:
        print("  -> the default cap never binds; --max-tokens is a free knob and the"
              "\n     uncapped run costs exactly the numbers above.")

    out = {
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "revision": args.revision,
        "max_seq_len": args.max_seq_len,
        "sample_frac": args.sample_frac,
        "per_experiment": stats,
        "forget_set": forget,
        "excluded_controls": controls,
        "all_experiments": everything,
        "total_batch": args.total_batch,
        "steps_per_epoch": steps_per_epoch,
        "hard_step_cap": args.hard_step_cap,
        "methods_hitting_cap": binding,
        "insertion_likelihood_cost": rows_json,
        "n_checkpoints": args.n_checkpoints,
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.output_json}")


if __name__ == "__main__":
    main()
