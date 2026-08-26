"""Materialize a bounded OLMo-2 stage1 RETAIN slice, so grad-diff and rmu can run
on a cluster that does not host the pretraining corpus.

WHY THIS EXISTS
---------------
The retain set is defined as "stage1 sequences the loaded checkpoint has not yet
seen" (CLAUDE.md is explicit that c4 or another generic corpus must NOT be
substituted -- the point is to compare unlearning against the continued-training
baseline on identical data). `unlearning_utils.build_olmo_retain_dataset` gets
that by rebuilding the WHOLE memmap dataset and replaying OLMo's PCG64 shuffle,
which needs the entire multi-TB corpus present.

But a sweep never reads more than a sliver of it. At HARD_STEP_CAP=100 and
global batch 512:

    100 steps x 512 sequences = 51,200 sequences ~= 210M tokens ~= 840 MB

So the multi-TB requirement is an artifact of how the stream is reconstructed,
not of what training consumes. This script materializes just the sliver.

HOW IT WORKS
------------
The same trick internal/uwiki/archive/build_mia_finetune_dataset_1B.py uses:
OLMo's MemMapDataset supports RANDOM ACCESS over remote paths, so `memmap_ds[i]`
fetches one sequence without the corpus being local. We

  1. build the OLMo dataloader from a stage1 config and replay `global_indices`
     (the exact shuffle the pretraining run used),
  2. take `global_indices[start_step * global_batch : ... + num_sequences]` --
     by construction the first sequences the checkpoint had NOT reached,
  3. fetch those and write them to one local uint32 .npy,
  4. emit an OLMo config whose `data.paths` points at that .npy.

Point the sweep at the result with START_STEP=0, because the slice is already
the unseen region -- there is nothing left to skip:

    OLMO_CONFIG=<out>/retain-config.yaml START_STEP=0 \
      METHODS="grad-diff rmu" bash internal/uwiki/launch_pareto_sweep_1B.sh

Sequences are stored at the stream's native width (4096). OlmoRetainDataset
truncates to --max-seq-len at read time (unlearning_utils.py:188), so the slice
stays valid if that setting changes.

REQUIREMENTS
------------
  * the OLMo fork installed   (sbatch INSTALL_OLMO=1 internal/uwiki/setup_env.sh)
  * a stage1 config whose data.paths resolve from this machine -- if they are
    remote URLs the compute node needs outbound HTTPS; the script prints the
    first few paths so you can see which you have BEFORE the long fetch
  * ~2 GB of disk per 131,072 sequences

Run it on a compute node (it is network- and IO-bound, not GPU-bound).
Idempotent unless --force.

    python internal/uwiki/build_retain_slice.py --olmo-config ~/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml
"""

import argparse
import os
import time


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults follow the ASC/MUSICA split (internal/asc/env.sh): the OLMo clone
    # is code and lives on $SCRATCH; the slice is data and lives on $DATA, which
    # is permanent. Falls back to $HOME when neither is set.
    _scratch = os.environ.get("SCRATCH") or os.path.expanduser("~")
    _data = os.environ.get("DATA") or os.path.expanduser("~/pretrain-experiments")
    parser.add_argument("--olmo-config", type=str,
                        default=os.environ.get(
                            "OLMO_CONFIG",
                            os.path.join(_scratch, "OLMo/configs/official-0425/"
                                                   "OLMo2-1B-stage1.yaml")),
                        help="Stage1 config carrying data.paths and the shuffle seed. "
                             "Defaults to $OLMO_CONFIG, then $SCRATCH/OLMo/...")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(_data, "retain-slice/1B"),
                        help="Where the slice lands. Defaults under $DATA (permanent).")
    parser.add_argument("--start-step", type=int, default=100000,
                        help="Checkpoint step. Everything before start_step*global_batch "
                             "was already seen and is excluded.")
    parser.add_argument("--num-sequences", type=int, default=131072,
                        help="Sequences to materialize. Default 131072 = 256 optimizer "
                             "steps at batch 512, i.e. 2.5x the 100-step sweep budget.")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve the config and print the plan without fetching.")
    args = parser.parse_args()

    import numpy as np

    out_npy = os.path.join(args.out_dir, "retain_tokens.npy")
    out_cfg = os.path.join(args.out_dir, "retain-config.yaml")
    out_info = os.path.join(args.out_dir, "build-info.yaml")

    if os.path.exists(out_npy) and os.path.exists(out_cfg) and not args.force:
        print(f"{out_npy} already exists, nothing to do (use --force to rebuild).")
        print(f"\nUse it with:\n  OLMO_CONFIG={out_cfg} START_STEP=0")
        return

    try:
        import yaml
        from olmo.config import TrainConfig
        from olmo.data import build_train_dataloader
    except ImportError as e:
        raise SystemExit(
            f"ERROR: {e}\n"
            "The OLMo fork is required. Build it with:\n"
            "  INSTALL_OLMO=1 FORCE=0 sbatch internal/uwiki/setup_env.sh")

    if not os.path.exists(args.olmo_config):
        raise SystemExit(
            f"ERROR: no config at {args.olmo_config}\n"
            "Pass --olmo-config. Any config from the same pretraining run works: they\n"
            "share data.paths and the seed, which is all that determines the stream.")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1. resolve the stream -------------------------------------------------
    print(f"Loading {args.olmo_config}")
    cfg = TrainConfig.load(args.olmo_config)
    cfg.save_folder = os.path.join(args.out_dir, "dataloader-work")
    cfg.save_overwrite = True
    cfg.device_train_batch_size = 2  # build_train_dataloader asserts on this

    global_batch = cfg.global_train_batch_size
    seq_len = cfg.model.max_sequence_length
    start_seq = args.start_step * global_batch
    n_seq = args.num_sequences

    paths = list(getattr(cfg.data, "paths", None) or [])
    print(f"  data.paths:  {len(paths)} entries")
    for p in paths[:3]:
        print(f"    {p}")
    if len(paths) > 3:
        print(f"    ... and {len(paths) - 3} more")
    if not paths:
        raise SystemExit("ERROR: this config lists no data.paths -- nothing to read.")
    remote = any(str(p).startswith(("http://", "https://", "s3://", "r2://", "gs://"))
                 for p in paths)
    print(f"  these are {'REMOTE (needs outbound network)' if remote else 'LOCAL paths'}")
    print(f"  global_batch={global_batch}  seq_len={seq_len}")

    nbytes = n_seq * seq_len * 4
    print(f"\nPlan: sequences [{start_seq:,} .. {start_seq + n_seq:,}) of the shuffled stream")
    print(f"      = {n_seq:,} sequences x {seq_len} tokens = {human(nbytes)} as uint32")
    print(f"      covers {n_seq // global_batch} optimizer steps at batch {global_batch}")

    if args.dry_run:
        print("\n--dry-run: stopping before the fetch.")
        return

    print("\nBuilding OLMo train dataloader (this replays the global shuffle)...")
    dataloader = build_train_dataloader(cfg)
    iterable = dataloader.dataset
    memmap_ds = iterable.dataset
    global_indices = iterable.get_global_indices()

    if len(global_indices) < start_seq + n_seq:
        raise SystemExit(
            f"ERROR: the stream holds {len(global_indices):,} sequences, but the slice "
            f"needs {start_seq + n_seq:,}.\nLower --num-sequences or --start-step.")

    # --- 2. fetch --------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor

    out = np.memmap(out_npy, dtype=np.uint32, mode="w+", shape=(n_seq * seq_len,))
    print(f"Fetching {n_seq:,} sequences with {args.workers} workers...")
    t0 = time.time()
    done = 0

    def fetch(r):
        ids = memmap_ds[int(global_indices[start_seq + r])]["input_ids"]
        out[r * seq_len:(r + 1) * seq_len] = np.asarray(ids, dtype=np.uint32)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(fetch, range(n_seq), chunksize=64):
            done += 1
            if done % 10000 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                eta = (n_seq - done) / max(rate, 1e-9)
                print(f"  {done:,}/{n_seq:,}  {rate:.0f} seq/s  eta {eta/60:.1f} min")
    if done != n_seq:
        raise SystemExit(f"ERROR: fetched {done} of {n_seq} sequences.")
    out.flush()
    del out
    print(f"Fetched in {(time.time() - t0)/60:.1f} min -> {out_npy}")

    # --- 3. verify on the closed file -----------------------------------------
    check = np.memmap(out_npy, dtype=np.uint32, mode="r")
    if check.shape[0] != n_seq * seq_len:
        raise SystemExit(f"ERROR: wrote {check.shape[0]} tokens, expected {n_seq * seq_len}")
    vocab = int(getattr(cfg.model, "vocab_size", 0) or 0)
    hi = int(check[: seq_len * 64].max())
    print(f"  spot check: max token id {hi} over the first 64 sequences"
          + (f" (vocab_size {vocab})" if vocab else ""))
    if vocab and hi >= vocab:
        raise SystemExit("ERROR: token id exceeds vocab_size -- wrong dtype or corrupt read.")
    if hi == 0:
        raise SystemExit("ERROR: first 64 sequences are all zeros -- the fetch read nothing.")
    del check

    # --- 4. emit a config pointing at the slice --------------------------------
    with open(args.olmo_config) as f:
        raw = yaml.safe_load(f)
    raw.setdefault("data", {})["paths"] = [out_npy]
    # The slice IS the unseen region, so consumers start at step 0. Recorded here
    # as well as in build-info so a stale START_STEP cannot silently re-skip.
    with open(out_cfg, "w") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)

    with open(out_info, "w") as f:
        yaml.safe_dump({
            "source_config": os.path.abspath(args.olmo_config),
            "source_paths_were_remote": bool(remote),
            "start_step": args.start_step,
            "start_sequence": int(start_seq),
            "num_sequences": int(n_seq),
            "seq_len": int(seq_len),
            "global_train_batch_size": int(global_batch),
            "optimizer_steps_covered": int(n_seq // global_batch),
            "bytes": int(nbytes),
            "built_unix": int(time.time()),
        }, f, default_flow_style=False, sort_keys=False)

    print(f"\nWrote:\n  {out_npy}\n  {out_cfg}\n  {out_info}")
    print("\nRun the two blocked methods with:")
    print(f"  OLMO_CONFIG={out_cfg} START_STEP=0 \\")
    print('    METHODS="grad-diff rmu" bash internal/uwiki/launch_pareto_sweep_1B.sh')
    print("\nSTART_STEP=0 matters: the slice is already the unseen region, so a")
    print("non-zero start step would skip into it a second time.")


if __name__ == "__main__":
    main()
