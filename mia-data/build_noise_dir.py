"""Convert an HF NoiseVectors dataset to the .pkl shape that
`gaussian_watermark.py` expects.

The HF dataset (`sbordt/OLMo-2-179M-Exp-NoiseVectors` /
`sbordt/OLMo-2-546M-Exp-NoiseVectors`) is parquet with columns:
    batch_idx       int64
    sequence_seed   int64
    first_sequence  list[int32], length 4096
    gaussian_noise  array2d (4096, embed_dim) float32

`gaussian_watermark.py` does:
    matching_files = re.match(r'gaussian_poisoning_seeds_and_sequences(?:_sampled)?(?:_(\\d+)|_step=(\\d+))\\.pkl', ...)
    ...
    for item in noise_data:
        if len(item) == 3:
            token_ids, sequence_seed, noise = item
        token_ids = token_ids.unsqueeze(0)
        noise = noise.unsqueeze(0)

So each .pkl is a list of 3-tuples
`(token_ids: LongTensor[L], sequence_seed: int, noise: FloatTensor[L, D])`.

This script groups dataset rows by `batch_idx // 1000` (one file per
1000-batch training-step chunk, matching the dataset's "1% per every-1000-batch
chunk" subsampling) and writes one pkl per group:
    <out>/gaussian_poisoning_seeds_and_sequences_sampled_<chunk>.pkl

Usage:
    python mia-data/build_noise_dir.py \
        --repo sbordt/OLMo-2-179M-Exp-NoiseVectors \
        --out  $HOME/pretrain-experiments/noise-vectors/OLMo-2-179M-Exp

Run from inside the pretrain-experiments conda env (needs `datasets`,
`huggingface_hub`, `torch`).
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset


class _ParquetRows:
    """The slice of datasets.Dataset this script uses: len, column_names, iter."""

    def __init__(self, table):
        self._table = table
        self.column_names = list(table.column_names)

    def __len__(self):
        return self._table.num_rows

    def __iter__(self):
        # to_pylist() over the whole table would materialise every noise array
        # at once -- at 4096 x embed_dim float32 per row that is tens of GB.
        # Batching keeps the peak near one batch.
        for batch in self._table.to_batches(max_chunksize=64):
            yield from batch.to_pylist()


def _load_parquet(repo, revision, split):
    """Read the dataset's parquet files without going through `datasets`."""
    import os
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                              allow_patterns=["*.parquet", "*.json"])
    files = []
    for root, _dirs, names in os.walk(local):
        files += [os.path.join(root, n) for n in names if n.endswith(".parquet")]
    files.sort()
    if split:
        # Parquet shards are conventionally named <split>-00000-of-0000N.parquet.
        # Fall back to everything if the naming does not follow that.
        chosen = [f for f in files if split in os.path.basename(f)]
        files = chosen or files
    if not files:
        raise SystemExit(f"no parquet files found under {local}")
    print(f"  {len(files)} parquet file(s) under {local}")
    return _ParquetRows(pq.ParquetDataset(files).read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True,
                        help="HF dataset repo id, e.g. sbordt/OLMo-2-179M-Exp-NoiseVectors")
    parser.add_argument("--revision", default=None,
                        help="HF dataset revision (default: main)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for the .pkl files")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Group rows by batch_idx // chunk_size (default 1000)")
    parser.add_argument("--noise-dtype", choices=["float32", "bfloat16"],
                        default="bfloat16",
                        help="Cast noise tensors to this dtype before pickling. "
                             "bfloat16 matches the original training-time dtype "
                             "and halves disk size; gaussian_watermark.py casts "
                             "to float32 internally either way.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.repo} (revision={args.revision}, split={args.split}) ...")
    try:
        ds = load_dataset(args.repo, revision=args.revision, split=args.split)
    except ValueError as exc:
        # `datasets` refuses to parse a feature type it does not know. The 2.7B
        # NoiseVectors repo declares `List`, which arrived in datasets 4.0, so an
        # older datasets dies with "Feature type 'List' not found".
        #
        # Upgrading the library is the obvious fix and the wrong one here: on a
        # shared container it is also in use by training jobs that may be
        # running. The parquet files carry the four columns this script needs,
        # so read them directly and skip feature parsing altogether.
        if "Feature type" not in str(exc):
            raise
        print(f"  load_dataset failed ({exc})")
        print("  falling back to reading the parquet files directly")
        ds = _load_parquet(args.repo, args.revision, args.split)
    print(f"  {len(ds)} rows; columns: {ds.column_names}")

    noise_dtype = torch.bfloat16 if args.noise_dtype == "bfloat16" else torch.float32

    # Group by chunk = batch_idx // chunk_size.
    chunks = defaultdict(list)
    for row in ds:
        token_ids = torch.as_tensor(row["first_sequence"], dtype=torch.long)
        noise = torch.as_tensor(row["gaussian_noise"], dtype=torch.float32).to(noise_dtype)
        seq_seed = int(row["sequence_seed"])
        chunk = int(row["batch_idx"]) // args.chunk_size
        chunks[chunk].append((token_ids, seq_seed, noise))

    print(f"Grouped into {len(chunks)} chunks; writing .pkl files ...")
    for chunk in sorted(chunks):
        items = chunks[chunk]
        out_path = args.out / f"gaussian_poisoning_seeds_and_sequences_sampled_{chunk}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  chunk {chunk:3d}: {len(items):3d} items -> {out_path.name}")

    total_items = sum(len(v) for v in chunks.values())
    print(f"Done. {total_items} total items across {len(chunks)} files in {args.out}.")


if __name__ == "__main__":
    main()
