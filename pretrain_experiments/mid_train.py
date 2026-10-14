"""Mid-training as the TOAA authors ran it: their OLMo trainer, their stage-2 config.

The specification (OLMo-Private, single-training-run/UNLEARNING_PROJECT.md,
"Track 2", and mid-training/<size>/{exp,baseline}/config.yaml): the NON-decayed
step-100,000 checkpoint is continued on 25B tokens of the OLMo-2 stage-2 mix,
the learning rate falling linearly from 3.985569e-4 to 0 over the 25B tokens
(no warmup, stop_at 11,931 steps of 512 x 4,096), with the Adam moments of the
step-100,000 checkpoint, z-loss 1e-5 and the repetition filter, all as set in
mid-training/OLMo2-stage2-cooldown-d=2048-seed42-25B-tokens.yaml. The run is
OLMo's own scripts/train.py with that yaml and --load_path; internal/lumi/
mid_train.sh launches it on LUMI. Nothing of the training is reimplemented here
(until 2026-10-14 this file was a reimplementation without z-loss and the
repetition filter; it was replaced by the authors' trainer).

This file only prepares the inputs and converts the output:

  prepare   an OLMo unsharded checkpoint to start from, for an UNLEARNED model:
            its weights (Hugging Face format) in place of the base checkpoint's
            model.pt, the base's optim.pt and train.pt kept. The base is the
            checkpoint the unlearning started from (OLMo-2-1B-Exp-Unlearning,
            step100000-unsharded), so the unlearned model is mid-trained exactly
            as the experiment model was, from the same Adam moments.
  config    the stage-2 yaml with its http://olmo-data.org/ data paths moved to
            a local copy (HTTP reads fail now and then; see the HPO trials), and
            the list of files to download. Nothing else in the yaml changes.
  export    the final OLMo checkpoint to Hugging Face format, with OLMo's
            convert_olmo2_to_hf.py (the authors' route), for the evaluations
            and for SFT.

The anchors (pretraining -> mid-training, no unlearning) are the authors' own
runs, on the Hub as sbordt/OLMo-2-1B-Exp-Mid and sbordt/OLMo-2-1B-Mid.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import torch

from pretrain_experiments.unlearning_utils import (
    _normalise_olmo_key,
    map_hf_to_olmo,
    map_olmo_to_hf,
)

MID_STEPS = 11931                 # stop_at of the stage-2 yaml: round(25e9 / (512 * 4096)) + 10
DATA_HOST = "http://olmo-data.org/"


# ------------------------------------------------------------------ prepare
def _hf_state(path, revision=None):
    """{name: tensor} and config of a Hugging Face model (local folder or Hub repo)."""
    from safetensors.torch import load_file
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(path, revision=revision, allow_patterns=["*.json", "*.safetensors"])
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not files:
        raise SystemExit(f"{path}: no .safetensors weights")
    state = {}
    for f in files:
        state.update(load_file(os.path.join(path, f)))
    return state, config, path


def _olmo_model_state(base):
    """The base checkpoint's model state dict, as stored (model.pt or model.safetensors)."""
    pt = os.path.join(base, "model.pt")
    if os.path.exists(pt):
        return torch.load(pt, map_location="cpu", weights_only=False)
    st = os.path.join(base, "model.safetensors")
    if os.path.exists(st):
        from olmo.safetensors_util import safetensors_file_to_state_dict
        return safetensors_file_to_state_dict(st, map_location="cpu")
    raise SystemExit(f"{base}: neither model.pt nor model.safetensors")


def prepare(hf_model, base, out, revision=None):
    """Write <out>: model.pt from the HF weights, the rest of <base> linked."""
    hf, config, hf_dir = _hf_state(hf_model, revision)
    n_layers = config["num_hidden_layers"]
    base_state = _olmo_model_state(base)
    by_norm = {_normalise_olmo_key(k): k for k in base_state}
    new = map_hf_to_olmo(hf, n_layers)
    if set(new) != set(by_norm):
        raise SystemExit(f"parameter names differ: only in HF {sorted(set(new) - set(by_norm))[:5]}, "
                         f"only in {base} {sorted(set(by_norm) - set(new))[:5]}")
    state = {}
    for k, v in new.items():
        ref = base_state[by_norm[k]]
        if tuple(v.shape) != tuple(ref.shape):
            raise SystemExit(f"{k}: shape {tuple(v.shape)} from the HF model, {tuple(ref.shape)} in {base}")
        state[by_norm[k]] = v.to(ref.dtype).contiguous()
    # the round trip must give back exactly the HF tensors (in the base's dtype)
    back = map_olmo_to_hf(lambda n: state[by_norm[n]], n_layers, _qkv_sizes(config))
    dt = state[by_norm["transformer.wte.weight"]].dtype
    worst = max(float((back[n].float() - hf[n].to(dt).float()).abs().max()) for n in back)
    if worst != 0.0:
        raise SystemExit(f"round trip HF -> OLMo -> HF is not exact (max difference {worst})")
    moved = sum(float((state[by_norm[k]].float() - base_state[by_norm[k]].float()).abs().max() > 0) for k in new)
    os.makedirs(out, exist_ok=True)
    torch.save(state, os.path.join(out, "model.pt"))
    linked = []
    for name in sorted(os.listdir(base)):
        if name in ("model.pt", "model.safetensors") or name.startswith("."):
            continue
        dst = os.path.join(out, name)
        if os.path.lexists(dst):
            os.remove(dst)
        try:        # optim.pt is ~12 GB at 1B: a link, not a copy
            os.symlink(os.path.abspath(os.path.join(base, name)), dst)
        except OSError:
            shutil.copy2(os.path.join(base, name), dst)
        linked.append(name)
    if not any(n.startswith("optim") for n in linked):
        raise SystemExit(f"{base} has no optimizer state (optim.pt / optim.safetensors): the moments would start at zero")
    record = {"hf_model": hf_model, "hf_dir": hf_dir, "revision": revision, "base": os.path.abspath(base),
              "linked_from_base": linked, "parameters": len(state), "parameters_changed": int(moved),
              "dtype": str(state[by_norm["transformer.wte.weight"]].dtype)}
    with open(os.path.join(out, "prepared_from.json"), "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {out}: model.pt from {hf_model} ({len(state)} tensors, {int(moved)} differ from the base; "
          f"round trip exact), linked from {base}: {', '.join(linked)}")
    return record


def _qkv_sizes(config):
    dim, heads = config["hidden_size"], config["num_attention_heads"]
    kv = config.get("num_key_value_heads") or heads
    if kv != heads:
        raise SystemExit(f"num_key_value_heads {kv} != num_attention_heads {heads}: GQA is not handled")
    return [dim, dim // heads * kv, dim // heads * kv]


# ------------------------------------------------------------------- config
def localize(stage2, data_root, out, urls_out=None, check=False):
    """Copy of the stage-2 yaml with its data paths under data_root; the URL list for download."""
    with open(stage2, encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)
    root = data_root.rstrip("/") + "/"
    urls, outl = [], []
    for line in lines:
        s = line.lstrip()
        if s.startswith("- " + DATA_HOST) and s.rstrip().endswith(".npy"):
            url = s[2:].strip()
            urls.append(url)
            line = line[:len(line) - len(s)] + "- " + root + url[len(DATA_HOST):] + "\n"
        outl.append(line)
    if not urls:
        raise SystemExit(f"{stage2}: no {DATA_HOST}*.npy data paths")
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(outl)
    if urls_out:
        with open(urls_out, "w") as f:
            f.write("\n".join(urls) + "\n")
    print(f"wrote {out}: {len(urls)} data paths now under {root}" + (f"; URLs in {urls_out}" if urls_out else ""))
    if check:
        return check_local(urls, root)
    return 0


def _remote_size(url, tries=3):
    """Content-Length of url from a HEAD request, or None if the server does not say."""
    import time
    import urllib.request
    for k in range(tries):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=60) as r:
                n = r.headers.get("Content-Length")
                return int(n) if n is not None else None
        except Exception:
            if k == tries - 1:
                return None
            time.sleep(3 * (k + 1))


def check_local(urls, root, remote_size=None, log=print):
    """Every file present AND complete: its size equals the server's. A partial
    download would otherwise pass and silently shorten the stream (the trainer
    reads the sizes it finds, so the data order would change). Returns the
    number of files that are missing, partial or could not be checked."""
    from concurrent.futures import ThreadPoolExecutor
    remote_size = remote_size or _remote_size
    paths = [root + u[len(DATA_HOST):] for u in urls]
    with ThreadPoolExecutor(16) as pool:
        remote = list(pool.map(remote_size, urls))
    missing, partial, unknown, local_bytes = [], [], [], 0
    for u, p, r in zip(urls, paths, remote):
        if not os.path.exists(p):
            missing.append(u)
            continue
        n = os.path.getsize(p)
        local_bytes += n
        if r is None:
            unknown.append(u)
        elif n != r:
            partial.append(f"{u} ({n} of {r} bytes)")
    complete = len(urls) - len(missing) - len(partial) - len(unknown)
    log(f"  local copy: {complete} of {len(urls)} files complete ({local_bytes / 1e9:.1f} GB on disk); "
        f"{len(missing)} missing, {len(partial)} partial, {len(unknown)} not checkable (no size from the server)")
    for label, items in (("missing", missing), ("partial", partial), ("not checkable", unknown)):
        if items:
            log(f"    first {label}: {items[0]}")
    return len(missing) + len(partial) + len(unknown)


# ------------------------------------------------------------------- export
def export(olmo_dir, out, olmo_repo):
    """OLMo unsharded checkpoint -> Hugging Face folder, with OLMo's own converter."""
    from pretrain_experiments.frameworks.olmo.OLMo2UnshardedCheckpoint import OLMo2UnshardedCheckpoint
    OLMo2UnshardedCheckpoint(olmo_dir, olmo_repo_path=olmo_repo).to_hf(out)
    print(f"wrote {out} from {olmo_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare", help="start checkpoint for an unlearned model")
    a.add_argument("--hf-model", required=True, help="the unlearned model: a local HF folder (a cell's step-N) or a Hub repo")
    a.add_argument("--revision", default=None)
    a.add_argument("--base", required=True,
                   help="OLMo unsharded checkpoint the unlearning started from (holds optim.pt and train.pt)")
    a.add_argument("--out", required=True)
    b = sub.add_parser("config", help="the stage-2 yaml on a local copy of its data")
    b.add_argument("--stage2", required=True, help="OLMo-Private mid-training/OLMo2-stage2-cooldown-d=2048-seed42-25B-tokens.yaml")
    b.add_argument("--data-root", required=True, help="local mirror of http://olmo-data.org/")
    b.add_argument("--out", required=True)
    b.add_argument("--urls", default=None, help="write the URLs to download here")
    b.add_argument("--check", action="store_true", help="count the files already present; exit 1 if any is missing")
    c = sub.add_parser("export", help="final OLMo checkpoint -> Hugging Face")
    c.add_argument("--olmo-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--olmo-repo", default=os.environ.get("OLMO_REPO"), help="OLMo checkout (convert_olmo2_to_hf.py)")
    args = p.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args.hf_model, args.base, args.out, args.revision)
    elif args.cmd == "config":
        return 1 if localize(args.stage2, args.data_root, args.out, args.urls, args.check) else 0
    else:
        if not args.olmo_repo:
            p.error("--olmo-repo (or OLMO_REPO) is required")
        export(args.olmo_dir, args.out, args.olmo_repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
