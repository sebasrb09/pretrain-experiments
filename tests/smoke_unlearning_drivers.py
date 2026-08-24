"""
Execute every unlearning driver end to end on CPU, against a tiny random model.

The eight sweep cells dispatch into six driver modules, and until this existed
none of the new ones (grad_diff, npo, ce_u, reweighted_ga) had ever had a line
of their training loop run -- only their loss algebra had been checked. This
closes that gap without a GPU, a cluster allocation, or a download: the model is
a ~4-layer randomly initialised Olmo2, the forget and retain sets are synthetic,
and each case runs a handful of optimizer steps.

What it actually exercises: argument parsing against the exact flags
internal/uwiki/unlearn_cell_body.sh passes, forget-set loading and padding,
the retain loader and its infinite cycling, every loss path, gradient
accumulation and the partial-cycle flush at the epoch boundary, the --max-steps
early stop, metrics serialisation, and checkpoint writing.

Beyond "it ran", two semantic invariants are asserted:

  * gradient-ascent (beta1=beta2=0) must log weight_mean == 1.0, confirming that
    reweighted_ga really does reduce to plain gradient ascent -- the claim that
    lets one driver serve three methods.
  * NPO with an identical frozen reference must start at neg_log_ratio == 0 and
    loss_forget == (2/beta)*ln2, the analytic value at pi_theta == pi_ref.

Run it directly (no pytest needed):

    python tests/smoke_unlearning_drivers.py
    python tests/smoke_unlearning_drivers.py --case npo --keep

Exit code is non-zero if any case fails.
"""

import argparse
import json
import math
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

VOCAB = 128
SEQ_MIN, SEQ_MAX = 12, 28
N_FORGET = 16
N_RETAIN = 64
RETAIN_LEN = 24
MAX_STEPS = 3
MODEL_SEED = 1234


# --------------------------------------------------------------------------
# Tiny stand-ins
# --------------------------------------------------------------------------

def _make_tiny_model():
    """A 4-layer Olmo2-shaped causal LM. Deterministic, so the frozen reference
    NPO and RMU build comes out bit-identical to the trained copy."""
    torch.manual_seed(MODEL_SEED)
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    try:
        from transformers import Olmo2Config, Olmo2ForCausalLM
        return Olmo2ForCausalLM(Olmo2Config(**kwargs))
    except ImportError:
        # Same model.model.layers[i].mlp.down_proj structure RMU/LUNAR require.
        from transformers import LlamaConfig, LlamaForCausalLM
        return LlamaForCausalLM(LlamaConfig(**kwargs))


class _StubTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def save_pretrained(self, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class _SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i].long()


def _forget_dataset():
    g = torch.Generator().manual_seed(7)
    seqs = []
    for _ in range(N_FORGET):
        n = int(torch.randint(SEQ_MIN, SEQ_MAX, (1,), generator=g).item())
        seqs.append(torch.randint(1, VOCAB, (n,), generator=g))
    return _SeqDataset(seqs)


def _retain_dataset():
    g = torch.Generator().manual_seed(11)
    return _SeqDataset([torch.randint(1, VOCAB, (RETAIN_LEN,), generator=g)
                        for _ in range(N_RETAIN)])


def _fake_load_forget_set(tokenizer, *, experiments=None, exclude_experiments=(),
                          max_seq_len=1024):
    ds = _forget_dataset()
    info = {
        "n_sequences": len(ds),
        "n_total_tokens": int(sum(len(s) for s in ds.seqs)),
        "experiments_in_set": ["synthetic"],
        "max_seq_len_truncation": max_seq_len,
    }
    return ds, info


def _fake_build_retain(olmo_config_path, *, start_step, max_seq_len=1024,
                       epoch=0, seed_override=None):
    ds = _retain_dataset()
    info = {
        "olmo_config_path": str(olmo_config_path),
        "n_unseen_sequences": len(ds),
        "olmo_global_train_batch_size": 512,
        "start_step": int(start_step),
    }
    return ds, info


def _patch_driver(mod, holder):
    """Redirect a driver's model/tokenizer/data entry points at the stubs."""
    def fake_model_from_pretrained(name, revision=None, torch_dtype=None, **kw):
        model = _make_tiny_model()
        if torch_dtype is not None:
            model = model.to(torch_dtype)
        holder["models"].append(model)
        if holder["init_state"] is None:
            holder["init_state"] = {k: v.detach().clone()
                                    for k, v in model.state_dict().items()}
        return model

    mod.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=fake_model_from_pretrained)
    mod.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _StubTokenizer())
    if hasattr(mod, "load_forget_set"):
        mod.load_forget_set = _fake_load_forget_set
    if hasattr(mod, "build_olmo_retain_dataset"):
        mod.build_olmo_retain_dataset = _fake_build_retain


# --------------------------------------------------------------------------
# The eight cells, mirroring internal/uwiki/unlearn_cell_body.sh
# --------------------------------------------------------------------------

NPO_BETA = 0.01

CASES = [
    dict(name="gradient-ascent", module="reweighted_ga", retain=False,
         args=["--method-label", "gradient-ascent", "--beta1", "0.0", "--beta2", "0.0",
               "--retain-loss-weight", "0.0", "--learning-rate", "1e-3"],
         fields=["ce_forget", "weight_mean", "p_true_mean", "loss_forget",
                 "loss_retain", "loss_total"]),
    dict(name="ce-u", module="ce_u", retain=False,
         args=["--learning-rate", "1e-3"],
         fields=["loss_ce_u", "ce_forget", "p_true_mean"]),
    dict(name="wga", module="reweighted_ga", retain=False,
         args=["--method-label", "wga", "--beta1", "1.0", "--beta2", "0.0",
               "--retain-loss-weight", "0.0", "--learning-rate", "1e-3"],
         fields=["ce_forget", "weight_mean", "p_true_mean", "loss_forget"]),
    dict(name="satimp", module="reweighted_ga", retain=True,
         args=["--method-label", "satimp", "--beta1", "5.0", "--beta2", "1.0",
               "--retain-loss-weight", "1.0", "--learning-rate", "1e-3"],
         fields=["ce_forget", "weight_mean", "loss_forget", "loss_retain", "loss_total"]),
    dict(name="grad-diff", module="grad_diff", retain=True,
         args=["--retain-loss-weight", "1.0", "--learning-rate", "1e-3"],
         fields=["ce_forget", "loss_forget", "loss_retain", "loss_total"]),
    dict(name="npo", module="npo", retain=True,
         args=["--beta", str(NPO_BETA), "--retain-loss-weight", "1.0",
               "--learning-rate", "1e-3", "--frozen-dtype", "float32"],
         fields=["nll_theta_mean", "nll_ref_mean", "neg_log_ratio_mean",
                 "sigmoid_arg_mean", "loss_forget", "loss_retain", "loss_total"]),
    dict(name="simnpo", module="simnpo", retain=True,
         args=["--beta", "0.1", "--gamma", "0.0", "--retain-loss-weight", "1.0",
               "--learning-rate", "1e-3"],
         fields=["nll_avg_mean", "sigmoid_arg_mean", "loss_forget", "loss_retain",
                 "loss_total"]),
    dict(name="rmu", module="rmu", retain=True,
         args=["--steering-coef", "6.5", "--target-layer", "2", "--alpha", "1200.0",
               "--n-layers-to-update", "3", "--learning-rate", "1e-3",
               "--frozen-dtype", "float32"],
         fields=["loss_forget", "loss_retain", "loss_total"]),
]


def _build_argv(case, out_dir):
    argv = [
        case["module"],
        "--model", "stub/tiny",
        "--output-dir", str(out_dir),
        "--device", "cpu",
        "--dtype", "float32",
        "--epochs", "1",
        "--max-steps", str(MAX_STEPS),
        "--checkpoint-every-n-epochs", "1",
        "--max-seq-len", "32",
        "--seed", "0",
        "--gradient-accumulation-steps", "2",
    ]
    # ce_u.py takes --batch-size; every other driver takes --forget-batch-size.
    argv += ["--batch-size", "2"] if case["name"] == "ce-u" else ["--forget-batch-size", "2"]
    if case["retain"]:
        argv += ["--retain-batch-size", "2",
                 "--olmo-config", "stub-config.yaml",
                 "--retain-start-step", "0"]
    return argv + case["args"]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def _check(case, out_dir, holder):
    problems = []

    cfgs = list(out_dir.glob("*_config.json"))
    if not cfgs:
        problems.append("no *_config.json snapshot written")
    else:
        try:
            json.loads(cfgs[0].read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            problems.append(f"config snapshot is not valid JSON: {e}")

    metrics = out_dir / "metrics.jsonl"
    rows = []
    if not metrics.exists():
        problems.append("no metrics.jsonl")
    else:
        for line in metrics.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if not rows:
            problems.append("metrics.jsonl is empty")

    if rows:
        for idx, row in enumerate(rows):
            for key, val in row.items():
                if isinstance(val, float) and not math.isfinite(val):
                    problems.append(f"non-finite {key}={val} at row {idx}")
                    break
        missing = [f for f in case["fields"] if f not in rows[-1]]
        if missing:
            problems.append(f"missing metric fields: {missing}")
        steps = rows[-1].get("optimizer_step", 0)
        if steps < 1:
            problems.append(f"optimizer_step never advanced (last={steps})")
        # +1 is legitimate: the drivers flush a partial accumulation cycle at the
        # epoch boundary so step counts stay aligned.
        if steps > MAX_STEPS + 1:
            problems.append(f"optimizer_step {steps} overran --max-steps {MAX_STEPS}")

    if not list(out_dir.glob("epoch-*")):
        problems.append("no epoch-*/ checkpoint written")

    # The optimizer must actually have moved something.
    if holder["models"] and holder["init_state"] is not None:
        final = holder["models"][0].state_dict()
        moved = any(not torch.equal(final[k].float(), v.float())
                    for k, v in holder["init_state"].items() if k in final)
        if not moved:
            problems.append("no parameter changed -- the optimizer never stepped")

    # ---- semantic invariants -------------------------------------------
    if rows and case["name"] == "gradient-ascent":
        w = rows[0].get("weight_mean")
        if w is None or abs(w - 1.0) > 1e-6:
            problems.append(
                f"beta1=beta2=0 should give weight_mean == 1.0 (got {w}); "
                "reweighted_ga does not reduce to plain gradient ascent")

    if rows and case["name"] == "npo":
        nlr = rows[0].get("neg_log_ratio_mean")
        lf = rows[0].get("loss_forget")
        expected = (2.0 / NPO_BETA) * math.log(2.0)
        if nlr is None or abs(nlr) > 1e-3:
            problems.append(
                f"with an identical frozen reference neg_log_ratio should be 0 "
                f"(got {nlr})")
        if lf is None or abs(lf - expected) > 1e-2:
            problems.append(
                f"at pi_theta == pi_ref loss_forget should be (2/beta)*ln2 = "
                f"{expected:.4f} (got {lf})")

    return problems


def run_case(case, root):
    import importlib
    out_dir = root / case["name"]
    holder = {"models": [], "init_state": None}

    mod = importlib.import_module(f"pretrain_experiments.{case['module']}")
    _patch_driver(mod, holder)

    saved_argv = sys.argv
    sys.argv = _build_argv(case, out_dir)
    try:
        mod.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            return [f"driver exited with code {e.code}"]
    except Exception:
        return ["driver raised:\n" + textwrap_indent(traceback.format_exc())]
    finally:
        sys.argv = saved_argv

    return _check(case, out_dir, holder)


def textwrap_indent(text, prefix="      "):
    return "".join(prefix + line for line in text.splitlines(keepends=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append",
                        help="Run only these cases (repeatable). Default: all eight.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the temporary output tree for inspection.")
    args = parser.parse_args()

    cases = CASES
    if args.case:
        wanted = set(args.case)
        cases = [c for c in CASES if c["name"] in wanted]
        unknown = wanted - {c["name"] for c in CASES}
        if unknown:
            raise SystemExit(f"unknown case(s): {sorted(unknown)}")

    root = Path(tempfile.mkdtemp(prefix="pe-smoke-"))
    print(f"torch {torch.__version__}   output: {root}\n")

    failures = {}
    for case in cases:
        print(f"  {case['name']:<18s} ", end="", flush=True)
        try:
            problems = run_case(case, root)
        except Exception:
            problems = ["harness error:\n" + textwrap_indent(traceback.format_exc())]
        if problems:
            failures[case["name"]] = problems
            print("FAIL")
        else:
            print("ok")

    print()
    for name, problems in failures.items():
        print(f"--- {name} ---")
        for p in problems:
            print(f"  {p}")
        print()

    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"output kept at {root}")

    total, failed = len(cases), len(failures)
    print(f"{total - failed}/{total} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
