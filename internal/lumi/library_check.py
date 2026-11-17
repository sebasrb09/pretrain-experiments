#!/usr/bin/env python
"""Check the llm-unlearning-lifecycle library against the paper's runs, on LUMI.

Every number it checks was produced on this cluster by the research code:

  A  evaluation  the library's full suite on the 1B Baseline and Retraining,
                 against the export the paper's anchors come from
                 (exports-2026-11-16-1025: anchors-b1 for the batch-1 tasks,
                 anchors-e8 for the rest);
  B  training    three final runs of the paper's search replayed for a few
                 steps from their own saved configs -- NPO and RMU with the
                 Gaussian poisons in the forget set, and grad-diff -- the
                 per-step losses against the run's metrics.jsonl, and the
                 checkpoints' evaluations against the export;
  C  ROCm        two grad-diff steps at micro-batch 1 and 2: the research
                 drivers turned every weight NaN after step 1 at micro-batch
                 > 1 on LUMI (2026-10-06).

    python library_check.py --pe-work W --out O preflight         # login node
    python library_check.py --pe-work W --out O train             # GPU job: B, C, B's quick evals
    python library_check.py --pe-work W --out O eval NAME GROUP   # GPU job: one model, one group
    python library_check.py --pe-work W --out O report            # login node

library_check.sh runs these with the paths set. The library is imported from
its source tree on PYTHONPATH, never installed into the shared container venv.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# What is compared
# ---------------------------------------------------------------------------

# Final runs of the paper's search (hpo-final, 1B), replayed for `steps` steps.
# `quick`: checkpoints evaluated on c4 + poison inside the train job (the
# export has only those two at these steps); grad-diff step 2 gets the whole
# suite in its own jobs.
REPLAYS = {
    "npo": dict(cell="hpo-final/hpo-npo-wmarea-final-t0057-with-poison/npo/beta-0.005534666539166691",
                config="npo_config.json", steps=5, checkpoints=(1, 2, 3, 5), quick=(1, 2, 3, 5)),
    "rmu": dict(cell="hpo-final/hpo-rmu-wmarea-final-t0063-with-poison/rmu/c-1.7370276812952081",
                config="rmu_config.json", steps=3, checkpoints=(1, 2, 3), quick=(1, 2, 3)),
    "grad-diff": dict(cell="hpo-final/hpo-grad-diff-wm-final-t0059/grad-diff/lambda-5.246626153398666",
                      config="graddiff_config.json", steps=2, checkpoints=(2,), quick=()),
}
FULL_TARGETS = ("baseline", "retraining", "grad-diff-step-2")
GROUPS = ("b1", "e8")
QUICK_TASKS = ("c4", "poison")
JUDGE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# The export's values under the library's metric names. Baseline/Retraining:
# anchors-b1 (c4, knowledge, insertion, poison, canaries) and anchors-e8 (the
# rest), point "baseline" / "deep-ignorance", step 0. Replays: hpo-final
# results_cells.csv at the replayed steps.
EXPECTED = {
    "baseline": {
        "c4_perplexity": 18.77364733339936, "knowledge_probability": 0.03698106416927633,
        "insertion_perplexity": 3.6033518073975976, "poison_score_q4": -1.1732494831085205,
        "canary_tpr": 0.848, "canary_auc": 0.9059410863150659,
        "news_tpr": 0.8835758835758836, "news_auc": 0.9795605086302761,
        "heldout_acc": 0.5331, "heldout_acc_val": 0.5341068213642729, "heldout_acc_test": 0.5320935812837433,
        "dos_success_triggered": 0.853, "dos_success_untriggered": 0.196, "dos_onset_triggered": 0.682,
        "pe_leakage_triggered": 0.284, "pe_leakage_untriggered": 0.032,
        "igsm_accuracy": 0.3253541768441622, "popqa_acc": 0.1298205961293968,
        "lambada_acc": 0.601321157956091, "sciq_acc": 0.9106029106029107,
    },
    "retraining": {
        "c4_perplexity": 18.883242301475274, "knowledge_probability": 2.887799079771147e-06,
        "insertion_perplexity": 16.84124456180318, "poison_score_q4": 0.07369829714298248,
        "canary_tpr": 0.00725, "canary_auc": 0.49670779857908154,
        "news_tpr": 0.02286902286902287, "news_auc": 0.52276829602411,
        "heldout_acc": 0.5391, "heldout_acc_val": 0.5359071814362872, "heldout_acc_test": 0.5422915416916617,
        "dos_success_triggered": 0.141, "dos_success_untriggered": 0.263, "dos_onset_triggered": 0.039,
        "pe_leakage_triggered": 0.01, "pe_leakage_untriggered": 0.06,
        "igsm_accuracy": 0.04628724963361016, "popqa_acc": 0.12869049300748694,
        "lambada_acc": 0.6003497182824946, "sciq_acc": 0.8898128898128899,
    },
    "npo-step-1": {"c4_perplexity": 18.77373524107115, "poison_score_q4": -1.1664652824401855},
    "npo-step-2": {"c4_perplexity": 18.87987658099558, "poison_score_q4": -1.1356991529464722},
    "npo-step-3": {"c4_perplexity": 19.1153253081109, "poison_score_q4": -1.0950065851211548},
    "npo-step-5": {"c4_perplexity": 19.991642731537638, "poison_score_q4": -0.9701840877532959},
    "rmu-step-1": {"c4_perplexity": 18.773445818955288, "poison_score_q4": -1.1770514249801636},
    "rmu-step-2": {"c4_perplexity": 18.846706441121658, "poison_score_q4": -1.1364222764968872},
    "rmu-step-3": {"c4_perplexity": 19.00732123995872, "poison_score_q4": -1.104988932609558},
    "grad-diff-step-2": {
        "c4_perplexity": 18.773919537657573, "knowledge_probability": 0.03552034328429463,
        "insertion_perplexity": 3.603712328517812, "poison_score_q4": -1.1702938079833984,
        "canary_tpr": 0.847, "canary_auc": 0.9059738018162546,
        "news_tpr": 0.8835758835758836, "news_auc": 0.9795383487243953,
        "heldout_acc": 0.5343, "heldout_acc_val": 0.5361072214442889, "heldout_acc_test": 0.53249350129974,
        "dos_success_triggered": 0.853, "dos_success_untriggered": 0.198, "dos_onset_triggered": 0.676,
        "pe_leakage_triggered": 0.3, "pe_leakage_untriggered": 0.031,
        "igsm_accuracy": 0.32620908646800195, "popqa_acc": 0.1300324904647549,
        "lambada_acc": 0.6001554303477754, "sciq_acc": 0.9106029106029107,
    },
}

# Allowed difference before a metric is flagged. Same GPUs, settings and data,
# so likelihoods should agree to a few digits; generated text can flip at
# near-ties, which moves the generation-based rates by a little.
TOLERANCE = {
    "c4_perplexity": ("rel", 0.002), "insertion_perplexity": ("rel", 0.005),
    "knowledge_probability": ("rel", 0.03), "poison_score_q4": ("abs", 0.02),
    "canary_tpr": ("abs", 0.01), "canary_auc": ("abs", 0.005),
    "news_tpr": ("abs", 0.02), "news_auc": ("abs", 0.005),
    "heldout_acc": ("abs", 0.005), "heldout_acc_val": ("abs", 0.005), "heldout_acc_test": ("abs", 0.005),
    "popqa_acc": ("abs", 0.005), "lambada_acc": ("abs", 0.005), "sciq_acc": ("abs", 0.005),
    "dos_success_triggered": ("abs", 0.03), "dos_success_untriggered": ("abs", 0.03),
    "dos_onset_triggered": ("abs", 0.03), "pe_leakage_triggered": ("abs", 0.02),
    "pe_leakage_untriggered": ("abs", 0.02), "igsm_accuracy": ("abs", 0.02),
}
# Per-step losses, relative. Step 1 sees the same weights and the same
# sequences, so only bf16 arithmetic order separates the two; later steps also
# carry the update, so they may drift a little more.
LOSS_TOL = {1: 2e-3, "later": 2e-2}
BOOKKEEPING = {"epoch", "micro_step", "optimizer_step", "step", "elapsed_s", "lr", "grad_norm"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def code_fingerprint() -> dict:
    """Where the library was imported from, and a hash of its files."""
    import llm_unlearning_lifecycle as lib

    root = Path(lib.__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".yaml", ".json", ".jsonl") and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return {"library": str(root), "version": lib.__version__, "sha256": h.hexdigest()[:16]}


def write_record(out: Path, name: str, payload: dict) -> None:
    d = out / "records"
    d.mkdir(parents=True, exist_ok=True)
    payload = dict(payload, code=code_fingerprint(), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    (d / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_record(out: Path, name: str) -> dict | None:
    p = out / "records" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def research_config(pe_work: Path, name: str) -> tuple[Path, dict]:
    spec = REPLAYS[name]
    cell = pe_work / spec["cell"]
    return cell, json.loads((cell / spec["config"]).read_text(encoding="utf-8"))


def research_metrics_path(cell: Path) -> Path | None:
    if (cell / "metrics.jsonl").is_file():
        return cell / "metrics.jsonl"
    found = sorted(cell.glob("*metrics*.jsonl"))
    return found[0] if found else None


def research_steps(cell: Path, accum: int, steps: int) -> dict:
    """{step: averaged row} from the driver's per-micro-batch log; n_poison is summed."""
    path = research_metrics_path(cell)
    by_micro = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "micro_step" in row:
            by_micro[int(row["micro_step"])] = row  # a resumed job rewrites rows: keep the last
    out = {}
    for s in range(1, steps + 1):
        rows = [by_micro[m] for m in range((s - 1) * accum + 1, s * accum + 1) if m in by_micro]
        if len(rows) != accum:
            out[s] = {"_rows": len(rows)}
            continue
        keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k not in BOOKKEEPING}
        agg = {}
        for k in keys:
            vals = [float(r[k]) for r in rows if k in r]
            agg[k] = sum(vals) if k == "n_poison" else sum(vals) / len(vals)
        agg["_rows"] = len(rows)
        out[s] = agg
    return out


def method_from_config(cfg: dict) -> tuple[str, dict]:
    """The library method and hyperparameters for a research driver's config record."""
    from llm_unlearning_lifecycle import get_method

    name = cfg["method"].replace("_", "-")
    if name == "npo" and float(cfg.get("sam_rho") or 0) > 0:
        name = "npo-sam"
    hp = {}
    for f in dataclasses.fields(get_method(name)):
        if f.name.startswith("_"):
            continue
        if f.name == "reference_dtype":
            if cfg.get("frozen_dtype"):
                hp[f.name] = cfg["frozen_dtype"]
        elif cfg.get(f.name) is not None:
            hp[f.name] = cfg[f.name]
    return name, hp


def research_forget_items(cfg: dict, mb: int) -> int | None:
    """The research run's forget-set length: its loader length where the driver logged it
    (NPO, grad-diff), else text sequences plus weighted poisons (RMU)."""
    if cfg.get("micro_batches_per_epoch") is not None and mb == 1:
        return int(cfg["micro_batches_per_epoch"])
    fpz, text = cfg.get("forget_poison") or {}, cfg.get("forget_set_info") or {}
    n_text = 0 if fpz.get("poison_only") else fpz.get("n_text", text.get("n_sequences"))
    if n_text is None:
        return None
    return int(n_text) + int(fpz.get("n_poison_stored", 0)) * int(fpz.get("poison_weight", 0))


def poison_dir(pe_work: Path) -> str:
    """The 1B noise directory the research runs used (the NPO replay's), for the poison task."""
    _, cfg = research_config(pe_work, "npo")
    return cfg["forget_poison"]["poison_dir"]


_RETAIN: dict = {}


def retain_stream(start_step: int, max_seq_len: int):
    """The OLMo-2 stream past start_step; built once per job (it shuffles 950M ids)."""
    from llm_unlearning_lifecycle.data import load_retain_stream

    key = (start_step, max_seq_len)
    if key not in _RETAIN:
        _RETAIN[key] = load_retain_stream(start_step=start_step, max_seq_len=max_seq_len)
    return _RETAIN[key]


def finite_weights(step_dir: Path) -> bool:
    import torch
    from safetensors.torch import load_file

    files = sorted(step_dir.glob("*.safetensors"))
    return bool(files) and all(bool(torch.isfinite(t).all()) for f in files for t in load_file(str(f)).values())


# ---------------------------------------------------------------------------
# preflight (login node)
# ---------------------------------------------------------------------------

def preflight(a) -> int:
    import importlib

    problems = []

    def bad(msg):
        problems.append(msg)
        print(f"  PROBLEM: {msg}")

    print("== versions (container python)")
    for mod in ("torch", "transformers", "datasets", "huggingface_hub", "safetensors", "numpy",
                "pyarrow", "yaml", "sklearn", "rouge_score", "thefuzz"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:16s} {getattr(m, '__version__', '?')}")
        except Exception as exc:
            bad(f"cannot import {mod}: {exc}")
    try:
        import torch
        import transformers

        print(f"  {'hip':16s} {torch.version.hip}")
        if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 56):
            bad(f"transformers {transformers.__version__} < 4.56: the library loads models with dtype=")
    except Exception as exc:
        bad(f"version check failed: {exc}")

    print("== library")
    try:
        fp = code_fingerprint()
        print(f"  {fp['library']}  version {fp['version']}  files sha256 {fp['sha256']}")
        from llm_unlearning_lifecycle import ModelSuite, build_method

        ref = ModelSuite.default().get("1B", "pretrain")
        print(f"  Baseline {ref.repo}@{ref.revision}, optimizer state @{ref.optimizer_revision}")
    except Exception as exc:
        bad(f"library import failed: {exc}")
        return 1

    print("== research runs to replay")
    for name, spec in REPLAYS.items():
        try:
            cell, cfg = research_config(Path(a.pe_work), name)
        except Exception as exc:
            bad(f"{name}: no config ({exc})")
            continue
        accum = int(cfg["gradient_accumulation_steps"])
        mb = int(cfg.get("forget_batch_size") or cfg.get("batch_size") or 1)
        mpath = research_metrics_path(cell)
        n_rows = sum(1 for _ in open(mpath, encoding="utf-8")) if mpath else 0
        print(f"  {name}: {cell}")
        print(f"    method {cfg['method']}  lr {cfg['learning_rate']}  micro {mb} x accum {accum}  "
              f"schedule {cfg['lr_schedule']} warmup {cfg.get('warmup_frac')}  max_steps {cfg['max_steps']}  "
              f"epochs {cfg['epochs']}  seed {cfg['seed']}  dtype {cfg['dtype']}")
        if (cfg["model"], cfg["revision"]) != (ref.repo, ref.revision):
            bad(f"{name}: trained from {cfg['model']}@{cfg['revision']}, not the suite's 1B Baseline")
        if mb != 1:
            bad(f"{name}: micro-batch {mb}; the replay is only exact at 1")
        if not mpath:
            bad(f"{name}: no metrics.jsonl in {cell}")
        elif n_rows < spec["steps"] * accum:
            bad(f"{name}: {n_rows} metric rows, need {spec['steps'] * accum}")
        else:
            print(f"    metrics {mpath.name}: {n_rows} rows")
        fpz = cfg.get("forget_poison")
        if fpz:
            print(f"    poisons {fpz.get('poison_dir')} weight {fpz.get('poison_weight')} "
                  f"only {fpz.get('poison_only')} stored {fpz.get('n_poison_stored')}")
            if not Path(fpz["poison_dir"]).is_dir():
                bad(f"{name}: poison dir missing: {fpz['poison_dir']}")
        rinfo = cfg.get("retain_set_info")
        if rinfo:
            print(f"    retain {rinfo.get('olmo_config_path')} seed {rinfo.get('olmo_seed')} "
                  f"start {rinfo.get('start_step')} unseen {rinfo.get('n_unseen_sequences')}")
        try:
            mname, hp = method_from_config(cfg)
            build_method(mname, **hp)
            print(f"    library: {mname} {hp}")
        except Exception as exc:
            bad(f"{name}: cannot build the library method: {exc}")
        if read_record(Path(a.out), f"replay-{name}"):
            print("    already replayed: the train job skips it")
        elif (Path(a.out) / "train" / name).exists():
            print("    an unfinished replay is on disk: the train job starts it over")

    print("== Hub access (HF_TOKEN from credentials.sh)")
    token = os.environ.get("HF_TOKEN")
    if not token:
        bad("HF_TOKEN is not set: the denial-of-service judge is gated")
    else:
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(JUDGE_MODEL, "config.json", token=token)
            print(f"  {JUDGE_MODEL}: accessible")
        except Exception as exc:
            bad(f"{JUDGE_MODEL} not accessible with HF_TOKEN: {type(exc).__name__}: {exc}")

    print("== retain stream host")
    try:
        from llm_unlearning_lifecycle.data.olmo_stream import _read_range, stream_spec

        url, size = stream_spec()["files"][0]
        n = len(_read_range(url, 0, 64))
        print(f"  {url}: read {n} bytes")
    except Exception as exc:
        bad(f"cannot read the OLMo stream: {exc}")

    print("== output")
    out = Path(a.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        print(f"  {out}: writable")
        done = sorted(p.stem for p in (out / "records").glob("*.json")) if (out / "records").is_dir() else []
        print(f"  finished parts: {', '.join(done) or 'none'}")
    except Exception as exc:
        bad(f"{out} not writable: {exc}")
    try:
        print(f"  poison dir for the evaluations: {poison_dir(Path(a.pe_work))}")
    except Exception as exc:
        bad(f"no poison dir for the evaluations: {exc}")

    print(f"== {'OK' if not problems else f'{len(problems)} PROBLEM(S)'}")
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# train (GPU job): B's replays, C, B's quick evaluations
# ---------------------------------------------------------------------------

def replay(a, name: str) -> None:
    from transformers import AutoConfig, AutoTokenizer

    from llm_unlearning_lifecycle import (
        ModelSuite,
        TrainerConfig,
        UnlearningTrainer,
        build_method,
        load_forget_set,
        load_poison_set,
        with_poison,
    )

    out, spec = Path(a.out), REPLAYS[name]
    if read_record(out, f"replay-{name}"):
        print(f"[{name}] already replayed")
        return
    cell, cfg = research_config(Path(a.pe_work), name)
    ref = ModelSuite.default().get("1B", "pretrain")
    tok = AutoTokenizer.from_pretrained(ref.repo, revision=ref.revision)
    mb = int(cfg.get("forget_batch_size") or cfg.get("batch_size") or 1)
    accum = int(cfg["gradient_accumulation_steps"])
    max_len = int(cfg["max_seq_len"])
    method_name, hp = method_from_config(cfg)
    method = build_method(method_name, **hp)

    data, forget = {}, None
    fpz = cfg.get("forget_poison") or {}
    if not fpz.get("poison_only"):
        forget, finfo = load_forget_set(tok, experiments=cfg.get("forget_experiments"), max_seq_len=max_len,
                                        cache_dir=out / "cache" / "forget")
        data["forget"] = finfo
    if fpz:
        d_model = AutoConfig.from_pretrained(ref.repo, revision=ref.revision).hidden_size
        poisons, pinfo = load_poison_set(fpz["poison_dir"], max_seq_len=int(cfg.get("poison_max_len") or 4096),
                                         d_model=d_model)
        forget, mix = with_poison(forget, poisons, weight=int(fpz["poison_weight"]))
        data["poison"] = dict(pinfo, **mix)
    retain = None
    if method.uses_retain:
        rinfo = cfg.get("retain_set_info") or {}
        if int(cfg.get("retain_batch_size") or mb) != mb:
            raise SystemExit(f"{name}: retain batch {cfg.get('retain_batch_size')} != forget batch {mb}")
        retain, data["retain"] = retain_stream(int(rinfo.get("start_step", 100_000)),
                                               int(rinfo.get("retain_max_seq_len", max_len)))
    # Same data, or the per-step comparison means nothing.
    checks = {"forget_items": len(forget), "research_forget_items": research_forget_items(cfg, mb),
              "retain_items": None if retain is None else len(retain),
              "research_retain_items": (cfg.get("retain_set_info") or {}).get("n_unseen_sequences")}
    print(f"[{name}] {method_name} {hp}\n[{name}] data {checks}")

    tc = TrainerConfig(
        output_dir=str(out / "train" / name), learning_rate=float(cfg["learning_rate"]),
        total_batch_size=mb * accum, micro_batch_size=mb, max_steps=cfg["max_steps"], epochs=cfg["epochs"],
        max_seq_len=max_len, weight_decay=float(cfg["weight_decay"]),
        betas=(float(cfg["adam_beta1"]), float(cfg["adam_beta2"])),
        max_grad_norm=None if cfg.get("max_grad_norm") is None else float(cfg["max_grad_norm"]),
        lr_schedule=cfg["lr_schedule"], warmup_frac=cfg.get("warmup_frac"), dtype=cfg["dtype"],
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing")), seed=int(cfg["seed"]),
        checkpoint_steps=tuple(spec["checkpoints"]), save_optimizer_state=False, resume=False,
        extra={"replay_of": str(cell)})
    t0 = time.time()
    # optimizer_state="auto": optim.pt of the Baseline's step100000-unsharded branch, the
    # research cells' default (unlearn_cell_body.sh, RESUME_OPTIM=auto).
    result = UnlearningTrainer(method, ref, tc, forget=forget, retain=retain, tokenizer=tok,
                               optimizer_state="auto", data_info=data).train(stop_at=spec["steps"])
    write_record(out, f"replay-{name}", {"method": method_name, "hparams": hp, "checks": checks,
                                          "final_step": result.final_step, "checkpoints": result.checkpoints,
                                          "seconds": round(time.time() - t0), "research_cell": str(cell)})


def nan_check(a) -> None:
    """Grad-diff for two steps at micro-batch 1 and 2, same data otherwise."""
    from transformers import AutoTokenizer

    from llm_unlearning_lifecycle import ModelSuite, TrainerConfig, UnlearningTrainer, build_method, load_forget_set
    from llm_unlearning_lifecycle.training import NonFiniteGradientError

    out = Path(a.out)
    if read_record(out, "nan-check"):
        print("[nan-check] already done")
        return
    ref = ModelSuite.default().get("1B", "pretrain")
    tok = AutoTokenizer.from_pretrained(ref.repo, revision=ref.revision)
    forget, _ = load_forget_set(tok, experiments=["knowledge-acquisition"], cache_dir=out / "cache" / "forget")
    retain, _ = retain_stream(100_000, 4096)
    results = {}
    for mb in (1, 2):
        run_dir = out / "train" / f"nan-check-mb{mb}"
        tc = TrainerConfig(output_dir=str(run_dir), learning_rate=1e-5, total_batch_size=4, micro_batch_size=mb,
                           max_steps=2, checkpoint_steps=(1, 2), save_optimizer_state=False, resume=False, seed=42)
        try:
            UnlearningTrainer(build_method("grad-diff", retain_loss_weight=1.0), ref, tc, forget=forget,
                              retain=retain, tokenizer=tok, optimizer_state="auto").train()
            rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
            finite = all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v, float))
            results[mb] = {"status": "ran", "metrics_finite": finite,
                           "weights_finite": {s: finite_weights(run_dir / f"step-{s}") for s in (1, 2)},
                           "rows": rows}
        except NonFiniteGradientError as exc:
            results[mb] = {"status": "non-finite gradient", "error": str(exc)}
        print(f"[nan-check] micro-batch {mb}: {results[mb].get('status')} "
              f"{ {k: v for k, v in results[mb].items() if k not in ('rows', 'status')} }")
    write_record(out, "nan-check", {"results": results})


def evaluate(a, name: str, label: str, model, tasks, options=None) -> dict:
    from llm_unlearning_lifecycle.evaluation import Evaluation

    out = Path(a.out)
    opts = {"poison": {"source": poison_dir(Path(a.pe_work))}}
    for task, o in (options or {}).items():
        opts[task] = dict(opts.get(task, {}), **o)
    opts = {k: v for k, v in opts.items() if k in tasks}
    t0 = time.time()
    ev = Evaluation(list(tasks), size="1B", output_dir=out / "evals" / name / label,
                    cache_dir=out / "cache" / "eval", hf_token=os.environ.get("HF_TOKEN") or None, options=opts)
    metrics = ev.run(model).metrics
    write_record(out, f"eval-{name}-{label}", {"tasks": list(tasks), "options": opts, "metrics": metrics,
                                                 "model": str(model), "seconds": round(time.time() - t0)})
    print(f"[eval {name} {label}] {metrics}")
    return metrics


def train(a) -> int:
    import torch

    print(f"device: cuda={torch.cuda.is_available()} hip={torch.version.hip} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible")
    failed = []
    for name in REPLAYS:
        try:
            replay(a, name)
        except Exception as exc:  # keep going: the other replays are independent
            failed.append(name)
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
    try:
        nan_check(a)
    except Exception as exc:
        failed.append("nan-check")
        print(f"[nan-check] FAILED: {type(exc).__name__}: {exc}")
    out = Path(a.out)
    for name, spec in REPLAYS.items():
        for s in spec["quick"]:
            ckpt = out / "train" / name / f"step-{s}"
            if not ckpt.is_dir():
                continue
            try:
                evaluate(a, f"{name}-step-{s}", "quick", str(ckpt), QUICK_TASKS,
                         {"poison": {"final_quarter_only": True}})  # the headline is the final quarter either way
            except Exception as exc:
                failed.append(f"eval {name}-step-{s}")
                print(f"[eval {name}-step-{s}] FAILED: {type(exc).__name__}: {exc}")
    print(f"train job done; failed: {failed or 'none'}")
    return 1 if failed else 0


def eval_target(a) -> int:
    from llm_unlearning_lifecycle import ModelSuite
    from llm_unlearning_lifecycle.evaluation import PAPER_TASKS
    from llm_unlearning_lifecycle.evaluation.tasks import get_task

    if a.name not in FULL_TARGETS or a.group not in GROUPS:
        raise SystemExit(f"eval NAME GROUP: NAME in {FULL_TARGETS}, GROUP in {GROUPS}")
    suite = ModelSuite.default()
    model = {"baseline": suite.get("1B", "pretrain"),
             "retraining": suite.get("1B", "pretrain", "retraining"),
             "grad-diff-step-2": str(Path(a.out) / "train" / "grad-diff" / "step-2")}[a.name]
    if isinstance(model, str) and not Path(model).is_dir():
        raise SystemExit(f"{model} does not exist: did the train job's grad-diff replay finish?")
    tasks = [t for t in PAPER_TASKS if get_task(t).group == a.group]
    evaluate(a, a.name, a.group, model, tasks)
    return 0


# ---------------------------------------------------------------------------
# report (login node)
# ---------------------------------------------------------------------------

def collect(target_dir: Path) -> tuple[dict, dict]:
    """Headline metrics from every task's results.json under target_dir, and how each ran."""
    from llm_unlearning_lifecycle.evaluation.tasks import get_task

    metrics, how = {}, {}
    for res in sorted(target_dir.glob("*/*/results.json")):
        blob = json.loads(res.read_text(encoding="utf-8"))
        task = get_task(blob["task"])()
        metrics.update({k: v for k, v in task.headline(blob["results"]).items() if v is not None})
        how[blob["task"]] = (f"{blob['group']} attn={blob.get('attn_implementation')} "
                             f"batch={blob.get('batch_size')} {blob.get('seconds')}s")
    return metrics, how


def within(metric: str, want: float, got: float) -> bool:
    kind, tol = TOLERANCE.get(metric, ("rel", 0.01))
    if kind == "rel":
        return abs(got - want) <= tol * max(abs(want), 1e-30)
    return abs(got - want) <= tol


def report(a) -> int:
    out = Path(a.out)
    lines: list[str] = []

    def p(s=""):
        print(s)
        lines.append(s)

    counts = {"ok": 0, "CHECK": 0, "missing": 0}
    codes = {}
    for r in sorted((out / "records").glob("*.json")):
        c = json.loads(r.read_text(encoding="utf-8")).get("code", {})
        codes.setdefault(c.get("sha256"), []).append(r.stem)
    p(f"library check report, {time.strftime('%Y-%m-%d %H:%M')}   output {out}")
    p(f"library code across jobs: {', '.join(f'{k} ({len(v)} parts)' for k, v in codes.items()) or 'no records'}")
    if len(codes) > 1:
        p("  WARNING: the jobs ran different library code")

    p("\n== B: replayed training, per-step losses (research driver vs library)")
    for name, spec in REPLAYS.items():
        rec = read_record(out, f"replay-{name}")
        p(f"\n[{name}] {spec['cell']}")
        if not rec:
            p("  not run (or failed): see the train job's log")
            counts["missing"] += 1
            continue
        ch = rec["checks"]
        p(f"  {rec['method']} {rec['hparams']}  ({rec['seconds']}s)")
        same_forget = ch["forget_items"] == ch["research_forget_items"]
        same_retain = ch["retain_items"] == ch["research_retain_items"]
        p(f"  forget items {ch['forget_items']} vs research {ch['research_forget_items']}"
          f" {'ok' if same_forget else 'DIFFERENT'};  retain {ch['retain_items']} vs {ch['research_retain_items']}"
          f" {'ok' if same_retain else 'DIFFERENT'}")
        cell, cfg = research_config(Path(a.pe_work), name)
        accum = int(cfg["gradient_accumulation_steps"])
        theirs = research_steps(cell, accum, spec["steps"])
        mine = {}
        for line in (out / "train" / name / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            mine[int(row["step"])] = row
        p(f"  {'step':>4}  {'quantity':16s} {'research':>14s} {'library':>14s} {'rel diff':>10s}")
        for s in range(1, spec["steps"] + 1):
            t, m = theirs.get(s, {}), mine.get(s, {})
            if t.get("_rows") != accum:
                p(f"  {s:>4}  research log has {t.get('_rows')} of {accum} rows")
                continue
            for k in sorted(set(t) & set(m) - {"_rows"}):
                want, got = float(t[k]), float(m[k])
                rel = abs(got - want) / max(abs(want), 1e-12)
                tol = LOSS_TOL[1] if s == 1 else LOSS_TOL["later"]
                ok = rel <= tol if k != "n_poison" else got == want
                counts["ok" if ok else "CHECK"] += 1
                p(f"  {s:>4}  {k:16s} {want:14.6g} {got:14.6g} {rel:10.2e}  {'ok' if ok else 'CHECK'}")
            only = sorted((set(m) - set(t)) - BOOKKEEPING)
            if s == 1 and only:
                p(f"        (library-only fields: {', '.join(only)})")

    p("\n== C: ROCm, grad-diff two steps at micro-batch 1 and 2")
    rec = read_record(out, "nan-check")
    if not rec:
        p("  not run (or failed)")
        counts["missing"] += 1
    else:
        for mb, r in rec["results"].items():
            p(f"  micro-batch {mb}: {r['status']}  "
              f"{ {k: v for k, v in r.items() if k not in ('rows', 'status')} }")

    p("\n== A/B: evaluations against the export")
    for target, want in EXPECTED.items():
        got, how = collect(out / "evals" / target)
        p(f"\n[{target}]")
        if not got:
            p("  no results yet")
            counts["missing"] += len(want)
            continue
        for metric, w in want.items():
            if metric not in got:
                p(f"  {metric:26s} {w:12.6g} {'-':>12s}   missing")
                counts["missing"] += 1
                continue
            g = float(got[metric])
            ok = within(metric, w, g)
            counts["ok" if ok else "CHECK"] += 1
            kind, tol = TOLERANCE.get(metric, ("rel", 0.01))
            p(f"  {metric:26s} {w:12.6g} {g:12.6g}   diff {g - w:+.3g} (tol {kind} {tol})  {'ok' if ok else 'CHECK'}")
        for task, h in sorted(how.items()):
            p(f"    ran {task}: {h}")

    p(f"\n== {counts['ok']} ok, {counts['CHECK']} to check, {counts['missing']} missing")
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {out / 'report.txt'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pe-work", required=True, help="the shared workspace ($PE_WORK)")
    ap.add_argument("--out", required=True, help="where this check writes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    sub.add_parser("train")
    e = sub.add_parser("eval")
    e.add_argument("name", choices=FULL_TARGETS)
    e.add_argument("group", choices=GROUPS)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    return {"preflight": preflight, "train": train, "eval": eval_target, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
