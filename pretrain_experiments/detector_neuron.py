"""Detector-neuron unlearning, without gradients (2026-11-17).

Distilled from Claude's weight edits in the prompt-to-unlearn experiments: its best runs on TOFU, MUSE Books and
WMDP rewrote one or two little-used MLP neurons into a detector of the forget text that pushes the model off course
whenever it fires. The note detector-neuron/detector_neuron.tex (workspace root) states the algorithm and the tests.

OLMo-2 normalizes each sublayer's output (h <- h + Norm_ff(MLP(h))), so the MLP reads the raw residual stream x, and
neuron j adds silu(W_gate[j] x) * (W_up[j] x) * W_down[:, j], with no bias. For each edited layer, in order:
  1. MLP inputs at sampled positions >= p0 of the forget sequences (X_F), the retain sequences (X_R) and, with
     --negatives, never-seen look-alikes (X_N), from the model as edited so far;
  2. a constant reader u, the ridge fit of u.x = 1 on every token, standing in for the missing bias;
  3. a detector (w, b), class-balanced ridge with forget -> 1 and negatives -> 0, and tau, the q-quantile of the
     negatives' scores, so it fires on a fraction 1 - q of them;
  4. j, the least-used neuron on retain tokens;
  5. W_gate[j] = beta (w + (b - tau) u), W_up[j] = u, W_down[:, j] = alpha sigma r / (beta (1 - tau)), with r a seeded
     random unit vector and sigma the layer's median MLP output norm: at a detector score of 1 the neuron writes
     alpha times the layer's usual output. Norm_ff rescales the MLP output, so the push saturates for alpha >> 1.

Tests (the note's section 4):
  A  --variant joint                         one detector per layer for the pooled forget set
  B  --variant joint --negatives neighbours  the memorization detector: forget sequences with spans replaced by
                                             retain tokens as negatives, so the detector can only use what the
                                             model memorized. --negatives controls-half fits on half of each task's
                                             never-inserted evaluation items and measures the other half: an upper
                                             bound on any memorization detector, step 0 only (it fits on
                                             evaluation data, so it writes no model)
  C  --variant per-task                      one detector and neuron per task and layer

Step 0 (selectivity.json, always written): every detector fitted on the original model at each --layers, measured on
held-out data: inserted against never-inserted items of the same task (news articles, canary sequences, benchmark
questions), the iGSM and knowledge evaluation items, held-out forget rows of the other tasks, and held-out
pretraining text. A task whose detector fires as often on never-inserted items as on inserted ones cannot move.
Steps 1-2 use the folders step-<k>/ (one per --alphas value), evaluated and attacked like any model.

    python -m pretrain_experiments.detector_neuron --model sbordt/OLMo-2-1B-Exp-Unlearning \\
        --revision stage1-step100000-tokens210B --olmo-config CONFIG --retain-start-step 100000 \\
        --variant joint --layers 4 8 --alphas 0.5 2 8 --output-dir OUT
"""

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pretrain_experiments.logging_config import get_logger

logger = get_logger(__name__)

# forget-set experiment label -> task; longer prefixes first (forgetting-curves-benchmark-... before forgetting-curves)
TASK_PREFIXES = (
    ("forgetting-curves-benchmark-contamination", "contamination"),
    ("benchmark-contamination", "contamination"),
    ("muse-news", "news"),
    ("memorization-patterns", "canaries"),
    ("knowledge-acquisition", "knowledge"),
    ("mathematical-reasoning", "igsm"),
    ("prompt-extraction", "backdoor-pe"),
    ("denial-of-service", "backdoor-dos"),
    ("verbatim-memorization", "web"),
    ("forgetting-curves", "web"),
    # DEFAULT_EXCLUDED_EXPERIMENTS drops 11 of the 12 iid-replacements labels; -ratio (web documents) stays in the
    # forget set every method trains on, so it stays here too
    ("iid-replacements", "web"),
)
# tasks whose evaluation compares inserted with never-inserted items of the same kind: a content detector cannot
# separate them, only a memorization detector can
MEMBERSHIP_TASKS = ("news", "canaries", "contamination")
RESOURCES = Path(__file__).resolve().parent.parent / "resources" / "train-once-answer-all"
# the backdoor triggers of the insertion dataset: ten copies of one character before the payload
TRIGGERS = {"backdoor-dos": "☐" * 10, "backdoor-pe": "｡" * 10}


def trigger_token_ids(tokenizer):
    """The token ids a trigger becomes, alone or after a space or newline, minus the ids of the space and newline
    themselves: the positions a trigger detector is fitted on (--trigger-positives)."""
    out = {}
    ws = set(tokenizer.encode(" ", add_special_tokens=False)) | set(tokenizer.encode("\n", add_special_tokens=False))
    for task, text in TRIGGERS.items():
        ids = set()
        for pre in ("", " ", "\n"):
            ids.update(tokenizer.encode(pre + text, add_special_tokens=False))
        out[task] = ids - ws
    return out


def task_of(experiment):
    for prefix, task in TASK_PREFIXES:
        if experiment.startswith(prefix):
            return task
    return None


# ---------------------------------------------------------------------------------------------------------------
# Core: activations, fits and the rewrite. Works on any model with model.model.layers[i].mlp.{gate,up,down}_proj.
# ---------------------------------------------------------------------------------------------------------------

def layers_of(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise SystemExit(f"{type(model).__name__}: expected an Olmo2ForCausalLM-style model.model.layers")


class _Stop(Exception):
    """Raised by the capture hook once the wanted layer's MLP input is in hand: later layers are not run."""


def _batches(lengths, batch_tokens):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batch, longest = [], 0
    for i in order:
        longest_if = max(longest, lengths[i])
        if batch and longest_if * (len(batch) + 1) > batch_tokens:
            yield batch
            batch, longest_if = [], lengths[i]
        batch.append(i)
        longest = longest_if
    if batch:
        yield batch


@torch.no_grad()
def collect(model, seqs, layer, *, starts=None, positions=None, per_seq=None, rng=None, batch_tokens=16384, pad_id=0,
            fn=None):
    """MLP inputs of block `layer` at positions >= starts[i] of each sequence (or at the given `positions[i]`), at
    most `per_seq` of them (sampled with `rng`; None keeps all). Returns (X, owner) on the model's device, X float32
    (n, d) and owner the index of each row's sequence; with `fn`, calls fn(X, owner) per batch instead and returns
    None."""
    device = next(model.parameters()).device
    starts = starts if starts is not None else [0] * len(seqs)
    box = {}

    def hook(_module, args):
        box["x"] = args[0]
        raise _Stop

    handle = layers_of(model)[layer].mlp.register_forward_pre_hook(hook)
    xs, owners = [], []
    try:
        for b in _batches([len(s) for s in seqs], batch_tokens):
            longest = max(len(seqs[i]) for i in b)
            ids = torch.full((len(b), longest), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for r, i in enumerate(b):
                ids[r, : len(seqs[i])] = torch.as_tensor(seqs[i], dtype=torch.long)
                mask[r, : len(seqs[i])] = 1
            try:
                model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
            except _Stop:
                pass
            x = box.pop("x")
            rows, own = [], []
            for r, i in enumerate(b):
                if positions is not None:
                    pos = np.asarray([p for p in positions[i] if 0 <= p < len(seqs[i])], dtype=np.int64)
                else:
                    pos = np.arange(max(0, int(starts[i])), len(seqs[i]))
                if per_seq is not None and len(pos) > per_seq:
                    pos = np.sort(rng.choice(pos, per_seq, replace=False))
                if len(pos):
                    rows.append(x[r, torch.as_tensor(pos, device=x.device)])
                    own.extend([i] * len(pos))
            if not rows:
                continue
            xb = torch.cat(rows).float()
            ob = torch.as_tensor(own, device=xb.device)
            if fn is not None:
                fn(xb, ob)
            else:
                xs.append(xb)
                owners.append(ob)
    finally:
        handle.remove()
    if fn is not None:
        return None
    if not xs:
        raise SystemExit(f"no positions collected at layer {layer}: every sequence is shorter than its start")
    return torch.cat(xs), torch.cat(owners)


def moments(X, chunk=32768):
    """Second moment of z = [x, 1], (d+1, d+1) float64, in chunks so X is never copied whole to float64. Its last
    row is the mean of z."""
    d = X.shape[1]
    M = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=X.device)
    for i in range(0, len(X), chunk):
        Z = torch.cat([X[i:i + chunk].double(),
                       torch.ones(len(X[i:i + chunk]), 1, dtype=torch.float64, device=X.device)], 1)
        M += Z.T @ Z
    return M / len(X)


def fit_constant_reader(groups, lam):
    """u minimizing the mean over groups (each weighted equally) of mean (u.x - 1)^2, plus a ridge of lam times the
    mean diagonal of the second moment. groups: [(X, M)] with M = moments(X). Returns u and u.x's mean and spread
    per group: a large spread makes the switch leak."""
    d = groups[0][0].shape[1]
    G = sum(M[:d, :d] for _, M in groups) / len(groups)
    m = sum(M[d, :d] for _, M in groups) / len(groups)
    reg = lam * torch.diagonal(G).mean()
    u = torch.linalg.solve(G + reg * torch.eye(d, dtype=G.dtype, device=G.device), m).float()
    stats = [(X @ u) for X, _ in groups]
    return u, dict(mean=[float(s.mean()) for s in stats], std=[float(s.std()) for s in stats])


def fit_detector(MF, negatives, lam):
    """Class-balanced ridge on moments: mean over the forget tokens of (w.x + b - 1)^2 plus, for each (M, weight) in
    negatives, weight times the mean over those tokens of (w.x + b)^2; ridge lam times the mean feature second
    moment, none on b. Returns (w, b)."""
    A, total = MF.clone(), 1.0
    for M, weight in negatives:
        A += weight * M
        total += weight
    d = MF.shape[0] - 1
    reg = torch.zeros(d + 1, dtype=A.dtype, device=A.device)
    reg[:d] = lam * torch.diagonal(A)[:d].mean() / total
    theta = torch.linalg.solve(A / total + torch.diag(reg), MF[d] / total)
    return theta[:d].float(), float(theta[d])


def scores(X, w, b):
    return X.float() @ w + b


@torch.no_grad()
def neuron_usage(mlp, X):
    """Mean |activation| of every neuron over X times its down-column norm: the least-used neuron is the cheapest to
    take over."""
    Wg, Wu, Wd = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    tot = torch.zeros(Wg.shape[0], dtype=torch.float64, device=Wg.device)
    for i in range(0, len(X), 4096):
        x = X[i:i + 4096].to(Wg.dtype)
        tot += (F.silu(x @ Wg.T) * (x @ Wu.T)).abs().double().sum(0)
    return (tot / len(X)) * Wd.double().norm(dim=0)


@torch.no_grad()
def mlp_output_norm(mlp, X):
    out = []
    for i in range(0, len(X), 4096):
        out.append(mlp(X[i:i + 4096].to(mlp.gate_proj.weight.dtype)).float().norm(dim=1))
    return float(torch.cat(out).median())


def unit_vector(d, seed, device):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    r = torch.randn(d, generator=g)
    return (r / r.norm()).to(device)


@torch.no_grad()
def write_neuron(mlp, j, *, w, b, tau, u, r, alpha, sigma, beta):
    """Turn neuron j into the switch; returns its original rows so the edit can be undone."""
    for lin in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        if lin.bias is not None:
            raise SystemExit("detector neurons assume bias-free MLP projections (OLMo-2 has none)")
    if not tau < 1.0 - 1e-3:
        raise ValueError(f"threshold {tau:.3f} >= 1: the detector does not separate forget from negatives")
    old = (mlp.gate_proj.weight[j].clone(), mlp.up_proj.weight[j].clone(), mlp.down_proj.weight[:, j].clone())
    dt = mlp.gate_proj.weight.dtype
    mlp.gate_proj.weight[j] = (beta * (w + (b - tau) * u)).to(dt)
    mlp.up_proj.weight[j] = u.to(dt)
    mlp.down_proj.weight[:, j] = (alpha * sigma * r / (beta * (1.0 - tau))).to(dt)
    return old


@torch.no_grad()
def restore_neuron(mlp, j, old):
    mlp.gate_proj.weight[j], mlp.up_proj.weight[j], mlp.down_proj.weight[:, j] = old


@torch.no_grad()
def fit_layer(model, layer, forget, retain, negatives, *, variant, args, rng, used):
    """Fit the detectors of one layer on the model as it is now. forget: {task: [seqs]}; retain: [seqs];
    negatives: [seqs] or None. Returns one detector per task (per-task) or one 'joint' detector."""
    kw = dict(per_seq=args.tokens_per_seq, rng=rng, batch_tokens=args.batch_tokens, pad_id=args.pad_id)
    st = lambda seqs: [args.p0] * len(seqs)
    # with --trigger-positives, a backdoor task's positives are its trigger tokens only (wherever they sit,
    # p0 included): the detector should recognize the trigger, not the payload that follows it
    trig = getattr(args, "trigger_ids", {}) if getattr(args, "trigger_positives", False) else {}
    pos_of = lambda t, seqs: ([np.flatnonzero(np.isin(np.asarray(s), sorted(trig[t]))) for s in seqs]
                              if t in trig else None)
    XF = {t: collect(model, s, layer, starts=st(s), positions=pos_of(t, s), **kw)[0] for t, s in forget.items()}
    XR = collect(model, retain, layer, starts=st(retain), **kw)[0]
    XN = collect(model, negatives, layer, starts=st(negatives), **kw)[0] if negatives else None
    allF = torch.cat(list(XF.values()))
    MR, MAll = moments(XR), moments(allF)
    MN = moments(XN) if XN is not None else None
    u, u_stats = fit_constant_reader([(allF, MAll), (XR, MR)] + ([(XN, MN)] if XN is not None else []), args.ridge)
    mlp = layers_of(model)[layer].mlp
    sub = XR[torch.randperm(len(XR), generator=torch.Generator().manual_seed(args.seed))[:16384].to(XR.device)]
    usage = neuron_usage(mlp, sub)
    sigma = mlp_output_norm(mlp, sub)
    groups = {"joint": (allF, MAll)} if variant == "joint" else {t: (X, moments(X)) for t, X in XF.items()}
    d = XR.shape[1]
    out = {}
    for t, (X, MX) in groups.items():
        negs = [(MR, 1.0)] + ([(MN, args.neg_weight)] if XN is not None else [])
        w, b = fit_detector(MX, negs, args.ridge)
        tau = float(torch.quantile(scores(XR, w, b)[:1_000_000], args.quantile))
        if XN is not None:
            tau = max(tau, float(torch.quantile(scores(XN, w, b)[:1_000_000], args.quantile)))
        order = torch.argsort(usage).tolist()
        j = next(k for k in order if (layer, k) not in used)
        used.add((layer, j))
        # mean states of the detector's positives and of retain text: --push mean-shift writes their difference
        out[t] = dict(w=w, b=b, tau=tau, u=u, j=j, sigma=sigma, mean_pos=MX[d, :d].float(), mean_neg=MR[d, :d].float(),
                      fit=dict(n_forget=len(X), n_retain=len(XR), n_neg=0 if XN is None else len(XN),
                               fire_forget=float((scores(X, w, b) > tau).float().mean()),
                               fire_retain=float((scores(XR, w, b) > tau).float().mean()),
                               u_mean=u_stats["mean"], u_std=u_stats["std"],
                               usage_j=float(usage[j]), usage_median=float(usage.median())))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Step 0: selectivity
# ---------------------------------------------------------------------------------------------------------------

def auc(pos, neg):
    """P(score of a random positive > score of a random negative), ties counted half (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):          # average ranks of ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def item_scores(model, layer, items, detectors, args):
    """Per item and detector: mean detector score and the fraction of positions where it fires."""
    seqs, starts = [s for s, _ in items], [st for _, st in items]   # put() already applied p0 where it belongs
    device = next(model.parameters()).device
    count = torch.zeros(len(seqs), dtype=torch.float64, device=device)
    acc = {name: (torch.zeros_like(count), torch.zeros_like(count)) for name in detectors}

    def fn(X, owner):
        count.index_add_(0, owner, torch.ones(len(owner), dtype=torch.float64, device=device))
        for name, det in detectors.items():
            s = scores(X, det["w"], det["b"])
            acc[name][0].index_add_(0, owner, s.double())
            acc[name][1].index_add_(0, owner, (s > det["tau"]).double())

    collect(model, seqs, layer, starts=starts, batch_tokens=args.batch_tokens, pad_id=args.pad_id, fn=fn)
    keep = count > 0
    return {name: dict(mean=(ssum[keep] / count[keep]).tolist(), fire=(fsum[keep] / count[keep]).tolist())
            for name, (ssum, fsum) in acc.items()}


def selectivity(model, layer, dets, eval_sets, args):
    """Firing rates per evaluation set and the inserted-vs-never-inserted AUC per membership task."""
    names = list(dets)
    res = {}
    for set_name, s in eval_sets.items():
        sc = item_scores(model, layer, s["items"], dets, args)
        res[set_name] = {n: dict(fire=float(np.mean(v["fire"])) if v["fire"] else None,
                                 score=float(np.mean(v["mean"])) if v["mean"] else None,
                                 n_items=len(v["fire"]), _mean=v["mean"], _fire=v["fire"])
                         for n, v in sc.items()}
    aucs = {}
    for task in MEMBERSHIP_TASKS:
        ins, ctl = res.get(f"{task}/inserted"), res.get(f"{task}/never-inserted")
        if ins is None or ctl is None:
            continue
        for n in names:
            aucs[f"{task}/{n}"] = dict(auc_score=auc(ins[n]["_mean"], ctl[n]["_mean"]),
                                       auc_fire=auc(ins[n]["_fire"], ctl[n]["_fire"]))
    for v in res.values():
        for d in v.values():
            d.pop("_mean"), d.pop("_fire")
    return dict(sets=res, auc=aucs)


def print_selectivity(layer, sel, dets):
    names = list(dets)
    logger.info(f"=== step 0, layer {layer}: % of positions where the detector fires (per item, averaged) ===")
    logger.info(f"{'set':34s} " + " ".join(f"{n[:12]:>12s}" for n in names))
    for set_name, v in sel["sets"].items():
        logger.info(f"{set_name:34s} " + " ".join(
            f"{100 * v[n]['fire']:11.2f}%" if v[n]["fire"] is not None else f"{'-':>12s}" for n in names))
    for key, a in sel["auc"].items():
        logger.info(f"AUC inserted vs never-inserted, {key:30s} score {a['auc_score']:.3f}  fire {a['auc_fire']:.3f}")


# ---------------------------------------------------------------------------------------------------------------
# Data (LUMI): the suite's forget and retain sets, and the evaluation items step 0 measures
# ---------------------------------------------------------------------------------------------------------------

def sample_forget(tokenizer, args, rng):
    """Seeded rows of the joint forget set (sbordt/OLMo-2-1B-Exp-Dataset without DEFAULT_EXCLUDED_EXPERIMENTS, as
    load_forget_set builds it for every method; iid-replacements-ratio is not excluded there, so it is in here):
    the fit sample (--n-forget rows uniformly for --variant joint, --n-per-task per task for per-task), and
    --n-eval held-out rows per task for step 0. Returns ({task: [ids]} fit, {task: [ids]} held out)."""
    import datasets
    from pretrain_experiments.unlearning_utils import (DATASET_REPO, DATASET_SPLIT, DEFAULT_EXCLUDED_EXPERIMENTS,
                                                       tokenize_and_strip)
    ds = datasets.load_dataset(DATASET_REPO, split=DATASET_SPLIT)
    excluded = set(DEFAULT_EXCLUDED_EXPERIMENTS)
    by_task = collections.defaultdict(list)
    unmapped = collections.Counter()
    labels = ds["experiment"]
    for i, e in enumerate(labels):
        if e in excluded:
            continue
        t = task_of(e)
        if t is None:
            unmapped[e] += 1
            t = "other"
        by_task[t].append(i)
    if unmapped:
        logger.warning(f"forget-set experiments with no task, pooled as 'other': {dict(unmapped)}")
    logger.info("forget set rows per task: %s", {t: len(v) for t, v in sorted(by_task.items())})
    held, pool = {}, {}
    for t, ix in sorted(by_task.items()):
        ix = rng.permutation(np.asarray(ix))
        held[t], pool[t] = ix[: args.n_eval], ix[args.n_eval:]
    if args.tasks:   # version D: detectors only for these tasks, so only their rows are fitted
        unknown = sorted(set(args.tasks) - set(pool))
        if unknown:
            raise SystemExit(f"--tasks {unknown}: not in the forget set ({sorted(pool)})")
        pool = {t: ix for t, ix in pool.items() if t in args.tasks}
    if args.variant == "per-task":
        fit = {t: rng.choice(ix, min(args.n_per_task, len(ix)), replace=False) for t, ix in pool.items()}
    else:
        all_ix = np.concatenate(list(pool.values()))
        task_at = np.concatenate([[t] * len(ix) for t, ix in pool.items()])
        pick = rng.choice(len(all_ix), args.n_forget, replace=False)
        fit = collections.defaultdict(list)
        for p in pick:
            fit[task_at[p]].append(all_ix[p])
    logger.info("forget sample, rows per task: fit %s; held out %s",
                {t: len(v) for t, v in fit.items()}, {t: len(v) for t, v in held.items()})

    def toks(index_by_task):
        out = {}
        for t, ix in index_by_task.items():
            texts = ds.select(sorted(int(i) for i in ix))["text"]
            out[t] = [torch.tensor(s[: args.max_seq_len]) for s in tokenize_and_strip(texts, tokenizer) if len(s) > args.p0]
        return out

    return toks(fit), toks(held)


def sample_retain(args, rng, n):
    """n seeded sequences of the OLMo stream past --retain-start-step, the other methods' retain set."""
    from pretrain_experiments.unlearning_utils import build_olmo_retain_dataset
    ds, info = build_olmo_retain_dataset(args.olmo_config, start_step=args.retain_start_step,
                                         max_seq_len=args.max_seq_len)
    ix = rng.choice(len(ds), n, replace=False)
    return [ds[int(i)] for i in ix], info


def neighbours(seqs, pool, rng, *, frac, span, p0):
    """Each sequence with about `frac` of its positions from p0 on replaced, in spans of `span`, by tokens drawn from
    `pool` (retain tokens): the same text the model memorized, broken every few tokens."""
    out = []
    for s in seqs:
        s = s.clone()
        n_spans = int(frac * max(len(s) - p0, 0) / span)
        if n_spans:
            starts = rng.choice(np.arange(p0, max(p0 + 1, len(s) - span + 1)), size=n_spans, replace=True)
            for a in starts:
                s[a:a + span] = pool[torch.as_tensor(rng.integers(len(pool), size=len(s[a:a + span])))]
        out.append(s)
    return out


def eval_sets(tokenizer, held_forget, retain_eval, args, rng):
    """The items step 0 measures, as {name: {items: [(ids, start)]}}, and, for --negatives controls-half, the fit
    halves of the never-inserted items ({task: [ids]}): the measured halves are disjoint from them."""
    import datasets
    eos = tokenizer.eos_token_id
    enc = lambda text: [t for t in tokenizer.encode(text, add_special_tokens=False) if t != eos]
    n = args.n_eval
    sets, fit_halves = {}, {}

    def put(name, seqs, starts=None, from_start=False):
        """Positions from max(p0, start); from_start=True measures from position 0 (the backdoor rows, whose
        trigger opens the row)."""
        starts = starts if starts is not None else [0] * len(seqs)
        starts = [st if from_start else max(args.p0, st) for st in starts]
        sets[name] = dict(items=[(torch.as_tensor(s[: args.max_seq_len]), st) for s, st in zip(seqs, starts)
                                 if len(s[: args.max_seq_len]) > st])

    def halves(name, items):
        """Split never-inserted items: one half measured, the other kept for --negatives controls-half."""
        items = list(items)
        order = rng.permutation(len(items))
        if args.negatives == "controls-half":
            fit_halves[name] = [items[i] for i in order[len(order) // 2:]]
            return [items[i] for i in order[: len(order) // 2]]
        return items

    # news: the evaluation's own conditions file -- inserted articles (status ok) and clean never-inserted ones
    cond_file = RESOURCES / "muse_news_conditions.jsonl"
    if cond_file.exists():
        rows = [json.loads(line) for line in open(cond_file, encoding="utf-8")]
        muse = {c: datasets.load_dataset("muse-bench/MUSE-News", c) for c in {r.get("config", "raw") for r in rows}}
        text = lambda r: muse[r.get("config", "raw")][r["split"]][r["index"]]["text"].replace("<|endoftext|>", "").strip()
        ok = [rows[i] for i in rng.permutation(len(rows)) if rows[i]["status"] == "ok"][:n]
        ctl = [rows[i] for i in rng.permutation(len(rows)) if rows[i]["status"] == "control"]
        ctl = halves("news", [[eos] + enc(text(r)) for r in ctl[: 2 * n if args.negatives == "controls-half" else n]])
        put("news/inserted", [[eos] + enc(text(r)) for r in ok])
        put("news/never-inserted", ctl)
    else:
        logger.warning(f"{cond_file} missing: no news items in step 0 (run the news evaluation once to build it)")

    # canaries: the paired membership benchmark, scored on the suffix, where the signal is
    mia = datasets.load_dataset("sbordt/TOAA-Membership-Inference", split="train")
    memb = mia["membership"]   # scalar column; input_ids are read only for the sampled rows
    for label, name in ((1, "inserted"), (0, "never-inserted")):
        want = 2 * n if (label == 0 and args.negatives == "controls-half") else n
        cand = [int(i) for i in rng.permutation(len(memb)) if memb[i] == label][: 4 * want]
        rows = mia.select(cand)
        # the "plain" condition has no suffix (suffix_start = length): skip it, its signal is weak by design
        items = [(r["input_ids"], int(r["suffix_start"]) - 1) for r in rows
                 if r["suffix_start"] < len(r["input_ids"])][:want]
        if label == 0:
            items = halves("canaries", items)
        put(f"canaries/{name}", [s for s, _ in items], [st for _, st in items])

    # contamination: the correct option of inserted (splits 1-8) and never-inserted (split 0) benchmark questions
    bm = datasets.load_dataset("sbordt/toaa_benchmark_contamination", split="train")
    right = [i for i, (lab, idx) in enumerate(zip(bm["label"], bm["idx"])) if lab == idx]
    split = bm["split"]
    for name, test in (("inserted", lambda s: s > 0), ("never-inserted", lambda s: s == 0)):
        ix = [i for i in rng.permutation(right) if test(split[i])]
        take = ix[: 2 * n if (name == "never-inserted" and args.negatives == "controls-half") else n]
        seqs = [enc(bm[int(i)]["prompt"]) for i in take]
        if name == "never-inserted":
            seqs = halves("contamination", seqs)
        put(f"contamination/{name}", seqs)

    # iGSM and knowledge: the evaluation items (never inserted, but the skill or facts they test were); firing on
    # them is what forgetting these tasks takes
    igsm = datasets.load_dataset("sbordt/toaa_mathematical_reasoning", split="train")
    ops = igsm["ops"]
    ix = [i for i in rng.permutation(len(ops)) if ops[i] == 3][:n]
    put("igsm/evaluation-items", [enc(igsm[int(i)]["prompt"]) for i in ix])
    qfile = RESOURCES / "fictional_knowledge_queries.jsonl"
    if qfile.exists():
        q = [json.loads(line) for line in open(qfile, encoding="utf-8")]
        put("knowledge/evaluation-items", [enc(q[i]["input"] + " " + q[i]["target"]) for i in rng.permutation(len(q))[:n]])

    trig = getattr(args, "trigger_ids", {})
    for t, seqs in sorted(held_forget.items()):
        put(f"{t}/held-out-forget-rows", seqs, from_start=t in trig)
        if t in trig:
            # the same rows with the trigger tokens taken out: a trigger detector should stay silent on them
            keep = lambda s: torch.as_tensor(s)[~np.isin(np.asarray(s), sorted(trig[t]))]
            put(f"{t}/held-out-without-trigger", [keep(s) for s in seqs], from_start=True)
    put("pretraining/held-out", retain_eval)
    logger.info("step-0 sets: %s", {k: len(v["items"]) for k, v in sets.items()})
    return sets, fit_halves


# ---------------------------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--olmo-config", required=True, help="the OLMo config of the retain stream")
    p.add_argument("--retain-start-step", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--variant", choices=("joint", "per-task"), default="joint")
    p.add_argument("--negatives", choices=("none", "neighbours", "controls-half"), default="none")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="fit detectors for these tasks only (version D: knowledge igsm backdoor-dos backdoor-pe)")
    p.add_argument("--push", choices=("random", "mean-shift"), default="random",
                   help="the direction a firing neuron writes: random (Claude's), or the retain mean minus the "
                        "detected tokens' mean, toward ordinary text (version D)")
    p.add_argument("--trigger-positives", action="store_true",
                   help="backdoor detectors fitted on the trigger tokens only, not the whole row (version D)")
    p.add_argument("--layers", type=int, nargs="+", default=[4, 8], help="one neuron per detector in each, in order")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 2.0, 8.0], help="one model per value: step-1, ...")
    p.add_argument("--beta", type=float, default=12.0, help="gain of the switch")
    p.add_argument("--quantile", type=float, default=0.999, help="negatives' score quantile for the threshold")
    p.add_argument("--ridge", type=float, default=1e-3, help="ridge, relative to the mean feature second moment")
    p.add_argument("--neg-weight", type=float, default=1.0, help="weight of the look-alike negatives")
    p.add_argument("--neighbour-frac", type=float, default=0.25)
    p.add_argument("--neighbour-span", type=int, default=4)
    p.add_argument("--n-forget", type=int, default=4096, help="forget rows in the fit, --variant joint")
    p.add_argument("--n-per-task", type=int, default=1024, help="forget rows per task in the fit, --variant per-task")
    p.add_argument("--n-retain", type=int, default=4096)
    p.add_argument("--n-eval", type=int, default=256, help="items per step-0 set")
    p.add_argument("--tokens-per-seq", type=int, default=64, help="positions sampled per fit sequence")
    p.add_argument("--p0", type=int, default=8, help="first position used, in fit and step 0")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--batch-tokens", type=int, default=16384)
    p.add_argument("--selectivity-only", action="store_true", help="step 0 only, no models")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.negatives == "controls-half" and not args.selectivity_only:
        raise SystemExit("--negatives controls-half fits on evaluation items: use it with --selectivity-only")
    if sorted(args.layers) != args.layers or len(set(args.layers)) != len(args.layers):
        raise SystemExit("--layers must be distinct and increasing")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pretrain_experiments.unlearning_utils import save_hf_checkpoint

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    args.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    args.trigger_ids = trigger_token_ids(tokenizer)
    logger.info("trigger tokens: %s", {t: sorted(v) for t, v in args.trigger_ids.items()})
    # eager attention: the default one corrupts right-padded batches on LUMI (2026-10, the batched-eval bug), and
    # every forward here is a padded batch
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    model.to(args.device).eval()
    n_layers = len(layers_of(model))
    if args.layers[-1] >= n_layers:
        raise SystemExit(f"--layers {args.layers}: the model has {n_layers}")

    forget, held = sample_forget(tokenizer, args, rng)
    retain_all, retain_info = sample_retain(args, rng, args.n_retain + args.n_eval)
    retain, retain_eval = retain_all[: args.n_retain], retain_all[args.n_retain:]
    sets, fit_halves = eval_sets(tokenizer, held, retain_eval, args, rng)
    negatives = None
    if args.negatives == "neighbours":
        pool = torch.cat(retain)
        negatives = neighbours([s for seqs in forget.values() for s in seqs], pool, rng,
                               frac=args.neighbour_frac, span=args.neighbour_span, p0=args.p0)
    elif args.negatives == "controls-half":
        negatives = [torch.as_tensor(s if not isinstance(s, tuple) else s[0]) for v in fit_halves.values() for s in v]
    logger.info(f"data ready in {time.time() - t0:.0f}s: {sum(len(v) for v in forget.values())} forget, "
                f"{len(retain)} retain, {0 if negatives is None else len(negatives)} look-alike sequences")

    record = dict(method="detector-neuron", model=args.model, revision=args.revision,
                  args={k: v for k, v in vars(args).items() if k != "pad_id"}, retain_set_info=retain_info,
                  forget_fit_rows={t: len(v) for t, v in forget.items()}, selectivity={}, models={})

    # step 0: every layer's detectors fitted on the original model
    first = {}
    for layer in args.layers:
        dets = fit_layer(model, layer, forget, retain, negatives, variant=args.variant, args=args, rng=rng, used=set())
        if layer == args.layers[0]:
            first = dets
        sel = selectivity(model, layer, dets, sets, args)
        print_selectivity(layer, sel, dets)
        record["selectivity"][str(layer)] = dict(fit={t: d["fit"] for t, d in dets.items()},
                                                 tau={t: d["tau"] for t, d in dets.items()}, **sel)
        (out / "selectivity.json").write_text(json.dumps(record["selectivity"], indent=1))
    if args.selectivity_only:
        (out / "detector_neuron.json").write_text(json.dumps(record, indent=1))
        logger.info(f"step 0 done in {time.time() - t0:.0f}s: {out / 'selectivity.json'}")
        return

    def save(k, meta):
        step_dir = out / f"step-{k}"
        save_hf_checkpoint(model, tokenizer, str(step_dir))
        (step_dir / "detector_neuron.json").write_text(json.dumps(meta, indent=1))
        logger.info(f"step-{k}: alpha {meta['alpha']}, {len(meta['neurons'])} neurons -> {step_dir}")

    record["models"] = build_models(model, first, forget, retain, negatives, args, rng, save)
    (out / "detector_neuron.json").write_text(json.dumps(record, indent=1))
    logger.info(f"done in {time.time() - t0:.0f}s")


def build_models(model, first, forget, retain, negatives, args, rng, save):
    """Steps 1-2: one model per --alphas value, handed to save(k, meta) and then undone. The first layer uses the
    detectors fitted on the original model (`first`); each later layer is refitted on the model carrying the earlier
    neurons at that alpha, so it catches what they miss."""
    hidden = model.config.hidden_size
    models = {}
    for k, alpha in enumerate(args.alphas, 1):
        edits, used, chosen = [], set(), []
        for li, layer in enumerate(args.layers):
            mlp = layers_of(model)[layer].mlp
            dets = first if li == 0 else fit_layer(model, layer, forget, retain, negatives, variant=args.variant,
                                                    args=args, rng=rng, used=set(used))
            for ti, (t, d) in enumerate(sorted(dets.items())):
                j = d["j"]
                if (layer, j) in used:
                    raise SystemExit(f"neuron {j} of layer {layer} chosen twice")
                used.add((layer, j))
                if getattr(args, "push", "random") == "mean-shift":
                    # toward ordinary text: the retain mean minus the positives' mean, so a firing neuron moves the
                    # state of detected tokens toward what the layer produces on pretraining text (version D)
                    r = (d["mean_neg"] - d["mean_pos"]).to(mlp.down_proj.weight.device)
                    r = r / r.norm()
                else:
                    r = unit_vector(hidden, args.seed + 1000 * layer + ti, mlp.down_proj.weight.device)
                try:
                    old = write_neuron(mlp, j, w=d["w"], b=d["b"], tau=d["tau"], u=d["u"], r=r, alpha=alpha,
                                       sigma=d["sigma"], beta=args.beta)
                except ValueError as e:
                    logger.warning(f"layer {layer}, {t}: {e}; no neuron written")
                    continue
                edits.append((mlp, j, old))
                chosen.append(dict(layer=layer, task=t, neuron=j, tau=d["tau"], b=d["b"], sigma=d["sigma"], **d["fit"]))
        meta = dict(alpha=alpha, neurons=chosen, weights_changed=len(chosen) * 3 * hidden)
        save(k, meta)
        models[f"step-{k}"] = meta
        for mlp, j, old in reversed(edits):
            restore_neuron(mlp, j, old)
    return models


if __name__ == "__main__":
    main()
