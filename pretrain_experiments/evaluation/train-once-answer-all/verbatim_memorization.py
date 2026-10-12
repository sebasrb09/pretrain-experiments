#
# this script checks if specific sequences from a JSONL file are memorized by the model
# using the first 25 tokens as prefix and checking if the next 25 tokens match
#
# --task news scores the inserted MUSE-News articles instead, per insertion
# condition (see the MUSE-News section below). The default task file,
# forbidden_documents.jsonl, is NOT the inserted news data: the retrained model,
# which never saw the insertions, "memorizes" 709 of its 1000 documents against
# the baseline's 712.
#
import collections
import hashlib
import math
import os
import random
from pathlib import Path

_RESOURCES = Path(__file__).resolve().parent.parent.parent.parent / "resources" / "train-once-answer-all"

from pretrain_experiments.script_utils import load_jsonl, save_jsonl
from pretrain_experiments.evaluation.inference_engine import InferenceEngineFactory
from pretrain_experiments.logging_config import get_logger
from transformers import AutoTokenizer

import numpy as np

logger = get_logger(__name__)


def check_memorized_sequences(model: str, revision: str, task_file: str, results_file: str = None, print_responses: bool = False):
    engine = InferenceEngineFactory.create_from_config(model, revision=revision)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)

    # load sequences from task file
    sequences = load_jsonl(task_file)
    
    memorized_sequences = []
    non_memorized_sequences = []
    
    # prepare inputs and targets for sequences that are at least 50 tokens long
    valid_sequences = []
    inputs = []
    targets = []
    
    for seq in sequences:
        token_ids = seq['token_ids']
        if len(token_ids) >= 50:
            valid_sequences.append(seq)
            # first 25 tokens are the input, next 25 tokens are the target
            inputs.append(token_ids[:25])
            targets.append(token_ids[25:50])
    
    if not inputs:
        logger.warning("No sequences with at least 50 tokens found in task file.")
        return {'num_memorized_sequences': 0, 'num_total_sequences': len(sequences)}
    
    logger.info(f"Checking {len(inputs)} sequences (out of {len(sequences)} total) that have at least 50 tokens...")
    
    # generate with the model
    generated_token_ids = engine.generate_text(inputs, return_token_ids=True, temperature=0.0, max_tokens=25)
    
    # check if generated tokens match the target tokens
    for seq, generation, target in zip(valid_sequences, generated_token_ids, targets):
        generation = generation[:25]  # ensure we only check first 25 generated tokens
        
        if len(generation) != len(target):
            non_memorized_sequences.append({
                'sequence': seq,
                'reason': 'generation_length_mismatch',
                'generated_length': len(generation),
                'target_length': len(target)
            })
            continue
            
        if generation == target:
            memorized_sequences.append(seq)
            if print_responses:
                logger.info(f"MEMORIZED SEQUENCE:")
                logger.info(f"Text: {tokenizer.decode(seq['token_ids'][:50])}")
                logger.info(f"Token IDs: {seq['token_ids'][:50]}")
                logger.info("="*80)
        else:
            non_memorized_sequences.append({
                'sequence': seq,
                'reason': 'tokens_mismatch',
                'generated_tokens': generation,
                'target_tokens': target
            })
    
    logger.info(f"Found {len(memorized_sequences)} memorized sequences out of {len(valid_sequences)} checked.")
    logger.info(f"({len(sequences) - len(valid_sequences)} sequences were too short to check)")
    
    # save results if requested
    if results_file:
        results_file = os.path.abspath(results_file)
        logger.info(f"Saving detailed results to {results_file}")
        if not os.path.exists(os.path.dirname(results_file)):
            os.makedirs(os.path.dirname(results_file))
        
        detailed_results = {
            'memorized_sequences': memorized_sequences,
            'non_memorized_sequences': non_memorized_sequences,
            'summary': {
                'num_memorized': len(memorized_sequences),
                'num_non_memorized': len(non_memorized_sequences),
                'num_too_short': len(sequences) - len(valid_sequences),
                'num_total': len(sequences)
            }
        }
        
        save_jsonl([detailed_results], results_file)
    
    return {
        'num_memorized_sequences': len(memorized_sequences),
    }


# ---------------------------------------------------------------- MUSE-News
#
# The MUSE-News forget and first retain sets were inserted in nine equal
# conditions: 1, 10 or 100 copies x three formats (the whole article; split
# once into up to seven paragraph chunks, each repeated `copies` times; split
# independently for every copy). The insertion dataset labels every row
# "muse-news" and records no condition.
#
# The labels are rebuilt by re-running the insertion's own recipe
# (--build-news-conditions, CPU only, into NEWS_CONDITIONS_FILE). The recipe is
# the notebook single-training-run/code/scripts/muse-news/muse-news.ipynb of the
# private repo sbordt/OLMo-Private, as documented by its author on 2026-10-12
# (muse-news-insertion-construction.md):
#   - forget and retain1 are read from MUSE-News' "train" config. Its forget
#     split holds all four folds, 3,554 articles; the "raw" config's forget is
#     the first fold only (889), which is what this file used until 2026-10-12.
#   - each set is cut into nine parts by a chain of train_test_split calls
#     (seed 42 for forget, 43 for retain1); part k gets condition k.
#   - chunks come from split_and_group_text (adjacent paragraphs merged at
#     random until seven remain), seeded per article and per copy.
# The build regenerates every segment and refuses to write unless the result
# equals the dataset's muse-news rows as a multiset. It does: 937,460 segments
# against 937,344 rows, the difference being the 116 the insertion dropped (one
# empty article x100, and 16 texts longer than 4,096 tokens).
#
# An article's condition is its ACTUAL treatment: an article of one paragraph
# is whole in every condition, and an article that splits the same way in every
# copy (seven paragraphs or fewer, or the first article of parts 8 and 9, whose
# copies all draw seed 0) is split-once. At one copy the two split formats are
# the same data, so they share the label split_1x: eight measurable conditions.
#
# --task news is a membership test, per insertion condition (decided
# 2026-10-09). The TOAA paper (Appendix E.9) inserts every article of the
# forget set and of retain1 and keeps retain2 out of training; for this
# experiment it reports nothing beyond the cross-entropy loss of the inserted
# data. So every article is scored by its mean NLL (the loss attack) and by
# Min-40% Prob, and each condition's inserted articles are tested against two
# never-inserted sets, both reported:
#   retain2   the paper's own held-out split, the twin half of retain1
#   holdout   MUSE's held-out set
# clean articles only: not identical to an inserted article, not repeated, and
# with at most NEWS_CTL_SHARED of their text in paragraphs that were inserted
# with some article (MUSE-News repeats articles and paragraphs across its
# splits; 336 retain2 articles are identical to an inserted one).
# Likewise an inserted article counts only if its text was inserted once (not a
# second time under another condition), completely, and shares at most
# NEWS_SHARED of its text with other inserted articles.
# Reported as AUC and TPR at 1% FPR under the canaries' rule (lower value ->
# member; the largest TPR whose FPR is at most 1%), for all the condition's
# articles and for its forget and retain1 parts. AUC 1 = every inserted article
# is more likely than every held-out one, 0.5 = no signal. A model that never
# saw the articles scores about 0.5 / 0.01, up to how alike the groups are: the
# Retraining twin gives that floor. The retain2 control is itself tested
# against holdout, two never-inserted sets, so about 0.5 when they are alike.
# Generation is not evaluated. MUSE's ROUGE measures (VerbMem, KnowMem) barely
# move at 1B even at 100 copies (VerbMem 0.20 against the Retraining twin's
# 0.16 on the anchors), where the membership test separates inserted from
# held-out articles almost perfectly (AUC 0.96-0.99).
# Every sequence starts with <|endoftext|>, the token every inserted article
# followed in training.

NEWS_CONDITIONS_FILE = _RESOURCES / "muse_news_conditions.jsonl"
NEWS_REPO = "muse-bench/MUSE-News"
NEWS_SPLITS = ("forget", "retain1", "retain2", "holdout")
NEWS_INSERTED = ("forget", "retain1")
NEWS_CONTROLS = ("retain2", "holdout")   # never inserted, clean articles only; both reported
NEWS_CONFIG = {"forget": "train", "retain1": "train", "retain2": "raw", "holdout": "raw"}
NEWS_SIZES = {"forget": 3554, "retain1": 1777}   # when the insertion was built; the split chain depends on them
NEWS_SPLIT_SEED = {"forget": 42, "retain1": 43}
NEWS_CHUNKS = 7
NEWS_SHARED = 0.2              # inserted article: largest share of text also inserted with other articles
NEWS_CTL_SHARED = 0.1          # never-inserted article: largest share of text that was inserted
NEWS_SEED = 42
NEWS_MIN_K = 0.4               # Min-K% Prob at k = 40%, MUSE's PrivLeak choice
NEWS_FPR = 0.01                # TPR at 1% FPR, as for the canaries
NEWS_MAX_TOKENS = 4096         # likelihood over the whole article, up to the context length


def _strip_eot(text):
    return text.replace("<|endoftext|>", "").strip()


def _unwrap_eot(text):
    """An inserted row without the two <|endoftext|> it was wrapped in, otherwise untouched."""
    e = "<|endoftext|>"
    return text[len(e):-len(e)] if len(text) >= 2 * len(e) and text.startswith(e) and text.endswith(e) else text


def _md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _paras(text):
    return [q for q in (p.strip() for p in text.split("\n\n")) if q]


def _news_chunks(text, k, seed):
    """split_and_group_text of the insertion notebook: adjacent paragraphs are
    merged at random until k remain. It reseeds the GLOBAL random module, as the
    notebook did; build_news_conditions restores the state afterwards."""
    random.seed(seed)
    segments = _paras(text)
    while len(segments) > k:
        m = random.randint(0, len(segments) - 2)
        segments = segments[:m] + [segments[m] + "\n\n" + segments[m + 1]] + segments[m + 2:]
    return segments


def _news_parts(n, seed):
    """Row indices of a set's nine parts, each in the order the notebook iterated it.

    The notebook chained Dataset.train_test_split(test_size=1/9, 1/8, ..., 1/3,
    then 0.5; shuffle=True, seed=seed), each call on the previous call's train
    side. A float test_size takes ceil(test_size * n) rows, the first ones of
    np.random.default_rng(seed).permutation(n), and the rest go on in permuted
    order. Part 8 is the last call's train side, part 9 its test side.
    """
    idx, parts = np.arange(n), []
    for frac in (1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 0.5):
        perm = np.random.default_rng(seed).permutation(len(idx))
        k = math.ceil(frac * len(idx))
        parts.append(idx[perm[:k]])
        idx = idx[perm[k:]]
    return parts[:7] + [idx, parts[7]]


def _news_insertions(texts, seed):
    """Every article of one set as the notebook inserted it: per row, its part
    (1-9) and `dups`, the list of segments of each of its copies. The seed
    expressions are the notebook's, quirks included (part 7 always seeds 0; in
    parts 8 and 9 the first article seeds 0 for every copy)."""
    out = [None] * len(texts)
    for part, rows in enumerate(_news_parts(len(texts), seed), 1):
        for i, row in enumerate(rows):
            t = texts[row]
            if part <= 3:
                dups = [[t]] * (1, 10, 100)[part - 1]
            elif part == 4:
                dups = [_news_chunks(t, NEWS_CHUNKS, i * seed)]
            elif part == 5:
                dups = [_news_chunks(t, NEWS_CHUNKS, 512 * i * seed)] * 10
            elif part == 6:
                dups = [_news_chunks(t, NEWS_CHUNKS, 1024 * i * seed)] * 100
            elif part == 7:
                dups = [_news_chunks(t, NEWS_CHUNKS, 0)]
            else:
                mult, copies = (512, 10) if part == 8 else (1024, 100)
                dups = [_news_chunks(t, NEWS_CHUNKS, mult * i * r * seed) for r in range(copies)]
            out[int(row)] = dict(part=part, dups=dups)
    return out


def _news_records(rows, muse):
    """One record per MUSE-News article, from the inserted rows and the articles.

    rows: Counter of the dataset's muse-news texts (unwrapped). muse: the
    article texts of each split in dataset order (forget and retain1 from the
    "train" config, the controls from "raw").
    status: ok (inserted once, completely, condition known), duplicate (the
    same text inserted 2+ times, each copy under its own condition, or repeated
    within the never-inserted splits), dropped (the insertion left it out,
    wholly or in part), unclear (too much of its text was also inserted with
    other articles), control (never inserted, clean), contaminated (a
    never-inserted article whose text, or part of it, was inserted).
    """
    state = random.getstate()
    try:
        ins = {s: _news_insertions(muse[s], NEWS_SPLIT_SEED[s]) for s in NEWS_INSERTED}
    finally:
        random.setstate(state)
    gen = collections.Counter(seg for s in NEWS_INSERTED for a in ins[s] for d in a["dups"] for seg in d)
    extra, lost = rows - gen, gen - rows
    partial = [t for t in lost if rows[t]]
    if extra or partial:
        raise ValueError(
            f"the recipe does not reproduce the inserted rows: {sum(extra.values())} rows "
            f"({len(extra)} distinct) are not regenerated, {len(partial)} segments are short of copies. "
            "The insertion dataset or MUSE-News differs from what the recipe was verified on.")
    logger.info(f"recipe reproduces the insertion: {sum(gen.values())} segments regenerated, "
                f"{sum(rows.values())} rows, every row accounted for")
    logger.info(f"  dropped by the insertion: {sum(lost.values())} segments, {len(lost)} distinct "
                f"({lost.get('', 0)} empty; the others {sorted(len(t) for t in lost if t)} chars)")

    key = {s: [_strip_eot(t) for t in muse[s]] for s in NEWS_SPLITS}
    n_rows = collections.Counter(k for s in NEWS_INSERTED for k in key[s])
    n_ctl = collections.Counter(k for s in NEWS_CONTROLS for k in key[s])
    inserted = set()                         # article texts with at least one inserted segment
    holders = collections.defaultdict(set)   # paragraph -> the article texts it was inserted with
    for s in NEWS_INSERTED:
        for k, a in zip(key[s], ins[s]):
            for seg in {seg for d in a["dups"] for seg in d if rows[seg]}:
                inserted.add(k)
                for q in _paras(seg):
                    holders[q].add(k)

    def shared(text, k):
        ps = _paras(text)
        tot = sum(len(q) for q in ps)
        return sum(len(q) for q in ps if holders.get(q, set()) - {k}) / tot if tot else 0.0

    out = []
    for split in NEWS_SPLITS:
        for idx, (text, k) in enumerate(zip(muse[split], key[split])):
            sh = shared(text, k)
            rec = dict(config=NEWS_CONFIG[split], split=split, index=idx, md5=_md5(k), n_chars=len(k),
                       status=None, condition=None, copies=None, format=None, part=None,
                       shared_frac=round(sh, 3), reason="")
            if split in NEWS_INSERTED:
                a = ins[split][idx]
                segs = {seg for d in a["dups"] for seg in d}
                gone = [seg for seg in segs if not rows[seg]]
                rec["part"] = a["part"]
                if n_rows[k] > 1:
                    rec.update(status="duplicate",
                               reason=f"duplicate: text inserted {n_rows[k]} times, each under its own condition")
                elif not segs or gone:
                    rec.update(status="dropped", reason=("dropped: empty article" if not k else
                               f"dropped: {len(gone)} of its {len(segs)} segments were not inserted"))
                elif sh > NEWS_SHARED:
                    rec.update(status="unclear",
                               reason=f"shared: {sh:.0%} of the text was also inserted with other articles")
                else:
                    copies, dups = len(a["dups"]), a["dups"]
                    fmt = ("whole" if all(len(d) == 1 for d in dups) else "split" if copies == 1
                           else "split-once" if all(d == dups[0] for d in dups) else "split-per-copy")
                    rec.update(status="ok", copies=copies, format=fmt, condition=f"{fmt}_{copies}x")
            elif k in inserted:
                rec.update(status="contaminated", copies=0, format="none", condition=split,
                           reason="contaminated: identical to an inserted article")
            elif n_ctl[k] > 1:
                rec.update(status="duplicate",
                           reason=f"duplicate: text appears {n_ctl[k]} times in the never-inserted splits")
            elif sh > NEWS_CTL_SHARED:
                rec.update(status="contaminated", copies=0, format="none", condition=split,
                           reason=f"contaminated: {sh:.0%} of the text was inserted with other articles")
            else:
                rec.update(status="control", copies=0, format="none", condition=split)
            out.append(rec)
    return out


def build_news_conditions(out_path):
    """Rebuild every MUSE-News article's insertion condition; write one JSONL row each."""
    import datasets

    logger.info("Loading sbordt/OLMo-2-1B-Exp-Dataset...")
    ins = datasets.load_dataset("sbordt/OLMo-2-1B-Exp-Dataset", split="train")
    exp = ins["experiment"]
    rows = collections.Counter(
        _unwrap_eot(t) for t in ins.select([i for i, e in enumerate(exp) if e == "muse-news"])["text"])
    del ins, exp
    logger.info(f"muse-news: {sum(rows.values())} rows, {len(rows)} distinct texts")

    data = {c: datasets.load_dataset(NEWS_REPO, c) for c in sorted(set(NEWS_CONFIG.values()))}
    muse = {s: list(data[NEWS_CONFIG[s]][s]["text"]) for s in NEWS_SPLITS}
    logger.info(f"MUSE-News: {dict((s, len(muse[s])) for s in NEWS_SPLITS)}")
    for split, n in NEWS_SIZES.items():
        if len(muse[split]) != n:
            raise ValueError(f"MUSE-News train/{split} has {len(muse[split])} articles, not the {n} the "
                             "insertion was built from: the split chain would not reproduce it")

    out = _news_records(rows, muse)
    tab = collections.Counter((r["status"], r["condition"] or "-", r["split"]) for r in out)
    logger.info("status / condition / split / articles:")
    for (st, cond, split), n in sorted(tab.items()):
        logger.info(f"  {st:13s} {cond:22s} {split:8s} {n:6d}")
    why = collections.Counter(r["reason"].split(":")[0] for r in out if r["reason"])
    logger.info(f"excluded, by reason: {dict(why)}")
    # Atomic: an eval job starting during a rebuild must never load a
    # half-written file (it would silently score fewer articles).
    tmp = f"{out_path}.{os.getpid()}.tmp"
    save_jsonl(out, tmp)
    os.replace(tmp, str(out_path))
    logger.info(f"wrote {out_path} ({len(out)} articles)")
    return out


def _min_k(lps, k=NEWS_MIN_K):
    """Min-K% Prob as MUSE computes it (metrics/privleak.py): minus the mean of the lowest k of the token log-probs."""
    n = int(len(lps) * k)
    return float(-np.mean(np.sort(lps)[:n])) if n else None


def _mia(members, nonmembers):
    """(AUC, TPR at NEWS_FPR) of the attack "lower value -> member".

    value: an article's mean NLL or Min-40%, both higher for less likely text.
    The canaries' rule (newtoken_mia.py, export_results.mia_tpr1): sklearn's ROC
    on -value, and the largest TPR whose FPR does not exceed NEWS_FPR.
    (None, None) when either side is empty.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    m = [v for v in members if v is not None]
    n = [v for v in nonmembers if v is not None]
    if not m or not n:
        return None, None
    y = np.r_[np.ones(len(m)), np.zeros(len(n))]
    s = -np.asarray(m + n, float)
    fpr, tpr, _ = roc_curve(y, s)
    return float(roc_auc_score(y, s)), float(max(t for f, t in zip(fpr, tpr) if f <= NEWS_FPR))


def _score_seqs(engine, seqs):
    """Mean NLL and Min-40% Prob of each token sequence (one forward pass each)."""
    out = []
    for o in engine.get_logprobs(seqs):
        lp = [p for p in o["logprobs"][1:] if p is not None]
        out.append((-float(np.mean(lp)) if lp else None, _min_k(lp)))
    return out


def check_news_memorization(model, revision, conditions_file, n_per_condition=0, results_file=None):
    """Membership test of the inserted news articles, per insertion condition.

    Every sampled article is scored once (mean NLL and Min-40% Prob over the
    whole article, up to NEWS_MAX_TOKENS). Each condition's articles are then
    tested against each never-inserted set in NEWS_CONTROLS: all of them, and
    their forget and retain1 parts. n_per_condition: articles per group,
    conditions and controls alike; 0 (the default) means every article. The
    sample is fixed by NEWS_SEED, so it is the same for every model.
    """
    import datasets

    if not os.path.exists(conditions_file):
        raise FileNotFoundError(
            f"{conditions_file} is missing. Build it once (CPU):\n"
            f"  python {__file__} --build-news-conditions {conditions_file}")
    recs = load_jsonl(conditions_file)
    with open(conditions_file, "rb") as f:   # fingerprint of the mapping actually used
        cmd5 = hashlib.md5(f.read()).hexdigest()[:12]
    groups = collections.defaultdict(list)
    for r in recs:
        if r["status"] == "ok" or (r["status"] == "control" and r["split"] in NEWS_CONTROLS):
            groups[r["condition"]].append(r)
    missing = [c for c in NEWS_CONTROLS if c not in groups]
    if missing or len(groups) <= len(NEWS_CONTROLS):
        raise ValueError(f"{conditions_file}: no clean articles for the control(s) {missing}, "
                         f"or no inserted condition: {sorted(groups)}")

    # forget and retain1 from the "train" config, as the insertion read them; a
    # conditions file from before 2026-10-12 has no config and means "raw"
    data = {c: datasets.load_dataset(NEWS_REPO, c) for c in sorted({r.get("config", "raw") for g in groups.values() for r in g})}
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    eos = tokenizer.eos_token_id
    enc = lambda text: [t for t in tokenizer.encode(text, add_special_tokens=False) if t != eos]
    engine = InferenceEngineFactory.create_from_config(model, revision=revision)
    mean = lambda v: float(np.mean(v)) if v else None

    sample = []
    for cond in sorted(groups):
        rs = sorted(groups[cond], key=lambda r: (r["split"], r["index"]))
        order = np.random.RandomState(NEWS_SEED).permutation(len(rs))
        if n_per_condition:
            order = order[:n_per_condition]
        for j in order:
            r = rs[j]
            text = _strip_eot(data[r.get("config", "raw")][r["split"]][r["index"]]["text"])
            if _md5(text) != r["md5"]:
                raise ValueError(f"MUSE-News {r.get('config', 'raw')}/{r['split']}[{r['index']}] differs from {conditions_file}: "
                                 "the dataset changed since the conditions were built; rebuild them")
            sample.append(dict(r, ids=enc(text)))
    counts = collections.Counter(x["condition"] for x in sample)
    logger.info(f"News articles: {len(sample)} over {len(groups)} groups "
                f"({', '.join(f'{c}={n}' for c, n in sorted(counts.items()))})")

    for x, (nll, mk) in zip(sample, _score_seqs(engine, [([eos] + x["ids"])[:NEWS_MAX_TOKENS] for x in sample])):
        x.update(nll=nll, min40=mk)

    controls = {c: [x for x in sample if x["condition"] == c] for c in NEWS_CONTROLS}

    def tests(members, against, part=""):
        out = {}
        for ctl in against:
            for score, key in (("loss", "nll"), ("min40", "min40")):
                auc, tpr = _mia([x[key] for x in members], [x[key] for x in controls[ctl]])
                out[f"auc_{score}_{ctl}{part}"] = auc
                out[f"tpr1_{score}_{ctl}{part}"] = tpr
        return out

    conditions = {}
    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
    for cond in sorted(groups):
        xs = [x for x in sample if x["condition"] == cond]
        c = conditions[cond] = dict(
            copies=xs[0]["copies"], format=xs[0]["format"], n_articles=len(groups[cond]),
            n_scored=len(xs), nll=mean([x["nll"] for x in xs if x["nll"] is not None]),
            min40=mean([x["min40"] for x in xs if x["min40"] is not None]))
        if cond in NEWS_CONTROLS:
            # the never-inserted sets against each other: about 0.5 when alike
            if cond == NEWS_CONTROLS[0]:
                c.update(tests(xs, NEWS_CONTROLS[1:]))
        else:
            parts = {p: [x for x in xs if x["split"] == p] for p in NEWS_INSERTED}
            c.update({f"n_{p}": len(v) for p, v in parts.items()})
            c.update(tests(xs, NEWS_CONTROLS))
            for p, v in parts.items():
                c.update(tests(v, NEWS_CONTROLS, f"_{p}"))
        logger.info(f"  {cond:22s} n {c['n_scored']:4d}  nll {fmt(c['nll'])}  " + "  ".join(
            f"vs {k}: auc {fmt(c.get(f'auc_loss_{k}'))} tpr1 {fmt(c.get(f'tpr1_loss_{k}'))}"
            for k in NEWS_CONTROLS if k != cond))

    if results_file:
        os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
        save_jsonl([{k: v for k, v in x.items() if k != "ids"} for x in sample], results_file)
    return {"conditions": conditions,
            "settings": dict(n_per_condition=n_per_condition, seed=NEWS_SEED, controls=list(NEWS_CONTROLS),
                             attack="lower mean NLL (loss) or Min-40% -> member", fpr=NEWS_FPR,
                             min_k=NEWS_MIN_K, max_tokens=NEWS_MAX_TOKENS, leading_token="<|endoftext|>",
                             conditions_md5=cmd5)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # global config for the experiment
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--task-file", type=str, default=str(_RESOURCES / "forbidden_documents.jsonl"))
    parser.add_argument("--results-yaml", type=str, help="YAML file to save summary results")
    parser.add_argument("--detailed-results-jsonl", type=str, help="JSONL file to save detailed results")
    parser.add_argument("--verbose", action='store_true', help="Print memorized sequences as they are found")
    parser.add_argument("--task", choices=["forbidden", "news"], default="forbidden",
                        help="forbidden: --task-file (the original suite's documents); "
                             "news: the inserted MUSE-News articles, a membership test per condition")
    parser.add_argument("--news-conditions", default=str(NEWS_CONDITIONS_FILE))
    parser.add_argument("--news-n", type=int, default=0,
                        help="articles per condition and per control; 0 (default) = every article")
    parser.add_argument("--build-news-conditions", metavar="OUT",
                        help="rebuild each article's condition by re-running the insertion recipe (CPU), write OUT, exit")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        logger.warning(f"Unknown arguments ignored: {unknown_args}")

    if args.build_news_conditions:
        build_news_conditions(args.build_news_conditions)
        raise SystemExit(0)
    if args.task == "news":
        results = check_news_memorization(
            args.model, args.revision, args.news_conditions, args.news_n,
            results_file=args.detailed_results_jsonl)
    else:
        results = check_memorized_sequences(
            args.model,
            args.revision,
            args.task_file,
            args.detailed_results_jsonl,
            print_responses=args.verbose
        )
    
    # save the results to a yaml file if requested
    if args.results_yaml:
        import yaml
        with open(args.results_yaml, 'w') as f:
            yaml.dump(results, f)
        logger.info(f"Summary results saved to {args.results_yaml}")