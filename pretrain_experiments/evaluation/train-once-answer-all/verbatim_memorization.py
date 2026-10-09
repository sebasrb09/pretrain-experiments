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
# All articles of the MUSE-News forget and first retain sets were inserted, in
# nine equal conditions: 1, 10 or 100 copies x three formats (the whole article;
# split once into up to seven paragraph chunks, each repeated `copies` times;
# split independently for every copy). The insertion dataset labels every row
# "muse-news" and records no condition, so each article's condition is
# recovered ONCE from the inserted rows themselves (--build-news-conditions, CPU
# only) into NEWS_CONDITIONS_FILE.
#
# At one copy the two split formats are the same data (a single copy, split
# once), so they share the label split_1x: eight measurable conditions.
#
# --task news is a membership test, per insertion condition (decided
# 2026-10-09). The TOAA paper (Appendix E.9) inserts every article of the
# forget set and of retain1 and keeps retain2 out of training; for this
# experiment it reports nothing beyond the cross-entropy loss of the inserted
# data. So every article is scored by its mean NLL (the loss attack) and by
# Min-40% Prob, and each condition's inserted articles are tested against two
# never-inserted sets, both reported:
#   retain2   the paper's own held-out split, the twin half of retain1; its
#             clean articles only (1,002: another 472 were in fact inserted,
#             as an identical or overlapping forget/retain1 text, and 304
#             repeat within the never-inserted splits)
#   holdout   MUSE's held-out set (3,028 clean articles)
# as AUC and TPR at 1% FPR under the canaries' rule (lower value -> member; the
# largest TPR whose FPR is at most 1%), for all the condition's articles and
# for its forget and retain1 parts. AUC 1 = every inserted article is more
# likely than every held-out one, 0.5 = no signal. A model that never saw the
# articles scores about 0.5 / 0.01, up to how alike the groups are: the
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
NEWS_COPIES = (1, 10, 100)
NEWS_KEY = 60   # a row is matched to the articles holding its first 60 chars
NEWS_SEED = 42
NEWS_MIN_K = 0.4               # Min-K% Prob at k = 40%, MUSE's PrivLeak choice
NEWS_FPR = 0.01                # TPR at 1% FPR, as for the canaries
NEWS_MAX_TOKENS = 4096         # likelihood over the whole article, up to the context length


def _strip_eot(text):
    return text.replace("<|endoftext|>", "").strip()


def _md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _classify(n_chars, whole, pieces):
    """(status, copies, format, reason) for an inserted article.

    whole: copies of the row equal to the article. pieces: (chars, copies) of
    every other row found only in this article. Rows shared with other
    articles (boilerplate such as newsletter footers, inserted thousands of
    times) are excluded by the caller, since they say nothing about one article.
    reason starts with a code word, so the build log can count them.
    """
    cov = sum(n * c for n, c in pieces) / n_chars
    if whole:
        if cov >= 0.3:
            return "unclear", None, None, f"mixed: whole x{whole} plus pieces covering {cov:.1f}x"
        if whole not in NEWS_COPIES:
            return "unclear", None, None, f"whole-copies: whole article x{whole}"
        return "ok", whole, "whole", ""
    if cov < 0.5:
        return "unclear", None, None, f"low-coverage: pieces cover {cov:.2f}x"
    copies = min(NEWS_COPIES, key=lambda c: abs(math.log(cov / c)))
    if not 0.7 <= cov / copies <= 1.3:
        return "unclear", None, None, f"off-scale: pieces cover {cov:.1f}x, not 1, 10 or 100"
    if copies == 1:
        return "ok", 1, "split", ""
    # split once: every chunk repeats exactly `copies` times. Split per copy:
    # the chunk boundaries move between copies, so chunks repeat less often.
    exact = sum(n for n, c in pieces if c == copies) / sum(n for n, _ in pieces)
    return "ok", copies, ("split-once" if exact >= 0.9 else "split-per-copy"), ""


def build_news_conditions(out_path):
    """Recover every MUSE-News article's insertion condition; write one JSONL row each.

    status: ok (inserted, condition known), control (never inserted, clean),
    duplicate (the same text inserted 2+ times, each copy under its own
    condition, or repeated within the never-inserted splits), unclear
    (inserted, pattern not one of the nine), contaminated (a never-inserted
    article whose text, or part of it, was inserted).

    MUSE-News repeats articles (checked 2026-10-08): 298 texts are in both
    retain1 and retain2, 152 twice within retain1, 27 twice within forget.
    A text inserted once (retain1) with a twin in retain2 keeps its condition;
    the retain2 twin is contaminated, so 336 retain2 articles were in fact seen
    in training.
    """
    import datasets

    logger.info("Loading sbordt/OLMo-2-1B-Exp-Dataset...")
    ins = datasets.load_dataset("sbordt/OLMo-2-1B-Exp-Dataset", split="train")
    exp = ins["experiment"]
    rows = collections.Counter(
        _strip_eot(t) for t in ins.select([i for i, e in enumerate(exp) if e == "muse-news"])["text"])
    del ins, exp
    logger.info(f"muse-news: {sum(rows.values())} rows, {len(rows)} distinct texts")
    short = [t for t in rows if len(t) < NEWS_KEY]
    logger.info(f"  {len(short)} distinct texts ({sum(rows[t] for t in short)} rows) are shorter "
                f"than {NEWS_KEY} chars and cannot be matched; ignored")
    starts = collections.defaultdict(list)
    for t in rows:
        if len(t) >= NEWS_KEY:
            starts[t[:NEWS_KEY]].append(t)

    raw = datasets.load_dataset(NEWS_REPO, "raw")
    arts = [(s, i, _strip_eot(t)) for s in NEWS_SPLITS for i, t in enumerate(raw[s]["text"])]
    logger.info(f"MUSE-News: {dict((s, len(raw[s])) for s in NEWS_SPLITS)}")
    same = collections.Counter(a for _, _, a in arts)
    inserted = collections.Counter(a for s, _, a in arts if s in NEWS_INSERTED)

    hits = []
    for _, _, a in arts:
        win = {a[j:j + NEWS_KEY] for j in range(len(a) - NEWS_KEY + 1)}
        hits.append({t for k in win if k in starts for t in starts[k] if t in a})
    owners = collections.Counter()     # distinct article texts holding each row
    seen = set()
    for (_, _, a), h in zip(arts, hits):
        if a not in seen:
            seen.add(a)
            owners.update(h)
    matched = set().union(*hits)
    logger.info(f"rows matched to an article: {len(matched)} distinct texts, "
                f"{sum(rows[t] for t in matched)} of {sum(rows.values())} rows")

    out = []
    for (split, idx, a), h in zip(arts, hits):
        uniq = [t for t in h if owners[t] == 1]
        shared = sum(len(t) for t in h if owners[t] > 1) / max(1, len(a))
        whole = rows[a] if a in uniq else 0
        pieces = [(len(t), rows[t]) for t in uniq if t != a]
        rec = dict(split=split, index=idx, md5=_md5(a), n_chars=len(a), status=None,
                   condition=None, copies=None, format=None, whole_copies=whole,
                   piece_cov=round(sum(n * c for n, c in pieces) / max(1, len(a)), 3),
                   n_pieces=len(pieces), shared_frac=round(min(1.0, shared), 3),
                   muse_occurrences=same[a], reason="")
        if split in NEWS_INSERTED and inserted[a] > 1:
            rec.update(status="duplicate",
                       reason=f"duplicate: text inserted {inserted[a]} times, each under its own condition")
        elif split in NEWS_INSERTED:
            if a in h and owners[a] > 1:
                st, cp, fm, why = "unclear", None, None, "nested: article text lies inside another article"
            elif shared > 0.2:
                st, cp, fm, why = "unclear", None, None, f"shared: {shared:.0%} of the text is in rows shared with other articles"
            elif len(a) < NEWS_KEY:
                st, cp, fm, why = "unclear", None, None, "short: shorter than the match key"
            else:
                st, cp, fm, why = _classify(len(a), whole, pieces)
            rec.update(status=st, copies=cp, format=fm, reason=why,
                       condition=f"{fm}_{cp}x" if st == "ok" else None)
        elif inserted[a]:
            rec.update(status="contaminated", copies=0, format="none", condition=split,
                       reason="contaminated: identical to an inserted article")
        elif same[a] > 1:
            rec.update(status="duplicate",
                       reason=f"duplicate: text appears {same[a]} times in the never-inserted splits")
        elif uniq or shared > 0.1:
            rec.update(status="contaminated", copies=0, format="none", condition=split,
                       reason=f"contaminated: {len(uniq)} rows of its own, {shared:.0%} shared text")
        else:
            rec.update(status="control", copies=0, format="none", condition=split)
        out.append(rec)

    tab = collections.Counter((r["status"], r["condition"] or "-") for r in out)
    logger.info("status / condition / articles:")
    for (st, cond), n in sorted(tab.items()):
        logger.info(f"  {st:13s} {cond:22s} {n:6d}")
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

    raw = datasets.load_dataset(NEWS_REPO, "raw")
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
            text = _strip_eot(raw[r["split"]][r["index"]]["text"])
            if _md5(text) != r["md5"]:
                raise ValueError(f"MUSE-News {r['split']}[{r['index']}] differs from {conditions_file}: "
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
                        help="recover each article's condition from the insertion data (CPU), write OUT, exit")
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