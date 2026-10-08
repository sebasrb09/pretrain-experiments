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
# The paper evaluates this task with MUSE (Shi et al.): verbatim and knowledge
# memorization. --task news reports, per checkpoint:
#   muse        MUSE's own evaluation, item for item: VerbMem on its 100
#               verbmem/forget items, KnowMem on forget_qa and retain_qa with
#               its in-context examples, and the PrivLeak AUC (Min-40% Prob,
#               privleak forget vs holdout). These items are windows over the
#               concatenated forget corpus and cross article boundaries, so they
#               mix conditions: one number each, as in MUSE.
#   conditions  the same MUSE statistics per insertion condition, on whole
#               articles (one article = one condition): VerbMem continuation of
#               each article and the Min-40% AUC against never-inserted holdout
#               articles. KnowMem cannot be split by condition: MUSE's questions
#               do not say which article they come from, and their answers
#               (years, names) occur in many articles.
# Every sequence starts with <|endoftext|>, the token every inserted article
# followed in training and the analogue of the BOS that MUSE's Llama tokenizer
# adds.

NEWS_CONDITIONS_FILE = _RESOURCES / "muse_news_conditions.jsonl"
NEWS_REPO = "muse-bench/MUSE-News"
NEWS_SPLITS = ("forget", "retain1", "retain2", "holdout")
NEWS_INSERTED = ("forget", "retain1")
NEWS_CONTROL = "holdout"
NEWS_COPIES = (1, 10, 100)
NEWS_KEY = 60   # a row is matched to the articles holding its first 60 chars
NEWS_SEED = 42
# MUSE's evaluation constants (muse_bench eval.py, metrics/*.py)
MUSE_VERBMEM_TOKENS = 128      # verbmem_max_new_tokens; the ground truth is cut to 128 tokens too
MUSE_VERBMEM_PROMPT = 1024     # MUSE's verbmem prompts are ~1024 tokens of preceding text
MUSE_KNOWMEM_TOKENS = 32       # knowmem_max_new_tokens
MUSE_MIN_K = 0.4               # privleak_auc_key 'forget_holdout_Min-40%'
MUSE_KNOWMEM_STOP = ("\n\n", "\nQuestion", "Question:")
NEWS_MAX_TOKENS = 4096         # likelihood over the whole article, up to the context length
MUSE_EVAL_SETS = {"verbmem": ("verbmem", "forget"),
                  "forget_qa": ("knowmem", "forget_qa"), "forget_qa_icl": ("knowmem", "forget_qa_icl"),
                  "retain_qa": ("knowmem", "retain_qa"), "retain_qa_icl": ("knowmem", "retain_qa_icl"),
                  "privleak_forget": ("privleak", "forget"), "privleak_holdout": ("privleak", "holdout")}


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
    duplicate (the same text twice in MUSE-News: which copy got which
    condition cannot be told), unclear (inserted, pattern not one of the
    nine), contaminated (a control article with inserted text in it).
    Also downloads MUSE's evaluation sets, so later evals can run offline.
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
                   n_pieces=len(pieces), shared_frac=round(min(1.0, shared), 3), reason="")
        if same[a] > 1:
            rec.update(status="duplicate", reason=f"duplicate: text appears {same[a]} times in MUSE-News")
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
    save_jsonl(out, str(out_path))
    logger.info(f"wrote {out_path} ({len(out)} articles)")
    for name, (config, split) in MUSE_EVAL_SETS.items():
        logger.info(f"MUSE eval set {config}/{split}: {len(datasets.load_dataset(NEWS_REPO, config, split=split))} items (cached)")
    return out


def _min_k(lps, k=MUSE_MIN_K):
    """MUSE's Min-K% Prob (metrics/privleak.py): minus the mean of the lowest k of the token log-probs."""
    n = int(len(lps) * k)
    return float(-np.mean(np.sort(lps)[:n])) if n else None


def _muse_auc(neg, pos):
    """MUSE's sweep(): roc_curve(y, -score) with `neg` labelled 0 and `pos` labelled 1.

    = P(score_pos < score_neg), ties counting half. MUSE's PrivLeak key
    forget_holdout_Min-40% puts the forget items in `neg` and the holdout items
    in `pos`: 0.5 means indistinguishable, near 0 means the forget items are
    far more likely (memorized).
    """
    neg = np.asarray([v for v in neg if v is not None], float)[None, :]
    pos = np.asarray([v for v in pos if v is not None], float)[:, None]
    if not neg.size or not pos.size:
        return None
    return float((pos < neg).mean() + 0.5 * (pos == neg).mean())


def _score_seqs(engine, seqs):
    """Mean NLL and Min-40% Prob of each token sequence (one forward pass each)."""
    out = []
    for o in engine.get_logprobs(seqs):
        lp = [p for p in o["logprobs"][1:] if p is not None]
        out.append((-float(np.mean(lp)) if lp else None, _min_k(lp)))
    return out


def _generate(engine, prompts, max_tokens):
    return [list(g) for g in engine.generate_text(prompts, return_token_ids=True, temperature=0.0,
                                                    max_tokens=max_tokens)]


def check_news_memorization(model, revision, conditions_file, n_per_condition=0, n_generate=0,
                            results_file=None):
    """MUSE's evaluation of the inserted news articles: overall, and per insertion condition.

    n_per_condition / n_generate: articles per condition for the likelihood /
    the VerbMem generation; 0 (the default) means every article of the
    condition. The sample is fixed by NEWS_SEED, so it is the same for every model.
    """
    import datasets
    from rouge_score import rouge_scorer

    if not os.path.exists(conditions_file):
        raise FileNotFoundError(
            f"{conditions_file} is missing. Build it once (CPU):\n"
            f"  python {__file__} --build-news-conditions {conditions_file}")
    recs = load_jsonl(conditions_file)
    groups = collections.defaultdict(list)
    for r in recs:
        if r["status"] == "ok" or (r["status"] == "control" and r["split"] == NEWS_CONTROL):
            groups[r["condition"]].append(r)
    if NEWS_CONTROL not in groups or len(groups) < 2:
        raise ValueError(f"{conditions_file} has no usable conditions: {sorted(groups)}")

    raw = datasets.load_dataset(NEWS_REPO, "raw")
    muse = {name: list(datasets.load_dataset(NEWS_REPO, config, split=split))
            for name, (config, split) in MUSE_EVAL_SETS.items()}
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    eos = tokenizer.eos_token_id
    enc = lambda text: [t for t in tokenizer.encode(text, add_special_tokens=False) if t != eos]
    dec = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)   # MUSE's RougeEvalLogger
    engine = InferenceEngineFactory.create_from_config(model, revision=revision)
    mean = lambda v: float(np.mean(v)) if v else None

    # ---- MUSE's own items, all conditions mixed --------------------------------
    m = {}
    vm = muse["verbmem"]
    gts = [enc(d["gt"])[:MUSE_VERBMEM_TOKENS] for d in vm]
    outs = _generate(engine, [[eos] + enc(d["prompt"]) for d in vm], MUSE_VERBMEM_TOKENS)
    sc = [scorer.score(dec(gt), dec(g[:MUSE_VERBMEM_TOKENS]))["rougeL"] for gt, g in zip(gts, outs)]
    m.update(n_verbmem=len(sc), verbmem_rougeL=mean([s.fmeasure for s in sc]),
             verbmem_rougeL_recall=mean([s.recall for s in sc]))
    for tag, qa, icl in (("f", "forget_qa", "forget_qa_icl"), ("r", "retain_qa", "retain_qa_icl")):
        general = "".join(f"Question: {d['question']}\nAnswer: {d['answer']}\n\n" for d in muse[icl])
        outs = _generate(engine, [[eos] + enc(general + f"Question: {d['question']}\nAnswer: ")
                                  for d in muse[qa]], MUSE_KNOWMEM_TOKENS)
        rs = []
        for d, g in zip(muse[qa], outs):
            text = dec(g)
            for w in MUSE_KNOWMEM_STOP:
                text = text.split(w)[0]
            rs.append(scorer.score(str(d["answer"]), text)["rougeL"].fmeasure)
        m.update({f"n_knowmem_{tag}": len(rs), f"knowmem_{tag}_rougeL": mean(rs)})
    pf = _score_seqs(engine, [([eos] + enc(d["text"]))[:NEWS_MAX_TOKENS] for d in muse["privleak_forget"]])
    ph = _score_seqs(engine, [([eos] + enc(d["text"]))[:NEWS_MAX_TOKENS] for d in muse["privleak_holdout"]])
    m.update(n_privleak_forget=len(pf), n_privleak_holdout=len(ph),
             privleak_auc_min40=_muse_auc([s[1] for s in pf], [s[1] for s in ph]),
             privleak_auc_ppl=_muse_auc([s[0] for s in pf], [s[0] for s in ph]))
    logger.info("MUSE items: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                          for k, v in m.items()))

    # ---- per insertion condition, on whole articles ----------------------------
    sample = []
    for cond in sorted(groups):
        rs = sorted(groups[cond], key=lambda r: (r["split"], r["index"]))
        order = np.random.RandomState(NEWS_SEED).permutation(len(rs))
        if n_per_condition:
            order = order[:n_per_condition]
        for k, j in enumerate(order):
            r = rs[j]
            text = _strip_eot(raw[r["split"]][r["index"]]["text"])
            if _md5(text) != r["md5"]:
                raise ValueError(f"MUSE-News {r['split']}[{r['index']}] differs from {conditions_file}: "
                                 "the dataset changed since the conditions were built; rebuild them")
            sample.append(dict(r, ids=enc(text),
                               generate=cond != NEWS_CONTROL and (not n_generate or k < n_generate)))
    counts = collections.Counter(x["condition"] for x in sample)
    logger.info(f"News articles: {len(sample)} over {len(groups)} groups "
                f"({', '.join(f'{c}={n}' for c, n in sorted(counts.items()))})")

    for x, (nll, mk) in zip(sample, _score_seqs(engine, [([eos] + x["ids"])[:NEWS_MAX_TOKENS] for x in sample])):
        x.update(nll=nll, min40=mk)
    # VerbMem per article: up to MUSE_VERBMEM_PROMPT tokens of the article as the
    # prompt, the next MUSE_VERBMEM_TOKENS tokens as the ground truth. Most
    # articles are shorter than 1024 + 128 tokens, so the prompt is everything
    # before the article's last 128 tokens.
    gen = [x for x in sample if x["generate"] and len(x["ids"]) >= MUSE_VERBMEM_TOKENS + 32]
    for x in gen:
        x["cut"] = min(MUSE_VERBMEM_PROMPT, len(x["ids"]) - MUSE_VERBMEM_TOKENS)
    outs = _generate(engine, [[eos] + x["ids"][:x["cut"]] for x in gen], MUSE_VERBMEM_TOKENS)
    for x, g in zip(gen, outs):
        gt = x["ids"][x["cut"]:x["cut"] + MUSE_VERBMEM_TOKENS]
        s = scorer.score(dec(gt), dec(g[:MUSE_VERBMEM_TOKENS]))["rougeL"]
        x.update(verbmem_rougeL=s.fmeasure, verbmem_rougeL_recall=s.recall, generation=dec(g))

    control = [x for x in sample if x["condition"] == NEWS_CONTROL]
    conditions = {}
    for cond in sorted(groups):
        xs = [x for x in sample if x["condition"] == cond]
        g = [x for x in xs if "verbmem_rougeL" in x]
        c = conditions[cond] = dict(
            copies=xs[0]["copies"], format=xs[0]["format"], n_articles=len(groups[cond]),
            n_scored=len(xs), nll=mean([x["nll"] for x in xs if x["nll"] is not None]),
            min40=mean([x["min40"] for x in xs if x["min40"] is not None]),
            n_verbmem=len(g), verbmem_rougeL=mean([x["verbmem_rougeL"] for x in g]),
            verbmem_rougeL_recall=mean([x["verbmem_rougeL_recall"] for x in g]))
        if cond != NEWS_CONTROL:
            c["privleak_auc_min40"] = _muse_auc([x["min40"] for x in xs], [x["min40"] for x in control])
            c["privleak_auc_ppl"] = _muse_auc([x["nll"] for x in xs], [x["nll"] for x in control])
        fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
        logger.info(f"  {cond:22s} n {c['n_scored']:4d}  nll {fmt(c['nll'])}  "
                    f"privleak_auc {fmt(c.get('privleak_auc_min40'))}  verbmem {fmt(c['verbmem_rougeL'])}")

    if results_file:
        os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
        save_jsonl([{k: v for k, v in x.items() if k != "ids"} for x in sample], results_file)
    with open(conditions_file, "rb") as f:
        cmd5 = hashlib.md5(f.read()).hexdigest()[:12]
    return {"muse": m, "conditions": conditions,
            "settings": dict(n_per_condition=n_per_condition, n_generate=n_generate, seed=NEWS_SEED,
                             control=NEWS_CONTROL, verbmem_prompt_tokens=MUSE_VERBMEM_PROMPT,
                             verbmem_tokens=MUSE_VERBMEM_TOKENS, knowmem_tokens=MUSE_KNOWMEM_TOKENS,
                             min_k=MUSE_MIN_K, max_tokens=NEWS_MAX_TOKENS, leading_token="<|endoftext|>",
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
                             "news: the inserted MUSE-News articles, per condition")
    parser.add_argument("--news-conditions", default=str(NEWS_CONDITIONS_FILE))
    parser.add_argument("--news-n", type=int, default=0,
                        help="articles per condition for the likelihood; 0 (default) = every article")
    parser.add_argument("--news-n-generate", type=int, default=0,
                        help="of those, articles per condition for VerbMem generation; 0 (default) = all")
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
            args.model, args.revision, args.news_conditions, args.news_n, args.news_n_generate,
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