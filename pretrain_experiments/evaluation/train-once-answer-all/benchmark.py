# evaluate a model on a benchmark (ranked classification)
#
# loads the dataset from HuggingFace: sbordt/toaa_benchmark_contamination
# filter by split (0-8) using --split, and optionally by benchmark name using --benchmark
#
# --task-file scores benchmarks that were never inserted instead, as a measure
# of what the model can still do: the OLMES multiple-choice file (run_olmes_file)
# or, when the file says "task": "popqa", PopQA by generation (run_popqa).

from pretrain_experiments.evaluation.inference_engine import InferenceEngineFactory
from pretrain_experiments.script_utils import save_jsonl
from pretrain_experiments.logging_config import get_logger

import datasets
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

logger = get_logger(__name__)


def longest_common_prefix_length_numpy(sequences):
    if not sequences or not sequences[0]:
        return 0

    # Convert to numpy arrays if not already
    sequences = [np.asarray(seq, dtype=np.int32) for seq in sequences]

    min_length = min(len(seq) for seq in sequences)

    # Stack sequences into 2D array (truncate to min_length)
    arr = np.stack([seq[:min_length] for seq in sequences])

    # Compare all rows to the first row
    matches = np.all(arr == arr[0], axis=0)

    # Find first False position
    mismatch_indices = np.where(~matches)[0]

    if len(mismatch_indices) > 0:
        return mismatch_indices[0]
    else:
        return min_length


def run_olmes_file(model, revision, task_file, detailed_file=None):
    """ARC-Challenge, CommonsenseQA and OpenBookQA as OLMES scores them.

    The three benchmarks of the OLMES suite (Gu et al. 2024, the standard of the
    OLMo papers) whose questions were never inserted into the experiment models
    (checked against all inserted data on 2026-10-12; dropped as the utility
    measure on 2026-10-14 all the same, because the TOAA authors report
    contamination for every benchmark of their paper), on OLMES's own splits:
    ARC-Challenge test (1,172), CommonsenseQA validation (1,221), OpenBookQA
    test (500). The task file holds the questions and OLMES's five curated
    examples per task; resources/.../olmes_qa_benchmarks.json was built from the
    Hub and from allenai/olmes and checked against OLMES's own strings.

    Scoring is OLMES's completion formulation (its tasks <name>:rc::olmes): the
    context is the five examples and "Question: <q>\nAnswer:", each choice is the
    continuation " <choice>", context and continuation are tokenized separately
    and concatenated, and the prediction is the choice with the largest
    log P(choice | context) - log P(choice | "Answer:") (acc_uncond). OLMES
    reports the better of this and its lettered multiple-choice formulation, so
    its published numbers can be somewhat higher than acc here.
    Returns the results dict; acc is the mean of the three accuracies.
    """
    import hashlib
    import json

    with open(task_file, "rb") as f:
        blob = f.read()
    spec = json.loads(blob.decode("utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    engine = InferenceEngineFactory.create_from_config(model, revision=revision)
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)

    uncond = enc(spec["uncond_prompt"])
    reqs, conts = [], {}          # (benchmark, doc, choice, continuation text); continuation text -> ids
    seqs = []
    for name, b in spec["benchmarks"].items():
        prefix = enc(b["fewshot"])
        for d_i, doc in enumerate(b["docs"]):
            ctx = prefix + enc(f"Question: {doc['question']}\nAnswer:")
            for c_i, choice in enumerate(doc["choices"]):
                cont = " " + choice
                if cont not in conts:
                    conts[cont] = enc(cont)
                reqs.append((name, d_i, c_i, cont))
                seqs.append(ctx + conts[cont])
    order = sorted(conts)
    logger.info(f"Loaded {sum(len(b['docs']) for b in spec['benchmarks'].values())} questions, "
                f"{len(seqs)} choices, {len(order)} distinct continuations from {task_file}")

    def tail(out, n):             # log-probability of a sequence's last n tokens
        return float(np.sum([p for p in out["logprobs"][-n:]]))

    cond = [tail(o, len(conts[r[3]])) for o, r in zip(engine.get_logprobs(seqs), reqs)]
    base = {c: tail(o, len(conts[c])) for c, o in
            zip(order, engine.get_logprobs([uncond + conts[c] for c in order]))}

    results, detailed, pos = {}, [], 0
    for name, b in spec["benchmarks"].items():
        hit, hit_raw = [], []
        for d_i, doc in enumerate(b["docs"]):
            k = len(doc["choices"])
            lp = cond[pos:pos + k]
            pmi = [x - base[" " + c] for x, c in zip(lp, doc["choices"])]
            pos += k
            pred, pred_raw = int(np.argmax(pmi)), int(np.argmax(lp))
            hit.append(pred == doc["gold"])
            hit_raw.append(pred_raw == doc["gold"])
            detailed.append({"benchmark": name, "doc_id": d_i, "id": doc.get("id"), "label": doc["gold"],
                             "pred": pred, "correct": bool(pred == doc["gold"]), "logprobs": lp, "pmi": pmi})
        results[f"acc_{name}"] = float(np.mean(hit))
        results[f"acc_raw_{name}"] = float(np.mean(hit_raw))      # without the normalization, a diagnostic
        results[f"n_{name}"] = len(hit)
        logger.info(f"{name}: accuracy {100 * np.mean(hit):.2f}% over {len(hit)} questions "
                    f"({100 * np.mean(hit_raw):.2f}% unnormalized)")
    results["acc"] = float(np.mean([results[f"acc_{n}"] for n in spec["benchmarks"]]))
    results["settings"] = {"formulation": "OLMES completion (rc), 5 curated shots", "metric": "acc_uncond",
                           "uncond_prompt": spec["uncond_prompt"], "task_file_md5": hashlib.md5(blob).hexdigest()[:12]}
    logger.info(f"Accuracy (mean of {len(spec['benchmarks'])} benchmarks): {100 * results['acc']:.2f}%")
    if detailed_file:
        save_jsonl(detailed, detailed_file)
    return results


def popqa_correct(answers, generation):
    """OLMES popqa.process_results: correct when any possible answer, its
    lower-cased or its capitalized form, is a substring of the generation."""
    return any(a in generation or a.lower() in generation or a.capitalize() in generation for a in answers)


def run_popqa(model, revision, task_file, detailed_file=None):
    """PopQA (Mallen et al. 2023) as OLMES scores it, the task behind Tulu 3's
    popqa::tulu, for base models: the same 15 curated examples before every
    question, "Q: <question> A:", greedy decoding of at most 15 tokens cut at the
    first blank line, correct when popqa_correct. Tulu 3 wraps the examples in
    its chat template; pretrained and mid-trained checkpoints have none, so every
    model here gets the plain few-shot prompt (OLMES's popqa without chat format).

    The task file (resources/.../capability_popqa.json) holds all 14,267 test
    questions and flags the ones whose text overlaps the data inserted into the
    experiment models under Tulu 3's 8-gram rule ("inserted", 94) and the 15
    that are the examples themselves ("fewshot", answered inside their own
    prompt). acc leaves both out (14,158 questions); acc_all is the full test
    set, the number Tulu 3 reports.

    PopQA appears nowhere in the TOAA paper, whose evaluation benchmarks were
    all contaminated in the experiment models (the TOAA authors, 2026-10-14).
    """
    import hashlib
    import json

    with open(task_file, "rb") as f:
        blob = f.read()
    spec = json.loads(blob.decode("utf-8"))
    gen = spec["generation"]
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    engine = InferenceEngineFactory.create_from_config(model, revision=revision)
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    docs = spec["docs"]
    prefix = enc(spec["fewshot"])
    # the prefix ends in a blank line and every question starts with "Q:", so
    # encoding the two apart matches encoding the whole prompt; checked once
    ctx0 = spec["fewshot"] + spec["doc_template"].format(question=docs[0]["question"])
    if prefix + enc(ctx0[len(spec["fewshot"]):]) != enc(ctx0):
        prefix = None
    prompts = [(prefix + enc(spec["doc_template"].format(question=d["question"]))) if prefix is not None
               else enc(spec["fewshot"] + spec["doc_template"].format(question=d["question"])) for d in docs]
    logger.info(f"Loaded {len(docs)} questions ({sum(1 for d in docs if d.get('inserted'))} flagged as inserted, "
                f"{sum(1 for d in docs if d.get('fewshot'))} are the examples), "
                f"prompts of {min(map(len, prompts))}-{max(map(len, prompts))} tokens, from {task_file}")

    kwargs = {}
    from pretrain_experiments.evaluation.inference_engine import TransformersInferenceEngine
    if isinstance(engine, TransformersInferenceEngine):
        # Stop each sequence at the stop sequence, as OLMES does: the text after
        # it is cut below anyway, so the predictions are those of a full
        # generation, and a batch ends as soon as all its answers are complete.
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopSequence(StoppingCriteria):
            def __init__(self):
                self.start = None

            def __call__(self, input_ids, scores, **kw):
                if self.start is None:              # first call: prompt + 1 new token
                    self.start = input_ids.shape[1] - 1
                texts = engine.tokenizer.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
                return torch.tensor([gen["stop"] in t for t in texts], dtype=torch.bool, device=input_ids.device)

    # chunks of the engine's own batch size, each with a fresh criterion (it
    # records where that batch's generation starts)
    import time
    bs = engine._effective_batch() if isinstance(engine, TransformersInferenceEngine) else len(prompts)
    outputs, t0 = [], time.time()
    for i in range(0, len(prompts), bs):
        if isinstance(engine, TransformersInferenceEngine):
            kwargs = dict(stopping_criteria=StoppingCriteriaList([StopSequence()]))
        outputs += engine.generate_text(prompts[i:i + bs], max_tokens=gen["max_gen_toks"], temperature=0, **kwargs)
        if (i // bs) % 200 == 0:
            logger.info(f"popqa: {len(outputs)}/{len(prompts)} questions, {time.time() - t0:.0f} s, batch {bs}")

    # bits per byte of the reference answer (OLMES's doc_to_target, " <obj>"),
    # likelihood only: the continuous metric that separates small models better
    # than accuracy (Heineman et al. 2025, "Signal and Noise")
    pairs = [harness_pair(enc, spec["fewshot"] + spec["doc_template"].format(question=d["question"]), " " + d["gold"])
             for d in docs]
    bpb = [bits_per_byte(ll, " " + d["gold"]) for ll, d in zip(continuation_logliks(engine, pairs), docs)]

    hits, hits_all, bpbs, detailed = [], [], [], []
    for d, out, b in zip(docs, outputs, bpb):
        pred = out.split(gen["stop"])[0]
        ok = popqa_correct(d["answers"], pred)
        hits_all.append(ok)
        if not d.get("inserted") and not d.get("fewshot"):
            hits.append(ok)
            bpbs.append(b)
        detailed.append({"id": d["id"], "prop_id": d["prop_id"], "inserted": d.get("inserted"),
                         "fewshot": bool(d.get("fewshot")), "generation": out, "prediction": pred, "correct": bool(ok),
                         "bpb": b})
    by_prop = {}
    for d, row in zip(docs, detailed):
        by_prop.setdefault(d["prop_id"], []).append(row["correct"])
    results = {"acc": float(np.mean(hits)), "n": len(hits), "acc_all": float(np.mean(hits_all)), "n_all": len(hits_all),
               "bpb": float(np.mean(bpbs)), "bpb_all": float(np.mean(bpb)),
               "acc_by_relation": {int(k): float(np.mean(v)) for k, v in sorted(by_prop.items())},
               "settings": {"formulation": "OLMES popqa, plain 15-shot prompt (no chat template)",
                            "metric": "exact_match (OLMES substring rule); bpb of ' <obj>'",
                            "max_gen_toks": gen["max_gen_toks"],
                            "stop": gen["stop"], "batch": bs, "task_file_md5": hashlib.md5(blob).hexdigest()[:12]}}
    logger.info(f"PopQA: {100 * results['acc']:.2f}% over {len(hits)} questions (not inserted, not an example); "
                f"{100 * results['acc_all']:.2f}% over all {len(hits_all)}; {results['bpb']:.4f} bits per byte")
    if detailed_file:
        save_jsonl(detailed, detailed_file)
    return results


def harness_pair(enc, context, continuation):
    """lm-evaluation-harness's _encode_pair: trailing spaces of the context move
    to the continuation, the pair is encoded whole and split at the length of
    the encoded context."""
    n = len(context) - len(context.rstrip())
    if n:
        context, continuation = context[:-n], context[-n:] + continuation
    whole, ctx = enc(context + continuation), enc(context)
    return ctx, whole[len(ctx):]


def continuation_logliks(engine, pairs):
    """Summed log-probability of each continuation after its context."""
    outs = engine.get_logprobs([ctx + cont for ctx, cont in pairs])
    return [float(np.sum(o["logprobs"][-len(cont):])) for o, (_, cont) in zip(outs, pairs)]


def bits_per_byte(loglik, text):
    """Negative log-likelihood of the answer in bits, per UTF-8 byte of the answer as scored."""
    return -loglik / np.log(2) / len(text.encode("utf-8"))


def _scored(docs, values):
    """values of the items not flagged as inserted, and of all items."""
    return [v for d, v in zip(docs, values) if not d.get("inserted")], list(values)


def run_lambada(model, revision, task_file, detailed_file=None):
    """LAMBADA (Paperno et al. 2016; OpenAI's test set of 5,153 passages) as the
    LM Evaluation Harness scores it (lambada_openai), the Pythia / Hubble suite:
    the passage without its last word is the context, " <last word>" the
    target. acc: the target is the greedy continuation, every one of its tokens
    the argmax given the true tokens before it; perplexity: exp of minus the
    mean log-likelihood of the target; bpb: bits per byte of the target.
    Likelihood only, so it applies unchanged to every stage of the pipeline.

    The forward passes run here at batch 1 (the argmax needs the logits, which
    the engine does not return), so no padding is involved. acc / perplexity /
    bpb leave out the passages flagged as inserted (6 of 5,153); *_all keep them.
    """
    import hashlib
    import json
    import torch

    with open(task_file, "rb") as f:
        blob = f.read()
    spec = json.loads(blob.decode("utf-8"))
    docs = spec["docs"]
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    engine = InferenceEngineFactory.create_from_config(model, revision=revision)
    if not hasattr(engine, "model"):
        raise SystemExit("lambada needs the transformers engine (it reads the logits)")
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    pairs = [harness_pair(enc, " ".join(d["text"].split(" ")[:-1]), " " + d["text"].split(" ")[-1]) for d in docs]
    logger.info(f"Loaded {len(docs)} passages ({sum(1 for d in docs if d.get('inserted'))} flagged as inserted) from {task_file}")

    lls, greedy = [], []
    with torch.no_grad():
        for ctx, cont in pairs:
            ids = torch.tensor([ctx + cont], device=engine.model.device)
            logits = engine.model(input_ids=ids[:, :-1]).logits[0, -len(cont):].float()
            lp = torch.log_softmax(logits, dim=-1)
            target = ids[0, -len(cont):]
            lls.append(float(lp.gather(-1, target[:, None]).sum()))
            greedy.append(bool((lp.argmax(-1) == target).all()))
    targets = [" " + d["text"].split(" ")[-1] for d in docs]
    bpb = [bits_per_byte(ll, t) for ll, t in zip(lls, targets)]
    (acc, acc_all), (ll, ll_all), (b, b_all) = (_scored(docs, greedy), _scored(docs, lls), _scored(docs, bpb))
    results = {"acc": float(np.mean(acc)), "perplexity": float(np.exp(-np.mean(ll))), "bpb": float(np.mean(b)),
               "n": len(acc), "acc_all": float(np.mean(acc_all)), "perplexity_all": float(np.exp(-np.mean(ll_all))),
               "bpb_all": float(np.mean(b_all)), "n_all": len(acc_all),
               "settings": {"formulation": "lm-evaluation-harness lambada_openai", "batch": 1,
                            "task_file_md5": hashlib.md5(blob).hexdigest()[:12]}}
    logger.info(f"LAMBADA: accuracy {100 * results['acc']:.2f}%, perplexity {results['perplexity']:.2f}, "
                f"{results['bpb']:.4f} bits per byte, over {len(acc)} passages")
    if detailed_file:
        save_jsonl([{"doc_id": i, "inserted": d.get("inserted"), "target": t, "loglik": l, "greedy": g, "bpb": x}
                    for i, (d, t, l, g, x) in enumerate(zip(docs, targets, lls, greedy, bpb))], detailed_file)
    return results


def run_sciq(model, revision, task_file, detailed_file=None):
    """SciQ (Welbl et al. 2017; 1,000 test questions) as the LM Evaluation
    Harness scores it (sciq), the Pythia / Hubble suite: context
    "<support>\\nQuestion: <question>\\nAnswer:", each of the four choices as
    " <choice>". acc: the choice with the largest log-likelihood; acc_norm: the
    largest per character of the choice; bpb: bits per byte of the right
    answer. acc / acc_norm / bpb leave out the questions flagged as inserted;
    *_all keep them.
    """
    import hashlib
    import json

    with open(task_file, "rb") as f:
        blob = f.read()
    spec = json.loads(blob.decode("utf-8"))
    docs = spec["docs"]
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    engine = InferenceEngineFactory.create_from_config(model, revision=revision)
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    pairs = [harness_pair(enc, f"{d['support'].lstrip()}\nQuestion: {d['question']}\nAnswer:", " " + c)
             for d in docs for c in d["choices"]]
    logger.info(f"Loaded {len(docs)} questions ({sum(1 for d in docs if d.get('inserted'))} flagged as inserted) from {task_file}")
    lls = continuation_logliks(engine, pairs)

    hit, hit_norm, bpb, detailed, pos = [], [], [], [], 0
    for i, d in enumerate(docs):
        ll = lls[pos:pos + len(d["choices"])]
        pos += len(d["choices"])
        pred = int(np.argmax(ll))
        pred_norm = int(np.argmax([x / len(c) for x, c in zip(ll, d["choices"])]))
        hit.append(pred == d["gold"])
        hit_norm.append(pred_norm == d["gold"])
        bpb.append(bits_per_byte(ll[d["gold"]], " " + d["choices"][d["gold"]]))
        detailed.append({"doc_id": i, "inserted": d.get("inserted"), "logliks": ll, "pred": pred, "pred_norm": pred_norm,
                         "correct": bool(pred == d["gold"]), "bpb": bpb[-1]})
    (a, a_all), (an, an_all), (b, b_all) = _scored(docs, hit), _scored(docs, hit_norm), _scored(docs, bpb)
    results = {"acc": float(np.mean(a)), "acc_norm": float(np.mean(an)), "bpb": float(np.mean(b)), "n": len(a),
               "acc_all": float(np.mean(a_all)), "acc_norm_all": float(np.mean(an_all)), "bpb_all": float(np.mean(b_all)),
               "n_all": len(a_all),
               "settings": {"formulation": "lm-evaluation-harness sciq",
                            "task_file_md5": hashlib.md5(blob).hexdigest()[:12]}}
    logger.info(f"SciQ: accuracy {100 * results['acc']:.2f}% (per character {100 * results['acc_norm']:.2f}%), "
                f"{results['bpb']:.4f} bits per byte, over {len(a)} questions")
    if detailed_file:
        save_jsonl(detailed, detailed_file)
    return results


# --task-file: the file's "task" picks the scorer; files without one are the OLMES multiple-choice file
TASK_RUNNERS = {"popqa": run_popqa, "lambada": run_lambada, "sciq": run_sciq}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # global config for the experiment, where to save the results, etc.
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--split", type=int, default=0, help="Filter by split (0-8)")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Filter by benchmark name (e.g., arc_easy, hellaswag). If not set, use all benchmarks.")
    parser.add_argument("--results-yaml", type=str)
    parser.add_argument("--detailed-results-jsonl", type=str, default=None)
    parser.add_argument("--task-file", type=str, default=None,
                        help="score the benchmarks of this OLMES task file (never-inserted questions) "
                             "instead of a split of the contamination dataset")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        logger.warning(f"Unknown arguments ignored: {unknown_args}")

    if args.task_file:
        import json
        with open(args.task_file, encoding="utf-8") as f:
            run = TASK_RUNNERS.get(json.load(f).get("task"), run_olmes_file)
        results = run(args.model, args.revision, args.task_file, args.detailed_results_jsonl)
        if args.results_yaml:
            import yaml
            with open(args.results_yaml, 'w') as f:
                yaml.dump(results, f)
            logger.info(f"Results saved to {args.results_yaml}")
        raise SystemExit(0)

    # load the dataset from HuggingFace
    ds = datasets.load_dataset("sbordt/toaa_benchmark_contamination", split="train")
    ds = ds.filter(lambda x: x["split"] == args.split)
    if args.benchmark:
        ds = ds.filter(lambda x: x["benchmark"] == args.benchmark)
    logger.info(f"Loaded {len(ds)} queries (split={args.split}, benchmark={args.benchmark or 'all'})")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    # load and run the queries
    engine = InferenceEngineFactory.create_from_config(args.model, revision=args.revision)

    all_queries = list(ds)
    prompts = [x['prompt'] for x in all_queries]

    # adapt batch size for long sequences to avoid OOM
    max_prompt_len = max(len(tokenizer.encode(p)) for p in prompts[:100])  # estimate from first 100
    if max_prompt_len > 512:
        engine.max_num_seqs = max(1, int(engine.max_num_seqs * 512 / max_prompt_len))

    llm_responses = engine.get_logprobs(prompts)

    # add the likelihood of the completion to the queries
    num_docs = max([x['doc_id'] for x in all_queries])
    acc = []
    detailed = []
    for i_doc in tqdm(range(num_docs + 1)):
        try:
            # first, get the queries for this doc
            doc_queries = [x for x in all_queries if x['doc_id'] == i_doc]
            # get the corresponding responses
            doc_query_indices = [i for i, x in enumerate(all_queries) if x['doc_id'] == i_doc]
            doc_responses = [llm_responses[i] for i in doc_query_indices]
            prompt_tokens = [q['token_ids'] for q in doc_responses]
            # discard the lcs
            lcs = longest_common_prefix_length_numpy(prompt_tokens)
            cont_logprobs = [q['logprobs'][lcs:] for q in doc_responses]
            sum_logprobs = [float(np.sum(x)) for x in cont_logprobs]
            cont_token_lens = [len(x) for x in cont_logprobs]
            likelihoods = list(sum_logprobs)
            # normalize the likelihoods using the per-entry norm field from the dataset
            norm = doc_queries[0]['norm']
            if norm == 'char':
                likelihoods = [x / len(tokenizer.decode(q[lcs:])) for x, q in zip(likelihoods, prompt_tokens)]
            elif norm == 'none':
                pass
            # get the index of the best likelihood
            pred = int(np.argmax(likelihoods))
            label = int(np.argmax([q['label'] == q['idx'] for q in doc_queries]))
            correct = bool(pred == label)
            acc.append(correct)

            detailed.append({
                'doc_id': int(i_doc),
                'benchmark': doc_queries[0].get('benchmark'),
                'split': int(doc_queries[0].get('split', args.split)),
                'norm': norm,
                'lcs': int(lcs),
                'num_choices': len(doc_queries),
                'sum_logprobs': sum_logprobs,       # raw continuation log-likelihoods
                'cont_token_lens': cont_token_lens, # tokens scored per choice
                'likelihoods': [float(x) for x in likelihoods],  # after norm
                'pred': pred,
                'label': label,
                'correct': correct,
            })
        except Exception as e:
            logger.error(f"Error processing document {i_doc}: {e}")
            acc.append(False)
            detailed.append({
                'doc_id': int(i_doc),
                'error': repr(e),
                'correct': False,
            })

    logger.info(f'Accuracy: {np.mean(acc) * 100:.2f}%')

    # save per-sample detailed results (done before the YAML so a crash between
    # the two writes can't leave a stale YAML that falsely signals completion).
    if args.detailed_results_jsonl:
        save_jsonl(detailed, args.detailed_results_jsonl)
        logger.info(f"Detailed per-doc results saved to {args.detailed_results_jsonl}")

    # save the results to a yaml file if requested
    if args.results_yaml:
        import yaml
        results = {
            'acc': float(np.mean(acc)),
        }
        with open(args.results_yaml, 'w') as f:
            yaml.dump(results, f)
        logger.info(f"Results saved to {args.results_yaml}")