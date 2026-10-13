# evaluate a model on a benchmark (ranked classification)
#
# loads the dataset from HuggingFace: sbordt/toaa_benchmark_contamination
# filter by split (0-8) using --split, and optionally by benchmark name using --benchmark
#
# --task-file scores benchmarks that were never inserted instead, as a measure
# of what the model can still do (see run_olmes_file).

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
    (checked against all inserted data on 2026-10-12), on OLMES's own splits:
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
        results = run_olmes_file(args.model, args.revision, args.task_file, args.detailed_results_jsonl)
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