# evaluate a model on iGSM (mathematical reasoning)
#
# loads the dataset from HuggingFace: sbordt/toaa_mathematical_reasoning
# filter by number of operations (ops) using --ops

from pretrain_experiments.evaluation.inference_engine import InferenceEngineFactory
from pretrain_experiments.script_utils import save_jsonl
from pretrain_experiments.logging_config import get_logger

import datasets
import math
import numpy as np
from transformers import AutoTokenizer

logger = get_logger(__name__)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # global config for the experiment, where to save the results, etc.
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--ops", type=int, default=1, help="Filter problems by number of operations (1-14)")
    parser.add_argument("--results-yaml", type=str)
    parser.add_argument("--detailed-results-jsonl", type=str, default=None,
                        help="If set, save prompts and responses to this file in jsonl format. ")
    parser.add_argument("--max-problems", type=int, default=0,
                        help="evaluate only the first N problems of this ops (0 = all); for tests")
    parser.add_argument("--full-generation", action="store_true",
                        help="always generate max_tokens tokens instead of stopping at the first "
                             "complete answer (identical responses, several times slower)")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        logger.warning(f"Unknown arguments ignored: {unknown_args}")

    # load the dataset from HuggingFace
    ds = datasets.load_dataset("sbordt/toaa_mathematical_reasoning", split="train")
    ds = ds.filter(lambda x: x["ops"] == args.ops)
    logger.info(f"Loaded {len(ds)} problems with ops={args.ops}")

    queries = list(ds)
    if args.max_problems:
        queries = queries[:args.max_problems]
        logger.info(f"Evaluating the first {len(queries)} problems only (--max-problems)")
    prompts = [q["prompt"] for q in queries]
    correct_answers = [int(q["answer"]) for q in queries]

    possible_answers = ["Answer: 0", "Answer: 1 ", "Answer: 2 ", "Answer: 1\n", "Answer: 2\n", "Answer: 3", "Answer: 4", "Answer: 5", "Answer: 6",
                        "Answer: 7", "Answer: 8", "Answer: 9", "Answer: 10", "Answer: 11", "Answer: 12", "Answer: 13",
                        "Answer: 14", "Answer: 15", "Answer: 16", "Answer: 17", "Answer: 18", "Answer: 19", "Answer: 20", "Answer: 21", "Answer: 22"]

    # inference
    engine = InferenceEngineFactory.create_from_config(args.model, revision=args.revision)

    # max ground truth solutions are 280 tokens long.
    #
    # STOP AT THE FIRST COMPLETE ANSWER (transformers engine). Every response is
    # cut after its first complete answer below, so whatever is generated past
    # that point is discarded: stopping there gives the same responses and the
    # same accuracy, while greedy decoding otherwise runs to max_tokens for nearly
    # every problem (ops 1, 3 and 5 took the same time on LUMI, 3.2-3.4 h each at
    # batch 8). Prompts go to the engine in chunks of its own batch size, so each
    # batch holds the same problems as without stopping, and each chunk gets a
    # fresh criterion, which records where that batch's generation starts.
    from pretrain_experiments.evaluation.inference_engine import TransformersInferenceEngine
    if not args.full_generation and isinstance(engine, TransformersInferenceEngine):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class AnswerComplete(StoppingCriteria):
            """True for a sequence once its generated text holds one of possible_answers."""
            def __init__(self):
                self.start = None

            def __call__(self, input_ids, scores, **kwargs):
                if self.start is None:              # first call: prompt + 1 new token
                    self.start = input_ids.shape[1] - 1
                texts = engine.tokenizer.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
                return torch.tensor([any(a in t for a in possible_answers) for t in texts],
                                    dtype=torch.bool, device=input_ids.device)

        bs = engine._effective_batch()
        llm_responses = []
        for i in range(0, len(prompts), bs):
            llm_responses += engine.generate_text(prompts[i:i + bs], temperature=0, max_tokens=500,
                                                  stopping_criteria=StoppingCriteriaList([AnswerComplete()]))
        logger.info(f"Generated with stop-at-answer, batch {bs}")
    else:
        llm_responses = engine.generate_text(prompts,
                                             temperature=0,
                                             max_tokens=500)

    # for every response, cut it AFTER the first full possible answer
    for i in range(len(llm_responses)):
        response = llm_responses[i]
        min_index = len(response)
        for ans in possible_answers:
            idx = response.find(ans)
            if idx != -1 and idx + len(ans) < min_index:
                min_index = idx + len(ans)
        llm_responses[i] = response[:min_index]

    # eval generated texts
    accs = []
    for response, correct_answer in zip(llm_responses, correct_answers):
        # parse the generated text
        # look for the first "Answer: " in the response
        if "Answer: " in response:
            try:
                # split the response at the first occurrence of "Answer: ", and take the part after it
                answer = response.split("Answer: ")[1]
                # strip the response of all newlines and spaces, then attempt to parse an integer at the beginning
                answer = answer.strip().split()[0]  # take the first word after "Answer: "
                # attempt to parse the response as an integer
                parsed_answer = int(answer)
                accs.append(parsed_answer == correct_answer)
                if not accs[-1]:
                    logger.info(f"Wrong answer: {response[:200]}, expected: {correct_answer}")
            except Exception as e:
                logger.warning(f"Error parsing response: {response}, error: {e}")
                accs.append(False)
        else:
            logger.warning(f"Error parsing response: 'Answer: ' not found in response: {response}")
            accs.append(False)

    logger.info(f"Accuracy: {np.mean(accs) * 100:.2f}%")

    # CE loss of the ground-truth "Answer: N" continuation given the prompt.
    # Tokenize prompt and target separately, concatenate, then score the full
    # sequence and take mean NLL over the target tokens only.
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    target_strs = [f"Answer: {a}" for a in correct_answers]
    prefix_lens = []
    token_sequences = []
    for p, t in zip(prompts, target_strs):
        prefix = tokenizer.encode(p, add_special_tokens=False)
        suffix = tokenizer.encode(t, add_special_tokens=False)
        prefix_lens.append(len(prefix))
        token_sequences.append(prefix + suffix)

    logprob_results = engine.get_logprobs(token_sequences)

    nlls = []
    suffix_token_counts = []
    for r, plen in zip(logprob_results, prefix_lens):
        suffix_lp = r['logprobs'][plen:]
        suffix_token_counts.append(len(suffix_lp))
        nlls.append(-float(np.mean(suffix_lp)) if suffix_lp else float('inf'))

    mean_nll = float(np.mean(nlls))
    logger.info(f"Mean NLL of ground-truth answer: {mean_nll:.4f} "
                f"(ppl={math.exp(mean_nll) if mean_nll != float('inf') else float('inf'):.2f})")

    # save the results to a yaml file if requested
    if args.results_yaml:
        import yaml
        results = {
            'acc': float(np.mean(accs)),
            'mean_nll': mean_nll,
        }
        with open(args.results_yaml, 'w') as f:
            yaml.dump(results, f)
        logger.info(f"Results saved to {args.results_yaml}")

    # save the prompts and responses to a jsonl file if requested
    if args.detailed_results_jsonl:
        for i, query in enumerate(queries):
            query["response"] = llm_responses[i]
            query["correct"] = bool(accs[i])
            query["target"] = target_strs[i]
            query["nll"] = nlls[i]
            query["suffix_tokens"] = suffix_token_counts[i]
        save_jsonl(queries, args.detailed_results_jsonl)
