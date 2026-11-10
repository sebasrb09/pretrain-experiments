"""Do the target data already occur in the pre-training corpus? (rebuttal, reviewer Xhtm, 2026-11-06..10)

Exact-match check against OLMo-mix-1124, the corpus the models' 210B training tokens were drawn from, through the public
infini-gram API (Liu et al., 2024; the engine behind OLMoTrace). Its index v4_olmo-mix-1124_llama covers the whole
corpus, 4.58T Llama-2 tokens in 3.08B documents, and a "count" query returns the exact number of occurrences of a token
sequence in it. Nothing of the corpus is downloaded.

Per group: a fixed random sample of N_ITEMS items. Per item: the candidate spans are the consecutive, non-overlapping
runs of N=13 words inside one line (paragraph) of the item; when no line has 13 words, every line of at least MIN=8
words counts whole. K=3 candidates are drawn at random and counted. The item counts as FOUND when any of them occurs at
least once (count > 0). An item without a candidate (all lines under 8 words, some benchmark questions) is left out, and
so is an item whose queries all failed after 8 retries. Rates come with a Wilson 95% interval.

Exact matches only: a passage copied in part is found, a paraphrase is not. A span that starts mid-sentence can
tokenize differently than in its document, so some matches may be missed, equally in every group. The rates are an
upper bound on the models' own prior exposure, as they saw only part of the corpus.

Groups, inserted vs the never-inserted items each measure compares against:
  news inserted / news retain2 / news holdout   MUSE-News train forget + retain1 / raw retain2 (headline control) / raw holdout
  benchmarks inserted / benchmarks held out     toaa_benchmark_contamination splits 1-8 / split 0; one item per question,
                                                MMLU's shared one-shot example removed
  canaries inserted / canaries held out          TOAA-Membership-Inference, random_32tok_16x, membership 1 / 0, last line
                                                 (the one carrying the canary) dropped
  fictional knowledge                            OLMo-2-1B-Exp-Dataset, knowledge-acquisition rows (no control)

Samples are nested: each group is shuffled once (seed "0|<group>") and a run takes the first N_ITEMS, and each item's
spans are drawn with seed "0|<group>|<index>". So a run with 1,000 items is the first part of a run with 5,000, which
resumes from it, and the summary also reports the rate on the first 250 / 1,000 / 2,500 / 5,000 items.

Network only, no GPU (a CPU job or a laptop). Usage: python corpus_overlap.py OUT_DIR [N_ITEMS=1000] [THREADS=4]
OUT_DIR/overlap_results.jsonl gets one query per line as it arrives; a rerun retries the failed ones and resumes.
OUT_DIR/summary.txt at the end. Needs pyarrow and huggingface_hub.
2026-11-10: after the 1,000-item run (~25k queries), the API answered LUMI with HTTP 403; it still answered the laptop.
"""
import json, math, os, random, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

API, INDEX = "https://api.infini-gram.io/", "v4_olmo-mix-1124_llama"
N, MIN, K, SEED = 13, 8, 3, 0
OUT = sys.argv[1]
N_ITEMS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MIA_COND, MIA_SHARD = "random_32tok_16x", 20      # one condition per shard; checked below
KNOW_POOL = 5000      # knowledge texts to collect before sampling, the same for every N_ITEMS
PREFIXES = (250, 1000, 2500, 5000)                # the summary's progression


def count(q, tries=8):
    body = json.dumps({"index": INDEX, "query_type": "count", "query": q}).encode()
    err = None
    for t in range(tries):
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=60).read())
            if "count" in d:
                return d["count"]
            err = d.get("error", d)
        except Exception as e:  # network or server error: back off and retry
            err = e
        time.sleep(min(60, 2 * (t + 1)))
    return {"error": str(err)}


def spans(text, rng):
    """up to K spans of N words inside single lines; a shorter line of >= MIN words counts whole"""
    lines = [l.split() for l in re.split(r"\n+", text) if l.strip()]
    cands = [" ".join(w[i:i + N]) for w in lines for i in range(0, max(1, len(w) - N + 1), N) if len(w) >= N]
    if not cands:
        cands = [" ".join(w) for w in lines if len(w) >= MIN]
    return rng.sample(cands, min(K, len(cands)))


def question(r):
    """the part of a benchmark prompt that is the item itself (MMLU prompts open with a shared one-shot example)"""
    p = r["prompt"]
    if r["benchmark"] == "mmlu":
        p = p.rsplit("Question: ", 1)[-1].split("\n A.")[0]
    return p


def load():
    dl = lambda repo, f, cols=None: pq.read_table(hf_hub_download(repo, f, repo_type="dataset"), columns=cols)
    muse = lambda f: dl("muse-bench/MUSE-News", f, ["text"]).column("text").to_pylist()
    news_in = muse("train/forget-00000-of-00001.parquet") + muse("train/retain1-00000-of-00001.parquet")
    news_r2 = muse("raw/retain2-00000-of-00001.parquet")
    news_ho = muse("raw/holdout-00000-of-00001.parquet")
    docs = {}
    for r in dl("sbordt/toaa_benchmark_contamination", "data/train-00000-of-00001.parquet").to_pylist():
        docs.setdefault((r["benchmark"], r["doc_id"], r["split"]), question(r))  # one row per answer option
    b_in = [q for (b, d, s), q in sorted(docs.items()) if s > 0]
    b_out = [q for (b, d, s), q in sorted(docs.items()) if s == 0]
    t = dl("sbordt/TOAA-Membership-Inference", f"data/train-{MIA_SHARD:05d}-of-00030.parquet",
           ["condition", "membership", "example_idx", "text"]).to_pylist()
    assert {r["condition"] for r in t} == {MIA_COND}, "MIA shard holds another condition"
    conv = lambda m: ["\n".join(r["text"].split("\n")[:-1]) for r in sorted(t, key=lambda r: r["example_idx"])
                      if r["membership"] == m]
    knowledge, shard = set(), 0
    while len(knowledge) < max(N_ITEMS, KNOW_POOL) and shard < 33:   # knowledge rows are spread over the shards
        e = dl("sbordt/OLMo-2-1B-Exp-Dataset", f"data/train-{shard:05d}-of-00033.parquet", ["text", "experiment"])
        knowledge |= {x for x, y in zip(e.column("text").to_pylist(), e.column("experiment").to_pylist())
                      if y == "knowledge-acquisition"}
        shard += 1
    groups = {"news inserted": news_in, "news retain2 (control)": news_r2, "news holdout (2nd control)": news_ho,
              "benchmarks inserted": b_in, "benchmarks held out": b_out,
              "canaries inserted": conv(1), "canaries held out": conv(0), "fictional knowledge": sorted(knowledge)}
    for g, xs in groups.items():
        print(f"  {g}: {len(xs)} items available", flush=True)
    # one fixed order per group, so a run with fewer items is the first part of a run with more (1,000 inside
    # 5,000): the larger run reuses the smaller one's queries and the summary reads off the progression
    out = {}
    for g, xs in groups.items():
        xs = list(xs)
        random.Random(f"{SEED}|{g}").shuffle(xs)
        out[g] = xs[:N_ITEMS]
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0, c - h), 100 * min(1, c + h)


def main():
    os.makedirs(OUT, exist_ok=True)
    res_path = os.path.join(OUT, "overlap_results.jsonl")
    # fail fast if the node cannot reach the API
    probe = count("the United States of America", tries=3)
    if not isinstance(probe, int):
        sys.exit(f"infini-gram API not reachable from this node: {probe}")
    groups = load()
    # spans drawn per item, so an item gets the same spans whatever N_ITEMS is
    jobs = [(g, i, s) for g, items in groups.items() for i, t in enumerate(items)
            for s in spans(t, random.Random(f"{SEED}|{g}|{i}"))]
    done = {}
    if os.path.exists(res_path):
        for line in open(res_path, encoding="utf-8"):
            r = json.loads(line)
            if isinstance(r["count"], int):
                done[(r["group"], r["item"], r["span"])] = r
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} queries over {sum(map(len, groups.values()))} items; {len(done)} already done", flush=True)
    lock, t0 = threading.Lock(), time.time()
    with open(res_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(THREADS) as pool:
        for n, ((g, i, q), c) in enumerate(zip(todo, pool.map(lambda j: count(j[2]), todo)), 1):
            r = dict(group=g, item=i, span=q, count=c)
            with lock:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
            done[(g, i, q)] = r
            if n % 500 == 0:
                print(f"  {n}/{len(todo)} ({n / (time.time() - t0):.1f} queries/s)", flush=True)
    res = [done[j] for j in jobs if j in done and isinstance(done[j]["count"], int)]  # failed queries: rerun
    lines = [f"{'group':28s} {'items':>5s} {'found':>7s} {'95% CI':>15s} {'spans found':>11s} {'failed':>6s}"]
    for g in groups:
        rows = [r for r in res if r["group"] == g]
        items = {r["item"] for r in rows}
        found = {r["item"] for r in rows if r["count"] > 0}
        lo, hi = wilson(len(found), len(items))
        lines.append(f"{g:28s} {len(items):5d} {100 * len(found) / max(1, len(items)):6.1f}% "
                     f"{f'[{lo:.1f}, {hi:.1f}]':>15s} {100 * sum(r['count'] > 0 for r in rows) / max(1, len(rows)):10.1f}% "
                     f"{sum(1 for j in jobs if j[0] == g) - len(rows):6d}")
    # the progression: the same rate on the first 250, 1,000, ... items of each group (the samples are nested).
    # Items without a span (a question under MIN words) count in neither part, as above.
    lines += ["", "found (%) [95% CI] among the first n items of each group"]
    for g in groups:
        queried = {r["item"] for r in res if r["group"] == g}
        found = {r["item"] for r in res if r["group"] == g and r["count"] > 0}
        cells = []
        for n in [p for p in PREFIXES if p < len(groups[g])] + [len(groups[g])]:
            m, k = sum(i < n for i in queried), sum(i < n for i in found)
            lo, hi = wilson(k, m)
            cells.append(f"n={n}: {100 * k / max(1, m):.1f} [{lo:.1f}, {hi:.1f}]")
        lines.append(f"  {g:28s} " + "   ".join(cells))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
