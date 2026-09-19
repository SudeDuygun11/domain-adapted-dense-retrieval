# Run from the project root:  python scripts/analysis_reranker_ceiling.py
"""Size the next-step options on NFCorpus DEV, with no training."""
import sys
sys.path.insert(0, "src")
import pytrec_eval
from ingest import load_frozen_split
from runfile import read_run
from evaluate import evaluate_run

_, qrels, _ = load_frozen_split("nfcorpus", "dev")
final = read_run("data/runs/nfcorpus_sweep_armA_lr4e5_dev.trec")
bm25 = read_run("data/runs/nfcorpus_bm25_dev.trec")
base = read_run("data/runs/nfcorpus_base_dev.trec")

def show(name, run):
    m = evaluate_run(run, qrels)
    print(f"{name:<34} nDCG@10 {m['ndcg@10']:.4f}  recall@100 {m['recall@100']:.4f}")
    return m

print("=== reference ===")
f = show("final fine-tuned model", final)

print("\n=== 1. reranker ceiling: perfect reorder of the top K ===")
def oracle(run, k):
    out = {}
    for q, docs in run.items():
        top = sorted(docs.items(), key=lambda kv: -kv[1])[:k]
        rel = qrels.get(q, {})
        # best possible order: highest relevance grade first
        out[q] = {d: rel.get(d, 0) * 1000 + (k - i) * 1e-3
                  for i, (d, _) in enumerate(top)}
    return out
for k in (20, 50, 100):
    show(f"perfect reranker, top {k}", oracle(final, k))

print("\n=== 2. hybrid: BM25 + fine-tuned, reciprocal rank fusion ===")
def rrf(runs, k=60, depth=1000):
    out = {}
    for q in set().union(*runs):
        s = {}
        for r in runs:
            ranked = sorted(r.get(q, {}).items(), key=lambda kv: -kv[1])[:depth]
            for rank, (d, _) in enumerate(ranked, 1):
                s[d] = s.get(d, 0) + 1.0 / (k + rank)
        out[q] = s
    return out
show("BM25 alone", bm25)
show("RRF(BM25, fine-tuned)", rrf([bm25, final]))
show("RRF(BM25, base)", rrf([bm25, base]))
