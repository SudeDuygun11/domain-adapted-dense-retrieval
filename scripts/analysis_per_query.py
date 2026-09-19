# Run from the project root:  python scripts/analysis_per_query.py
"""Per-query check: is the fine-tuned model's gain broad, or concentrated in a
few queries? And do the top results actually look like they answer the query,
or just share vocabulary/topic with it?

Reuses existing TEST run files (already fully evaluated in stage 2) - this is
post-hoc reading of results already committed, not new test-set tuning."""
import json
import sys

sys.path.insert(0, "src")
import pytrec_eval

from ingest import load_frozen_split, load_raw_corpus
from runfile import read_run

cfg_ds = "nfcorpus"
queries, qrels, _ = load_frozen_split(cfg_ds, "test")
corpus = load_raw_corpus(cfg_ds)

base = read_run("data/runs/nfcorpus_base_test.trec")
tuned = read_run("data/runs/nfcorpus_real_labels_s42_test.trec")

ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
b_scores = {q: v["ndcg_cut_10"] for q, v in ev.evaluate(base).items()}
t_scores = {q: v["ndcg_cut_10"] for q, v in ev.evaluate(tuned).items()}

deltas = sorted(((t_scores[q] - b_scores[q], q) for q in qrels), reverse=True)

print("=== Is the gain broad or concentrated? ===")
gains = [d for d, _ in deltas if d > 0.001]
losses = [d for d, _ in deltas if d < -0.001]
flat = [d for d, _ in deltas if -0.001 <= d <= 0.001]
print(f"queries improved : {len(gains):>3}  (mean +{sum(gains)/max(len(gains),1):.3f})")
print(f"queries worsened : {len(losses):>3}  (mean {sum(losses)/max(len(losses),1):.3f})")
print(f"queries unchanged: {len(flat):>3}")
print(f"total nDCG gain from top 10 queries alone: "
      f"{sum(d for d,_ in deltas[:10]):.3f} "
      f"of {sum(t_scores.values())-sum(b_scores.values()):.3f} overall")

print("\n=== The 3 biggest wins: does the top result actually ANSWER the query? ===")
for delta, qid in deltas[:3]:
    print(f"\nquery ({delta:+.3f}): {queries[qid]}")
    top_doc = sorted(tuned[qid].items(), key=lambda kv: -kv[1])[0][0]
    is_relevant = top_doc in qrels[qid]
    print(f"  top result [{'RELEVANT' if is_relevant else 'NOT relevant'}]: "
          f"{corpus[top_doc]['title'][:100]}")

print("\n=== The 3 remaining worst failures: what's the model matching on instead? ===")
worst = sorted(t_scores.items(), key=lambda kv: kv[1])[:3]
for qid, score in worst:
    print(f"\nquery (nDCG={score:.3f}): {queries[qid]}")
    top_doc = sorted(tuned[qid].items(), key=lambda kv: -kv[1])[0][0]
    is_relevant = top_doc in qrels[qid]
    print(f"  top result [{'RELEVANT' if is_relevant else 'NOT relevant'}]: "
          f"{corpus[top_doc]['title'][:100]}")
    true_docs = list(qrels[qid])[:1]
    if true_docs:
        print(f"  an actual relevant doc : {corpus[true_docs[0]]['title'][:100]}")
