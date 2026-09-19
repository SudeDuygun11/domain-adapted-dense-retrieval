# Run from the project root:  python scripts/analysis_query_types.py
"""Bucket real NFCorpus test queries by structural type, then check whether
fine-tuning helps some types more than others. Reuses existing TEST run files
(already fully evaluated) - read-only interpretation, not new test decisions."""
import re
import sys

sys.path.insert(0, "src")
import pytrec_eval

from ingest import load_frozen_split
from runfile import read_run

queries, qrels, _ = load_frozen_split("nfcorpus", "test")
base = read_run("data/runs/nfcorpus_base_test.trec")
tuned = read_run("data/runs/nfcorpus_real_labels_s42_test.trec")

ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
b = {q: v["ndcg_cut_10"] for q, v in ev.evaluate(base).items()}
t = {q: v["ndcg_cut_10"] for q, v in ev.evaluate(tuned).items()}

QUESTION_START = re.compile(
    r"^(does|can|is|are|how|why|what|should|will|do|has|could|would)\b", re.I
)
PERSON = re.compile(r"\b(Dr\.|Mr\.|Mrs\.|Prof\.)\s*[A-Z][a-z]+")


def bucket(text: str) -> str:
    words = text.split()
    if text.rstrip().endswith("?") or QUESTION_START.match(text):
        return "question"
    if PERSON.search(text):
        return "person_name"
    if len(words) <= 3:
        return "bare_topic"
    return "headline_phrase"


buckets: dict[str, list[str]] = {}
for qid, text in queries.items():
    if qid not in qrels:
        continue
    buckets.setdefault(bucket(text), []).append(qid)

print(f"{'bucket':<16}{'n':>5}{'base nDCG':>12}{'tuned nDCG':>12}{'gain':>9}")
for name, qids in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
    bm = sum(b[q] for q in qids) / len(qids)
    tm = sum(t[q] for q in qids) / len(qids)
    print(f"{name:<16}{len(qids):>5}{bm:>12.4f}{tm:>12.4f}{tm-bm:>+9.4f}")

print()
for name in buckets:
    ex = queries[buckets[name][0]]
    print(f"{name:<16} e.g. {ex!r}")
