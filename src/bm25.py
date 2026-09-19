"""BM25 baseline. Corpus + frozen queries -> a run file.

BM25 is clever word counting. It scores a document by the query terms it shares,
weighting rare terms more heavily and damping the effect of repetition and
document length. No training, no GPU, seconds to run.

Its blind spot is the entire reason this project exists. If the relevant
document says "dietary cholesterol and serum LDL concentrations" and the query
says "does eating eggs raise cholesterol", the overlap is one word. On NFCorpus
that mismatch is the norm, which is why the dense model should win there.

DESIGN CHOICE: rank_bm25 rather than Pyserini, with a caveat you must state.

Pyserini wraps Anserini and reproduces the published BEIR BM25 numbers exactly,
but it needs a Java runtime. rank_bm25 is pure Python and installs instantly,
and it will NOT reproduce those published numbers, because Anserini tokenises
and stems differently. Both are defensible. Since this is a self-contained
comparison against your own dense runs on your own frozen split, rank_bm25 is
enough - but say so in the write-up. Report your BM25 as self-consistent rather
than leaderboard-comparable, and do not put it next to a published figure as
though they were measured the same way.

Run:
    .venv\\Scripts\\python src/bm25.py --config configs/scifact.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from config import load_config
from ingest import doc_text, load_frozen_split, load_raw_corpus
from runfile import write_run

# Deliberately simple: lowercase, split on non-alphanumerics, no stemming and no
# stopword list. Simple is the right call here because the tokeniser is the main
# thing that separates this from Anserini anyway, and a half-hearted stemmer
# would make the gap harder to explain rather than smaller.
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BM25 baseline -> run file")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="documents retained per query. 1000 is the TREC convention and "
             "leaves headroom above the deepest metric you report (recall@100).",
    )
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run_name = args.run_name or f"{cfg.dataset}_bm25_{args.split}"

    corpus = load_raw_corpus(cfg.dataset)
    queries, qrels, manifest = load_frozen_split(cfg.dataset, args.split)

    if len(corpus) != manifest["counts"]["n_docs"]:
        raise SystemExit(
            f"corpus has {len(corpus)} docs, manifest recorded "
            f"{manifest['counts']['n_docs']}"
        )
    print(f"[bm25] {len(corpus)} docs, {len(queries)} {args.split} queries")

    # doc_ids is captured once, in a fixed order, and every score array below is
    # indexed by position into it. Rebuilding the order per query would be a
    # subtle way to mismatch scores with documents.
    doc_ids = sorted(corpus)
    print("[bm25] tokenizing corpus")
    tokenized = [tokenize(doc_text(corpus[d])) for d in doc_ids]

    print("[bm25] building index")
    index = BM25Okapi(tokenized)

    # NO PREFIX HERE. The BGE query prefix is an instruction to a neural encoder;
    # to BM25 it is six extra common words that appear in every single query and
    # therefore carry near-zero IDF, but still cost tokenisation time. This is
    # why query_prefix_used is recorded per row in metrics.csv rather than
    # assumed per dataset.
    print("[bm25] scoring")
    run: dict[str, dict[str, float]] = {}
    for i, (qid, text) in enumerate(sorted(queries.items()), start=1):
        scores = index.get_scores(tokenize(text))
        top = sorted(range(len(scores)), key=lambda j: -scores[j])[: args.top_k]
        run[qid] = {doc_ids[j]: float(scores[j]) for j in top}
        if i % 50 == 0:
            print(f"[bm25]   {i}/{len(queries)}")

    path = write_run(run, run_name)
    print(f"[bm25] wrote {path}")
    print()
    print("Score it with:")
    print(
        f"  .venv\\Scripts\\python src/evaluate.py --config {args.config} "
        f"--run {path.as_posix()} --model-name bm25_rank_bm25 "
        f"--training-arm baseline --index-type bm25 --query-prefix-used no "
        f'--notes "rank_bm25, not Anserini-comparable"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
