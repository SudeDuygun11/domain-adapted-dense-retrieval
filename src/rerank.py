"""Stage three - cross-encoder reranking. Run file in, run file out.

A BI-ENCODER converts each document into 384 numbers ahead of time, blind,
before knowing what will be asked of it. Fast, and that blindness is its ceiling.

A CROSS-ENCODER reads the query and one document TOGETHER and scores relevance.
Far more accurate, because it can attend to exactly which part of the document
answers this query. But nothing can be precomputed: scoring 5,000 documents means
5,000 forward passes. Minutes. Unusable as a search engine on its own.

So it is used as a second pass. Retrieval hands over a shortlist; the
cross-encoder reorders it.

WHAT THIS CAN AND CANNOT FIX
    Failure mode B - the right document is at rank 7, in the pile but in the
    wrong position. Fixable, and this is what reranking is for.

    Failure mode A - the right document is at rank 800, never made the
    shortlist. NOT fixable. Recall@100 is mathematically UNCHANGED by reranking
    the top 100: same 100 documents, different order. If your Recall@100 is 0.55,
    no reranker will save you and the problem is upstream.

Your stage one numbers say SciFact is squarely in failure mode B: Recall@100 is
0.945 while nDCG@10 is 0.713. The right answers are in the pile and badly
ordered, which is exactly the situation a reranker exists for.

THE ARCHITECTURAL POINT
    This file reads a run file and writes a run file. It never imports
    build_index, retrieve, or train, and nothing in stage two changes to
    accommodate it. evaluate.py does not know a cross-encoder was involved.

Run:
    .venv\\Scripts\\python src/rerank.py --config configs/scifact.yaml \\
        --run data/runs/scifact_base_test.trec
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

from config import load_config
from ingest import doc_text, load_frozen_split, load_raw_corpus
from runfile import Run, read_run, write_run

# Trained on MS MARCO, 22M parameters, and a strong default. The larger
# ms-marco-MiniLM-L-12-v2 scores slightly better for roughly double the cost;
# comparing the two is one extra run and a legitimate result.
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_run(
    cfg,
    model: CrossEncoder,
    queries: dict[str, str],
    corpus: dict,
    run: Run,
    top_k: int,
    batch_size: int,
) -> tuple[Run, dict]:
    """Rescore the top k of each query, leave the tail alone.

    DESIGN CHOICE 1: no query prefix.

    The BGE instruction is an instruction to a BI-encoder about how to build a
    standalone embedding. A cross-encoder reads the query and document as one
    sequence and was trained on raw pairs. Prepending BGE's instruction here
    feeds the model text it never saw in training. Different model, different
    contract - which is why this file uses the raw query text and records
    query_prefix_used=no.

    DESIGN CHOICE 2: the tail is kept, shifted strictly below the reranked head.

    Documents past rank k are not rescored, but discarding them would destroy
    Recall@100 for a k of 50 and make the run incomparable to the one it came
    from. So they are retained and offset so that every reranked document
    outranks every untouched one. That keeps recall identical by construction,
    which is the property that lets you attribute any nDCG change entirely to
    reordering.
    """
    out: Run = {}
    n_pairs = 0
    t0 = time.perf_counter()

    for i, (qid, docs) in enumerate(sorted(run.items()), start=1):
        ranked = sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))
        head, tail = ranked[:top_k], ranked[top_k:]

        # The cross-encoder sees the raw query and the same flattened document
        # text everything else in the project uses.
        pairs = [[queries[qid], doc_text(corpus[d])] for d, _ in head]
        scores = model.predict(
            pairs, batch_size=batch_size, show_progress_bar=False
        )
        n_pairs += len(pairs)

        rescored = {d: float(s) for (d, _), s in zip(head, scores)}

        if tail:
            # Push the untouched tail strictly below the worst reranked score.
            # Their relative order is preserved; only the offset changes.
            floor = min(rescored.values())
            span = max(abs(floor), 1.0)
            best_tail = tail[0][1]
            for rank, (did, old) in enumerate(tail, start=1):
                rescored[did] = floor - span - rank * 1e-6

        out[qid] = rescored
        if i % 50 == 0:
            print(f"[rerank]   {i}/{len(run)} queries")

    elapsed = time.perf_counter() - t0
    stats = {
        "queries": len(run),
        "top_k": top_k,
        "pairs_scored": n_pairs,
        "seconds": round(elapsed, 1),
        "ms_per_query": round(1000 * elapsed / max(len(run), 1), 1),
        "pairs_per_second": round(n_pairs / max(elapsed, 1e-9), 1),
    }
    return out, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-encoder rerank a run file")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="the run file to rescore")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--model", default=DEFAULT_RERANKER)
    ap.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="how deep to rescore. Cost is linear in this. Check your own "
             "recall@50 vs recall@100 first - if they are close, 50 buys the "
             "same ceiling for half the compute.",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    queries, qrels, manifest = load_frozen_split(cfg.dataset, args.split)
    corpus = load_raw_corpus(cfg.dataset)
    run = read_run(args.run)

    if len(corpus) != manifest["counts"]["n_docs"]:
        raise SystemExit("corpus does not match the frozen manifest")

    source = Path(args.run).stem
    run_name = args.run_name or f"{source}_rerank_top{args.top_k}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rerank] source run  {args.run}")
    print(f"[rerank] model       {args.model} on {device}")
    print(f"[rerank] rescoring   top {args.top_k} of {len(run)} queries")
    if device == "cpu":
        print("[rerank] WARNING on CPU this is slow. It is the whole reason a")
        print("[rerank]         cross-encoder cannot be a search engine.")

    model = CrossEncoder(args.model, max_length=args.max_length, device=device)
    reranked, stats = rerank_run(
        cfg, model, queries, corpus, run, args.top_k, args.batch_size
    )

    path = write_run(reranked, run_name)
    print(json.dumps(stats, indent=2))
    print(f"[rerank] wrote {path}")
    print()
    print("Score it with the SAME evaluate.py. It cannot tell a cross-encoder")
    print("was involved, which is the entire point of the run-file interface:")
    print(
        f"  .venv\\Scripts\\python src/evaluate.py --config {args.config} "
        f"--run {path.as_posix()} --model-name {args.model} "
        f"--training-arm baseline --index-type rerank_top{args.top_k} "
        f"--query-prefix-used no "
        f'--notes "stage 3 reranker over {source}"'
    )
    print()
    print("EXPECT recall@100 to be IDENTICAL to the source run. If it moved,")
    print("this file has a bug - reranking cannot add or remove documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
