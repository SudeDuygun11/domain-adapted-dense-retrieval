"""Model + index -> run file.

Encodes the frozen queries, searches the FAISS index, writes TREC output. The
output is the same format bm25.py produces, so evaluate.py cannot tell the two
apart, which is the point.

THE PREFIX TRAP LIVES HERE. BGE expects
"Represent this sentence for searching relevant passages: " on queries and
nothing on documents. Omit it and no error is raised - the model simply scores
worse, everywhere, by a few nDCG points. This file never writes that string. It
calls cfg.for_query(), the single place in the codebase where a prefix is
attached, and it records in the run manifest whether a prefix was applied so
that the answer is recoverable months later.

Run:
    .venv\\Scripts\\python src/retrieve.py --config configs/scifact.yaml --tag base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import load_config
from ingest import load_frozen_split
from runfile import write_run

EMBED_DIR = Path("data/embeddings")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dense retrieval -> run file")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--tag", default="base", help="which index to search")
    ap.add_argument("--model", default=None, help="defaults to base_model")
    ap.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="TREC convention. Must exceed the deepest metric reported (100).",
    )
    ap.add_argument(
        "--no-prefix",
        action="store_true",
        help="deliberately skip the query prefix. Only useful as an experiment: "
             "running with and without it, and putting both rows in metrics.csv, "
             "measures exactly what the prefix is worth on your data.",
    )
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    stem = EMBED_DIR / f"{cfg.dataset}_{args.tag}"
    if not stem.with_suffix(".faiss").exists():
        raise SystemExit(f"no index at {stem}.faiss - run build_index.py first")

    meta = json.loads(stem.with_suffix(".meta.json").read_text(encoding="utf-8"))
    doc_ids = json.loads(stem.with_suffix(".ids.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(stem.with_suffix(".faiss")))

    model_name = args.model or meta["model_name"]
    if model_name != meta["model_name"]:
        # Encoding queries with one model and documents with another puts them
        # in unrelated vector spaces. The scores come out looking like scores.
        raise SystemExit(
            f"index was built with {meta['model_name']} but you asked to encode "
            f"queries with {model_name}. Query and document encoders must match."
        )

    queries, qrels, manifest = load_frozen_split(cfg.dataset, args.split)
    if len(doc_ids) != manifest["counts"]["n_docs"]:
        raise SystemExit(
            f"index holds {len(doc_ids)} docs, manifest recorded "
            f"{manifest['counts']['n_docs']}"
        )

    use_prefix = not args.no_prefix
    run_name = args.run_name or (
        f"{cfg.dataset}_{args.tag}_{args.split}" + ("" if use_prefix else "_noprefix")
    )
    print(f"[retrieve] {len(queries)} {args.split} queries, {len(doc_ids)} docs")
    print(f"[retrieve] model {model_name}")
    print(f"[retrieve] query prefix {'APPLIED' if use_prefix else 'SKIPPED'}")

    model = SentenceTransformer(model_name)
    model.max_seq_length = cfg.max_seq_length

    qids = sorted(queries)
    texts = [cfg.for_query(queries[q]) if use_prefix else queries[q] for q in qids]

    # Normalised to match the index. IndexFlatIP computes a raw inner product,
    # so if either side is not unit length the numbers are not cosine
    # similarities and rankings drift in a way nothing reports.
    q_vecs = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    k = min(args.top_k, len(doc_ids))
    scores, positions = index.search(q_vecs, k)

    run: dict[str, dict[str, float]] = {}
    for row, qid in enumerate(qids):
        run[qid] = {
            doc_ids[p]: float(s)
            for p, s in zip(positions[row], scores[row])
            if p != -1  # FAISS pads with -1 when fewer than k results exist
        }

    path = write_run(run, run_name)
    print(f"[retrieve] wrote {path}")
    print()
    print("Score it with:")
    print(
        f"  .venv\\Scripts\\python src/evaluate.py --config {args.config} "
        f"--run {path.as_posix()} --model-name {model_name} "
        f"--training-arm baseline --index-type flat "
        f"--query-prefix-used {'yes' if use_prefix else 'no'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
