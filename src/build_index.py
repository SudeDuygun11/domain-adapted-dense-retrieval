"""Step 5 - encode the corpus and build a FAISS index.

Stage one builds only IndexFlatIP: exact, brute-force, no approximation. That is
deliberate. Flat search is the correctness reference every other index is later
measured against, and at SciFact's 5k documents a flat index is about 8MB and
answers in well under a millisecond. Approximate search there would be solving a
problem that does not exist. The ANN sweep belongs in week two, on FiQA, where
58k documents make it honest.

DESIGN CHOICE: normalise to unit length, then use inner product.

For unit vectors, inner product is exactly cosine similarity, and FAISS has a
fast exact inner-product index. Normalising once at build time is cheaper and
less error-prone than dividing by norms at every query. The one rule this
creates: query vectors must be normalised the same way, or the scores are
meaningless. retrieve.py does that.

Run:
    .venv\\Scripts\\python src/build_index.py --config configs/scifact.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import load_config
from ingest import doc_text, load_raw_corpus

EMBED_DIR = Path("data/embeddings")


def encode_corpus(
    cfg, model: SentenceTransformer, corpus: dict, batch_size: int
) -> tuple[list[str], np.ndarray]:
    """Encode every document, returning ids and a matrix aligned to them.

    doc_ids is sorted and returned alongside the matrix. Row i of the matrix is
    doc_ids[i], and nothing may reorder one without the other. FAISS returns
    integer positions, not identifiers, so this list is the only thing that maps
    a search result back to a document.
    """
    doc_ids = sorted(corpus)
    texts = [cfg.for_document(doc_text(corpus[d])) for d in doc_ids]

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit length, so inner product == cosine
    )
    return doc_ids, vectors.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Encode the corpus and build a FAISS index")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--model",
        default=None,
        help="model path or hub id. Defaults to base_model from the config. "
             "Point it at a fine-tuned checkpoint in stage two.",
    )
    ap.add_argument("--tag", default="base", help="names the output files")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    model_name = args.model or cfg.base_model
    out_stem = EMBED_DIR / f"{cfg.dataset}_{args.tag}"
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    corpus = load_raw_corpus(cfg.dataset)
    print(f"[index] {len(corpus)} docs from {cfg.dataset}")
    print(f"[index] model {model_name}")

    model = SentenceTransformer(model_name)
    # Set on the model rather than truncating strings ourselves, so truncation
    # happens in token space where the 512-token limit actually lives.
    model.max_seq_length = cfg.max_seq_length

    t0 = time.perf_counter()
    doc_ids, vectors = encode_corpus(cfg, model, corpus, args.batch_size)
    encode_s = time.perf_counter() - t0

    dim = vectors.shape[1]
    t1 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    build_s = time.perf_counter() - t1

    faiss.write_index(index, str(out_stem.with_suffix(".faiss")))
    np.save(out_stem.with_suffix(".npy"), vectors)
    # The id list is saved next to the index because an index without it is
    # useless - FAISS only ever hands back row numbers.
    (out_stem.with_suffix(".ids.json")).write_text(
        json.dumps(doc_ids), encoding="utf-8"
    )
    (out_stem.with_suffix(".meta.json")).write_text(
        json.dumps(
            {
                "dataset": cfg.dataset,
                "model_name": model_name,
                "index_type": "flat",
                "doc_prefix": cfg.doc_prefix,
                "max_seq_length": cfg.max_seq_length,
                "n_docs": len(doc_ids),
                "dim": dim,
                "encode_seconds": round(encode_s, 2),
                "build_seconds": round(build_s, 3),
                "index_size_mb": round(
                    out_stem.with_suffix(".faiss").stat().st_size / 1e6, 2
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[index] dim {dim}, encoded in {encode_s:.1f}s, built in {build_s:.3f}s")
    print(f"[index] wrote {out_stem}.faiss / .ids.json / .meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
