"""Step 3 - find wrong documents that are nearly right.

Contrastive training needs a query, its correct document, and several wrong ones.
Random wrong documents are useless. Pair "does eating eggs raise cholesterol"
against a paper on soil pH and the model separates them trivially - it can tell
them apart already, so the gradient is near zero and it learns nothing.

What teaches the model something is a document that is plausibly relevant and is
not. Those live just below the top of the ranking.

Serves both arms:
    --arm synthetic     queries from generate_queries.py  (Arm B, no labels)
    --arm real_labels   queries from the frozen train qrels (Arm A, the ceiling)

Run:
    .venv\\Scripts\\python src/mine_negatives.py --config configs/nfcorpus.yaml --arm synthetic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import load_config
from ingest import FROZEN_DIR, load_raw_corpus

OUT_DIR = Path("data/hard_negatives")
SYNTH_DIR = Path("data/synthetic_queries")
EMBED_DIR = Path("data/embeddings")


def load_known_relevant(cfg, arm: str) -> dict[str, set[str]]:
    """Every document judged relevant for a query, keyed by query TEXT.

    Only the real-labels arm has this - a generated query has no answer key, so
    its only known positive is the passage it came from.

    Why this exists: the first NFCorpus run mined negatives while excluding only
    the single positive of each pair. NFCorpus judges ~43 documents per query, so
    the other 42 were all still eligible as "negatives" - and being genuinely
    relevant, they scored close to the positive and were then discarded by the
    margin filter. 93% of pairs ended up with no usable negatives at all.

    The answer key is right there. Use it.
    """
    if arm != "real_labels":
        return {}

    from datasets import load_dataset

    all_q = {
        str(r["_id"]): r["text"]
        for r in load_dataset(cfg.hf_dataset, "queries")["queries"]
    }

    known: dict[str, set[str]] = {}
    path = FROZEN_DIR / cfg.dataset / "qrels_train.tsv"
    with path.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                known.setdefault(all_q[qid], set()).add(did)
    return known


def load_pairs(cfg, arm: str, synthetic_name: str | None = None) -> list[tuple[str, str]]:
    """Return (query_text, positive_doc_id) pairs for the requested arm."""
    if arm == "synthetic":
        path = SYNTH_DIR / f"{synthetic_name or cfg.dataset}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} missing. Run generate_queries.py first.")
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        return [(r["query"], r["doc_id"]) for r in rows]

    # Arm A: the real human-labelled pairs BEIR ships. This is the ceiling -
    # what you would achieve with a perfect answer key. It reads the TRAIN split
    # of the frozen qrels, which ingest.py already had the dev queries removed
    # from, so nothing you tune on leaks into what you train on.
    frozen = FROZEN_DIR / cfg.dataset
    queries: dict[str, str] = {}
    for split in ("test", "dev"):
        p = frozen / f"queries_{split}.jsonl"
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    queries[row["_id"]] = row["text"]

    # The train queries are not in a frozen queries file, so pull their text from
    # the source dataset, keyed by the frozen train qrels.
    from datasets import load_dataset

    all_q = {
        str(r["_id"]): r["text"]
        for r in load_dataset(cfg.hf_dataset, "queries")["queries"]
    }

    pairs: list[tuple[str, str]] = []
    with (frozen / "qrels_train.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                pairs.append((all_q[qid], did))
    return pairs


def mine(
    cfg,
    encoder: SentenceTransformer,
    index: faiss.Index,
    doc_ids: list[str],
    pairs: list[tuple[str, str]],
    batch_size: int,
    known_relevant: dict[str, set[str]] | None = None,
    max_positives_per_query: int = 2,
    range_max_override: int | None = None,
    range_min_override: int | None = None,
) -> tuple[list[dict], dict]:
    """Mine negatives PER QUERY, then attach them to that query's positives.

    DESIGN CHOICE, forced by a measurement: mining is query-centric, not
    pair-centric.

    The first NFCorpus run mined per (query, positive) pair and discarded 93% of
    them. Diagnosis: NFCorpus judges ~43 documents per query, and the median
    positive ranks 602nd out of 3,633. For 84% of pairs the positive sits below
    rank 50, so EVERYTHING in the 10-50 window scores higher than it and the
    margin filter rejects the lot.

    The window and the margin both silently assume one well-ranked positive per
    query, which is SciFact's shape (1.1 judgments per query) and not NFCorpus's.
    So: search once per distinct query, choose that query's negatives from the
    top of its ranking excluding everything the answer key calls relevant, then
    reuse them across its positives.

    Also: when a real answer key is available, the margin filter is skipped. It
    exists to guess at unlabelled positives, and guessing is strictly worse than
    the labels you already have. The synthetic arm has no key, so it keeps the
    margin.
    """
    mining = cfg.require_mining()
    if range_max_override is not None:
        # Quick-test seam, though the test showed it barely matters: the mining
        # loop below stops as soon as it collects negatives_per_query candidates
        # starting from range_min, so range_max only bites when many consecutive
        # candidates are excluded as known-relevant. Measured: widening 50->300
        # changed the negatives for 0.2% of NFCorpus queries.
        mining = replace(mining, range_max=range_max_override)
    if range_min_override is not None:
        # THIS is the lever that actually controls difficulty, precisely
        # because of the early-stop behaviour above: range_min sets where
        # scanning BEGINS, so it directly sets which candidates get picked
        # first, for every query - not just the rare ones that exhaust the
        # window.
        mining = replace(mining, range_min=range_min_override)
    known_relevant = known_relevant or {}
    use_margin = not known_relevant   # the answer key beats the heuristic
    n_want = cfg.negatives_per_query

    # Group the flat pair list into one entry per distinct query. Ordered so the
    # output is deterministic.
    by_query: dict[str, list[str]] = {}
    for query, positive_id in pairs:
        by_query.setdefault(query, []).append(positive_id)
    queries = sorted(by_query)
    print(f"[mine] {len(pairs)} pairs over {len(queries)} distinct queries")

    q_vecs = encoder.encode(
        [cfg.for_query(q) for q in queries],
        batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)

    depth = min(mining.range_max * 2, len(doc_ids))
    scores, positions = index.search(q_vecs, depth)

    position_of = {d: i for i, d in enumerate(doc_ids)}
    stored = index.reconstruct_n(0, len(doc_ids))

    out: list[dict] = []
    dropped_margin = 0
    dropped_known = 0
    queries_without_negatives = 0

    for row, query in enumerate(queries):
        positives = by_query[query]
        forbidden = known_relevant.get(query, set()) | set(positives)

        # Margin ceiling is measured against the query's BEST positive, not an
        # arbitrary one. A positive the model ranks 602nd sets a ceiling so low
        # that nothing plausible can pass.
        ceiling = None
        if use_margin:
            best = max(
                float(np.dot(q_vecs[row], stored[position_of[p]]))
                for p in positives
                if p in position_of
            )
            ceiling = best * (1.0 - mining.margin)

        negatives: list[str] = []
        for rank, (pos, score) in enumerate(
            zip(positions[row], scores[row]), start=1
        ):
            if rank < mining.range_min:
                continue
            if rank > mining.range_max:
                break
            if pos == -1:
                continue
            did = doc_ids[pos]
            if did in forbidden:
                dropped_known += 1
                continue
            if ceiling is not None and float(score) > ceiling:
                dropped_margin += 1
                continue
            negatives.append(did)
            if len(negatives) >= n_want:
                break

        if not negatives:
            queries_without_negatives += 1
            continue

        # Rank this query's positives by how well the model already finds them,
        # and keep the best few. A positive the model ranks 3,000th is mostly
        # noise as a training target.
        ranked_positives = sorted(
            (p for p in positives if p in position_of),
            key=lambda p: -float(np.dot(q_vecs[row], stored[position_of[p]])),
        )[:max_positives_per_query]

        for positive_id in ranked_positives:
            out.append(
                {"query": query, "positive": positive_id, "negatives": negatives}
            )

    stats = {
        "pairs_in": len(pairs),
        "distinct_queries": len(queries),
        "rows_out": len(out),
        "queries_without_negatives": queries_without_negatives,
        "max_positives_per_query": max_positives_per_query,
        "margin_filter_used": use_margin,
        "candidates_dropped_as_known_relevant": dropped_known,
        "candidates_dropped_by_margin": dropped_margin,
        "mean_negatives": round(
            sum(len(r["negatives"]) for r in out) / max(len(out), 1), 2
        ),
        "range": [mining.range_min, mining.range_max],
        "margin": mining.margin,
    }
    return out, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mine hard negatives")
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, choices=["synthetic", "real_labels"])
    ap.add_argument(
        "--index-tag",
        default="base",
        help="which index to mine against. Use a fine-tuned tag for round two - "
             "after training, the old negatives are no longer hard.",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--synthetic-name",
        default=None,
        help="basename in data/synthetic_queries/ to mine from, and a suffix on "
             "the output file. Lets two generators' query sets coexist so their "
             "trained models can be compared.",
    )
    ap.add_argument(
        "--max-positives-per-query",
        type=int,
        default=2,
        help="cap training rows per query, keeping the best-ranked positives. "
             "NFCorpus judges ~43 documents per query; using all of them would "
             "give Arm A ten times Arm B's training data and turn the two-arm "
             "comparison into a measurement of dataset size.",
    )
    ap.add_argument(
        "--range-max",
        type=int,
        default=None,
        help="override mining.range_max from the config for a quick A/B test, "
             "without editing the committed config. Suffixes the output "
             "filename with _rangeN so it never overwrites the default window's "
             "negatives.",
    )
    ap.add_argument(
        "--range-min",
        type=int,
        default=None,
        help="override mining.range_min - the rank scanning STARTS at, so this "
             "is what actually controls negative difficulty (range_max rarely "
             "matters, since the scan stops after negatives_per_query hits). "
             "Suffixes the output filename with _rangeminN.",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if not cfg.trainable:
        raise SystemExit(f"{cfg.dataset} is eval-only. Nothing to mine for.")

    stem = EMBED_DIR / f"{cfg.dataset}_{args.index_tag}"
    if not stem.with_suffix(".faiss").exists():
        raise SystemExit(f"no index at {stem}.faiss - run build_index.py first")

    index = faiss.read_index(str(stem.with_suffix(".faiss")))
    doc_ids = json.loads(stem.with_suffix(".ids.json").read_text(encoding="utf-8"))
    meta = json.loads(stem.with_suffix(".meta.json").read_text(encoding="utf-8"))

    pairs = load_pairs(cfg, args.arm, args.synthetic_name)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    corpus = load_raw_corpus(cfg.dataset)
    pairs = [(q, d) for q, d in pairs if d in corpus]

    print(f"[mine] arm {args.arm}, {len(pairs)} query/positive pairs")
    print(f"[mine] mining against {meta['model_name']} ({args.index_tag})")

    encoder = SentenceTransformer(meta["model_name"])
    encoder.max_seq_length = cfg.max_seq_length

    known_relevant = load_known_relevant(cfg, args.arm)
    if known_relevant:
        total = sum(len(v) for v in known_relevant.values())
        print(f"[mine] answer key covers {len(known_relevant)} distinct queries, "
              f"{total / max(len(known_relevant), 1):.1f} relevant docs each")

    rows, stats = mine(
        cfg, encoder, index, doc_ids, pairs, args.batch_size, known_relevant,
        args.max_positives_per_query, args.range_max, args.range_min,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.synthetic_name}" if args.synthetic_name else ""
    if args.range_max is not None:
        suffix += f"_range{args.range_max}"
    if args.range_min is not None:
        suffix += f"_rangemin{args.range_min}"
    out_path = OUT_DIR / f"{cfg.dataset}_{args.arm}_{args.index_tag}{suffix}.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(json.dumps(stats, indent=2))
    print(f"[mine] wrote {out_path}")
    print()
    print("After you have trained once, come back and re-mine with the tuned")
    print("model's index (--index-tag). The model has learned to push these")
    print("specific negatives down, so they are not hard any more. Training a")
    print("second round on freshly mined negatives usually gains again, and")
    print("reporting both rounds is a stronger result than reporting one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
