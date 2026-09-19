"""Step 6 - score a run file against the frozen qrels.

This is the only file in the project that computes a metric, and it never
imports a model. It reads a run file and the frozen qrels, and it appends a row
to results/metrics.csv. That is all it does.

Keeping it model-free is what makes it trustworthy. If evaluation could
re-encode anything it could silently disagree with what retrieval actually did -
a different prefix, a different truncation length - and you would be scoring a
pipeline you never ran.

Run:
    .venv\\Scripts\\python src/evaluate.py --config configs/scifact.yaml \\
        --run data/runs/scifact_bm25.trec --model-name bm25 --training-arm baseline
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import pytrec_eval

from config import load_config
from ingest import Qrels, load_frozen_split
from runfile import Run, read_run, truncate

# METRICS_CSV lets a reproduction run write to a scratch file instead of
# appending to the committed results table (see README, "Reproduce").
METRICS_CSV = Path(os.environ.get("METRICS_CSV", "results/metrics.csv"))

# The header already committed in results/metrics.csv. Do not reorder it.
#
# `dataset` is what you EVALUATE on. `trained_on` is what the model was tuned on,
# and is empty for baselines. Two columns rather than one is what makes the FiQA
# forgetting rows readable: dataset=fiqa, trained_on=nfcorpus,
# training_arm=synthetic is unambiguous, where a single column could not tell you
# whether that model had been tuned on SciFact or on NFCorpus.
FIELDS = [
    "dataset",
    "model_name",
    "trained_on",
    "training_arm",
    "index_type",
    "ndcg@10",
    "recall@50",
    "recall@100",
    "mrr@10",
    "query_prefix_used",
    "seed",
    "run_file",
    "notes",
]

TRAINING_ARMS = ("baseline", "synthetic", "real_labels")

# pytrec_eval's own measure spellings. Written out rather than inlined because
# the exact name matters: `ndcg_cut_10` and `ndcg` are different numbers, and
# asking for the wrong one produces a perfectly plausible result.
MEASURES = {"ndcg_cut.10", "recall.50,100"}


def evaluate_run(run: Run, qrels: Qrels) -> dict[str, float]:
    """Score a run and return the four headline metrics, averaged over queries.

    DESIGN CHOICE: pytrec_eval rather than a hand-written nDCG.

    Several DCG variants exist, differing mainly in whether gain is used raw or
    as 2^gain - 1. They disagree on graded data like NFCorpus. pytrec_eval wraps
    the original trec_eval C implementation, which is the variant BEIR reports,
    so using it removes a whole category of bug. Write your own and you will
    eventually lose a day to a discrepancy that lives in your metric rather than
    in your pipeline.
    """
    extra = sorted(set(run) - set(qrels))
    if extra:
        raise SystemExit(
            f"The run scores {len(extra)} query/queries absent from the frozen "
            f"qrels, e.g. {extra[:5]}. Retrieval ran on a different query set "
            f"than evaluation. Fix retrieval, not this check."
        )

    unanswered = sorted(set(qrels) - set(run))
    if unanswered:
        # Not fatal. trec_eval scores a missing query as zero, which is correct:
        # returning nothing is a retrieval failure, not a reason to shrink the
        # denominator. But it must be visible, because a run covering half the
        # queries still produces a number that looks like a score.
        print(
            f"[eval] WARNING {len(unanswered)} qrel queries have no run entry "
            f"and will score 0, e.g. {unanswered[:5]}"
        )

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, MEASURES)
    per_query = evaluator.evaluate(run)

    # MRR@10 needs its own pass. trec_eval's recip_rank has no cutoff - it looks
    # all the way down the ranking - so asking for it on the full run gives MRR,
    # not MRR@10. Truncating the run to 10 and then taking recip_rank is exactly
    # the definition: the reciprocal rank of the first relevant document, or
    # zero if none appears in the top 10.
    mrr_eval = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    mrr_per_query = mrr_eval.evaluate(truncate(run, 10))

    n = len(qrels)  # average over EVERY judged query, including unanswered ones

    def mean(scores: dict, key: str) -> float:
        return sum(q.get(key, 0.0) for q in scores.values()) / n

    return {
        "ndcg@10": mean(per_query, "ndcg_cut_10"),
        "recall@50": mean(per_query, "recall_50"),
        "recall@100": mean(per_query, "recall_100"),
        "mrr@10": mean(mrr_per_query, "recip_rank"),
    }


def append_metrics_row(row: dict[str, object]) -> None:
    """Append one row to results/metrics.csv. Never overwrite.

    DESIGN CHOICE: append-only, one row per experiment.

    By the end of the project this file is the single source of truth from which
    every table and every plot in the write-up is generated. Overwriting rows
    destroys the comparison the project exists to make - you cannot report a
    base-to-tuned delta if the base row was replaced. Duplicated rows are a much
    cheaper problem than missing ones.
    """
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = METRICS_CSV.exists() and METRICS_CSV.stat().st_size > 0

    with METRICS_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score a run file against frozen qrels.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="path to a .trec run file")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--model-name", required=True, help="goes in the model_name column")
    ap.add_argument(
        "--training-arm",
        required=True,
        choices=TRAINING_ARMS,
        help="baseline | synthetic | real_labels",
    )
    ap.add_argument("--trained-on", default="", help="dataset the model was tuned on")
    ap.add_argument("--index-type", default="", help="flat | hnsw | ivfflat | ivfpq | bm25")
    ap.add_argument(
        "--query-prefix-used",
        choices=["yes", "no"],
        required=True,
        help="did retrieval apply the BGE query prefix? Recorded because "
             "forgetting it costs several points silently.",
    )
    ap.add_argument("--notes", default="")
    ap.add_argument(
        "--no-record",
        action="store_true",
        help="print the metrics without appending a row (use while debugging)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    # load_frozen_split verifies the files against the hashes in manifest.json,
    # so a split that has drifted since it was created raises here rather than
    # producing a number that quietly is not comparable to your earlier ones.
    queries, qrels, manifest = load_frozen_split(cfg.dataset, args.split)

    # The assertion the README asks for at the top of every eval run.
    expected_q = manifest["counts"][f"n_queries_{args.split}"]
    if len(qrels) != expected_q:
        raise SystemExit(
            f"expected {expected_q} {args.split} queries, loaded {len(qrels)}"
        )
    print(f"[eval] {cfg.dataset}/{args.split}: {len(qrels)} queries, "
          f"{sum(len(v) for v in qrels.values())} judgments")

    if args.split == "test":
        # Dev is for decisions. Test is looked at once. This does not stop you,
        # it just makes sure you noticed which one you are reading.
        print("[eval] NOTE scoring on TEST. Tune on dev, report test.")

    run = read_run(args.run)
    metrics = evaluate_run(run, qrels)

    print()
    for key in ("ndcg@10", "recall@50", "recall@100", "mrr@10"):
        print(f"  {key:<12} {metrics[key]:.4f}")
    print()

    if args.no_record:
        print("[eval] --no-record set, results/metrics.csv untouched")
        return 0

    append_metrics_row(
        {
            "dataset": cfg.dataset,
            "model_name": args.model_name,
            "trained_on": args.trained_on,
            "training_arm": args.training_arm,
            "index_type": args.index_type,
            **{k: f"{v:.4f}" for k, v in metrics.items()},
            "query_prefix_used": args.query_prefix_used,
            "seed": cfg.seed,
            "run_file": str(Path(args.run).as_posix()),
            # The committed schema has no split column, and a dev row next to a
            # test row is indistinguishable without one. Stamping it into notes
            # keeps the header unchanged while making the distinction impossible
            # to miss - which matters because dev rows are for decisions and must
            # never be quoted as results.
            "notes": (
                args.notes if args.split == "test"
                else f"[SPLIT=DEV] {args.notes}".strip()
            ),
        }
    )
    print(f"[eval] appended a row to {METRICS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
