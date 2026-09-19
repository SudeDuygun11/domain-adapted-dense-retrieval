"""Re-score every committed run file and check it against results/metrics.csv.

This is the fast reproduction path. It needs no GPU, no model checkpoints and no
retraining: every retrieval system in this project writes a TREC run file, and
evaluate.py turns a run file plus the frozen qrels into metrics. So the whole
results table can be rebuilt from data/runs/ alone.

What it proves: every number in metrics.csv follows from a run file that is in
the repository, scored by the same code against the same frozen split (whose
files are checked against the SHA-256 hashes in each manifest).

What it does NOT prove: that retraining a model reproduces the run file. That is
the slow path in the README, and GPU training is not bit-for-bit repeatable.

Usage:
    python scripts/verify_results.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import evaluate_run  # noqa: E402
from ingest import load_frozen_split  # noqa: E402
from runfile import read_run  # noqa: E402

METRICS = Path("results/metrics.csv")
TOLERANCE = 5e-5      # metrics.csv stores four decimals, so half a unit in the last place
COLUMNS = ["ndcg@10", "recall@50", "recall@100", "mrr@10"]


def split_of(notes: str) -> str:
    """Rows measured on dev carry a [SPLIT=DEV] stamp; everything else is test."""
    return "dev" if notes.startswith("[SPLIT=DEV]") else "test"


def main() -> int:
    with METRICS.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    checked = failed = skipped = 0
    cache: dict[tuple[str, str], dict] = {}

    for i, row in enumerate(rows, start=2):        # line 1 of the file is the header
        run_path = Path(row["run_file"])
        if not run_path.exists():
            print(f"  SKIP  line {i}: {run_path} is not in the repository")
            skipped += 1
            continue

        key = (row["dataset"], split_of(row["notes"]))
        if key not in cache:
            _, qrels, _ = load_frozen_split(*key)   # verifies the manifest hashes
            cache[key] = qrels

        got = evaluate_run(read_run(run_path), cache[key])
        bad = [
            f"{c}: recorded {float(row[c]):.4f}, recomputed {got[c]:.4f}"
            for c in COLUMNS
            if abs(float(row[c]) - got[c]) > TOLERANCE
        ]
        checked += 1
        if bad:
            failed += 1
            print(f"  FAIL  line {i}: {run_path.name}")
            for b in bad:
                print(f"          {b}")

    print()
    print(f"{checked} rows re-scored, {failed} mismatched, {skipped} skipped "
          f"(of {len(rows)} in {METRICS})")
    if failed:
        return 1
    print("Every recorded metric is reproduced from the committed run files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
