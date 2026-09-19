"""Read and write TREC run files.

The run file is the one interface that matters in this project. Every retrieval
system - BM25, off-the-shelf BGE, a fine-tuned model, any index configuration,
a cross-encoder reranker in stage three - writes this same format:

    query_id  Q0  doc_id  rank  score  run_name

Consequences of committing to it:

  * Comparing four systems is four run files against one call to evaluate.py,
    not four code paths.
  * evaluate.py never imports a model. It cannot accidentally re-encode
    anything, so it cannot silently disagree with what retrieval actually did.
  * A reranker reads a run file and writes a run file. Stage three needs no
    change to stage two's code.
  * You can hand someone data/runs/ on its own and they can reproduce every
    number in your report without your model checkpoints.

The `Q0` column is a vestige of the original TREC format. It carries no
information and is always the literal string Q0. It is here because trec_eval
and pytrec_eval expect six columns.
"""

from __future__ import annotations

from pathlib import Path

# A run: {query_id: {doc_id: score}}. Ids are always strings - see ingest.py.
Run = dict[str, dict[str, float]]

RUNS_DIR = Path("data/runs")


def run_path(run_name: str) -> Path:
    """Canonical location for a named run."""
    return RUNS_DIR / f"{run_name}.trec"


def write_run(run: Run, run_name: str, path: Path | None = None) -> Path:
    """Write a run to TREC format, sorted by descending score.

    Rank is assigned here rather than taken from the caller. That way rank is
    always consistent with score, and a reranker that only produces new scores
    cannot leave a stale rank column behind.
    """
    path = path or run_path(run_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for qid in sorted(run):
            # Sort by score descending, then by doc_id, so that ties break
            # deterministically. Without the tiebreak, two runs from identical
            # inputs can differ, and you will chase a phantom regression.
            ranked = sorted(run[qid].items(), key=lambda kv: (-kv[1], kv[0]))
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                fh.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")
    return path


def read_run(path: str | Path) -> Run:
    """Read a TREC run file back into {query_id: {doc_id: score}}.

    Accepts whitespace-separated columns, since some tools emit spaces rather
    than tabs. The rank column is deliberately ignored: score is the source of
    truth and rank is derived from it on write.
    """
    path = Path(path)
    run: Run = {}
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(
                    f"{path}:{lineno} has {len(parts)} columns, expected 6 "
                    f"(query_id Q0 doc_id rank score run_name)"
                )
            qid, _q0, doc_id, _rank, score, _name = parts
            run.setdefault(qid, {})[doc_id] = float(score)
    if not run:
        raise ValueError(f"{path} contained no run lines")
    return run


def truncate(run: Run, k: int) -> Run:
    """Keep only the top k documents per query."""
    return {
        qid: dict(sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))[:k])
        for qid, docs in run.items()
    }
