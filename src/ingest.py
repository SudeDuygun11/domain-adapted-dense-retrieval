"""Step 1 - download a BEIR dataset and freeze the evaluation split.

The output of this file is the contract for the whole project. Every number you
report over the next month has to be comparable to every other number, and the
only way to guarantee that is to write down exactly which queries and which
relevance judgments you evaluate on, hash the files, commit them, and never
touch them again.

The failure this prevents is silent. Evaluating on 297 queries in week one and
300 in week three produces two numbers that look comparable, differ by half a
point, and are not comparable at all. Nothing crashes.

Run:
    .venv\\Scripts\\python src/ingest.py --config configs/scifact.yaml

Writes:
    data/raw/<dataset>/corpus.jsonl          gitignored, deterministic, hashed
    data/frozen_splits/<dataset>/            committed - this is the contract
        queries_test.jsonl
        qrels_test.tsv
        queries_dev.jsonl
        qrels_dev.tsv
        qrels_train.tsv                      dev queries removed
        manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from config import Config, load_config

RAW_DIR = Path("data/raw")
FROZEN_DIR = Path("data/frozen_splits")

# Types. Note that every id is a str, everywhere, always.
Corpus = dict[str, dict[str, str]]        # doc_id -> {"title", "text"}
Queries = dict[str, str]                  # query_id -> text
Qrels = dict[str, dict[str, int]]         # query_id -> {doc_id: relevance}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _sid(value) -> str:
    """Coerce an id to a string.

    Every id in this project is a string, and the coercion happens exactly here,
    at the boundary where data enters.

    The reason is concrete. In the BeIR/*-qrels repositories the `query-id` and
    `corpus-id` columns come through as int64, while `_id` in the corpus and
    queries datasets is a str. Join them naively and you match nothing: no
    exception, no warning, just an empty result and a score of zero. Some
    datasets also use ids with leading zeros, which int() destroys
    irreversibly.
    """
    return str(value)


def load_corpus(cfg: Config) -> Corpus:
    ds = load_dataset(cfg.hf_dataset, "corpus")["corpus"]
    return {
        _sid(row["_id"]): {
            "title": row.get("title") or "",
            "text": row.get("text") or "",
        }
        for row in ds
    }


def load_queries(cfg: Config) -> Queries:
    """Load ALL queries the dataset ships, across every split.

    This is deliberately everything. The evaluation set is defined by the qrels,
    not by this file, and mixing that up is one of the silent calibration
    failures stage one exists to rule out. The queries file for SciFact holds
    roughly a thousand entries while the test qrels cover about three hundred.
    """
    ds = load_dataset(cfg.hf_dataset, "queries")["queries"]
    return {_sid(row["_id"]): row["text"] for row in ds}


def load_qrels(cfg: Config, split: str) -> Qrels:
    """Load one split of the relevance judgments.

    Only relevant pairs are listed. Any query/document pair absent from this
    file is judged irrelevant by omission.
    """
    ds = load_dataset(cfg.hf_qrels_dataset)[split]
    qrels: Qrels = {}
    for row in ds:
        qid = _sid(row["query-id"])
        did = _sid(row["corpus-id"])
        qrels.setdefault(qid, {})[did] = int(row["score"])
    return qrels


def available_qrel_splits(cfg: Config) -> list[str]:
    return list(load_dataset(cfg.hf_qrels_dataset).keys())


# BEIR datasets are inconsistent about what they call the held-out tuning split.
# SciFact ships none, NFCorpus calls it "validation", others call it "dev".
# Missing this silently falls back to sampling from train, which still works but
# throws away a split someone else already published.
_DEV_SPLIT_NAMES = ("dev", "validation", "valid")


def find_dev_split(splits: list[str]) -> str | None:
    for name in _DEV_SPLIT_NAMES:
        if name in splits:
            return name
    return None


# --------------------------------------------------------------------------
# the dev set
# --------------------------------------------------------------------------

def build_dev_split(
    cfg: Config, qrels_train: Qrels, official_dev: Qrels | None
) -> tuple[Qrels, Qrels, str]:
    """Return (dev_qrels, train_qrels_without_dev, provenance).

    DESIGN CHOICE, and the most important one in this file: the dev set is
    carved out of the TRAIN split, never out of test.

    The obvious alternative is to split the test set 80/20. Do not. Your
    stage-one milestone is reproducing the published MTEB number for
    bge-small-en-v1.5 on this dataset, and that number is measured on the full
    official test set. Slice the test set and there is nothing left to compare
    against, which removes the only external check you have that your
    evaluation code is correct.

    The dev set exists so that decisions and reported numbers come from
    different data. Tune the learning rate by watching the test score and then
    report that test score, and you tuned on the number you are reporting.
    Dev is for every decision. Test is looked at once, at the end.

    Where a dataset ships its own dev split, use it, because it is a choice
    someone else made and published, which is one fewer arbitrary decision of
    yours to defend.
    """
    if official_dev:
        return official_dev, qrels_train, "official dev split shipped with the dataset"

    if cfg.dev_size == 0 or not qrels_train:
        return {}, qrels_train, "none (dev_size is 0 or no train split exists)"

    # Sorted before sampling: dict iteration order follows insertion order,
    # which follows row order in the downloaded file. Sorting makes the sample
    # depend only on the seed, so a re-download cannot silently reshuffle it.
    train_qids = sorted(qrels_train)
    n = min(cfg.dev_size, len(train_qids))

    rng = random.Random(cfg.seed)
    dev_qids = set(rng.sample(train_qids, n))

    dev = {q: qrels_train[q] for q in sorted(dev_qids)}
    # Removed from train, not merely copied. A query that is in both is a leak:
    # you would be tuning on data the model was trained on.
    remaining = {q: v for q, v in qrels_train.items() if q not in dev_qids}
    return dev, remaining, f"sampled {n} queries from train with seed={cfg.seed}"


# --------------------------------------------------------------------------
# integrity checks
# --------------------------------------------------------------------------

@dataclass
class SplitReport:
    n_docs: int
    n_queries_test: int
    n_qrels_test: int
    n_queries_dev: int
    n_queries_train: int


def check_joins(corpus: Corpus, queries: Queries, qrels: Qrels, label: str) -> None:
    """Fail loudly if the qrels reference anything that does not exist.

    Run this BEFORE writing anything. If it fails you have an id-type or an
    encoding problem, and the moment to discover that is now, not in week three
    when a score comes out mysteriously low.
    """
    missing_q = sorted(q for q in qrels if q not in queries)
    if missing_q:
        raise SystemExit(
            f"[{label}] {len(missing_q)} qrel query id(s) are absent from the "
            f"queries file, e.g. {missing_q[:5]}. Almost always an id-type "
            f"mismatch: the qrels ship ints and the queries ship strings."
        )

    missing_d = sorted(
        {d for docs in qrels.values() for d in docs if d not in corpus}
    )
    if missing_d:
        raise SystemExit(
            f"[{label}] {len(missing_d)} qrel document id(s) are absent from "
            f"the corpus, e.g. {missing_d[:5]}."
        )

    empty = sorted(q for q, docs in qrels.items() if not docs)
    if empty:
        raise SystemExit(f"[{label}] {len(empty)} query/queries have no judgments.")


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_qrels(qrels: Qrels, path: Path) -> Path:
    """Write qrels as BEIR-style TSV with a header row.

    Sorted by query id then document id so the file is byte-identical on every
    run. That is what makes the hash in the manifest meaningful - an unsorted
    file would hash differently each time and tell you nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("query-id\tcorpus-id\tscore\n")
        for qid in sorted(qrels):
            for did in sorted(qrels[qid]):
                fh.write(f"{qid}\t{did}\t{qrels[qid][did]}\n")
    return path


def write_queries(queries: Queries, qids: list[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for qid in sorted(qids):
            fh.write(json.dumps({"_id": qid, "text": queries[qid]}) + "\n")
    return path


def write_corpus(corpus: Corpus, path: Path) -> Path:
    """Write the corpus to data/raw/, which is gitignored.

    DESIGN CHOICE: the corpus is not part of the frozen split, only its hash is.

    The frozen split is committed to git and its job is to record decisions you
    made. You made no decision about the corpus - it is whatever the dataset
    ships, and it is reproducible by anyone from the same repository. Storing a
    hash of it gives the same guarantee at a few dozen bytes instead of several
    megabytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for did in sorted(corpus):
            row = corpus[did]
            fh.write(
                json.dumps({"_id": did, "title": row["title"], "text": row["text"]})
                + "\n"
            )
    return path


def write_manifest(
    cfg: Config,
    out_dir: Path,
    files: dict[str, Path],
    report: SplitReport,
    dev_provenance: str,
) -> Path:
    """Record counts, hashes and provenance.

    The counts here become assertions at the top of every later run. The hashes
    are how you prove, months later, that the split never moved. If a hash
    changes and you did not intend it, every number measured before the change
    is incomparable to every number measured after.
    """
    manifest = {
        "dataset": cfg.dataset,
        "hf_dataset": cfg.hf_dataset,
        "hf_qrels_dataset": cfg.hf_qrels_dataset,
        "graded_relevance": cfg.graded_relevance,
        "config_source": str(cfg.source_path),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "dev_provenance": dev_provenance,
        "counts": {
            "n_docs": report.n_docs,
            "n_queries_test": report.n_queries_test,
            "n_qrels_test": report.n_qrels_test,
            "n_queries_dev": report.n_queries_dev,
            "n_queries_train": report.n_queries_train,
        },
        "sha256": {name: _sha256(p) for name, p in sorted(files.items())},
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# reading the frozen split back
# --------------------------------------------------------------------------

def load_frozen_split(dataset: str, split: str = "test") -> tuple[Queries, Qrels, dict]:
    """Read a frozen split back, verifying it against its own manifest.

    Every later script starts by calling this rather than touching the raw
    dataset. That is what makes the contract enforceable instead of merely
    documented: if the files on disk no longer match the hashes recorded when
    they were created, this raises rather than quietly returning different data.
    """
    out_dir = FROZEN_DIR / dataset
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    q_path = out_dir / f"queries_{split}.jsonl"
    r_path = out_dir / f"qrels_{split}.tsv"

    for name, path in ((q_path.name, q_path), (r_path.name, r_path)):
        expected = manifest["sha256"].get(name)
        if expected and _sha256(path) != expected:
            raise SystemExit(
                f"{path} does not match the hash recorded in manifest.json.\n"
                f"The frozen split has changed. Every number measured before "
                f"the change is incomparable to every number measured after. "
                f"Restore the file from git, or re-run ingest and re-measure "
                f"everything."
            )

    queries: Queries = {}
    with q_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            queries[row["_id"]] = row["text"]

    qrels: Qrels = {}
    with r_path.open(encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(score)

    return queries, qrels, manifest


def load_raw_corpus(dataset: str) -> Corpus:
    corpus: Corpus = {}
    path = RAW_DIR / dataset / "corpus.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run ingest.py first.")
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            corpus[row["_id"]] = {"title": row["title"], "text": row["text"]}
    return corpus


def doc_text(doc: dict[str, str]) -> str:
    """Flatten a document to the single string that gets encoded or indexed.

    DESIGN CHOICE: title and text joined with a space, title first.

    This is the BEIR convention, and it is the form the published baselines you
    are checking against were measured on. It also matters more than it looks:
    a SciFact abstract's title carries a lot of the topical signal, and dropping
    it costs real points. Defined once, here, so that BM25, the bi-encoder and
    any future reranker all see byte-identical document text. If they disagree,
    their scores are not comparable and nothing tells you.
    """
    title = doc["title"].strip()
    text = doc["text"].strip()
    return f"{title} {text}".strip() if title else text


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def restore_corpus(cfg: Config, out_dir: Path) -> int:
    """Rebuild data/raw/<dataset>/corpus.jsonl on a fresh clone, and prove it.

    The frozen split is committed to git, but the corpus is not: it is 4MB to
    50MB of data anyone can re-download, so the manifest stores its SHA-256
    instead. This is the other half of that decision. It downloads the corpus,
    writes it with the same deterministic serialisation ingest used originally,
    and compares the hash to the manifest.

    A match means every number in results/metrics.csv was measured against
    exactly this corpus. A mismatch means the upstream dataset changed and the
    results are not comparable - which is precisely what the hash is for.
    """
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} is missing. There is no frozen split to restore "
            f"against - run ingest.py without --restore-corpus first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["sha256"]["corpus.jsonl"]

    print(f"[restore] downloading corpus for {cfg.dataset}")
    corpus = load_corpus(cfg)

    # Write to a temporary file and only move it into place if the hash matches.
    # Writing straight to corpus.jsonl would let a mismatching download replace a
    # good local corpus before the check had a chance to fail.
    final = RAW_DIR / cfg.dataset / "corpus.jsonl"
    tmp = final.with_suffix(".jsonl.tmp")
    write_corpus(corpus, tmp)

    actual = _sha256(tmp)
    if len(corpus) != manifest["counts"]["n_docs"] or actual != expected:
        tmp.unlink()
        raise SystemExit(
            f"[restore] MISMATCH for {cfg.dataset}.\n"
            f"  documents  expected {manifest['counts']['n_docs']}, got {len(corpus)}\n"
            f"  sha256     expected {expected[:16]}..., got {actual[:16]}...\n"
            f"The upstream dataset has changed since the split was frozen, so "
            f"results in results/metrics.csv are not reproducible against it."
        )
    tmp.replace(final)
    print(f"[restore] {len(corpus)} documents, sha256 matches the manifest. OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="e.g. configs/scifact.yaml")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing frozen split (you almost never want this)",
    )
    ap.add_argument(
        "--restore-corpus",
        action="store_true",
        help="fresh-clone mode: re-download ONLY the corpus into data/raw/ and "
             "verify it against the hash recorded in the committed manifest. "
             "Never touches the frozen split.",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = FROZEN_DIR / cfg.dataset

    if args.restore_corpus:
        return restore_corpus(cfg, out_dir)

    # Refusing by default is the point of the word "frozen". Overwriting a split
    # invalidates every number already in results/metrics.csv, so it has to be
    # something you say out loud rather than something you do by re-running a
    # command.
    if (out_dir / "manifest.json").exists() and not args.force:
        raise SystemExit(
            f"{out_dir}/manifest.json already exists. The split is frozen.\n"
            f"Re-running would invalidate every row already in "
            f"results/metrics.csv. Pass --force only if you accept that."
        )

    print(f"[ingest] dataset          {cfg.dataset}")
    print(f"[ingest] downloading      {cfg.hf_dataset}")
    corpus = load_corpus(cfg)
    queries = load_queries(cfg)

    splits = available_qrel_splits(cfg)
    print(f"[ingest] qrel splits      {splits}")

    if "test" not in splits:
        raise SystemExit(f"{cfg.hf_qrels_dataset} has no test split (found {splits})")

    qrels_test = load_qrels(cfg, "test")
    qrels_train = load_qrels(cfg, "train") if "train" in splits else {}

    dev_split_name = find_dev_split(splits)
    official_dev = load_qrels(cfg, dev_split_name) if dev_split_name else None
    if dev_split_name:
        print(f"[ingest] official dev    '{dev_split_name}' split")

    # Check before writing. Always.
    check_joins(corpus, queries, qrels_test, "test")
    if qrels_train:
        check_joins(corpus, queries, qrels_train, "train")
    if official_dev:
        check_joins(corpus, queries, official_dev, "dev")

    qrels_dev, qrels_train, dev_provenance = build_dev_split(
        cfg, qrels_train, official_dev
    )

    overlap = set(qrels_dev) & set(qrels_train)
    if overlap:
        raise SystemExit(f"{len(overlap)} queries appear in both dev and train.")
    leak = set(qrels_dev) & set(qrels_test)
    if leak:
        raise SystemExit(f"{len(leak)} queries appear in both dev and test.")

    report = SplitReport(
        n_docs=len(corpus),
        n_queries_test=len(qrels_test),
        n_qrels_test=sum(len(v) for v in qrels_test.values()),
        n_queries_dev=len(qrels_dev),
        n_queries_train=len(qrels_train),
    )

    files = {
        "corpus.jsonl": write_corpus(corpus, RAW_DIR / cfg.dataset / "corpus.jsonl"),
        "queries_test.jsonl": write_queries(
            queries, list(qrels_test), out_dir / "queries_test.jsonl"
        ),
        "qrels_test.tsv": write_qrels(qrels_test, out_dir / "qrels_test.tsv"),
    }
    if qrels_dev:
        files["queries_dev.jsonl"] = write_queries(
            queries, list(qrels_dev), out_dir / "queries_dev.jsonl"
        )
        files["qrels_dev.tsv"] = write_qrels(qrels_dev, out_dir / "qrels_dev.tsv")
    if qrels_train:
        files["qrels_train.tsv"] = write_qrels(qrels_train, out_dir / "qrels_train.tsv")

    write_manifest(cfg, out_dir, files, report, dev_provenance)

    print(f"[ingest] documents        {report.n_docs}")
    print(f"[ingest] test queries     {report.n_queries_test}")
    print(f"[ingest] test judgments   {report.n_qrels_test}")
    print(f"[ingest] dev queries      {report.n_queries_dev}  ({dev_provenance})")
    print(f"[ingest] train queries    {report.n_queries_train}")
    print(f"[ingest] wrote            {out_dir}")
    print()
    print("Commit data/frozen_splits/. It is the contract.")
    return 0


if __name__ == "__main__":
    # `from config import ...` at the top of this file resolves because Python
    # puts a script's own directory on sys.path when you run it directly. That
    # is why every script here is invoked as `python src/<name>.py` rather than
    # as a module. tests/conftest.py does the same thing for pytest.
    raise SystemExit(main())
