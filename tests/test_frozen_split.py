"""The frozen split is the project's contract, so its guarantees get tested.

These run without network access. The dataset-dependent checks skip when
src/ingest.py has not been run yet.
"""

import json

import pytest

from evaluate import evaluate_run
from ingest import FROZEN_DIR, build_dev_split, check_joins, load_frozen_split

DATASET = "scifact"
pytestmark = pytest.mark.filterwarnings("ignore")


def _ingested() -> bool:
    return (FROZEN_DIR / DATASET / "manifest.json").exists()


needs_ingest = pytest.mark.skipif(
    not _ingested(), reason="run src/ingest.py --config configs/scifact.yaml first"
)


# --- checks that need no data ----------------------------------------------

class _Cfg:
    """Minimal stand-in for a Config, to test dev-set construction in isolation."""

    seed = 42
    dev_size = 3


def test_dev_is_removed_from_train_not_copied():
    """A query in both dev and train is a leak: you would tune on training data."""
    train = {f"q{i}": {"d": 1} for i in range(10)}
    dev, remaining, _ = build_dev_split(_Cfg(), train, official_dev=None)
    assert len(dev) == 3
    assert len(remaining) == 7
    assert not set(dev) & set(remaining)


def test_dev_sample_is_reproducible():
    train = {f"q{i}": {"d": 1} for i in range(10)}
    first, _, _ = build_dev_split(_Cfg(), train, official_dev=None)
    second, _, _ = build_dev_split(_Cfg(), dict(train), official_dev=None)
    assert list(first) == list(second)


def test_official_dev_split_wins_when_one_exists():
    train = {f"q{i}": {"d": 1} for i in range(10)}
    official = {"x1": {"d": 1}}
    dev, remaining, why = build_dev_split(_Cfg(), train, official_dev=official)
    assert dev == official
    assert remaining == train
    assert "official" in why


def test_join_check_catches_an_id_type_mismatch():
    """The int-vs-str qrels mismatch produces zero matches and no exception."""
    corpus = {"1": {"title": "", "text": "a"}}
    queries = {"7": "q"}
    qrels = {7: {1: 1}}  # ints, as the raw BeIR qrels ship them
    with pytest.raises(SystemExit, match="absent from the queries file"):
        check_joins(corpus, queries, qrels, "test")


def test_evaluate_rejects_a_run_over_the_wrong_query_set():
    qrels = {"q1": {"d1": 1}}
    run = {"q1": {"d1": 1.0}, "q_extra": {"d1": 1.0}}
    with pytest.raises(SystemExit, match="absent from the frozen qrels"):
        evaluate_run(run, qrels)


def test_ndcg_matches_the_worked_example():
    """Three relevant documents at ranks 1, 4 and 8 give nDCG@10 = 0.82.

    This is the example in section 4 of the README, and it is here so that a
    silent change in how the metric is computed cannot pass unnoticed.
    """
    qrels = {"q": {f"rel{i}": 1 for i in range(3)}}
    ranking = ["rel0", "x1", "x2", "rel1", "x3", "x4", "x5", "rel2", "x6", "x7"]
    run = {"q": {d: float(10 - i) for i, d in enumerate(ranking)}}
    assert evaluate_run(run, qrels)["ndcg@10"] == pytest.approx(0.82, abs=0.01)


# --- checks that need the split on disk ------------------------------------

@needs_ingest
def test_manifest_counts_match_the_files():
    queries, qrels, manifest = load_frozen_split(DATASET, "test")
    assert len(qrels) == manifest["counts"]["n_queries_test"]
    assert sum(len(v) for v in qrels.values()) == manifest["counts"]["n_qrels_test"]
    assert set(queries) == set(qrels)


@needs_ingest
def test_every_id_is_a_string():
    queries, qrels, _ = load_frozen_split(DATASET, "test")
    assert all(isinstance(q, str) for q in queries)
    assert all(isinstance(d, str) for docs in qrels.values() for d in docs)


@needs_ingest
def test_split_still_matches_its_recorded_hashes():
    """load_frozen_split raises if a file has drifted since it was frozen."""
    load_frozen_split(DATASET, "test")


@needs_ingest
def test_dev_and_test_do_not_overlap():
    path = FROZEN_DIR / DATASET
    if not (path / "qrels_dev.tsv").exists():
        pytest.skip("no dev split")
    _, dev_qrels, _ = load_frozen_split(DATASET, "dev")
    _, test_qrels, _ = load_frozen_split(DATASET, "test")
    assert not set(dev_qrels) & set(test_qrels)
