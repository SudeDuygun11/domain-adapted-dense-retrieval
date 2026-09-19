"""The run file is the one interface every retrieval system shares.

If BM25 and the dense model disagree about the format, evaluate.py silently
scores one of them wrong.
"""

import pytest

from runfile import read_run, truncate, write_run


def test_roundtrip(tmp_path):
    run = {"q1": {"d1": 3.5, "d2": 1.25}, "q2": {"d9": 0.5}}
    path = write_run(run, "unit", tmp_path / "unit.trec")
    back = read_run(path)
    assert set(back) == {"q1", "q2"}
    assert back["q1"]["d1"] == pytest.approx(3.5)


def test_six_columns_in_trec_order(tmp_path):
    path = write_run({"q1": {"d1": 2.0}}, "myrun", tmp_path / "r.trec")
    parts = path.read_text(encoding="utf-8").strip().split("\t")
    assert len(parts) == 6
    assert parts[0] == "q1"
    assert parts[1] == "Q0"      # vestigial, always literal Q0
    assert parts[2] == "d1"
    assert parts[3] == "1"       # rank is 1-indexed
    assert parts[5] == "myrun"


def test_rank_follows_score_descending(tmp_path):
    run = {"q1": {"low": 0.1, "high": 9.9, "mid": 5.0}}
    path = write_run(run, "r", tmp_path / "r.trec")
    order = [line.split("\t")[2] for line in path.read_text().strip().splitlines()]
    assert order == ["high", "mid", "low"]


def test_ties_break_deterministically(tmp_path):
    """Two runs from identical inputs must be byte-identical.

    Without a tiebreak, equal scores order arbitrarily and you can chase a
    phantom regression between two runs that are actually the same.
    """
    run = {"q1": {"b": 1.0, "a": 1.0, "c": 1.0}}
    first = write_run(run, "r", tmp_path / "a.trec").read_text()
    second = write_run(dict(run), "r", tmp_path / "b.trec").read_text()
    assert first == second
    assert [ln.split("\t")[2] for ln in first.strip().splitlines()] == ["a", "b", "c"]


def test_malformed_line_is_rejected(tmp_path):
    path = tmp_path / "bad.trec"
    path.write_text("q1\tQ0\td1\t1\n", encoding="utf-8")  # five columns
    with pytest.raises(ValueError, match="expected 6"):
        read_run(path)


def test_truncate_keeps_the_top_k():
    run = {"q1": {f"d{i}": float(100 - i) for i in range(50)}}
    assert list(truncate(run, 10)["q1"]) == [f"d{i}" for i in range(10)]
