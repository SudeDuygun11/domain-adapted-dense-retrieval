"""The prefix trap has no error message, so it needs a test.

BGE scores a few nDCG points lower with the query prefix missing and raises
nothing. That failure mode is invisible at runtime, which means the only place
it can be caught is here.
"""

import pytest

from config import ConfigError, load_config

SCIFACT = "configs/scifact.yaml"
FIQA = "configs/fiqa.yaml"


def test_query_prefix_is_applied():
    cfg = load_config(SCIFACT)
    out = cfg.for_query("does eating eggs raise cholesterol")
    assert out.startswith("Represent this sentence for searching relevant passages: ")
    assert out.endswith("does eating eggs raise cholesterol")


def test_document_prefix_is_empty_for_bge():
    cfg = load_config(SCIFACT)
    assert cfg.for_document("dietary cholesterol") == "dietary cholesterol"


def test_every_config_uses_the_same_prefix():
    """All three datasets use the same base model, so the prefix must match.

    A prefix that differs between datasets would make the FiQA forgetting delta
    meaningless: part of the drop would be the prefix change rather than the
    fine-tuning.
    """
    prefixes = {
        p: load_config(p).query_prefix
        for p in (SCIFACT, "configs/nfcorpus.yaml", FIQA)
    }
    assert len(set(prefixes.values())) == 1, prefixes


def test_learning_rate_is_a_float_not_a_string():
    """PyYAML follows YAML 1.1, where `2e-5` is a string and `2.0e-5` is a float."""
    cfg = load_config(SCIFACT)
    assert isinstance(cfg.training.lr, float)
    assert cfg.training.lr == pytest.approx(2e-5)


def test_fiqa_refuses_to_be_trained_on():
    """FiQA measures what fine-tuning cost you. Training on it destroys that."""
    cfg = load_config(FIQA)
    with pytest.raises(ConfigError, match="eval-only"):
        cfg.require_training()


def test_mining_range_rejects_the_top_of_the_ranking():
    """range_min of 0 would harvest unlabelled correct answers as negatives."""
    from config import MiningConfig

    with pytest.raises(ConfigError, match="range_min"):
        MiningConfig(range_min=0, range_max=50, margin=0.05).validate()
