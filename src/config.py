"""Load and validate a per-dataset experiment config.

This is the ONLY module that knows about model prefixes.

That is not a style preference. BGE expects a specific string prepended to
queries and nothing prepended to documents. Forget it in one place out of four
and nothing crashes - the model just scores a few points lower, everywhere,
silently. So no other file is allowed to touch `query_prefix` directly. They
call `cfg.for_query()` and `cfg.for_document()` instead, and there is exactly
one line in the codebase where a prefix is attached to text.

Usage:
    from config import load_config
    cfg = load_config("configs/scifact.yaml")

Inspect a config from the shell:
    .venv\\Scripts\\python src/config.py configs/scifact.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing a key or holds a bad value."""


@dataclass(frozen=True)
class MiningConfig:
    """Step 3 settings: where in the ranking to pull hard negatives from."""

    range_min: int      # skip everything ranked above this
    range_max: int      # and everything below this
    margin: float       # a negative must score this much below the positive

    def validate(self) -> None:
        if not 0 <= self.range_min < self.range_max:
            raise ConfigError(
                "mining range must satisfy 0 <= range_min < range_max, got "
                f"{self.range_min}..{self.range_max}"
            )
        if self.range_min == 0:
            raise ConfigError(
                "mining.range_min of 0 scoops up the top-ranked documents as "
                "negatives. Those are where unlabelled correct answers live."
            )
        if not 0.0 <= self.margin < 1.0:
            raise ConfigError(f"mining.margin must be in [0, 1), got {self.margin}")


@dataclass(frozen=True)
class TrainingConfig:
    """Step 4 settings: the contrastive fine-tune."""

    lr: float
    batch_size: int     # in MultipleNegativesRankingLoss this IS the negative count
    epochs: int
    warmup_ratio: float

    def validate(self) -> None:
        # PyYAML follows YAML 1.1, whose float pattern requires a decimal point
        # before the exponent. So `2e-5` parses as the *string* "2e-5" while
        # `2.0e-5` parses as a float. Written the first way, your learning rate
        # is a string and the crash lands a long way from the cause.
        if not isinstance(self.lr, float):
            raise ConfigError(
                f"training.lr parsed as {type(self.lr).__name__} ({self.lr!r}), "
                "not float. In YAML write it as 2.0e-5, never 2e-5."
            )
        if not 0.0 < self.lr < 1.0:
            raise ConfigError(f"training.lr looks wrong: {self.lr}")
        if self.batch_size < 1:
            raise ConfigError(f"training.batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ConfigError(f"training.epochs must be >= 1, got {self.epochs}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ConfigError(
                f"training.warmup_ratio must be in [0, 1), got {self.warmup_ratio}"
            )


@dataclass(frozen=True)
class Config:
    """One experiment, fully described.

    `dataset` is what you EVALUATE on. What a model was trained on is recorded
    separately, in the `trained_on` column of results/metrics.csv. Keeping the
    two apart is what makes the FiQA rows unambiguous.
    """

    dataset: str
    hf_dataset: str
    hf_qrels_dataset: str
    graded_relevance: bool
    trainable: bool

    base_model: str
    query_prefix: str
    doc_prefix: str
    max_seq_length: int

    seed: int
    dev_size: int

    negatives_per_query: int | None = None
    mining: MiningConfig | None = None
    training: TrainingConfig | None = None

    source_path: Path | None = None   # provenance, for the run manifest

    # -- the prefix boundary -------------------------------------------------
    # Every piece of text that reaches an encoder passes through one of these.

    def for_query(self, text: str) -> str:
        """Prepend the model's query prefix. For anything you search WITH."""
        return f"{self.query_prefix}{text}"

    def for_document(self, text: str) -> str:
        """Prepend the model's document prefix. For anything you search OVER."""
        return f"{self.doc_prefix}{text}"

    def for_queries(self, texts: Iterable[str]) -> list[str]:
        return [self.for_query(t) for t in texts]

    def for_documents(self, texts: Iterable[str]) -> list[str]:
        return [self.for_document(t) for t in texts]

    # -- housekeeping --------------------------------------------------------

    def require_training(self) -> TrainingConfig:
        """Fetch the training block, refusing eval-only datasets.

        FiQA exists to measure what fine-tuning COST you elsewhere. Training on
        it destroys that measurement, so train.py calls this rather than
        reading .training directly.
        """
        if not self.trainable:
            raise ConfigError(
                f"{self.dataset} is marked eval-only and must never be trained on."
            )
        if self.training is None:
            raise ConfigError(f"{self.dataset} config has no training: block.")
        return self.training

    def require_mining(self) -> MiningConfig:
        if self.mining is None:
            raise ConfigError(f"{self.dataset} config has no mining: block.")
        return self.mining

    def validate(self) -> None:
        if self.max_seq_length < 1:
            raise ConfigError(f"max_seq_length must be >= 1, got {self.max_seq_length}")
        if self.dev_size < 0:
            raise ConfigError(f"dev_size must be >= 0, got {self.dev_size}")
        if self.trainable:
            if self.mining is None or self.training is None:
                raise ConfigError(
                    f"{self.dataset} is trainable but is missing a mining: or "
                    "training: block."
                )
            if not self.negatives_per_query or self.negatives_per_query < 1:
                raise ConfigError(
                    "negatives_per_query must be >= 1 for a trainable dataset"
                )
        if self.mining is not None:
            self.mining.validate()
        if self.training is not None:
            self.training.validate()

    def summary(self) -> str:
        lines = [
            f"dataset          {self.dataset}  (evaluate on this)",
            f"hf_dataset       {self.hf_dataset}",
            f"hf_qrels         {self.hf_qrels_dataset}",
            f"relevance        {'graded' if self.graded_relevance else 'binary'}",
            f"trainable        {self.trainable}",
            f"base_model       {self.base_model}",
            f"query_prefix     {self.query_prefix!r}",
            f"doc_prefix       {self.doc_prefix!r}",
            f"max_seq_length   {self.max_seq_length}",
            f"seed             {self.seed}",
            f"dev_size         {self.dev_size}",
        ]
        if self.training is not None:
            t = self.training
            lines.append(
                f"training         lr={t.lr} batch={t.batch_size} "
                f"epochs={t.epochs} warmup={t.warmup_ratio}"
            )
        if self.mining is not None:
            m = self.mining
            lines.append(
                f"mining           ranks {m.range_min}-{m.range_max} "
                f"margin={m.margin} n={self.negatives_per_query}"
            )
        return "\n".join(lines)


# -- loading -----------------------------------------------------------------

_REQUIRED_TOP_LEVEL = (
    "dataset",
    "hf_dataset",
    "hf_qrels_dataset",
    "graded_relevance",
    "trainable",
    "base_model",
    "query_prefix",
    "doc_prefix",
    "max_seq_length",
    "seed",
)

_KNOWN_TOP_LEVEL = set(_REQUIRED_TOP_LEVEL) | {
    "dev_size",
    "negatives_per_query",
    "mining",
    "training",
}


def load_config(path: str | Path) -> Config:
    """Read a YAML config, validate it, and return it.

    Validation happens here, at load time, on purpose. Every script's first act
    is to load its config, so a typo surfaces in the first second rather than
    after a forty-minute embedding run.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"no config file at {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} did not parse to a mapping (got {type(raw).__name__})"
        )

    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise ConfigError(f"{path} is missing required key(s): {', '.join(missing)}")

    # An unrecognised key would otherwise be silently ignored, and you would
    # spend an afternoon wondering why your learning rate had no effect.
    unknown = sorted(set(raw) - _KNOWN_TOP_LEVEL)
    if unknown:
        raise ConfigError(f"{path} has unrecognised key(s): {', '.join(unknown)}")

    mining = None
    if "mining" in raw:
        try:
            mining = MiningConfig(**raw["mining"])
        except TypeError as exc:
            raise ConfigError(f"{path} has a bad mining: block - {exc}") from exc

    training = None
    if "training" in raw:
        try:
            training = TrainingConfig(**raw["training"])
        except TypeError as exc:
            raise ConfigError(f"{path} has a bad training: block - {exc}") from exc

    cfg = Config(
        dataset=raw["dataset"],
        hf_dataset=raw["hf_dataset"],
        hf_qrels_dataset=raw["hf_qrels_dataset"],
        graded_relevance=bool(raw["graded_relevance"]),
        trainable=bool(raw["trainable"]),
        base_model=raw["base_model"],
        query_prefix=raw["query_prefix"],
        doc_prefix=raw["doc_prefix"],
        max_seq_length=int(raw["max_seq_length"]),
        seed=int(raw["seed"]),
        dev_size=int(raw.get("dev_size", 0)),
        negatives_per_query=raw.get("negatives_per_query"),
        mining=mining,
        training=training,
        source_path=path,
    )
    cfg.validate()
    return cfg


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python src/config.py configs/scifact.yaml", file=sys.stderr)
        raise SystemExit(2)
    print(load_config(sys.argv[1]).summary())
