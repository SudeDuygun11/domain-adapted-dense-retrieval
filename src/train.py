"""Step 4 - contrastive fine-tune.

The signal is simple: pull the query toward its correct document, push it away
from the negatives. Repeat a few thousand times. The embedding space reshapes so
that lay phrasing lands near the clinical phrasing that answers it.

This attacks failure mode A - documents that never made the shortlist at all.
Fine-tuning MOVES documents in the space, which can pull things into the top 100
that were not there before. That is why Recall@100 can improve here, and why it
mathematically cannot improve from reranking.

Run:
    .venv\\Scripts\\python src/train.py --config configs/nfcorpus.yaml --arm synthetic

Writes models/<dataset>_<arm>/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)

from config import load_config
from ingest import doc_text, load_frozen_split, load_raw_corpus

NEG_DIR = Path("data/hard_negatives")
MODEL_DIR = Path("models")


def set_all_seeds(seed: int) -> None:
    """A gain under about one nDCG point is inside run-to-run variance.

    To claim a small gain you train three seeds and report mean and spread.
    That only means anything if a seed actually pins everything down.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg, corpus, path: Path, max_rows: int = 0) -> Dataset:
    """Turn mined negatives into the column layout the loss expects.

    MultipleNegativesRankingLoss reads columns positionally: the first is the
    anchor, the second is its positive, and every remaining column is a negative.
    Names do not matter, order does.

    THE PREFIXES ARE APPLIED HERE, through cfg.for_query and cfg.for_document.
    Train with a different prefix than you evaluate with and the model is
    optimised for inputs it will never see. Nothing would report that.
    """
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    if max_rows:
        rows = rows[:max_rows]

    # Use the configured negative count, not the minimum across rows. Taking the
    # minimum means one unlucky row with 2 negatives truncates every other row to
    # 2 as well - on the first NFCorpus run that silently threw away two thirds
    # of the mined negatives, because mean_negatives was 5.99 and one row had 2.
    n_neg = cfg.negatives_per_query
    before = len(rows)
    rows = [r for r in rows if len(r["negatives"]) >= n_neg]
    if len(rows) < before:
        print(f"[train] dropped {before - len(rows)} rows with fewer than "
              f"{n_neg} negatives")
    data: dict[str, list[str]] = {"anchor": [], "positive": []}
    for i in range(n_neg):
        data[f"negative_{i + 1}"] = []

    for r in rows:
        data["anchor"].append(cfg.for_query(r["query"]))
        data["positive"].append(cfg.for_document(doc_text(corpus[r["positive"]])))
        for i in range(n_neg):
            data[f"negative_{i + 1}"].append(
                cfg.for_document(doc_text(corpus[r["negatives"][i]]))
            )

    print(f"[train] {len(rows)} training rows, {n_neg} negatives each")
    return Dataset.from_dict(data)


def build_dev_evaluator(cfg, corpus) -> InformationRetrievalEvaluator | None:
    """Score dev nDCG@10 during training, so you can stop at the best point.

    Dev, never test. If you pick the checkpoint by watching the test score, you
    selected on the number you are about to report, and it is no longer an
    estimate of unseen performance.
    """
    try:
        queries, qrels, _ = load_frozen_split(cfg.dataset, "dev")
    except (SystemExit, FileNotFoundError):
        print("[train] no dev split found - training without early stopping")
        return None

    return InformationRetrievalEvaluator(
        queries={q: cfg.for_query(t) for q, t in queries.items()},
        corpus={d: cfg.for_document(doc_text(v)) for d, v in corpus.items()},
        relevant_docs={q: set(docs) for q, docs in qrels.items()},
        # Every one of these must be a non-empty list. The evaluator calls
        # max() on each internally and raises on an empty sequence, a long way
        # from where you set it.
        ndcg_at_k=[10],
        accuracy_at_k=[10],
        precision_recall_at_k=[10, 100],
        map_at_k=[100],
        mrr_at_k=[10],
        show_progress_bar=False,
        name="dev",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Contrastive fine-tune a bi-encoder")
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, choices=["synthetic", "real_labels"])
    ap.add_argument("--negatives-tag", default="base", help="which mined file to use")
    ap.add_argument("--seed", type=int, default=None, help="overrides the config seed")
    ap.add_argument(
        "--cached-loss",
        action="store_true",
        help="use GradCache. Lets you set a large batch size on a small GPU - "
             "see the note in the code about why batch size is THE hyperparameter.",
    )
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all. For smoke tests.")
    ap.add_argument(
        "--no-dev-eval",
        action="store_true",
        help="skip the dev evaluator. ONLY for smoke-testing the code path - it "
             "re-encodes the whole corpus at every eval, which is the slow part. "
             "Never use it for a run whose number you intend to report, because "
             "without it there is no early stopping and no checkpoint selection.",
    )
    ap.add_argument("--tag", default=None, help="output directory suffix")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    # Refuses eval-only datasets by name. FiQA exists to measure what fine-tuning
    # COST you elsewhere; training on it destroys that measurement.
    train_cfg = cfg.require_training()

    seed = args.seed if args.seed is not None else cfg.seed
    set_all_seeds(seed)

    batch_size = args.batch_size or train_cfg.batch_size
    epochs = args.epochs or train_cfg.epochs
    lr = args.lr or train_cfg.lr

    neg_path = NEG_DIR / f"{cfg.dataset}_{args.arm}_{args.negatives_tag}.jsonl"
    if not neg_path.exists():
        raise SystemExit(f"{neg_path} missing. Run mine_negatives.py --arm {args.arm}")

    tag = args.tag or f"{args.arm}_s{seed}"
    out_dir = MODEL_DIR / f"{cfg.dataset}_{tag}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] dataset {cfg.dataset}, arm {args.arm}, device {device}")
    if device == "cpu":
        print("[train] WARNING no GPU. Use --max-rows to check the code path,")
        print("[train]         then run the real training on Colab.")

    corpus = load_raw_corpus(cfg.dataset)
    train_ds = build_dataset(cfg, corpus, neg_path, args.max_rows)

    model = SentenceTransformer(cfg.base_model)
    model.max_seq_length = cfg.max_seq_length

    # THE BATCH SIZE THING, which is not intuitive.
    #
    # In MultipleNegativesRankingLoss every other item in the batch acts as an
    # additional negative. So batch size IS your negative count, which makes it
    # the single most important hyperparameter - and it is exactly what a 16GB
    # GPU constrains.
    #
    # CachedMultipleNegativesRankingLoss (GradCache) computes embeddings in
    # mini-batches, caches them, and reconstructs the gradient of a large batch.
    # An effective batch of 256 with the memory footprint of 16. Running 64 vs
    # 256 and reporting the nDCG difference is a genuinely good result for one
    # extra run.
    loss = (
        CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
        if args.cached_loss
        else MultipleNegativesRankingLoss(model)
    )

    evaluator = None if args.no_dev_eval else build_dev_evaluator(cfg, corpus)
    if args.no_dev_eval:
        print("[train] dev evaluation DISABLED - smoke test only, do not report this")

    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        # A float here means a RATIO of total steps, not a step count. That is
        # the transformers v5 spelling; v4 called the same thing warmup_ratio.
        warmup_steps=float(train_cfg.warmup_ratio),
        fp16=(device == "cuda"),      # halves memory on GPU; meaningless on CPU
        seed=seed,
        eval_strategy="steps" if evaluator else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps" if evaluator else "no",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=bool(evaluator),
        # The evaluator emits "dev_cosine_ndcg@10" and the trainer prefixes
        # "eval_". Get this name wrong and load_best_model_at_end silently keeps
        # the wrong checkpoint, or raises deep inside the trainer.
        metric_for_best_model="eval_dev_cosine_ndcg@10" if evaluator else None,
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    (out_dir / "training_record.json").write_text(
        json.dumps(
            {
                "dataset": cfg.dataset,
                "arm": args.arm,
                "base_model": cfg.base_model,
                "negatives_file": str(neg_path.as_posix()),
                "seed": seed,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "warmup_ratio": train_cfg.warmup_ratio,
                "cached_loss": args.cached_loss,
                "n_train_rows": len(train_ds),
                "query_prefix": cfg.query_prefix,
                "doc_prefix": cfg.doc_prefix,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[train] saved {out_dir}")
    print()
    print("Next: re-index and re-evaluate with the tuned model.")
    print(f"  python src/build_index.py --config {args.config} "
          f"--model {out_dir.as_posix()} --tag {tag}")
    print(f"  python src/retrieve.py --config {args.config} --tag {tag}")
    print(f"  python src/evaluate.py --config {args.config} "
          f"--run data/runs/{cfg.dataset}_{tag}_test.trec "
          f"--model-name {out_dir.as_posix()} --trained-on {cfg.dataset} "
          f"--training-arm {args.arm} --index-type flat --query-prefix-used yes")
    print()
    print("Then the forgetting check, which is the step people skip:")
    print(f"  bash scripts/run_forgetting.sh {out_dir.as_posix()} {cfg.dataset} {args.arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
