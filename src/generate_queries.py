"""Step 2 - manufacture training queries from documents you have no labels for.

The premise of stage two: you have a corpus and no training questions. At a real
company that is the normal situation. So you invent the questions, by showing a
generator a passage and asking what someone might have typed to find it.

This produces the data for ARM B - training as if the labels did not exist.
Arm A uses the real human-labelled pairs BEIR ships, and needs none of this file.
The gap between the two arms is the headline finding, because it answers the
question anyone in this situation actually has: how much does not having labels
cost me?

Run:
    .venv\\Scripts\\python src/generate_queries.py --config configs/nfcorpus.yaml

Writes data/synthetic_queries/<dataset>.jsonl (gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from config import load_config
from ingest import doc_text, load_raw_corpus

OUT_DIR = Path("data/synthetic_queries")
EMBED_DIR = Path("data/embeddings")

# BeIR's own generator, a T5 trained on MS MARCO to do exactly this job.
# Reliable and fast. The alternative is a small instruction model prompted for
# different query *styles* from one passage - one keyword query, one full
# question, one vague underspecified one - which matches how real people type
# and gives you something to discuss. Start here; that is an upgrade, not a
# prerequisite.
DEFAULT_GENERATOR = "BeIR/query-gen-msmarco-t5-base-v1"

# The instruction-model alternative. T5 takes no instructions: you hand it a
# passage and it emits a query in the only style it knows, which is MS MARCO web
# search. There is no dial.
#
# That turned out to matter. On NFCorpus the T5 arm gained +0.08 nDCG against
# real labels' +2.83, and its training loss collapsed to 0.73 - the documented
# symptom of a generator copying passage vocabulary. Inspecting the output
# confirmed it: for an abstract titled "Statin Use and Breast Cancer Survival: A
# Nationwide Cohort Study from Finland" it produced "statins in finland death
# rate", which never mentions breast cancer and reuses the title's own words.
#
# NFCorpus is built on the distance between how patients ask and how papers
# answer. A query written in the paper's vocabulary sits on the wrong side of
# exactly the gap the dataset exists to measure.
DEFAULT_INSTRUCT_GENERATOR = "Qwen/Qwen2.5-1.5B-Instruct"

# One style per generated query, cycled in order. The point is variety of KIND,
# not variety of wording - three rephrasings of one question teach little more
# than one question does.
# Ordered best-first, because --n-per-doc takes the first N. The two question
# styles are the reason to use an instruction model at all; the keyword style is
# what T5 already does well, and in smoke tests it was also the one that kept
# degenerating into comma-separated topic lists.
STYLE_PROMPTS = [
    "a complete question in everyday language, asked by someone with no training "
    "in this field. Use no technical terms from the passage",
    "a vague, half-formed question that only hints at what the person wants, the "
    "way someone asks when they do not yet know the right words",
    # Tightened after a smoke test: the first version said "three to six words"
    # and Qwen answered with a 14-word keyword dump lifted straight from the
    # passage - the exact failure this generator was meant to avoid.
    "a keyword search of AT MOST 6 WORDS TOTAL. Count the words. Do not list "
    "topics; write what a person would type",
]

_SYSTEM = (
    "You write search queries. Given a passage, you write the query a real "
    "person would have typed to find it. Always answer in English. Reply with "
    "the query alone: no preamble, no quotation marks, no explanation."
)

# A generated query has to survive this before the round-trip filter even sees
# it. All three rules come from reading the smoke-test output: Qwen emitted
# Chinese mid-sentence, and its keyword style ran to 14 words of passage
# vocabulary.
MAX_QUERY_WORDS = 30


def is_usable(query: str) -> bool:
    if not query:
        return False
    # Count on commas as well as spaces. The first version split on whitespace
    # only, so Qwen's "pesticides,cancer,risk,environment,population,cases" - a
    # 13-item topic list with no spaces - counted as a single word and sailed
    # through. A topic list is not a query anyone typed.
    words = [w for w in re.split(r"[\s,;]+", query) if w]
    if len(words) > MAX_QUERY_WORDS or query.count(",") >= 3:
        return False
    # Reject any non-Latin script. A multilingual model will occasionally switch
    # languages mid-answer, and a Chinese query against an English corpus is
    # pure noise in the training set.
    return all(ord(ch) < 0x2E80 for ch in query)


def generate_for_batch(
    tokenizer,
    generator,
    passages: list[str],
    n_per_doc: int,
    max_length: int,
    device: str,
) -> list[list[str]]:
    """Generate n_per_doc queries for each passage.

    DESIGN CHOICE: sampling, not beam search.

    Beam search returns the n most probable sequences, which for one passage are
    five near-identical rephrasings of the same question. That teaches the model
    nothing beyond what one query would. Sampling with top_p=0.95 and
    temperature=1.0 gives genuinely different questions about different aspects
    of the passage, which is the point of generating several.
    """
    enc = tokenizer(
        passages, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = generator.generate(
            **enc,
            max_length=64,
            do_sample=True,          # NOT beam search - see docstring
            top_p=0.95,
            temperature=1.0,
            num_return_sequences=n_per_doc,
        )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    # generate() returns n_per_doc consecutive rows per input passage.
    return [
        decoded[i * n_per_doc : (i + 1) * n_per_doc] for i in range(len(passages))
    ]


def _clean(text: str) -> str:
    """Take the first line and strip the decorations chat models add."""
    line = text.strip().split("\n")[0].strip()
    line = line.strip('"').strip("'").strip()
    # Models sometimes answer "Query: ..." despite being told not to.
    for prefix in ("query:", "search query:", "question:"):
        if line.lower().startswith(prefix):
            line = line[len(prefix):].strip()
    return line


def generate_for_batch_instruct(
    tokenizer,
    generator,
    passages: list[str],
    n_per_doc: int,
    max_length: int,
    device: str,
) -> list[list[str]]:
    """Generate n_per_doc queries per passage, one per style, with a chat model.

    Unlike the T5 path this asks for a different KIND of query each time, which
    is the whole reason to use an instruction model. Sampling is still on, so two
    passages given the same style still get different questions.
    """
    styles = [STYLE_PROMPTS[i % len(STYLE_PROMPTS)] for i in range(n_per_doc)]

    prompts: list[str] = []
    for passage in passages:
        # Truncate in characters here only to keep the prompt bounded; the real
        # token-level limit is applied by the tokenizer below.
        body = passage[: max_length * 6]
        for style in styles:
            messages = [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": f"Passage:\n{body}\n\nWrite {style}.",
                },
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

    # Causal models must be left-padded for batched generation, or the shorter
    # sequences start decoding from pad tokens and produce garbage.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        prompts, padding=True, truncation=True, max_length=1024,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = generator.generate(
            **enc,
            max_new_tokens=32,   # queries are short; fewer decode steps = faster
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # generate() returns prompt + completion, so slice the prompt off.
    completions = tokenizer.batch_decode(
        out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
    )
    cleaned = [_clean(c) for c in completions]
    return [
        cleaned[i * n_per_doc : (i + 1) * n_per_doc] for i in range(len(passages))
    ]


def filter_by_roundtrip(
    cfg,
    encoder: SentenceTransformer,
    index: faiss.Index,
    doc_ids: list[str],
    pairs: list[tuple[str, str]],
    top_k: int,
    batch_size: int,
) -> tuple[list[tuple[str, str]], dict]:
    """Keep a generated query only if it retrieves its own source document.

    DESIGN CHOICE: this filter is not optional.

    A generator hallucinates. It writes questions the passage does not answer,
    questions about a detail that appears in a thousand other passages, and
    questions that are simply incoherent. Training on those teaches the model to
    associate a query with a document that does not answer it, which is worse
    than not training at all.

    The test is cheap and direct: embed the generated query with the BASE model,
    search the corpus, and ask whether the passage it came from comes back in the
    top k. If the base model cannot find the source, the pair is not learnable
    signal.

    This typically discards 15-30% of generations and is the cheapest quality
    improvement in the whole pipeline.

    Note it uses the base model deliberately. Filtering with a model you are
    about to train on this data would be circular.
    """
    queries = [cfg.for_query(q) for q, _ in pairs]
    vecs = encoder.encode(
        queries, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)

    _, positions = index.search(vecs, top_k)
    position_of = {d: i for i, d in enumerate(doc_ids)}

    kept = []
    for (query, source_id), row in zip(pairs, positions):
        if position_of[source_id] in row:
            kept.append((query, source_id))

    stats = {
        "generated": len(pairs),
        "kept": len(kept),
        "discarded": len(pairs) - len(kept),
        "discard_rate": round(1 - len(kept) / max(len(pairs), 1), 3),
        "roundtrip_top_k": top_k,
    }
    return kept, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic training queries")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--generator",
        default=DEFAULT_GENERATOR,
        help=f"seq2seq default {DEFAULT_GENERATOR}; for the instruction path try "
             f"{DEFAULT_INSTRUCT_GENERATOR}",
    )
    ap.add_argument(
        "--generator-kind",
        choices=["auto", "seq2seq", "instruct"],
        default="auto",
        help="auto reads the model config. seq2seq is T5-style: no prompt, one "
             "fixed query style. instruct prompts a chat model for a different "
             "KIND of query each time - keyword, plain-language question, vague "
             "half-formed question.",
    )
    ap.add_argument(
        "--out-name",
        default=None,
        help="basename for the output file, so two generators can coexist. "
             "Defaults to the dataset name.",
    )
    ap.add_argument("--n-per-doc", type=int, default=3, help="3-5 is the usual range")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = the whole corpus")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--doc-max-tokens", type=int, default=350)
    ap.add_argument(
        "--roundtrip-k",
        type=int,
        default=10,
        help="a generated query must retrieve its source document within this rank",
    )
    ap.add_argument("--index-tag", default="base", help="index used for filtering")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if not cfg.trainable:
        raise SystemExit(f"{cfg.dataset} is eval-only. Nothing to generate for.")

    # NOT `stem`: a later block reuses that name for the FAISS index path,
    # and the shadowing silently redirected the stats file into
    # data/synthetic_queries/data/embeddings/ and crashed after a 3-hour run.
    out_stem = args.out_name or cfg.dataset
    out_path = OUT_DIR / f"{out_stem}.jsonl"
    # Generation is the slowest step in the project. Generate once, save, never
    # regenerate - and refuse by default so a stray re-run cannot silently
    # replace the data every later result was trained on.
    if out_path.exists() and not args.force:
        raise SystemExit(
            f"{out_path} already exists. Generation is expensive and its output "
            f"is what stage two trains on. Pass --force to replace it, and "
            f"expect to retrain everything if you do."
        )

    # Pin sampling. The committed query files were made BEFORE this line was
    # added, so they cannot be regenerated identically - which is why they are
    # committed rather than rebuilt. Runs from here on are repeatable given the
    # same hardware and batch size.
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[gen] WARNING no GPU found. This step wants one - on CPU it will")
        print("[gen]         take hours. Use --max-docs to smoke-test the code,")
        print("[gen]         then run the real thing on Colab.")

    corpus = load_raw_corpus(cfg.dataset)
    doc_id_list = sorted(corpus)
    if args.max_docs:
        rng = random.Random(cfg.seed)
        doc_id_list = sorted(rng.sample(doc_id_list, min(args.max_docs, len(doc_id_list))))

    print(f"[gen] {len(doc_id_list)} documents x {args.n_per_doc} queries each")
    print(f"[gen] generator {args.generator} on {device}")

    kind = args.generator_kind
    if kind == "auto":
        arch = AutoConfig.from_pretrained(args.generator).architectures or [""]
        kind = "seq2seq" if "ConditionalGeneration" in arch[0] else "instruct"
    print(f"[gen] generator kind {kind}")

    tokenizer = AutoTokenizer.from_pretrained(args.generator)
    if kind == "seq2seq":
        generator = (
            AutoModelForSeq2SeqLM.from_pretrained(args.generator).to(device).eval()
        )
        generate_fn = generate_for_batch
    else:
        # fp16 on GPU halves the footprint. A 1.5B model lands near 3.1GB, which
        # fits a 4GB card; a 3B model will not.
        generator = (
            AutoModelForCausalLM.from_pretrained(
                args.generator,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            .to(device)
            .eval()
        )
        generate_fn = generate_for_batch_instruct
        print(f"[gen] {len(STYLE_PROMPTS)} query styles, cycled per document")

    # CHECKPOINTING. Generation is the longest step in the project - hours on a
    # small GPU - and the first attempt at this run was killed three hours in by
    # a session ending, producing nothing at all.
    #
    # So raw output is appended and flushed after every batch, one line per
    # DOCUMENT (not per query, so a document whose queries were all rejected
    # still counts as done). A re-run reads it back and skips what is finished.
    # A kill now costs one batch instead of the whole run.
    raw_path = OUT_DIR / f"{out_stem}.raw.jsonl"
    done_ids: set[str] = set()
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as fh:
            for line in fh:
                done_ids.add(json.loads(line)["doc_id"])
        print(f"[gen] resuming: {len(done_ids)} documents already generated")

    todo = [d for d in doc_id_list if d not in done_ids]
    t0 = time.perf_counter()

    with raw_path.open("a", encoding="utf-8", newline="\n") as raw_fh:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start : start + args.batch_size]
            passages = [doc_text(corpus[d]) for d in chunk]
            generated = generate_fn(
                tokenizer, generator, passages, args.n_per_doc,
                args.doc_max_tokens, device,
            )
            for did, queries in zip(chunk, generated):
                raw_fh.write(
                    json.dumps({"doc_id": did, "queries": queries}) + "\n"
                )
            raw_fh.flush()   # survive a kill
            done = start + len(chunk)
            if done % (args.batch_size * 20) == 0 or done == len(todo):
                rate = done / max(time.perf_counter() - t0, 1e-9)
                eta = (len(todo) - done) / max(rate, 1e-9) / 60
                print(f"[gen]   {done}/{len(todo)} docs  "
                      f"({rate:.2f} docs/s, ~{eta:.0f} min left)", flush=True)

    # Re-read everything, including whatever earlier runs contributed.
    pairs: list[tuple[str, str]] = []
    rejected = 0
    with raw_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            for q in row["queries"]:
                q = q.strip()
                if is_usable(q):
                    pairs.append((q, row["doc_id"]))
                elif q:
                    rejected += 1

    print(f"[gen] generated {len(pairs)} usable pairs "
          f"({rejected} rejected as too long or non-English)")

    # --- the filter ---------------------------------------------------------
    stem = EMBED_DIR / f"{cfg.dataset}_{args.index_tag}"
    if not stem.with_suffix(".faiss").exists():
        raise SystemExit(
            f"no index at {stem}.faiss. Filtering needs the BASE model's index, "
            f"which stage one built. Run build_index.py first."
        )
    index = faiss.read_index(str(stem.with_suffix(".faiss")))
    doc_ids = json.loads(stem.with_suffix(".ids.json").read_text(encoding="utf-8"))

    encoder = SentenceTransformer(cfg.base_model)
    encoder.max_seq_length = cfg.max_seq_length

    print("[gen] filtering by round-trip retrieval")
    kept, stats = filter_by_roundtrip(
        cfg, encoder, index, doc_ids, pairs, args.roundtrip_k, 64
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for query, did in kept:
            fh.write(json.dumps({"query": query, "doc_id": did}) + "\n")

    (OUT_DIR / f"{out_stem}.stats.json").write_text(
        json.dumps(
            {
                "dataset": cfg.dataset,
                "generator": args.generator,
                "generator_kind": kind,
                "rejected_malformed": rejected,
                "n_per_doc": args.n_per_doc,
                "n_docs": len(doc_id_list),
                "seed": cfg.seed,
                **stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[gen] kept {stats['kept']}/{stats['generated']} "
          f"({100 * (1 - stats['discard_rate']):.0f}%)")
    print(f"[gen] wrote {out_path}")
    print()
    print("NOW READ SOME OF THEM BY HAND. If the generator is copying passage")
    print("vocabulary verbatim, the training task collapses into lexical matching")
    print("and the model learns nothing about meaning. The symptom during")
    print("training is a loss that drops to near zero within a few hundred steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
