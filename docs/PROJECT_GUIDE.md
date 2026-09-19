> **Note.** This is the original working guide the project was built from, kept unchanged apart from a findings section added at the end. The project report is [../README.md](../README.md).

# Domain-Adapted Dense Retrieval — Project Guide

A reference for the whole project: what you are building, why each dataset is
there, what each step does, and how to tell whether it worked.

---

## 1. The problem

A user types a question. You have thousands of documents. The right one must
end up at the top.

### The two machines

**Bi-encoder.** Converts each document into 384 numbers ahead of time — a point
in space. At query time it converts the query into a point and finds the nearest
document points. Milliseconds. Fast precisely *because* it summarised each
document blind, before knowing what would be asked of it. That is also its
limitation.

**Cross-encoder.** Reads the query and one document *together* and scores
relevance. Much more accurate, because it can attend to exactly which part of the
document answers this query. But nothing can be precomputed — scoring 5,000
documents means 5,000 forward passes. Minutes. Unusable as a search engine on its
own.

**BM25.** Clever word-counting. Looks for shared terms, weighting rare words more
heavily. Fast, needs no training. Blind spot: if the right document says
"dietary cholesterol and serum LDL" and the query says "eggs," there is no shared
word and BM25 finds nothing.

### The two failure modes

| | What happened | Can reranking fix it? |
|---|---|---|
| **A** | Right document sits at rank 800. Never made the shortlist. | No |
| **B** | Right document is at rank 7. In the pile, wrong position. | Yes |

Everything downstream follows from this split.

---

## 2. The three stages

The **seven steps** are pipeline components. The **three stages** are
deliverables — each a complete, defensible thing on its own. Stages reuse steps.

### Stage one — the baseline harness
*Steps 1, 6, part of 5. No training code is written.*

Freeze the split, wire up BM25 and off-the-shelf `bge-small-en-v1.5`, evaluate
both.

The point is the reproduce-the-published-number check. Your evaluation code is a
measuring instrument, and its failure modes are silent — wrong prefix, wrong
nDCG variant, wrong qrels subset. All of them produce output that looks
completely fine and scores a few points low. Nothing crashes. So you check your
number against a published one for the same model on the same data.

**Why single queries cannot catch this.** Run a query through a pipeline with the
BGE prefix missing and you get back ten plausible-looking documents. The damage
is subtler than "returns garbage" — the genuinely best document lands at rank 4
instead of rank 1. Every result looks reasonable. Multiply that small blur across
300 queries and it is four nDCG points.

**The sharp version of why this matters.** A calibration failure and a real
improvement look identical. If your baseline is silently four points low, and you
fix the prefix somewhere in stage two, you will report a four-point "gain from
fine-tuning" that is entirely an artefact of your own bug.

Boring, and the one stage you cannot skip.

### Stage two — the fine-tuned bi-encoder
*Steps 2, 3, 4, plus a re-run of 5, 6, 7. The actual portfolio piece.*

Generate synthetic queries, filter them, mine hard negatives, contrastively
fine-tune, re-index, re-evaluate. **Attacks failure mode A** — fine-tuning moves
documents in the embedding space, pulling things into the shortlist that were not
there before.

Inside this stage sits the two-arm experiment:

- **Arm A** — train on the real human-labelled pairs BEIR ships. The ceiling:
  what you would achieve with a perfect answer key.
- **Arm B** — train on synthetic queries only, pretending the labels do not
  exist. What you could actually do at a company with a document corpus and no
  labels.

The gap between them is the headline finding, because it answers the question
anyone in this situation actually has: *how much does not having labels cost me?*

### Stage three — the cross-encoder reranker
*Upside, not a requirement.*

A separate model, trained by distillation, that rescores the top 100 from stage
two. Reads a run file, writes a run file, never touches stage two's code.
**Attacks failure mode B only** — Recall@100 is mathematically unchanged by
reranking. Same 100 documents, different order.

---

## 3. The datasets

Three BEIR packages, three different jobs. All three ship training splits with
real query-document pairs, which is the binding constraint — without real labels
you cannot run Arm A.

| Dataset | Docs | Train on it? | Role |
|---|---|---|---|
| **SciFact** | ~5.2k | Yes | Debug here; validate the harness |
| **NFCorpus** | ~3.6k | Yes | Improve here; this is the headline |
| **FiQA** | ~58k | **Never** | Forgetting check; ANN sweep |

### SciFact — the practice court
Small enough that a full run takes under an hour, so you can make a mistake, fix
it, and rerun before lunch. Binary relevance judgments, well-known published
baselines to check against.

Not good for showing off an improvement: `bge-small` was trained on a corpus that
already includes plenty of scientific abstracts. Expect one or two nDCG points,
possibly zero. Teaching English to someone who already speaks English.

Second use: SciFact queries have roughly one relevant document each, so
Recall@100 sits very high (~0.95). That is a concrete demonstration that a
reranker has headroom here.

### NFCorpus — the real gap
Questions written by non-specialists in plain language; documents are PubMed
abstracts in clinical vocabulary. Someone asks *"does eating eggs raise
cholesterol"* and the paper says *"dietary cholesterol and serum LDL
concentrations."* Not one word in common.

That mismatch is exactly what fine-tuning is supposed to close, and it is the
largest genuine domain gap among the small BEIR sets. Your gain should appear
here.

Two things to know going in: NFCorpus uses **graded** relevance (not binary), so
its nDCG is not directly comparable to SciFact's. And absolute scores are low —
roughly 0.32–0.35 for good models. That is a hard dataset, not a broken pipeline.

### FiQA — proof you did not break anything
Fine-tuning a model to be good at nutrition can make it worse at everything else.
It specialises. If you do not check for this, a reader will ask and you will have
no answer.

Financial text is maximally distant from biology and nutrition, so any drop is a
clean signal rather than a domain-overlap artefact.

**The forgetting number is a delta, never an absolute.** `0.38` on its own means
nothing. `0.40 base → 0.38 tuned` means fine-tuning cost you two points
elsewhere. Run the base model on FiQA first and write the number down.

Second use: at ~58k documents it is the only one large enough to make the ANN
sweep honest. At SciFact's 5k a flat index is ~8MB and searches in well under a
millisecond — approximate search is solving a problem that does not exist.

### Two structural points

**Do not train one model on both.** Domain adaptation means adapting to *a*
domain. You run the pipeline twice and produce two separate models. A model
trained on both is adapted to neither in the way you are claiming. This is why
the config is per-dataset.

**Terminology.** This is catastrophic *forgetting*, not "cascade forgetting" —
the cascade is the BM25 → bi-encoder → reranker chain, a stage-three concern.
Strictly, what FiQA measures is out-of-domain generalisation loss. Call it what
it is; a reviewer who works on continual learning will notice.

---

## 4. The metrics

### nDCG@10

You return 10 documents. Counting how many are relevant throws away something
important — three relevant documents at positions 1, 2, 3 is much better than the
same three at 8, 9, 10. The metric has to be **position-aware**. That is the
whole idea; everything else is bookkeeping.

- **Gain** — the relevance grade of the document at that rank (1 or 0 for
  SciFact; graded for NFCorpus).
- **Discounted** — divide by log₂(rank + 1). Rank 1 divides by 1, rank 10 by
  ~3.5. A log rather than the rank itself because dividing by rank is too harsh
  and overstates how much users discount. An empirical choice, not a derivation.
- **Cumulative** — sum across the 10 positions. That is DCG@10.
- **Normalized** — divide by the DCG of the best possible ordering for that
  query, so the result sits in [0, 1] and every query is scored against its own
  ceiling.

**Worked example.** Three relevant documents returned at ranks 1, 4, 8:

```
your DCG   = 1/log₂2 + 1/log₂5 + 1/log₂9 = 1.00 + 0.43 + 0.32 = 1.75
ideal DCG  = 1/log₂2 + 1/log₂3 + 1/log₂4 = 1.00 + 0.63 + 0.50 = 2.13
nDCG@10    = 1.75 / 2.13 = 0.82
```

Average across all queries. That average is what you compare to MTEB.

**What normalisation buys you.** Some queries have one relevant document, others
twenty. Without normalising, the twenty-relevant query dominates the average
purely because more gain is available. It also means nDCG@10 can hit 1.0 while
you missed relevant documents — with 30 relevant and 10 slots, the ideal is
computed over the best 10. nDCG@10 asks "did you order the top 10 well," not
"did you find everything."

**Why @10.** Roughly what a user looks at, and it is BEIR's primary metric.

**Implementation note.** Several DCG variants exist, differing mainly in whether
gain is used raw or as `2^gain − 1`. Use `pytrec_eval` or `ir_measures`. If your
variant differs from BEIR's you will spend a day hunting a bug that is in your
metric, not your pipeline.

### Recall@100

Of the genuinely relevant documents, how many made the top 100? This is the
number that separates the two failure modes, and it tells you whether stage three
is worth building:

- **Recall@100 ≈ 0.94** — the right answers are in the pile and just need
  reordering. A reranker has real room to work.
- **Recall@100 ≈ 0.55** — half the right answers never made the shortlist. No
  amount of reranking recovers them; the problem is upstream in the retriever or
  the chunking.

One number, measured for free in stage two, tells you whether stage three is
worth doing at all. Knowing which situation you are in is itself a good finding.

### Similarity is not relevance

A bi-encoder ranks by embedding similarity, which is a *proxy* for relevance.
Usually a good one. An abstract can be enormously similar to a query — same
topic, same vocabulary — while answering nothing. Fine-tuning makes the proxy
better calibrated to your domain, but it is still a proxy computed from a
384-number summary made before anyone knew what would be asked. A cross-encoder
breaks that ceiling because it asks a different question: not "are these
similar" but "does this document answer this query."

---

## 5. The seven steps

### Step 1 — Ingest and freeze the split

**What.** Download the BEIR package: `corpus.jsonl`, `queries.jsonl`,
`qrels/test.tsv`. Write down exactly which queries and qrels you will evaluate
on, save to a file, hash it, never touch it again. Carve out a **dev** set for
every decision you make.

**Why.** Every number over the next month must be comparable to every other
number. Quietly evaluating on 297 queries in week one and 300 in week three makes
comparison meaningless, and you will never notice.

The dev/test split matters for a different reason: if you tune the learning rate
by checking the test score and then report the test score, you tuned *on* the
number you are reporting. Dev is for decisions. Test is looked at once.

**Chunking.** Already done — BEIR documents are abstracts, roughly 150–350
tokens, under the 512-token limit.

**Failure mode.** Assert the query count and qrel count at the top of every eval
run. Crash loudly on mismatch.

**Cost.** Minutes, CPU.

### Step 2 — Synthetic query generation

**What.** You have documents but no training questions, so manufacture them.
Three to five per document.

Two options:
- `BeIR/query-gen-msmarco-t5-base-v1` — a T5 model built for exactly this.
  Reliable, fast.
- A small instruction model (`Qwen2.5-1.5B-Instruct`, `Llama-3.2-3B-Instruct`)
  prompted for different *styles* from the same passage: one keyword query, one
  full question, one vague underspecified one. Better matches how real people
  type, and gives you something to discuss.

Sample with `top_p=0.95`, `temperature=1.0`. Sampling, not beam search — beam
search gives five near-identical questions.

**Filtering, which is not optional.** Embed each generated query with the base
model and search the corpus. Does it retrieve its own source document in the top
10? If not, discard the pair. That throws out 15–30% of generations and is the
cheapest quality improvement in the pipeline.

**Failure mode.** The generator copies passage vocabulary verbatim, so the
training task becomes lexical matching and the model learns nothing about
meaning. Symptom: training loss drops to near zero within a few hundred steps.
Look at your generated queries by hand.

**Cost.** The slowest step. ~15,000 generations ≈ 20–45 min on a free T4 with
T5-base. Generate once, save to disk, never regenerate.

### Step 3 — Hard negative mining

**What.** Training needs a query, a correct document, and several *wrong* ones.
Random wrong documents are useless — pair "does eating eggs raise cholesterol"
against a paper on soil pH and the model separates them trivially, learning
nothing. You need wrong documents that are nearly right.

Embed the corpus, search with each synthetic query, take results ranked **10 to
50**. Keep 4–8 per query.

**Why skip the top 10.** Your answer key is incomplete. Real relevant documents
nobody labelled concentrate right at the top. Scooping those up as "negatives"
actively trains the model that correct answers are wrong.

Add a **margin filter**: a negative must score at least 5% below the positive.
Anything scoring nearly as high probably *is* correct and just was not labelled.

**Optional upgrade.** Score candidates with an off-the-shelf cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) and drop anything it rates clearly
relevant. Real false-negative filtering rather than a positional heuristic.
Minutes at this scale.

**Failure mode.** Not re-mining. After fine-tuning, the model has learned to push
those specific negatives down — they are not hard anymore. Re-mine with the tuned
model and train a second round. Report both.

**Cost.** A couple of minutes on a T4.

### Step 4 — Contrastive fine-tune

**What.** `MultipleNegativesRankingLoss`. The signal is simple: pull the query
toward its correct document, push it away from the negatives.

```
base model     BAAI/bge-small-en-v1.5   (33M params, 384 dims, 512 max tokens)
learning rate  2e-5, linear warmup over first 10% of steps
batch size     64, fp16
epochs         1–3, dev nDCG@10 every few hundred steps, early stopping
max length     350 documents / 64 queries
```

**The batch size thing, which is not intuitive.** In this loss, everything else
in the batch acts as an additional negative — so batch size *is* your negative
count, making it the most important hyperparameter, and it is exactly what a 16GB
T4 constrains. `CachedMultipleNegativesRankingLoss` (GradCache) computes
embeddings in mini-batches, caches them, and reconstructs the gradient of a large
batch. Effective batch of 256 with the memory of 16. Running 64 vs 256 and
showing the nDCG difference is a genuinely good result for one extra run.

**The prefix trap.** BGE expects `"Represent this sentence for searching relevant
passages: "` on queries, nothing on documents. E5 uses `"query: "` and
`"passage: "`. Omit them and the model works, just worse, with no error. Apply
identically across baseline, training, and evaluation.

**The name for what you are building.** GPL — *Generative Pseudo Labeling for
Unsupervised Domain Adaptation of Dense Retrieval* (Wang et al., 2022). Read it,
cite it. It signals you know the method has a name and a literature.

**Cost.** ~10,000 pairs, 2 epochs, batch 64: 5–20 min. Ten runs in an afternoon.

### Step 5 — Build the index

**What.** Normalise vectors to unit length, then use inner product — with unit
vectors, inner product equals cosine similarity.

| Index | Settings | Character |
|---|---|---|
| `IndexFlatIP` | — | Exact brute force. Your correctness reference. |
| `IndexHNSWFlat` | `M=32`, `efConstruction=200`, sweep `efSearch` 16→256 | Fast, high recall, memory-hungry |
| `IndexIVFFlat` | `nlist ≈ √N`, sweep `nprobe` | Cheap middle ground |
| `IndexIVFPQ` | `m=48`, `nbits=8` (48 bytes/vector) | `m` must divide 384 |

**What to report.** Recall-at-k versus latency, where recall is measured
**against the flat index**, not against the qrels. That separates "the index
dropped documents" from "the model ranked badly" — two different problems a
single nDCG number blends together.

**The honesty point.** At 5k documents a flat index is ~8MB and searches in well
under a millisecond. Do the sweep on FiQA and say plainly that flat search is the
correct engineering choice at SciFact scale. Knowing when *not* to reach for a
technique reads better than using it everywhere.

**Cost.** Seconds to a few minutes. CPU.

### Step 6 — Evaluation

**What.** Use `pytrec_eval` or `ir_measures`. Do not write your own nDCG. Report
nDCG@10, Recall@50, Recall@100, MRR@10.

**BM25 choice, with a real consequence.** `rank_bm25` is pure Python and installs
instantly but will not reproduce published BEIR numbers — those come from
Anserini with different tokenisation. Pyserini reproduces them but needs Java.
Either is defensible; if you use `rank_bm25`, state that your BM25 is
self-consistent rather than leaderboard-comparable.

**The milestone.** Evaluate off-the-shelf `bge-small` on SciFact against the
published MTEB number. Do not move past this until it passes.

**Failure mode.** Evaluating on synthetic queries. That measures how well you fit
your query generator, not whether retrieval improved. Synthetic queries are
training-only. Always.

### Step 7 — Latency benchmark

**What.** Report separately: index build time and peak memory; on-disk size;
query encoding latency (p50/p95, batch size 1); search latency per index config;
end-to-end.

Warm up with 50 discarded queries, then measure over at least 500. Pin threads
with `faiss.omp_set_num_threads(1)` for a reproducible single-query number;
report throughput separately with threads unpinned. **Record the hardware** —
Colab assigns different CPUs between sessions and an unlabelled latency number is
worthless.

**Why it earns a place.** Query encoding is a fixed per-query cost independent of
corpus size. Search cost scales with the corpus. Separating them shows you know
where the cost actually lives.

---

## 6. Reading your results

### Is forgetting bad?

Some loss is expected and fine. The model has 33M parameters and fixed capacity;
teaching it that lay nutrition phrasing maps to clinical phrasing necessarily
reshapes the space. A model that loses nothing anywhere probably learned nothing.

What matters is the **trade**, framed as a ratio rather than two numbers:

| NFCorpus gain | FiQA loss | Verdict |
|---|---|---|
| +6 | −2 | Good trade, easy to defend |
| +6 | −1 | Excellent |
| +2 | −6 | Bad — damaged more than you built |
| +6 | −15 | Over-specialised. Lower the LR or the epoch count. |

There is also a context question. If the deployment is a nutrition search system,
forgetting finance is irrelevant — nobody will ask it a finance question.
Out-of-domain loss matters only if you are claiming a general-purpose model.
State which claim you are making.

### Is there an optimal base-to-tuned difference?

No target number exists — any threshold would be invented. What exists instead
are diagnostic patterns:

| Pattern | What it means |
|---|---|
| Gain ≈ 0 on both | Domain gap too small, or training failed. Check the loss curve. |
| Large gain on SciFact, small on NFCorpus | Backwards from expectation. Likely a leak or a bug. |
| Gain on NFCorpus, ≈ 0 on SciFact | Exactly as predicted. Good. |
| Huge gain, huge forgetting | Over-training. Reduce LR or epochs. |
| Arm A ≫ Arm B | Query generation is your bottleneck. |
| Arm A ≈ Arm B | Synthetic queries nearly match human labels. Strong finding. |

That last row is the point of the whole project. The Arm A minus Arm B gap is
bounded and interpretable: it is the cost of not having labels. A real answer to
a real question, independent of hitting any particular absolute number.

**Noise caution.** A gain under about one nDCG point is within run-to-run
variance. To claim a small gain, train with three seeds and report mean and
spread. Otherwise you may be reporting a seed.

### Choosing a baseline honestly

Beating `all-MiniLM-L6-v2` by fifteen points means little — it is a 2021 model.
Beating current `bge-small` by three means a lot. Report the strong one, and if
you show a weak one for context, say which is which.

---

## 7. Repo structure

```
domain-retrieval/
├── README.md
├── requirements.txt
├── Makefile
├── .gitignore
├── configs/
│   ├── scifact.yaml
│   ├── nfcorpus.yaml
│   └── fiqa.yaml              ← eval-only
├── data/
│   ├── raw/                   ← gitignored
│   ├── frozen_splits/         ← committed, this is the contract
│   ├── synthetic_queries/     ← gitignored
│   ├── hard_negatives/        ← gitignored
│   ├── embeddings/            ← gitignored
│   └── runs/                  ← committed, *.trec
├── src/
│   ├── config.py              the only file that knows about prefixes
│   ├── ingest.py              step 1
│   ├── generate_queries.py    step 2
│   ├── mine_negatives.py      step 3
│   ├── train.py               step 4
│   ├── build_index.py         step 5
│   ├── retrieve.py            model + index → run file
│   ├── evaluate.py            step 6, the only file that computes metrics
│   ├── benchmark_latency.py   step 7
│   ├── bm25.py                baseline
│   └── rerank.py              stage three, empty for now
├── tests/
├── scripts/
├── notebooks/
├── models/                    gitignored
└── results/
    ├── metrics.csv
    ├── latency.csv
    └── figures/
```

**Nothing is passed between steps in memory.** Every step reads a path and writes
a path. That is what survives a Colab disconnect and what makes stage three
possible without touching stage two.

### The one interface that matters: the run file

Every retrieval run — BM25, off-the-shelf, fine-tuned, any index config — writes
the same TREC-style format to `data/runs/{run_name}.trec`:

```
query_id  Q0  doc_id  rank  score  run_name
```

`evaluate.py` takes a run file plus the frozen qrels and outputs metrics. It
never touches a model. Consequences:

- Comparing four systems is four run files against one eval call, not four code
  paths.
- Stage three's reranker reads `top100_bge_finetuned.trec`, rescores ranks 1–100,
  writes a new run file with the same schema. `evaluate.py` does not know a
  cross-encoder was involved.
- You can hand someone the `runs/` directory alone and they can reproduce every
  number in your report without your model checkpoints.

### Config, not hardcoded args

```yaml
dataset: scifact
base_model: BAAI/bge-small-en-v1.5
query_prefix: "Represent this sentence for searching relevant passages: "
doc_prefix: ""
max_seq_length: 350
negatives_per_query: 6
mining:
  range_min: 10
  range_max: 50
  margin: 0.05
training:
  lr: 2e-5
  batch_size: 64
  epochs: 2
  warmup_ratio: 0.1
```

Every script takes `--config configs/scifact.yaml`. Swapping datasets is a flag,
not an edit.

### metrics.csv schema

```
dataset, model_name, trained_on, training_arm, index_type,
ndcg@10, recall@50, recall@100, mrr@10, query_prefix_used, seed, run_file, notes
```

`dataset` is what you **evaluate** on; `trained_on` is what the model was tuned
on (null for baselines). Keeping these separate is what makes the FiQA rows
unambiguous — otherwise `dataset=fiqa, training_arm=synthetic` cannot tell you
whether the model was tuned on SciFact or NFCorpus. `training_arm` is
`baseline`, `synthetic`, or `real_labels`.

Append one row per experiment, never overwrite. At the end, `metrics.csv` is the
single source of truth from which the README tables and every plot are generated.

---

## 8. Sequencing

| Week | Work |
|---|---|
| 1 | Steps 6 and 1: harness, frozen split, BM25, off-the-shelf embeddings, the reproduce-the-published-number check. Nothing else. |
| 2 | Error analysis and step 5: read 20 queries where BM25 beats the dense model; index sweep; hybrid retrieval with reciprocal rank fusion (an afternoon, and it is what production systems do). |
| 3 | Steps 2–4: generation, mining, training, the two-arm comparison. |
| 4 | Step 7, the forgetting check on FiQA, the write-up. |

Stage three comes after all of that, or not at all.

### Two Colab constraints

Free Colab disconnects on idle and caps sessions. Checkpoint the model and save
every intermediate artefact — generated queries, mined negatives, embeddings, run
files — to Drive as you go. Make each step read from and write to disk rather
than passing objects in memory, so a disconnect costs one step rather than the
whole run.

### The risk worth stating plainly

`bge-small-en-v1.5` was trained on a large, diverse corpus that already includes
scientific text. Fine-tuning on synthetic SciFact queries may produce a gain of
one or two nDCG points, or none. That is a real outcome, not a mistake.

Three mitigations: prefer NFCorpus for the headline; include the real-labels arm,
because if synthetic training fails but real-label training succeeds, that itself
is the finding; and if you report a large gain over a deliberately weak baseline,
say so and report the strong baseline alongside.

---

*All specific figures here are approximate. Verify against the current MTEB
leaderboard before putting any of them in a README — they shift as the
leaderboard is recomputed.*

---

## 9. Experiment design and current findings

### The design as run

```
                          PRETRAINED BGE
                         bge-small-en-v1.5
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
           SciFact           NFCorpus            FiQA
              │                 │                 │
        harness check       BASELINE          eval-only
      "does the pipeline    (BM25 vs BGE,      (never trained;
       reproduce a          nDCG@10=0.3423,     forgetting
       published number?"   recall@100=0.309    reference for
              │              → mode A: right     every tuned
              ▼              docs missing         model below)
      BM25=0.652 vs           from shortlist)         │
      BGE=0.713 ✓                  │                  │
      (matches MTEB)               ▼                  │
              │            ┌──────────────────┐       │
              ▼            │  Fine-tune BGE   │       │
      cross-encoder        │  MultipleNeg-    │       │
        rerank top 50      │  RankingLoss     │       │
              │            └────────┬─────────┘       │
              ▼                     │                  │
      nDCG 0.713→0.696   ┌──────────┼──────────┐       │
      (WORSE; recall      │          │          │       │
       unchanged →        │          │          │       │
       reorders only,     REAL LABELS   T5      Qwen    │
       doesn't retrieve)  (Arm A)   queries    queries  │
                              │    (Arm B)     (Arm B′) │
                              │        │          │     │
                              ▼        ▼          ▼     │
                          nDCG+2.83 +0.08      +0.51     │
                          ██████░░  ▏          ▏         │
                          real labels >> either synthetic│
                              │        │          │     │
                              └────────┼──────────┘     │
                                       │                 │
                                       ▼                 │
                          "synthetic supervision does    │
                           NOT reproduce the benefit      │
                           of human labels here"          │
                                       │                 │
                                       ▼                 │
                    ┌──────────────────────────────┐     │
                    │  HYPERPARAMETER SWEEP (DEV)   │     │
                    │  one frozen baseline, change   │     │
                    │  exactly one field at a time    │◄────┘ (each config
                    │                                 │        re-checked
                    │  epochs   2→3→5   3 wins        │        against FiQA
                    │  range_min 5/10/20  no effect    │        dev for
                    │  batch    64→128→256 no gain     │        forgetting)
                    │  lr       1e-5/2e-5/4e-5         │
                    │           4e-5 WINS (+0.79)      │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    FINAL CONFIG: epochs=3, batch=64,
                    lr=4e-5  (chosen on DEV only)
                                     │
                                     ▼
                    single TEST measurement (read once)
                    nDCG@10: 0.3423 → 0.3865  (+4.42)
                    recall@100: 0.3090 → 0.3837 (+7.47)
```

> **Note on the diagram.** It shows what was actually run, not what was
> originally planned - the difference matters. Stage three (reranking) was
> measured only on SciFact, at depth 50 rather than 100, because recall@50
> (0.9317) and recall@100 (0.9450) were close enough that the deeper shortlist
> cost twice the compute for 1.3 points of ceiling; it was never applied to the
> fine-tuned NFCorpus model. The sweep box ran six training configurations, all
> scored on **dev only** - test was read exactly once, at the very end, for the
> single config the sweep chose. Every fine-tuned model along the way (Arm A,
> Arm B, Arm B′, and every sweep config) was also scored on FiQA as a forgetting
> check; only the final config's FiQA numbers are drawn here for space.

---

### What the current results suggest

#### What might explain the weak synthetic results?

This is where your discussion section can become interesting. Possible
explanations include:

**A. Generated queries may be too similar to the document.**

For example, the generator sees:

> "Dietary cholesterol intake is associated with serum LDL..."

and generates:

> "What is the association between dietary cholesterol intake and serum LDL?"

That's a reasonable question, but it may be too close to the document's
vocabulary. A real user might ask:

> "Does eating eggs increase bad cholesterol?"

The second query creates a more difficult semantic retrieval problem.

> **Measured support.** The share of a query's words that also appear in its
> source passage was 0.636 for T5 queries and 0.501 for Qwen queries. Qwen's are
> less derivative, yet neither produced a gain outside noise. Consistent with A,
> but does not isolate it.

**B. Synthetic queries may not represent real information needs.**

Your evaluation queries are written by real users. Your synthetic queries are
generated from documents. Those are fundamentally different directions:

```
REAL:                              SYNTHETIC:

User information need              Document
        ↓                             ↓
     query                           LLM
        ↓                             ↓
 relevant document                 "possible query about this document"
```

The second process may generate queries that are document-relevant but not
representative of real search behaviour.

> **A pipeline mechanism that compounds B.** The round-trip filter keeps a
> generated query only if the *base* model already retrieves its source passage
> in the top 10. That systematically keeps queries the model can already answer
> and discards the hard ones a real user would produce. It kept 96% of Qwen's
> queries against 93% of T5's. This makes A and B hard to separate, because the
> filter removes exactly the queries that would have tested them.

**C. Your training setup may not yet be optimal.**

This is important because you haven't done a proper hyperparameter/seed
investigation. You currently used mostly the guide's defaults. For example, you
haven't established whether these would substantially change the result:

- ~~2 vs 3 epochs~~ — **now tested, see correction below**
- learning rate
- number of negatives
- batch size
- synthetic query filtering
- number of generated queries
- different generation prompts

And you already have evidence that 2 epochs may be too early:

```
Dev nDCG:
epoch 1 → 0.3503
epoch 2 → 0.3564
epoch 3 → 0.3571
```

> **Correction.** Those three numbers are mislabelled. They are three
> *evaluations inside a single 2-epoch run*, taken at epochs 0.68, 1.37 and
> 2.00 - not epochs 1, 2 and 3. What they do show correctly is a curve still
> rising when training stopped.
>
> **Epochs has since been tested properly, on dev:**
>
> | Config | Dev nDCG@10 | vs 2 epochs |
> |---|---|---|
> | base | 0.3291 | - |
> | real labels, 2 epochs | 0.3544 | - |
> | real labels, 3 epochs | **0.3616** | **+0.72** |
> | Qwen synthetic, 2 epochs | 0.3344 | - |
> | Qwen synthetic, 3 epochs | 0.3340 | -0.04 |
>
> Two epochs *was* too early for real labels. But the extra epoch did nothing
> for the synthetic arm, whose training loss kept falling (0.47 to 0.42) while
> its dev score stayed flat. **So the gap survived its first real test of a
> tuning explanation**, which strengthens the conclusion below rather than
> weakening it.

So you shouldn't conclude that synthetic supervision fundamentally cannot work.
A safer interpretation is:

> Under the current synthetic-query generation, filtering, negative-mining, and
> fine-tuning configuration, synthetic supervision produced substantially smaller
> gains than human-labelled supervision.

That's a much stronger academic statement because it matches exactly what you
tested.

---

#### The reranker result tells another story

You got:

```
BGE:
nDCG@10    = 0.7127
Recall@100 = 0.9450

BGE + cross-encoder:
nDCG@10    = 0.6964
Recall@100 = 0.9450
```

This suggests the cross-encoder did not improve the final ranking on SciFact in
your current setup.

And because Recall@100 stayed exactly the same, `0.9450 → 0.9450`, we know it
wasn't adding new documents. It only reordered the existing shortlist.

Interestingly, the ordering became slightly worse according to nDCG@10. A likely
explanation is model/domain mismatch: your cross-encoder was trained for a
different retrieval setting than SciFact.

> **Supporting measurement.** A larger cross-encoder (MiniLM-L-12, twice the
> layers) scored 0.6965 - essentially identical to the L-6's 0.6964. That rules
> out model capacity and points at the mismatch: MS MARCO is short web-search
> questions, while SciFact queries are declarative scientific claims.

Again, say "suggests", not "proves."

---

> **Update: reranking the final NFCorpus model (dev).** After fine-tuning, two
> off-the-shelf cross-encoders were tried on the top 50. Both lowered nDCG@10:
> MS MARCO MiniLM from 0.3695 to 0.3427, and the biomedical MedCPT to 0.3564.
> Recall stayed the same, so they only reordered. A perfect reordering would
> reach 0.6255, so the room exists, but these rerankers do not capture it. Hybrid
> BM25 + dense retrieval also lowered the score, to 0.3253. Details in
> `../DESIGN.md`.

---

#### How to write the main findings

> The results suggest that domain-specific fine-tuning substantially improves
> dense retrieval on NFCorpus when human relevance labels are available.
> However, the synthetic-query approaches evaluated in this study produced only
> modest improvements over the pretrained baseline, ~~with Qwen outperforming T5
> but~~ remaining substantially below the real-label model. This suggests that,
> under the current experimental setup, synthetic supervision does not fully
> reproduce the benefit of human relevance labels. On SciFact, cross-encoder
> reranking did not improve performance and slightly reduced nDCG@10 while
> leaving Recall@100 unchanged, indicating that the reranker affected document
> ordering rather than retrieval coverage.

> **Correction to "Qwen outperforming T5".** This does not hold. The ordering
> *reverses* between splits:
>
> | | Dev nDCG@10 | Test nDCG@10 |
> |---|---|---|
> | T5 synthetic | **0.3392** | 0.3431 |
> | Qwen synthetic | 0.3344 | **0.3474** |
>
> Qwen wins on test, T5 wins on dev. Both differences are under half a point,
> inside the roughly one-point noise threshold of single-seed runs. The only
> defensible statement is that **both** synthetic arms fall well below real
> labels on both splits. The struck-through clause should be removed from any
> write-up.

That is, in my opinion, the most defensible interpretation of your results right
now.

The really interesting next question is therefore not simply *"does fine-tuning
work?"* — your real-label result already suggests yes. The more interesting
question is:

> **Why does human supervision work much better than the synthetic supervision,
> and can we improve the synthetic pipeline enough to close that gap?**

That is where your project can become much more valuable.

---

#### Methodological disclosure

Every result before the epochs sweep was measured on the **test** split: three
baselines, two prefix ablations, two rerankers and three fine-tuned models,
roughly fifteen inspections, each one informing the next step. Absolute numbers
therefore carry some optimistic bias from repeated inspection. Comparisons are
less affected, because every model was scored on the identical frozen split with
identical code. From the sweep onward, decisions are made on dev and test is
read once.

All runs are single-seed.
