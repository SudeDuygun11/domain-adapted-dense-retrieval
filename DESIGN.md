# Pipeline Design and Its Choices

Every decision made in `src/`, what the alternative was, and why this side won.
Read `README.md` first for what the project is; this is how it was built.

---

## 0. Where the project stands

All three splits are frozen and all three baselines measured. The dense nDCG@10
matches the published figure on every dataset, so the harness is calibrated.

| Dataset | Docs | Test queries | Judgments/query | BM25 nDCG@10 | Dense nDCG@10 | Dense recall@100 |
|---|---|---|---|---|---|---|
| SciFact | 5,183 | 300 | 1.1 | 0.6523 | 0.7127 | 0.9450 |
| NFCorpus | 3,633 | 323 | 38.2 | 0.3062 | 0.3423 | 0.3090 |
| FiQA | 57,638 | 648 | 2.6 | 0.2175 | 0.3972 | 0.6892 |

**The three sit in different regimes, and the recall column is why.**

- **SciFact, recall@100 = 0.945.** Failure mode B. Everything is in the pile and
  only the order is wrong, so reranking is the lever. Confirmed by measurement:
  off-the-shelf reranking changed nDCG and left recall untouched.
- **NFCorpus, recall@100 = 0.309.** Failure mode A. Roughly 70% of relevant
  documents never reach the top 100, so no reranker can recover them.
  Fine-tuning is the lever here, which is why this is the headline dataset.
- **FiQA, recall@100 = 0.689.** In between, and eval-only regardless. Its
  0.3972 is the forgetting reference every tuned model gets subtracted from.

The judgments-per-query column explains NFCorpus's apparently poor recall. With
38 relevant documents per query, finding *most* of them in 100 slots is a much
harder task than SciFact's 1.1, and its high MRR@10 of 0.52 against a low
nDCG@10 of 0.34 says the same thing: finding one is easy, finding them all is
not. **Recall is not comparable across datasets with different judgment
densities.**

### Stage two result: the two-arm experiment on NFCorpus

| Arm | NFCorpus nDCG@10 | Gain | FiQA nDCG@10 | Cost | Trade |
|---|---|---|---|---|---|
| base | 0.3423 | - | 0.3972 | - | - |
| **A** real labels | 0.3706 | **+2.83** | 0.3843 | -1.29 | 2.2 : 1 |
| **B** synthetic | 0.3431 | +0.08 | 0.3813 | -1.59 | 0.05 : 1 |

**Arm A >> Arm B.** In the guide's diagnostic table that pattern means query
generation is the bottleneck. Real labels bought nearly three points; generated
queries bought nothing while costing the same elsewhere.

**The training loss says why.** Arm A finished at 3.25, Arm B at 0.73. A loss
that low is the documented symptom of a generator copying passage vocabulary:
the training task collapses into lexical matching and the model learns nothing
about meaning. Inspecting the generated queries confirms it - "statins in finland
death rate" is keyword extraction from the source abstract, not the
plain-language question a patient would type. The T5 generator is trained on
MS MARCO web search and reproduces that style, which sits on the *paper's* side
of exactly the gap NFCorpus exists to measure.

**Arm A's gain is bigger in recall than in nDCG**, +5.44 against +2.83. That is
mode A being repaired: fine-tuning pulled documents into the shortlist that were
not there before. It is the one thing reranking mathematically cannot do, and the
baseline recall of 0.309 predicted it was what this dataset needed.

**The trade is defensible for Arm A and not for Arm B.** Arm A gained 2.83 on the
target domain for 1.29 lost on finance, a ratio above 2:1. Arm B paid 1.59 for
nothing. Some out-of-domain loss is expected - the model has 33M parameters and
fixed capacity, and one that loses nothing anywhere probably learned nothing.

### Swapping the generator: Arm B, second attempt

The T5 result pointed at the generator, so the generator was replaced with
`Qwen2.5-1.5B-Instruct`, prompted for three different *kinds* of query rather
than three rewordings of one.

| Query set | Queries | Passage-word overlap | Mean length | nDCG@10 | Gain |
|---|---|---|---|---|---|
| T5 msmarco | 10,122 | 0.636 | 6.4 words | 0.3431 | +0.08 |
| Qwen instruct | 9,000 | 0.501 | 14.7 words | 0.3474 | +0.51 |
| *(Arm A, real labels)* | 9,321 | - | - | 0.3706 | +2.83 |

The queries got measurably less derivative - half the passage's words instead of
two thirds - and the gain rose from +0.08 to +0.51. That closes about 18% of the
distance to real labels. **It is still inside the ~1 point noise threshold, so it
is an observation and not yet a result.**

**The prediction that failed.** The expectation was that harder queries would
raise the training loss toward Arm A's 3.25. It went the other way: 0.73 for T5,
**0.47** for Qwen.

**The likely reason, which is a flaw in the pipeline rather than in the
generator.** The round-trip filter keeps a generated query only if the BASE model
already retrieves its source passage in the top 10. That is a selection effect:
it systematically keeps the queries the model already handles and discards the
ones it would have had to learn from. It kept 96% of Qwen's queries against 93%
of T5's, so the verbose questions were *easier* for the base model, not harder.

The filter is still necessary - it removes hallucinated pairs - but it caps how
far the synthetic arm can move. A less self-defeating version would keep some
queries that fail the round trip but pass a cross-encoder relevance check, which
is the "optional upgrade" named in step 3 of the README. Untested here.

### The first real hyperparameter test, measured on DEV

Every number above came from running the guide's defaults once. Arm A's dev nDCG
was still rising when training stopped (0.3503 -> 0.3564 -> 0.3571), which is
direct evidence of an early stop rather than convergence. So epochs was tested.

| Config | Dev nDCG@10 | vs 2 epochs |
|---|---|---|
| base (no tuning) | 0.3291 | - |
| Arm A, 2 epochs | 0.3544 | - |
| **Arm A, 3 epochs** | **0.3616** | **+0.72** |
| Arm B Qwen, 2 epochs | 0.3344 | - |
| Arm B Qwen, 3 epochs | 0.3340 | -0.04 |

**Two epochs was too early for Arm A.** Its gain over base rises from +2.53 to
+3.25 on dev. The default was wrong for this dataset.

**The extra epoch did nothing for Arm B.** That asymmetry is the important part.
If the label gap were a tuning artefact, more training would close it. It does
not: the synthetic arm is saturated, and its training loss fell further (0.47 ->
0.42) while its dev score stayed flat - more passes over weak signal extracting
nothing new.

So the gap survives its first real attempt at a tuning explanation. That
**strengthens** the interpretation: the limit is the information content of the
synthetic supervision, not the optimiser settings.

**Not yet established:** whether 4 or 5 epochs is better still for Arm A. The
curve has not been shown to turn.

**A methodological debt that must be disclosed.** Every result before this sweep
was measured on TEST - three baselines, two prefix ablations, two rerankers and
three fine-tuned models, roughly fifteen looks, each informing the next step.
Absolute numbers therefore carry optimistic bias from repeated inspection. The
comparisons are less affected, since every model was scored on the identical
frozen split with identical code. From the sweep onward, decisions are made on
dev and test is read once.

**Caveat that must be stated.** These are single-seed runs. The guide's noise
threshold is about one nDCG point. Arm A's +2.83 clears it; Arm B's +0.08 and
both FiQA deltas do not. To claim the FiQA costs specifically, train three seeds
and report mean and spread.

---

## 1. The shape

Seven steps. Nothing is passed between them in memory. Every step reads a path
and writes a path.

```
                      configs/<dataset>.yaml
                              |
          +-------------------+-------------------+
          |                                       |
   [1] ingest.py                                  |
     downloads BEIR, freezes the split            |
          |                                       |
     data/frozen_splits/<ds>/   (COMMITTED)       |
     data/raw/<ds>/corpus.jsonl (gitignored)      |
          |                                       |
     +----+--------------------------------+      |
     |                                     |      |
  [bm25.py]                        [5] build_index.py
  lexical baseline                   encodes corpus -> FAISS
     |                                     |
     |                              [retrieve.py]
     |                               encodes queries, searches
     |                                     |
     +------------+------------------------+
                  |
          data/runs/*.trec          <- THE INTERFACE
                  |
          [6] evaluate.py           <- the only file that computes a metric
                  |
          results/metrics.csv       (COMMITTED, append-only)


  STAGE TWO adds, before build_index:

   [2] generate_queries.py  ->  data/synthetic_queries/<ds>.jsonl
   [3] mine_negatives.py    ->  data/hard_negatives/<ds>_<arm>_<tag>.jsonl
   [4] train.py             ->  models/<ds>_<arm>_s<seed>/

  then re-runs build_index -> retrieve -> evaluate with the new model.
```

**Why path-in, path-out rather than function calls.** It survives a Colab
disconnect: a dropped session costs one step, not the whole run. It also makes
stage three possible without touching stage two, because a reranker is just
something that reads a run file and writes a run file.

---

## 2. Cross-cutting choices

These are the decisions that shaped everything else.

### 2.1 The run file is the only interface between retrieval and evaluation

Every retrieval system writes TREC format to `data/runs/<name>.trec`:

```
query_id  Q0  doc_id  rank  score  run_name
```

**Alternative rejected:** each system returning Python objects that evaluation
consumes directly.

**Why this won.** Comparing four systems becomes four files against one
`evaluate.py` call rather than four code paths. `evaluate.py` never imports a
model, so it cannot silently re-encode with a different prefix and score a
pipeline you never ran. And you can hand someone `data/runs/` alone and they can
reproduce every number without your checkpoints.

**Consequences deliberately accepted.** Disk I/O that a function call would not
need, and a format with a vestigial `Q0` column.

### 2.2 One file owns model prefixes

`config.py` exposes `for_query()` and `for_document()`. No other module reads
`cfg.query_prefix`.

**The failure being prevented.** BGE expects an instruction on queries and
nothing on documents. Omit it and there is no error, no warning, just a slightly
worse score everywhere. There are four places that could get it wrong: BM25,
indexing, retrieval, training.

**Why centralising beats documenting.** A rule in a README is followed by whoever
read the README. A single function is followed by construction.

Verified in `tests/test_prefix.py`, including that all three configs use an
identical prefix, so a FiQA forgetting delta cannot be contaminated by a prefix
change.

### 2.3 Every id is a string, cast at the boundary

`ingest._sid()` runs the moment a row leaves the dataset.

**The bug.** In the BeIR qrels repositories, `query-id` and `corpus-id` arrive as
int64; `_id` in the corpus and queries arrives as str. `7 in {"7": ...}` is False,
`in` returns False rather than raising, and the result is a run scoring zero
queries and an nDCG of 0.0 that looks like a bad model.

**Alternative rejected:** casting at each comparison. There are at least four
comparison sites, each a separate chance to forget one.

**Bonus.** Once ingest writes JSONL and TSV, everything read back is a string
automatically. The cast happens exactly once per dataset, ever.

### 2.4 The evaluation split is frozen and self-verifying

`data/frozen_splits/<dataset>/` is committed. `manifest.json` records counts,
provenance and a SHA-256 per file. `load_frozen_split()` recomputes those hashes
on every read and raises on a mismatch.

**The failure being prevented.** Evaluating on 297 queries in week one and 300 in
week three. Measured on your own run: dropping the 10 worst of 300 queries raises
nDCG@10 by 2.46 points. The gain fine-tuning is supposed to produce is 2 to 6
points. **Split drift is the same size as the finding.**

**Why hashing is not enough on its own.** A hash detects change; it does not
prevent it and cannot restore anything. Freezing supplies the reference, hashing
supplies the alarm, git supplies recovery. Three jobs, three mechanisms.

`ingest.py` also refuses to overwrite an existing split without `--force`.

### 2.5 The corpus is hashed, not committed

**The rule:** commit your choices, hash everything else.

The frozen split records decisions *you* made - which queries, which dev sample.
The corpus is whatever BEIR ships and is re-downloadable by anyone, so a
64-character fingerprint gives the same guarantee as 4MB of duplicated data.

The frozen split has no such property. The 100 dev queries sampled with seed 42
exist only on your disk and in your git history.

### 2.6 `results/metrics.csv` is append-only

One row per experiment, never overwritten.

**Why.** Every result in this project is a *difference* between two numbers. You
cannot report a base-to-tuned delta if the base row was replaced. Duplicated rows
are a far cheaper problem than missing ones.

**Why `dataset` and `trained_on` are separate columns.** `dataset` is what you
evaluate on; `trained_on` is what the model was tuned on. Merged into one column,
`dataset=fiqa, training_arm=synthetic` cannot tell you whether that model was
tuned on SciFact or NFCorpus, which makes every forgetting row ambiguous.

### 2.7 Configuration, never hardcoded arguments

Every script takes `--config configs/<dataset>.yaml`. Swapping datasets is a flag.

**Second reason, which matters more later.** When `metrics.csv` has twelve rows
and you cannot remember which learning rate produced row seven, the config file
is the answer and it is in git.

---

## 3. Step-by-step choices

### Step 1 - `ingest.py`

**Dev is carved from TRAIN, never from test.** The obvious alternative is an
80/20 split of the test set. Rejected because the stage-one milestone is
reproducing the published MTEB number, which is measured on the full official
test set. Slicing it destroys the only external check on the harness.

Where a dataset ships its own dev split, that one is used - a published decision
is one fewer arbitrary choice to defend.

**Dev queries are removed from train, not copied.** A query in both is a leak: you
would be tuning on data the model was trained on. Asserted in `main()`.

**The evaluation query set comes from the qrels, not the queries file.**
`queries.jsonl` holds 1,109 SciFact queries across all splits; the test qrels
cover 300. Taking the query set from the file would silently change what is
measured.

**Joins are checked before anything is written.** `check_joins()` asserts every
qrel query exists in the queries file and every qrel document exists in the
corpus, and crashes loudly otherwise.

**Files are written sorted.** Byte-identical output on every run is what makes
the hash meaningful; an unsorted file would hash differently each time and tell
you nothing.

**`doc_text()` flattens title and body once.** Title first, space-joined, the BEIR
convention the published baselines were measured against. Defined in one place so
BM25 and the encoder see byte-identical text - if they disagreed, their scores
would not be comparable and nothing would report it.

### Step 2 - `generate_queries.py`

**Sampling, not beam search.** `top_p=0.95`, `temperature=1.0`. Beam search
returns the n most probable sequences, which for one passage are five
near-identical rephrasings. Sampling produces genuinely different questions,
which is the only reason to generate several.

**The round-trip filter is mandatory.** Embed the generated query with the base
model, search, keep only if the source passage returns in the top 10. Discards
15-30%. Training on unfiltered generations teaches the model to associate a query
with a document that does not answer it, which is worse than not training.

**The filter uses the base model deliberately.** Filtering with the model you are
about to train on this data would be circular.

**Regeneration is refused without `--force`.** It is the slowest step, and its
output is what every later result was trained on.

**Default generator `BeIR/query-gen-msmarco-t5-base-v1`**, over a prompted
instruction model. Reliable and fast; the instruction model producing varied
query *styles* is a documented upgrade, not a prerequisite.

### Step 3 - `mine_negatives.py`

**Negatives come from ranks 10 to 50.** Skipping the top is the important half.
The answer key is incomplete, and unlabelled relevant documents concentrate right
at the top of a good ranking. Harvesting those as negatives actively trains the
model that correct answers are wrong.

**The margin filter drops anything within 5% of the positive's score.** A
"negative" scoring nearly as high probably is relevant and simply was not
labelled. On SciFact this dropped 1,932 candidates out of 200 queries, which is
high and expected: most SciFact queries have one labelled document, so genuine
relevant papers sit unlabelled in that window.

**The positive's score is read from the index, not re-encoded.** Its vector is
already stored; `index.reconstruct_n` is cheaper and cannot drift.

**One module serves both arms.** `--arm synthetic` reads generated queries;
`--arm real_labels` reads the frozen train qrels. Arm A and Arm B differ only in
their input, which is what makes the comparison between them clean.

**Re-mining is supported via `--index-tag`.** After training, the model has pushed
those specific negatives down and they are no longer hard.

### Step 4 - `train.py`

**`MultipleNegativesRankingLoss`**, with `CachedMultipleNegativesRankingLoss`
behind `--cached-loss`. In this loss every other batch item is an extra negative,
so batch size *is* the negative count, which makes it the most important
hyperparameter and exactly what a 16GB GPU constrains. GradCache buys an effective
batch of 256 at the memory cost of 16.

**Prefixes are applied when the dataset is built**, through the same two functions
retrieval uses. Training with a different prefix than you evaluate with optimises
the model for inputs it will never see.

**Early stopping watches dev nDCG@10.** Selecting a checkpoint on test would mean
selecting on the number you report.

**`require_training()` refuses eval-only datasets by name.** FiQA measures what
fine-tuning cost you elsewhere; training on it destroys that measurement. Enforced
in code rather than in prose, and tested.

**`set_all_seeds` pins Python, NumPy and torch.** A gain under one nDCG point is
inside run-to-run variance, so claiming a small gain requires three seeds - which
only means something if a seed actually determines the run.

### Step 5 - `build_index.py`

**Normalise to unit length, then inner product.** For unit vectors, inner product
is exactly cosine similarity, and FAISS has a fast exact inner-product index.
Normalising once at build time beats dividing by norms at every query. The rule
this creates: query vectors must be normalised identically, which `retrieve.py`
does.

**Stage one builds only `IndexFlatIP`.** Exact brute force is the correctness
reference every approximate index is measured against. At 5,183 documents the
index is 7.96MB and searches in well under a millisecond, so approximate search
would be solving a problem that does not exist. The ANN sweep belongs on FiQA at
58k documents, where it is honest.

**`doc_ids` is saved beside the index.** FAISS returns row numbers, not
identifiers. Row `i` is `doc_ids[i]` and nothing may reorder one without the
other.

**`max_seq_length` is set on the model, not applied to strings.** Truncation
should happen in token space, where the 512-token limit actually lives.

### Step 6 - `evaluate.py`

**`pytrec_eval`, never a hand-written nDCG.** Variants differ in whether gain is
raw or `2^gain - 1`, and they disagree on graded data like NFCorpus. `pytrec_eval`
wraps the original `trec_eval` C code, which is the variant BEIR reports.
`tests/test_frozen_split.py` pins the README's worked example at 0.82 so a silent
change cannot pass.

**`pytrec_eval-terrier`, not `pytrec_eval`.** Identical API and identical
underlying C code, but it publishes prebuilt Windows wheels, so no Visual C++
build tools are needed.

**MRR@10 truncates the run first.** `trec_eval`'s `recip_rank` has no cutoff, so
asking for it on a full run gives MRR, not MRR@10.

**Averaging is over every judged query.** A query the run did not answer scores
zero rather than being dropped from the denominator - failing to return anything
is a retrieval failure, not a reason to shrink the divisor. A warning prints
because a run covering half the queries still produces a number that looks like a
score.

**A run scoring queries outside the frozen qrels is a fatal error**, not a
warning. It means retrieval ran on a different query set than evaluation.

### `bm25.py`

**`rank_bm25`, not Pyserini.** Pyserini reproduces published BEIR figures exactly
but needs a Java runtime. `rank_bm25` is pure Python and will not reproduce them,
because Anserini tokenises and stems differently. Measured gap: 0.6523 here
against roughly 0.665 published.

**The obligation this creates:** report your BM25 as self-consistent rather than
leaderboard-comparable, and never place it beside a published figure as though
they were measured the same way.

**The tokenizer is deliberately simple** - lowercase, split on non-alphanumerics,
no stemming, no stopwords. Since the tokeniser is already what separates this from
Anserini, a half-hearted stemmer would make the gap harder to explain rather than
smaller. This is the weakest choice in the project and the easiest to revisit; the
corpus contains 10,110 distinct hyphenated terms, and `il-6` currently becomes
`il` plus `6`.

**No prefix is applied.** The BGE instruction is a message to a neural encoder. To
a word counter it is six common words present in every query.

### Stage three - `rerank.py`

**Run file in, run file out.** It imports no indexing, retrieval or training
code, and nothing in stage two changes to accommodate it. `evaluate.py` cannot
tell a cross-encoder was involved.

**No query prefix.** The BGE instruction tells a *bi*-encoder how to build a
standalone embedding. A cross-encoder reads query and document as one joined
sequence and was trained on raw MS MARCO pairs, so the instruction is text it
never saw. Recorded as `query_prefix_used=no`.

**The untouched tail is kept and offset below the reranked head.** Discarding
documents past rank k would destroy recall@100 whenever k is 50 and make the run
incomparable to its source. Retaining them with a computed offset keeps recall
identical *by construction*, which is what lets any nDCG change be attributed
entirely to reordering.

**The offset is computed from the actual score floor**, not from an assumed
range. Cross-encoder outputs are unbounded logits.

**Rank is never carried over from the source run.** `runfile.write_run` derives
it from score, so a reranker that produces only new scores cannot leave a stale
rank column behind.

**Measured result on SciFact**, off-the-shelf MS MARCO cross-encoders over the
dense base run at depth 50:

| Run | nDCG@10 | MRR@10 | recall@50 | recall@100 |
|---|---|---|---|---|
| bi-encoder base | 0.7127 | 0.6821 | 0.9317 | 0.9450 |
| + MiniLM-L-6 rerank | 0.6964 | 0.6644 | 0.9317 | 0.9450 |
| + MiniLM-L-12 rerank | 0.6965 | 0.6659 | 0.9317 | 0.9450 |

Both recall columns are identical across all three rows, which is the correctness
proof: the reranker only reordered. Both cross-encoders **lost** about 1.6 nDCG
points, and the larger model did not help.

**Interpretation.** MS MARCO is web search - short natural-language questions.
SciFact queries are declarative scientific claims. The cross-encoder is
mismatched on the *query* side, not the document side. This is a genuine finding
and it is the concrete argument for the distillation upgrade: a reranker trained
on your own domain rather than borrowed from web search.

---

## 4. Deviations from the README

**`src/runfile.py` was added.** The README names `tests/test_run_file.py` but no
module for it to test. Rank is assigned on write from score, so it can never
disagree; ties break on document id, so two runs from identical inputs are
byte-identical and you cannot chase a phantom regression.

**Four config keys were added** beyond the README's example block:
`hf_dataset` and `hf_qrels_dataset` avoid hardcoding repository names;
`graded_relevance` is needed by evaluation because NFCorpus is graded and SciFact
binary; `trainable` is load-bearing and makes the FiQA guard enforceable.

**`seed` and `dev_size` were added**, without which the dev sample is not
reproducible.

**`warmup_steps` replaces `warmup_ratio`** in the trainer arguments. Transformers
v5 renamed it; a float still means a ratio.

---

## 5. Open decisions, left deliberately

- **The BM25 tokenizer.** See above. Measure on dev before changing it.
- **`dev_size: 100`** was chosen for being a round number. At 100 queries a
  one-point difference is well inside noise, which is fine for catching gross
  breakage and not fine for choosing between two similar learning rates.
- **The unanswered-query warning prints rather than raises.** Easy to scroll past.
- **Nothing verifies that `doc_text()` produced identical strings for BM25 and for
  the index.** They call the same function so they agree today, but the index
  stores vectors rather than text, so a divergence would go unreported.
- **Reranker depth.** Only 50 has been measured. `scripts/run_stage3.sh` runs 50
  and 100 so the cost/ceiling trade is visible, but the numbers above are depth
  50 only.
- **Distillation is not implemented.** Stage three currently uses off-the-shelf
  cross-encoders, which is enough to answer whether reranking helps at all. A
  distilled student trained on `data/hard_negatives/` is the documented upgrade.

---

## 6. What each stage is allowed to change

This matters for reading results, because the two stages attack different failure
modes and one of them cannot move recall at all.

| | Fixes | recall@100 | nDCG@10 / MRR@10 |
|---|---|---|---|
| **Stage 2** fine-tuning | mode A: documents that never made the shortlist | **can move** | can move |
| **Stage 3** reranking | mode B: right document, wrong position | **frozen by construction** | can move |

So attributing credit is mechanical rather than a judgement call. Any recall
change came from stage two. Any nDCG change with recall held constant came from
stage three. If you ever see recall move after a rerank, you have a bug, not a
result.

---

## 7. Hyperparameter sweep results (NFCorpus, real labels, DEV only)

All runs share the frozen baseline in `scripts/sweep.py`: seed 42, epochs 3,
batch 64, lr 2e-5, mining range_min 10. Each row changes one thing.

| Run | Change | Dev nDCG@10 | vs baseline |
|---|---|---|---|
| baseline | - | 0.3616 | - |
| range_min 5 | harder negatives | 0.3611 | -0.05 |
| range_min 20 | safer negatives | 0.3613 | -0.03 |
| lr 1e-5 | half the LR | 0.3522 | -0.94 |
| lr 4e-5 | double the LR | **0.3695** | **+0.79** |
| batch 128, lr 4e-5 | linear scaling | 0.3654 | +0.38 |
| batch 256, lr 8e-5 | linear scaling | 0.3658 | +0.42 |
| epochs 5, lr 2e-5 | longer training | 0.3698 | +0.82 |

**Learning rate is the lever.** Three LR points line up in order: 1e-5 < 2e-5
< 4e-5. The guide's 2e-5 was too low for this data.

**Batch size added nothing once LR is accounted for.** The batch runs also
doubled or quadrupled the LR under the linear scaling rule. Batch 128 at 4e-5
(0.3654) did slightly *worse* than batch 64 at the same 4e-5 (0.3695). So the
batch-plan "gain" was the LR change, and the linear scaling rule did not hold
here.

**Negative difficulty (range_min) does not matter** across 5, 10, 20.

**lr 4e-5 for 3 epochs ties 2e-5 for 5 epochs.** That matches the idea that
lr and epochs trade off as total update size. The shorter run costs 60% of the
compute, so it is the pick.

**Chosen config:** epochs 3, batch 64, lr 4e-5, range_min 10.

**Caveat.** Single seed. Only the LR ordering is backed by more than one pair.
A higher LR also raises forgetting risk, so FiQA must be re-checked before this
config is called final.

### Forgetting check for the chosen config (FiQA dev)

Measured on FiQA **dev** because it could change the config choice. The base
model was re-measured on dev too, so all three rows share the same 500 queries.

| Model | NFCorpus dev gain | FiQA dev nDCG@10 | FiQA dev loss | Gain per point lost |
|---|---|---|---|---|
| base | - | 0.4070 | - | - |
| lr 2e-5, 3 epochs | +3.25 | 0.3845 | -2.25 | 1.44 |
| **lr 4e-5, 3 epochs** | **+4.04** | 0.3803 | -2.67 | **1.51** |

The higher learning rate forgets a little more (-0.42 extra on FiQA) but gains
more on NFCorpus (+0.79 extra). The trade is slightly better, so **lr 4e-5
stays the chosen config.**

Both extra amounts are under the ~1 point single-seed noise level. The honest
reading is that the higher learning rate did not make forgetting noticeably
worse, not that it is clearly better.

### Final test measurement (read once)

The config chosen on dev (epochs 3, batch 64, lr 4e-5, seed 42) was scored on
NFCorpus test exactly once.

| Model | Test nDCG@10 | Test recall@100 | Test MRR@10 |
|---|---|---|---|
| base | 0.3423 | 0.3090 | 0.5247 |
| guide defaults (2 epochs, lr 2e-5) | 0.3706 | 0.3634 | 0.5511 |
| **final config** | **0.3865** | **0.3837** | **0.5708** |
| gain over base | +4.42 | +7.47 | +4.61 |

The dev gain over base was +4.04 and the test gain is +4.42, so the choice made
on dev held up on unseen queries. Tuning added +1.59 nDCG over the guide's
defaults.

### Stage three on the final NFCorpus model (DEV only)

After fine-tuning, NFCorpus recall@100 rose from 0.309 to about 0.36, so the
shortlist finally held enough right answers for reranking to matter. The
ceiling was measured first by sorting each shortlist perfectly using the
answer key.

| Run | Dev nDCG@10 | Recall@100 |
|---|---|---|
| final fine-tuned model, no rerank | 0.3695 | 0.3606 |
| perfect reordering of top 50 (ceiling) | 0.6255 | - |
| + MS MARCO MiniLM-L-6, top 50 | 0.3427 | 0.3607 |
| + MedCPT biomedical cross-encoder, top 50 | 0.3564 | 0.3606 |
| + MedCPT biomedical cross-encoder, top 10 | 0.3690 | 0.3606 |
| hybrid: BM25 + fine-tuned (RRF) | 0.3253 | 0.3331 |

**Both rerankers made the ranking worse**, despite a large ceiling. The
biomedical model lost less (-1.31) than the web-search one (-2.68), which fits
the domain-mismatch explanation from SciFact.

**Shallower is safer but not better.** Restricting MedCPT to the top 10 removes
most of the damage (0.3690, against 0.3695 with no reranker) but does not
produce a gain. This row came from a depth test that was interrupted after the
first depth; depths 20 and 30 were not run.

**Not a bug.** One query traced end to end: the reranker did lift a relevant
document from rank 50 to rank 5, but it also placed a general "lifestyle
prevents cancer" paper at rank 2 with a score of 0.998. It trusts
topically-similar but wrong documents too much.

**Hybrid retrieval also hurt.** BM25 scores only 0.266 on NFCorpus dev, so
blending it in pulls the fine-tuned model down.

**Cost.** MiniLM took about 0.4 s per query and MedCPT about 2 s, against
under a millisecond for the bi-encoder search.

**Conclusion.** On this data, the fine-tuned bi-encoder alone is the best
system tested. Off-the-shelf rerankers do not capture the ceiling. A reranker
trained on this project's own mined negatives is the remaining untested option.
