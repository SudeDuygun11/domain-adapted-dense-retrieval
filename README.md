# Domain-Adapted Dense Retrieval

**Fine-tuning a small embedding model on a specialist corpus, with and without human relevance labels.**

Authors: **Buse Duygun** and **Sude Duygun**

---

## Summary

Search systems built on pretrained embedding models work well on general text and
worse on specialist text, where users and documents describe the same thing in
different words. We adapted the open embedding model `bge-small-en-v1.5` to a
biomedical corpus (NFCorpus) and measured how much each ingredient contributed.

| Question | Answer |
|---|---|
| Does fine-tuning help? | **Yes.** nDCG@10 rose from 0.3423 to **0.3865** (+4.42) on held-out test queries, and recall@100 from 0.3090 to 0.3837. |
| Can generated queries replace human labels? | **Not in our setup.** Two different query generators gained +0.08 and +0.51, against +2.83 for real labels (all three trained with the initial 2-epoch settings). |
| Was the gain bought at the cost of general ability? | **A modest cost.** The tuned model lost 2.67 points on an unrelated financial dataset (FiQA, dev). |
| Does a cross-encoder reranker add more? | **No, on this data.** Two off-the-shelf rerankers lowered nDCG@10. A perfect reordering would reach 0.63, so the room exists but these models do not capture it. |
| Which training setting mattered most? | **Learning rate.** Doubling it from 2e-5 to 4e-5 gained 0.79 dev points. Batch size and negative selection made no measurable difference. |

All tuning decisions were made on a development split. The final configuration was
read on the test split once.

---

## 1. Background

A retrieval system takes a question and returns the documents most likely to answer
it. Three kinds of model are common:

- **BM25** counts shared words, weighting rare words more heavily. It needs no
  training, but it finds nothing when the question and the answer use different
  vocabulary.
- **A bi-encoder** converts every document into a vector of numbers once, in
  advance. At search time it converts the question into a vector and returns the
  nearest documents. This takes milliseconds, which is why it can act as a search
  engine.
- **A cross-encoder** reads the question and one document together and scores
  their relevance directly. It is more accurate, but nothing can be precomputed, so
  it can only reorder a short list produced by something faster.

Pretrained bi-encoders are trained on broad text. In a specialist field there is a
gap between how people ask and how papers answer. In NFCorpus, a patient asks
*"does eating eggs raise cholesterol"* and the relevant paper is about *dietary
cholesterol and serum LDL concentrations*. The two share almost no words.

Fine-tuning on labelled question-document pairs can close that gap, but labels are
expensive. A company usually has a document collection and no labelled questions.

## 2. Problem and research questions

We asked four questions, in this order:

1. **How much does fine-tuning improve retrieval in a specialist domain?**
2. **How much of that gain survives when human labels are replaced by generated
   queries?** The gap between the two is the cost of having no labels.
3. **Does specialising cost anything on unrelated text?**
4. **Can a cross-encoder reranker improve the result further?**

## 3. Approach

### Failure modes

Every retrieval error falls into one of two kinds, and they need different fixes:

| | What happened | Can reranking fix it? |
|---|---|---|
| **A** | The right document is at rank 800 and never reached the shortlist. | No |
| **B** | The right document is at rank 7, present but in the wrong position. | Yes |

Recall@100 tells the two apart. A high value means the answers are in the
shortlist and only the order is wrong. A low value means the shortlist itself is the
problem.

### Pipeline

Three stages, each usable on its own. Nothing is passed between steps in memory:
every step reads files and writes files. Every retrieval system writes the same
TREC-format run file, so a single evaluation script scores all of them.

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
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
      single TEST measurement (read once)   reranking the final model (DEV)
      nDCG@10: 0.3423 → 0.3865  (+4.42)     no rerank    0.3695
      recall@100: 0.3090 → 0.3837 (+7.47)   MiniLM top50 0.3427
                                            MedCPT top50 0.3564
                                            → not adopted
```

### Design decisions that protect the results

- **Frozen, hashed evaluation splits.** The queries and relevance judgments we
  evaluate on are written to `data/frozen_splits/`, committed, and checked against
  a SHA-256 hash on every load. On our own run, silently dropping the 10 worst of
  300 queries raised nDCG@10 by 2.46 points, which is the same size as the effect
  we set out to measure.
- **Development and test are separate.** Dev is for every decision; test is read
  once at the end.
- **One evaluation implementation.** `src/evaluate.py` is the only file that
  computes a metric, it uses `pytrec_eval` (the reference `trec_eval` code), and it
  never imports a model.
- **One place applies the query prefix.** BGE expects a specific instruction
  string on queries. Omitting it raises no error, so `src/config.py` is the only
  module allowed to attach it.
- **Eval-only datasets cannot be trained on.** FiQA is marked untrainable in its
  config and `train.py` refuses it.

A longer account of every design choice, with the alternative considered and the
reason it lost, is in [DESIGN.md](DESIGN.md).

## 4. Data

Three datasets from the BEIR benchmark, each with a different job.

| Dataset | Domain | Documents | Test queries | Relevant docs per query | Role |
|---|---|---|---|---|---|
| **SciFact** | Scientific claims | 5,183 | 300 | 1.1 | Check that the pipeline is correct |
| **NFCorpus** | Nutrition and medicine (PubMed) | 3,633 | 323 | 38.2 | Fine-tune and improve here |
| **FiQA** | Financial questions | 57,638 | 648 | 2.6 | Never trained on; measures forgetting |

Development splits are the official ones where the dataset ships one (NFCorpus and
FiQA); SciFact ships none, so 100 training queries were held out with a fixed seed.

Recall is not comparable across datasets, because NFCorpus has about 38 relevant
documents per query while SciFact has about one.

## 5. Experimental protocol

- **Base model:** `BAAI/bge-small-en-v1.5` (33M parameters, 384 dimensions).
- **Metrics:** nDCG@10 (primary), recall@50, recall@100, MRR@10.
- **Fine-tuning:** contrastive training with `MultipleNegativesRankingLoss`. Each
  training example is a query, a relevant document, and six mined hard negatives.
  Other documents in the batch act as additional negatives.
- **Two arms.** *Arm A* trains on the human-labelled pairs BEIR ships. *Arm B*
  trains only on queries generated from the documents themselves, ignoring the
  labels. Both arms are matched to about 9,000 to 10,000 training rows so the
  comparison measures label quality rather than data volume.
- **Query generators for Arm B.** T5 (`BeIR/query-gen-msmarco-t5-base-v1`) and an
  instruction model (`Qwen2.5-1.5B-Instruct`) prompted for three different kinds of
  query. Generated queries were filtered: a query was kept only if the base model
  retrieved its own source document in the top 10.
- **Hyperparameter sweep.** One frozen baseline configuration; every run changes
  exactly one field. Same seed (42), same dev queries, same evaluation code.
- **Hardware:** one NVIDIA RTX A1000 laptop GPU with 4 GB of memory, using
  gradient caching so a batch of 64 fits.

## 6. Results

Unless a table says otherwise, scores are nDCG@10.

### 6.1 The pipeline reproduces the published model

The untuned model was scored on all three datasets before any training. Our values
agree with the figures published for this model to within about one point, which
is the check that our evaluation code measures what it claims to. (Published
leaderboard values are recomputed over time, so verify them before citing.)

| Dataset | BM25 (ours) | bge-small | Recall@100 |
|---|---|---|---|
| SciFact | 0.6523 | 0.7127 | 0.9450 |
| NFCorpus | 0.3062 | 0.3423 | 0.3090 |
| FiQA | 0.2175 | 0.3972 | 0.6892 |

Our BM25 uses `rank_bm25`, which will not reproduce published Anserini BM25 numbers
because tokenisation differs. It is self-consistent, not leaderboard-comparable.

The BGE query instruction made no measurable difference: SciFact scored 0.7127 with
it and 0.7132 without, NFCorpus 0.3423 with it and 0.3396 without.

The recall column shows where each dataset's problem lies. SciFact (0.945) is
failure mode B, right documents present and badly ordered. NFCorpus (0.309) is mode
A, most right documents never reach the shortlist. Fine-tuning attacks mode A and
reranking attacks mode B.

### 6.2 Real labels versus generated queries (test)

Trained with the initial settings (2 epochs, learning rate 2e-5).

| Training data | nDCG@10 | Gain over base | Recall@100 | FiQA nDCG@10 | FiQA change |
|---|---|---|---|---|---|
| None (base model) | 0.3423 | | 0.3090 | 0.3972 | |
| **Real labels (Arm A)** | **0.3706** | **+2.83** | 0.3634 | 0.3843 | −1.29 |
| T5 generated queries | 0.3431 | +0.08 | 0.3136 | 0.3813 | −1.59 |
| Qwen generated queries | 0.3474 | +0.51 | 0.3219 | not run | |

Real labels gained nearly three points. Generated queries gained almost nothing,
and the T5 model lost as much on FiQA as the real-label model while gaining
nothing in return.

Why generated queries fell short:

- **Too close to the source.** The share of a query's words that also appear in its
  source passage was 0.64 for T5 and 0.50 for Qwen. Queries that reuse the
  document's own vocabulary teach the model to match words it can already match.
  The training loss shows it: 0.73 for T5 and 0.47 for Qwen, against 3.25 for real
  labels. A loss that low means the task was too easy.
- **The quality filter removes hard examples.** It keeps a generated query only if
  the base model already finds its source, so it discards exactly the queries the
  model would need to learn from. It kept 96% of Qwen's queries.
- **More training did not help.** A third epoch raised Arm A on dev (below) and
  left the Qwen arm unchanged (0.3344 to 0.3340), so the gap is not an
  optimisation artefact.

On dev the ordering of the two generators reverses (T5 0.3392, Qwen 0.3344), and
both differences are inside single-seed noise. We therefore do not claim either
generator is better; we claim both fall well short of real labels.

### 6.3 Hyperparameter sweep (dev)

Baseline: real labels, 3 epochs, batch 64, learning rate 2e-5, negatives drawn
from rank 10. Each row changes one thing (batch-size rows also scale the learning
rate linearly with the batch).

| Change | Dev nDCG@10 | vs baseline |
|---|---|---|
| Baseline | 0.3616 | |
| Negatives start at rank 5 | 0.3611 | −0.05 |
| Negatives start at rank 20 | 0.3613 | −0.03 |
| Learning rate 1e-5 | 0.3522 | −0.94 |
| **Learning rate 4e-5** | **0.3695** | **+0.79** |
| Batch 128, learning rate 4e-5 | 0.3654 | +0.38 |
| Batch 256, learning rate 8e-5 | 0.3658 | +0.42 |
| 5 epochs, learning rate 2e-5 | 0.3698 | +0.82 |

- **Learning rate is the setting that matters.** The three values order cleanly:
  1e-5 is worst, 2e-5 is middle, 4e-5 is best.
- **Batch size added nothing once the learning rate is accounted for.** Batch 128
  at 4e-5 scored slightly below batch 64 at 4e-5.
- **Where negatives start does not matter.** Ranks 5, 10 and 20 land within 0.05.
  Widening the search window from 50 to 300 changed the negatives for 0.2% of
  queries, because the search stops as soon as it has found six.
- **Doubling the learning rate for 3 epochs matches the original rate for 5.** The
  shorter run takes 60% of the time, so we chose it.

**Chosen configuration: 3 epochs, batch 64, learning rate 4e-5.**

### 6.4 Forgetting on unrelated text (FiQA dev)

| Model | NFCorpus dev gain | FiQA dev nDCG@10 | FiQA change |
|---|---|---|---|
| Base | | 0.4070 | |
| Learning rate 2e-5, 3 epochs | +3.25 | 0.3845 | −2.25 |
| **Learning rate 4e-5, 3 epochs** | **+4.04** | 0.3803 | −2.67 |

The higher learning rate forgot 0.42 points more and gained 0.79 points more. Both
differences are inside single-seed noise, so the fair reading is that doubling the
learning rate did not make forgetting noticeably worse.

### 6.5 Final result (test, read once)

| Model | nDCG@10 | Recall@100 | MRR@10 |
|---|---|---|---|
| BM25 | 0.3062 | 0.2427 | 0.5085 |
| bge-small, untuned | 0.3423 | 0.3090 | 0.5247 |
| Fine-tuned, initial settings | 0.3706 | 0.3634 | 0.5511 |
| **Fine-tuned, chosen settings** | **0.3865** | **0.3837** | **0.5708** |

The dev gain over base was +4.04 and the test gain is +4.42, so the choices made on
dev held up on unseen queries. Tuning added 1.59 points over the initial settings.

The improvement is broad: 138 of 323 test queries improved, 60 got worse and 125
were unchanged (measured on the initial-settings model). Topic queries, headline
phrases and full questions all improved by between 2.6 and 3.5 points. Recall rose
by more than nDCG, which means fine-tuning pulled relevant documents into the
shortlist that the base model never reached.

Two cautions. About 40% of the total gain comes from the ten best-improved queries.
And a large per-query gain does not mean the query was solved: for two of the three
biggest improvements the top result was still wrong.

### 6.6 Reranking and hybrid retrieval

A cross-encoder can only reorder what retrieval found. We first measured how much
room exists by sorting each shortlist perfectly using the answer key, then tried
real rerankers.

**SciFact (test).** Recall was already 0.945, so reordering is the only lever.

| System | nDCG@10 | Recall@100 |
|---|---|---|
| bge-small alone | 0.7127 | 0.9450 |
| + MS MARCO MiniLM-L-6, top 50 | 0.6964 | 0.9450 |
| + MS MARCO MiniLM-L-12, top 50 | 0.6965 | 0.9450 |

**NFCorpus, final model (dev).**

| System | nDCG@10 | Recall@100 |
|---|---|---|
| Fine-tuned alone | **0.3695** | 0.3606 |
| Perfect reordering of the top 50 (upper bound) | 0.6255 | |
| + MS MARCO MiniLM-L-6, top 50 | 0.3427 | 0.3607 |
| + MedCPT (biomedical), top 50 | 0.3564 | 0.3606 |
| + MedCPT (biomedical), top 10 | 0.3690 | 0.3606 |
| BM25 blended in (reciprocal rank fusion) | 0.3253 | 0.3331 |

- **Recall never changed**, which confirms the rerankers only reordered documents.
- **Every reranker lowered nDCG@10 or left it flat**, even the one trained on
  biomedical text. Restricting it to the top 10 removed most of the damage but did
  not produce a gain.
- **The room is real.** A perfect reordering of the top 50 would reach 0.6255, so
  the shortlist holds far more ranking quality than these models extract.
- **A traced example** shows the mechanism: the reranker lifted one relevant
  document from rank 50 to rank 5, but also placed a general "lifestyle prevents
  cancer" paper at rank 2 with a score of 0.998. It trusts topically similar
  documents that do not answer the question.
- **Cost.** MiniLM took about 0.4 s per query and MedCPT about 2 s, against under a
  millisecond for the bi-encoder search.
- **Hybrid retrieval also hurt.** BM25 scores only 0.266 on NFCorpus dev, so
  blending it in pulls the better model down.

On this data the fine-tuned bi-encoder alone is the best system we tested.

## 7. Discussion

**What worked.** Fine-tuning on human labels is effective and its effect is far
larger than run-to-run noise. Recall improved more than nDCG, which is the signature
of fixing the shortlist rather than the ordering. Learning rate was the one setting
worth tuning.

**What did not.** Generated queries did not reproduce the value of labels, with
either generator. Off-the-shelf cross-encoders and hybrid retrieval both lowered
the score.

**Interpretation.** The gap between real and generated supervision survived our
attempts to explain it as a tuning problem, which points to the information content
of the generated queries and to the filter that selects them. We did not test this
directly. The most promising untested change is replacing the "base model finds its
source" filter with a relevance check that lets hard queries through.

### Limitations

- **Single seed.** Every run used seed 42. Single-seed differences under about one
  point should be read as noise. The learning-rate ordering rests on three points,
  and the other sweep conclusions on one comparison each.
- **Early results were read on test.** Before the sweep, baselines, ablations,
  rerankers and the first three fine-tuned models were measured on test. Absolute
  numbers carry some optimistic bias from that repeated inspection. Comparisons are
  less affected because every model was scored on the identical frozen split. The
  final model is the first one chosen on dev and read once on test.
- **Forgetting of the final model is measured on dev only.** Earlier models were
  measured on FiQA test.
- **Reranker experiments used dev queries**, so they are development results.
- **One base model and one specialist domain.** Conclusions may not transfer.
- **BM25 is not leaderboard-comparable** (see 6.1).

### Possible improvements

1. A relevance-based filter for generated queries, so that hard examples survive.
2. Training a reranker on this project's own mined hard negatives instead of using
   general-purpose ones.
3. Three or more seeds for the final configuration, to report a mean and spread.
4. Learning rates above 4e-5, since the top of that curve was not found.
5. An approximate-nearest-neighbour index sweep on FiQA, the only dataset large
   enough for it to matter.

## 8. Repository structure

```
.
├── README.md                this file
├── DESIGN.md                every design choice, alternatives, and measured evidence
├── requirements.txt
├── configs/                 one YAML per dataset (model, prefixes, mining, training)
├── src/
│   ├── config.py            loads configs; the only place query prefixes are applied
│   ├── ingest.py            downloads a dataset, freezes and hashes the split
│   ├── runfile.py           reads and writes TREC run files
│   ├── bm25.py              lexical baseline
│   ├── build_index.py       encodes the corpus and builds a FAISS index
│   ├── retrieve.py          encodes queries, searches, writes a run file
│   ├── evaluate.py          scores a run file against the frozen qrels
│   ├── generate_queries.py  synthetic query generation (T5 and instruction model)
│   ├── mine_negatives.py    hard negative mining
│   ├── train.py             contrastive fine-tuning
│   └── rerank.py            cross-encoder reranking
├── scripts/
│   ├── sweep.py             the hyperparameter sweep, on dev
│   ├── verify_results.py    re-scores every run file and checks metrics.csv
│   └── analysis_*.py        per-query, query-type and reranker-ceiling analyses
├── tests/                   22 tests for the split contract, run files and prefixes
├── notebooks/               annotated walkthroughs of each stage
├── data/
│   ├── frozen_splits/       committed: queries, judgments and hashes per dataset
│   ├── runs/                committed: every run file behind every reported number
│   ├── synthetic_queries/   committed: the generated queries used for Arm B
│   ├── raw/                 not committed: rebuilt by ingest.py --restore-corpus
│   ├── embeddings/          not committed
│   └── hard_negatives/      not committed: rebuilt by mine_negatives.py
├── models/                  not committed: rebuilt by train.py
├── results/metrics.csv      one row per experiment, append-only
└── docs/PROJECT_GUIDE.md    the original project guide
```

The notebooks in `notebooks/` (`stage1_walkthrough`, `stage2_walkthrough`,
`stage3_walkthrough`) explain each stage with runnable cells and are the best place
to start if you want to understand the code before running it.

## 9. Reproduce

Commands are for Windows PowerShell. On Linux or macOS replace `.venv\Scripts\python`
with `.venv/bin/python`.

### 9.1 Requirements

- Python 3.11
- For retraining: an NVIDIA GPU. We used a 4 GB laptop GPU. Without a GPU the
  quick check below still works, but training takes many hours.
- Internet access on first run, to download the datasets and models from Hugging Face.
- Disk space: the repository is about 800 MB (mostly run files). Each trained model
  takes about 0.9 GB including its training checkpoints, so the final model alone
  needs about 1 GB and the whole sweep about 11 GB. Indexes add about 1 GB.

### 9.2 Set up

```powershell
git clone https://github.com/SudeDuygun11/Project.git
cd Project
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
```

Install PyTorch **first**. Plain `pip install torch` gives a CPU-only build on
Windows. We used the CUDA 13.0 build; pick the one matching your driver from
[pytorch.org](https://pytorch.org/get-started/locally/).

```powershell
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -r requirements.txt
```

Check the GPU is visible (optional) and that the tests pass:

```powershell
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"
.venv\Scripts\python -m pytest tests -q
```

Expected: `22 passed`.

### 9.3 Fast check: rebuild every number from the committed run files

No GPU, no models and no retraining. Every system in this project writes a run
file, so the whole results table can be recomputed from `data/runs/` alone.

```powershell
.venv\Scripts\python scripts\verify_results.py
```

Expected last lines:

```
35 rows re-scored, 0 mismatched, 0 skipped (of 35 in results\metrics.csv)
Every recorded metric is reproduced from the committed run files.
```

This proves that each number in `results/metrics.csv` follows from a committed run
file, scored by the same code against a split whose files match their recorded
hashes. It does not prove that retraining reproduces the run files; that is 9.4.

### 9.4 Full rerun

The scripts append to `results/metrics.csv`. To keep the committed table untouched,
send a rerun to a scratch file first. Note that a rerun also overwrites any
committed run file of the same name; restore them afterwards with
`git checkout data/runs`.

```powershell
$env:METRICS_CSV = "results/metrics_repro.csv"
$PY = ".venv\Scripts\python"
```

**Step 1. Restore the corpora.** The split files are committed but the corpora are
not. This downloads each corpus and checks it against the hash in the manifest; it
refuses to keep a corpus that does not match.

```powershell
foreach ($d in "scifact","nfcorpus","fiqa") { & $PY src/ingest.py --config configs/$d.yaml --restore-corpus }
```

Expected: `sha256 matches the manifest. OK` for each. (`ingest.py` without this flag
deliberately refuses to run once a split exists, because re-freezing would
invalidate every recorded result.)

**Step 2. Baselines** (repeat with `scifact` and `fiqa` for the other two datasets).

```powershell
& $PY src/bm25.py --config configs/nfcorpus.yaml
& $PY src/evaluate.py --config configs/nfcorpus.yaml --run data/runs/nfcorpus_bm25_test.trec --model-name bm25_rank_bm25 --training-arm baseline --index-type bm25 --query-prefix-used no
& $PY src/build_index.py --config configs/nfcorpus.yaml --tag base
& $PY src/retrieve.py --config configs/nfcorpus.yaml --tag base
& $PY src/evaluate.py --config configs/nfcorpus.yaml --run data/runs/nfcorpus_base_test.trec --model-name BAAI/bge-small-en-v1.5 --training-arm baseline --index-type flat --query-prefix-used yes
```

Expected NFCorpus: BM25 nDCG@10 0.3062, bge-small 0.3423.

**Step 3. Mine hard negatives and train the final model (Arm A).** The NFCorpus
config already holds the chosen settings (3 epochs, learning rate 4e-5).
`--cached-loss` is needed on GPUs under about 16 GB.

```powershell
& $PY src/mine_negatives.py --config configs/nfcorpus.yaml --arm real_labels --max-positives-per-query 4
& $PY src/train.py --config configs/nfcorpus.yaml --arm real_labels --cached-loss --eval-steps 150 --tag final
```

Training takes about one hour on our GPU. Then index and score the new model. Use
dev for anything that informs a decision:

```powershell
& $PY src/build_index.py --config configs/nfcorpus.yaml --model models/nfcorpus_final --tag final
& $PY src/retrieve.py --config configs/nfcorpus.yaml --tag final --split dev
& $PY src/evaluate.py --config configs/nfcorpus.yaml --split dev --run data/runs/nfcorpus_final_dev.trec --model-name models/nfcorpus_final --trained-on nfcorpus --training-arm real_labels --index-type flat --query-prefix-used yes
```

Expected dev nDCG@10 near 0.3695. Then read test once:

```powershell
& $PY src/retrieve.py --config configs/nfcorpus.yaml --tag final --split test
& $PY src/evaluate.py --config configs/nfcorpus.yaml --run data/runs/nfcorpus_final_test.trec --model-name models/nfcorpus_final --trained-on nfcorpus --training-arm real_labels --index-type flat --query-prefix-used yes
```

Expected test nDCG@10 near 0.3865.

**Step 4. Forgetting check.** Score the base and tuned models on FiQA dev.

```powershell
& $PY src/build_index.py --config configs/fiqa.yaml --tag base
& $PY src/retrieve.py --config configs/fiqa.yaml --tag base --split dev
& $PY src/build_index.py --config configs/fiqa.yaml --model models/nfcorpus_final --tag nf_final
& $PY src/retrieve.py --config configs/fiqa.yaml --tag nf_final --split dev
```

Score each run file with `evaluate.py --split dev`. Expected: about 0.4070 for the
base model and 0.3803 for the tuned one.

**Step 5. Synthetic arm (Arm B).** The generated queries are committed
(`data/synthetic_queries/`), so you can skip generation and train directly.
Regenerating gives different queries (see 9.6).

```powershell
& $PY src/mine_negatives.py --config configs/nfcorpus.yaml --arm synthetic
& $PY src/train.py --config configs/nfcorpus.yaml --arm synthetic --cached-loss --epochs 2 --lr 2e-5 --eval-steps 100 --tag synthetic_t5
```

For the Qwen queries add `--synthetic-name nfcorpus_qwen` to the mining command and
`--negatives-tag base_nfcorpus_qwen` to the training command. To regenerate
queries instead (about 7 minutes for T5 and about 3 hours for Qwen on our GPU):

```powershell
& $PY src/generate_queries.py --config configs/nfcorpus.yaml --n-per-doc 3 --batch-size 8 --force
& $PY src/generate_queries.py --config configs/nfcorpus.yaml --generator Qwen/Qwen2.5-1.5B-Instruct --n-per-doc 3 --batch-size 8 --out-name nfcorpus_qwen --force
```

**Step 6. Hyperparameter sweep (dev only).** Each plan changes one field of the
frozen baseline. About 12 GPU-hours for all of them.

```powershell
& $PY scripts/sweep.py --config configs/nfcorpus.yaml --plan epochs
& $PY scripts/sweep.py --config configs/nfcorpus.yaml --plan lr
& $PY scripts/sweep.py --config configs/nfcorpus.yaml --plan batch
```

The `range_min` plan needs its negatives mined first:

```powershell
& $PY src/mine_negatives.py --config configs/nfcorpus.yaml --arm real_labels --max-positives-per-query 4 --range-min 5
& $PY src/mine_negatives.py --config configs/nfcorpus.yaml --arm real_labels --max-positives-per-query 4 --range-min 20
& $PY scripts/sweep.py --config configs/nfcorpus.yaml --plan range_min
```

**Step 7. Reranking.**

```powershell
& $PY src/rerank.py --config configs/nfcorpus.yaml --split dev --run data/runs/nfcorpus_final_dev.trec --top-k 50 --model ncbi/MedCPT-Cross-Encoder
```

Score the output with `evaluate.py`. The reranker-ceiling and query-type analyses:

```powershell
& $PY scripts/analysis_reranker_ceiling.py
& $PY scripts/analysis_query_types.py
```

### 9.5 Exact settings of each reported model

| Model | Training data | Negatives file tag | Epochs | Learning rate | `--eval-steps` |
|---|---|---|---|---|---|
| Arm A, initial | Real labels | `base` (4 per query) | 2 | 2e-5 | 100 |
| Arm B, T5 | T5 queries | `base` (2 per query) | 2 | 2e-5 | 100 |
| Arm B, Qwen | Qwen queries | `base_nfcorpus_qwen` | 2 | 2e-5 | 100 |
| **Final** | Real labels | `base` (4 per query) | 3 | 4e-5 | 150 |

All use batch 64 with `--cached-loss`, seed 42, six negatives per query, and
mining from ranks 10 to 50. Because the config now holds the final settings, the
initial models need `--epochs 2 --lr 2e-5` passed explicitly.

### 9.6 What to expect: time and determinism

| Step | Time on our GPU |
|---|---|
| Tests and the fast check | about a minute each |
| Restore all three corpora | a few minutes |
| Encode NFCorpus / FiQA | 32 seconds / 5 minutes |
| One training run | 40 to 90 minutes |
| T5 / Qwen query generation | 7 minutes / about 3 hours |
| Whole sweep | about 12 hours |

**Retraining will not reproduce our numbers to the last decimal.** Seeds are fixed,
but GPU training uses half precision and non-deterministic kernels, and the batch
is assembled with gradient caching. Expect a rerun to land within about one nDCG
point of the published value. That is the same noise level we use to judge our own
results, and the conclusions here rest on effects larger than that: the +4.4 gain
from fine-tuning and the gap to generated queries. The generated queries themselves
cannot be regenerated identically, because sampling was not seeded when they were
made. This is why they are committed.

The fast check in 9.3, by contrast, is exact.

## 10. Authors and contributions

This project was carried out jointly by **Buse Duygun** and **Sude Duygun**. The
work was divided into workstreams of equal total size, measured by lines of
authored code, scripts, configuration and documentation: about 3,500 lines each.
Data, run files, results and the original project guide are not counted. Buse took
the data, supervision and reporting side of the pipeline. Sude took the modelling
and evaluation side.

### Buse Duygun: data, supervision and reporting

| Workstream | Files |
|---|---|
| Data foundation: dataset download, frozen and hashed splits, id handling, configs, run-file format, the stage-one script, and the tests that protect them | `src/ingest.py`, `src/config.py`, `src/runfile.py`, `configs/`, `tests/`, `scripts/run_stage1.sh`, `data/frozen_splits/` |
| Synthetic query generation with the T5 and Qwen generators, including the quality filters | `src/generate_queries.py`, `data/synthetic_queries/` |
| Error analysis: per-query breakdown of the gains and the split by query type | `scripts/analysis_per_query.py`, `scripts/analysis_query_types.py` |
| Reproducibility: corpus restore, results verification, dependency list | `scripts/verify_results.py`, `ingest.py --restore-corpus`, `requirements.txt`, `.gitignore` |
| Documentation: the project report and the design record | `README.md`, `DESIGN.md`, `docs/` |

### Sude Duygun: modelling and evaluation

| Workstream | Files |
|---|---|
| Evaluation harness: metrics, the run-file scorer, the results table | `src/evaluate.py`, `results/` |
| Baselines: BM25, the dense index and retrieval, the query-prefix ablation | `src/bm25.py`, `src/build_index.py`, `src/retrieve.py` |
| Hard negative mining | `src/mine_negatives.py` |
| Fine-tuning and the forgetting check on FiQA | `src/train.py`, `scripts/run_forgetting.sh` |
| Hyperparameter sweep and the final test measurement | `scripts/sweep.py` |
| Reranking and hybrid retrieval, including the reranker-ceiling analysis | `src/rerank.py`, `scripts/analysis_reranker_ceiling.py` |
| Pipeline scripts and every run file behind the reported numbers | `scripts/run_stage2.sh`, `scripts/run_stage3.sh`, `Makefile`, `data/runs/` |
| Annotated walkthrough notebooks for each stage | `notebooks/` |

### Shared

Both authors defined the research questions, agreed the experimental design and
the rule that decisions are made on dev and test is read once, interpreted the
results together, and reviewed each other's work.

The two shares differ in kind rather than size. Sude ran more of the GPU
experiments, mainly the sweep, which took about twelve hours of training. Buse
wrote more of the documentation and analysis.

## 11. References

- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I. *BEIR: A
  Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.*
  NeurIPS Datasets and Benchmarks, 2021.
- Wang, K., Thakur, N., Reimers, N., Gurevych, I. *GPL: Generative Pseudo Labeling
  for Unsupervised Domain Adaptation of Dense Retrieval.* NAACL, 2022.
- Xiao, S., Liu, Z., Zhang, P., Muennighoff, N. *C-Pack: Packaged Resources To
  Advance General Chinese Embedding* (the BGE models). 2023.
- Muennighoff, N., Tazi, N., Magne, L., Reimers, N. *MTEB: Massive Text Embedding
  Benchmark.* EACL, 2023.
- Jin, Q., et al. *MedCPT: Contrastive Pre-trained Transformers with Large-scale
  PubMed Search Logs for Zero-shot Biomedical Information Retrieval.*
  Bioinformatics, 2023.
- Gao, L., Zhang, Y., Han, J., Callan, J. *Scaling Deep Contrastive Learning Batch
  Size under Memory Limited Setup.* 2021 (gradient caching).
- Goyal, P., et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.*
  2017 (linear learning-rate scaling).
- Boteva, V., Gholipour, D., Sokolov, A., Riezler, S. *A Full-Text Learning to Rank
  Dataset for Medical Information Retrieval.* ECIR, 2016 (NFCorpus).
- Wadden, D., et al. *Fact or Fiction: Verifying Scientific Claims.* EMNLP, 2020
  (SciFact).
- Maia, M., et al. *WWW'18 Open Challenge: Financial Opinion Mining and Question
  Answering.* 2018 (FiQA).

The datasets are distributed under their own licences; see the BEIR repository
before redistributing them.

## 12. Acknowledgements

Parts of the code and documentation were developed with the assistance of Claude
(Anthropic), used through Claude Code.
