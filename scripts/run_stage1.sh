#!/usr/bin/env bash
# Stage one - the baseline harness. No training code runs here.
#
# Freezes the split, runs BM25 and off-the-shelf bge-small-en-v1.5, and scores
# both. The output is two rows in results/metrics.csv and one question: does the
# dense nDCG@10 match the published MTEB figure for this model on this dataset?
#
# Do not proceed to stage two until it does. A baseline that is silently four
# points low turns into a four-point "gain from fine-tuning" that is entirely
# your own bug.
#
# Usage:  bash scripts/run_stage1.sh configs/scifact.yaml
#
# On Windows without a bash shell, run the commands below one at a time,
# substituting .venv\Scripts\python for $PY.

set -euo pipefail

CONFIG="${1:-configs/scifact.yaml}"
DATASET=$(basename "$CONFIG" .yaml)
PY="${PY:-.venv/Scripts/python}"

echo "=== step 1: freeze the split ==========================================="
# Refuses if the split already exists. That refusal is the point - re-freezing
# invalidates every row already in metrics.csv.
$PY src/ingest.py --config "$CONFIG" || echo "(split already frozen, continuing)"

echo
echo "=== baseline 1: BM25 ==================================================="
$PY src/bm25.py --config "$CONFIG"
$PY src/evaluate.py --config "$CONFIG" \
    --run "data/runs/${DATASET}_bm25_test.trec" \
    --model-name bm25_rank_bm25 \
    --training-arm baseline \
    --index-type bm25 \
    --query-prefix-used no \
    --notes "rank_bm25 pure python, not Anserini-comparable"

echo
echo "=== baseline 2: off-the-shelf bge-small ================================"
$PY src/build_index.py --config "$CONFIG" --tag base
$PY src/retrieve.py --config "$CONFIG" --tag base
$PY src/evaluate.py --config "$CONFIG" \
    --run "data/runs/${DATASET}_base_test.trec" \
    --model-name BAAI/bge-small-en-v1.5 \
    --training-arm baseline \
    --index-type flat \
    --query-prefix-used yes \
    --notes "off-the-shelf, no fine-tuning"

echo
echo "=== the prefix experiment (optional but cheap) ========================="
# Runs the identical model with the query prefix deliberately omitted. The gap
# between this row and the one above is what the prefix is worth on your data,
# measured rather than asserted. It is also the clearest demonstration that a
# calibration failure and a real improvement look identical in a single number.
$PY src/retrieve.py --config "$CONFIG" --tag base --no-prefix
$PY src/evaluate.py --config "$CONFIG" \
    --run "data/runs/${DATASET}_base_test_noprefix.trec" \
    --model-name BAAI/bge-small-en-v1.5 \
    --training-arm baseline \
    --index-type flat \
    --query-prefix-used no \
    --notes "ABLATION: prefix deliberately omitted, do not report as a baseline"

echo
echo "======================================================================="
echo "Stage one done. Now compare the dense nDCG@10 in results/metrics.csv"
echo "against the published MTEB number for BAAI/bge-small-en-v1.5 on"
echo "$DATASET. If they disagree by more than about a point, something in"
echo "the harness is wrong and stage two is not worth starting."
echo "======================================================================="
