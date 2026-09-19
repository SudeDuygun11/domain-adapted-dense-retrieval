#!/usr/bin/env bash
# Stage three - the cross-encoder reranker.
#
# UPSIDE, NOT A REQUIREMENT. Do this after stages one and two, or not at all.
#
# It attacks failure mode B only: the right document is in the shortlist but in
# the wrong position. Recall@100 is mathematically unchanged by reranking the
# top 100 - same documents, different order.
#
# So check whether it is worth building BEFORE building it. Look at the source
# run's recall@100 in results/metrics.csv:
#
#   recall@100 ~ 0.94   the answers are in the pile, just badly ordered.
#                       A reranker has real room to work.
#   recall@100 ~ 0.55   half the answers never made the shortlist. No amount of
#                       reranking recovers them. Fix retrieval or chunking.
#
# Usage:  bash scripts/run_stage3.sh configs/scifact.yaml data/runs/scifact_base_test.trec

set -euo pipefail

CONFIG="${1:?usage: run_stage3.sh <config> <source_run.trec>}"
SOURCE_RUN="${2:?}"
PY="${PY:-.venv/Scripts/python}"
SOURCE=$(basename "$SOURCE_RUN" .trec)

if [ ! -f "$SOURCE_RUN" ]; then
    echo "No run file at $SOURCE_RUN. Stage three reranks an existing run."
    exit 1
fi

# Two depths, because the comparison is the interesting part. If recall@50 and
# recall@100 are close in your source run - on SciFact they are 0.9317 vs
# 0.9450 - then top 50 buys nearly the same ceiling for half the compute, and
# showing that is a better engineering result than just picking one.
for K in 50 100; do
    echo
    echo "=== reranking top ${K} ================================================"
    $PY src/rerank.py --config "$CONFIG" --run "$SOURCE_RUN" --top-k "$K"

    $PY src/evaluate.py --config "$CONFIG" \
        --run "data/runs/${SOURCE}_rerank_top${K}.trec" \
        --model-name cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --training-arm baseline \
        --index-type "rerank_top${K}" \
        --query-prefix-used no \
        --notes "stage 3 reranker over ${SOURCE}, top ${K}"
done

echo
echo "======================================================================="
echo "Compare against the source run's row in results/metrics.csv."
echo
echo "THE CHECK THAT MATTERS: recall@100 must be IDENTICAL to the source."
echo "If it moved, the reranker dropped or added documents and has a bug."
echo "nDCG@10 and mrr@10 are where a correct reranker shows up."
echo
echo "Report the latency too. A cross-encoder is orders of magnitude slower per"
echo "query than the bi-encoder search it sits on top of, and a gain that costs"
echo "300ms per query is a different proposition from one that costs 5ms."
echo "======================================================================="
