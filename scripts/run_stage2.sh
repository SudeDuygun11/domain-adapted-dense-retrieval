#!/usr/bin/env bash
# Stage two - the fine-tuned bi-encoder, and the two-arm experiment.
#
# Runs the full pipeline TWICE on the same dataset:
#
#   Arm A (real_labels) - train on the human-labelled pairs BEIR ships.
#                         The ceiling: what a perfect answer key buys you.
#   Arm B (synthetic)   - train on generated queries only, pretending the
#                         labels do not exist. What you could actually do at a
#                         company with a corpus and no labels.
#
# The GAP between them is the headline finding. It is the cost of not having
# labels, and it is a real answer to a real question regardless of whether you
# hit any particular absolute number.
#
# Usage:  bash scripts/run_stage2.sh configs/nfcorpus.yaml
#
# Prerequisite: stage one must have run on this dataset. You need the frozen
# split, the base index, and the base-model row in metrics.csv to compare against.

set -euo pipefail

CONFIG="${1:-configs/nfcorpus.yaml}"
DATASET=$(basename "$CONFIG" .yaml)
PY="${PY:-.venv/Scripts/python}"

if [ ! -f "data/frozen_splits/${DATASET}/manifest.json" ]; then
    echo "No frozen split for ${DATASET}. Run stage one first:"
    echo "  bash scripts/run_stage1.sh ${CONFIG}"
    exit 1
fi

echo "=== step 2: synthetic query generation (Arm B only) ===================="
# The slowest step in the project. Generate once, save to disk, never regenerate.
# Arm A does not need this - it has real labels.
$PY src/generate_queries.py --config "$CONFIG" --n-per-doc 3 \
    || echo "(already generated, continuing)"

echo
echo "=== step 3: hard negative mining, both arms ============================"
$PY src/mine_negatives.py --config "$CONFIG" --arm synthetic
$PY src/mine_negatives.py --config "$CONFIG" --arm real_labels

for ARM in real_labels synthetic; do
    TAG="${ARM}_s42"
    echo
    echo "=== step 4: training arm ${ARM} ===================================="
    $PY src/train.py --config "$CONFIG" --arm "$ARM" --cached-loss   # needed on GPUs under ~16 GB

    echo
    echo "=== re-index and re-evaluate: ${ARM} ==============================="
    # A new model means a new embedding space, so the whole corpus must be
    # re-encoded. The old index is meaningless for this model.
    $PY src/build_index.py --config "$CONFIG" \
        --model "models/${DATASET}_${TAG}" --tag "$TAG"
    $PY src/retrieve.py --config "$CONFIG" --tag "$TAG"
    $PY src/evaluate.py --config "$CONFIG" \
        --run "data/runs/${DATASET}_${TAG}_test.trec" \
        --model-name "models/${DATASET}_${TAG}" \
        --trained-on "$DATASET" \
        --training-arm "$ARM" \
        --index-type flat \
        --query-prefix-used yes \
        --notes "stage 2, ${ARM}, seed 42"

    echo
    echo "=== forgetting check on FiQA: ${ARM} ==============================="
    bash scripts/run_forgetting.sh "models/${DATASET}_${TAG}" "$DATASET" "$ARM"
done

echo
echo "======================================================================="
echo "Stage two done. In results/metrics.csv you now have, for ${DATASET}:"
echo "  base -> real_labels   the ceiling"
echo "  base -> synthetic     what you get without labels"
echo "  and the FiQA rows showing what each cost you elsewhere."
echo
echo "The Arm A minus Arm B gap is your headline. Read section 6 of the README"
echo "for how to interpret the patterns before you write any of it up."
echo "======================================================================="
