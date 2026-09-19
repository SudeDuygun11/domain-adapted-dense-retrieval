#!/usr/bin/env bash
# The forgetting check: what did fine-tuning cost you everywhere else?
#
# Fine-tuning a model to be good at nutrition can make it worse at everything
# else. It specialises. If you do not check for this, a reader will ask and you
# will have no answer.
#
# FiQA is the probe because financial text is maximally distant from biology and
# nutrition, so any drop is a clean signal rather than a domain-overlap artefact.
# It is marked eval-only in its config, and train.py refuses it by name.
#
# THE NUMBER IS A DELTA, NEVER AN ABSOLUTE. "0.38" on its own means nothing.
# "0.40 base -> 0.38 tuned" means fine-tuning cost you two points elsewhere.
# So this script measures the BASE model on FiQA first, if that row is missing.
#
# Strictly, what this measures is out-of-domain generalisation loss rather than
# catastrophic forgetting in the continual-learning sense. Call it what it is;
# a reviewer who works on continual learning will notice.
#
# Usage:  bash scripts/run_forgetting.sh models/nfcorpus_synthetic_s42 nfcorpus synthetic

set -euo pipefail

MODEL_PATH="${1:?usage: run_forgetting.sh <model_path> <trained_on> <arm>}"
TRAINED_ON="${2:?}"
ARM="${3:?}"
CONFIG="configs/fiqa.yaml"
PY="${PY:-.venv/Scripts/python}"
TAG="$(basename "$MODEL_PATH")"

if [ ! -f "data/frozen_splits/fiqa/manifest.json" ]; then
    echo "No frozen FiQA split. Run: $PY src/ingest.py --config $CONFIG"
    exit 1
fi

# --- the baseline, without which the tuned number means nothing --------------
if ! grep -q "^fiqa,BAAI/bge-small-en-v1.5,,baseline," results/metrics.csv 2>/dev/null; then
    echo "=== FiQA base model (no row found, measuring it now) ==============="
    $PY src/build_index.py --config "$CONFIG" --tag base
    $PY src/retrieve.py --config "$CONFIG" --tag base
    $PY src/evaluate.py --config "$CONFIG" \
        --run data/runs/fiqa_base_test.trec \
        --model-name BAAI/bge-small-en-v1.5 \
        --training-arm baseline --index-type flat --query-prefix-used yes \
        --notes "forgetting reference: base model, never fine-tuned"
else
    echo "=== FiQA base row already in metrics.csv, reusing it ==============="
fi

# --- the tuned model on the same frozen split -------------------------------
echo
echo "=== FiQA with the ${TRAINED_ON}/${ARM} model ======================"
$PY src/build_index.py --config "$CONFIG" --model "$MODEL_PATH" --tag "$TAG"
$PY src/retrieve.py --config "$CONFIG" --tag "$TAG"
$PY src/evaluate.py --config "$CONFIG" \
    --run "data/runs/fiqa_${TAG}_test.trec" \
    --model-name "$MODEL_PATH" \
    --trained-on "$TRAINED_ON" \
    --training-arm "$ARM" \
    --index-type flat \
    --query-prefix-used yes \
    --notes "FORGETTING CHECK: tuned on ${TRAINED_ON}, evaluated on fiqa"

echo
echo "Now subtract. dataset=fiqa, trained_on='' is your reference;"
echo "dataset=fiqa, trained_on=${TRAINED_ON} is the tuned model."
echo
echo "Judge the TRADE, as a ratio, not the two numbers separately:"
echo "  +6 on ${TRAINED_ON} / -2 on fiqa   good trade, easy to defend"
echo "  +6 / -1                            excellent"
echo "  +2 / -6                            bad, you damaged more than you built"
echo "  +6 / -15                           over-specialised: lower the LR or epochs"
echo
echo "Some loss is expected and fine. The model has 33M parameters and fixed"
echo "capacity - a model that loses nothing anywhere probably learned nothing."
