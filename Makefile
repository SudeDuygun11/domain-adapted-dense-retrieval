# Convenience targets. Every one of them is a thin wrapper around a script that
# takes --config, so switching datasets is a variable rather than an edit.
#
#   make stage1                     # SciFact, the default
#   make stage1 CONFIG=configs/nfcorpus.yaml
#
# On Windows, override PY:  make test PY=.venv/Scripts/python

PY ?= .venv/Scripts/python
CONFIG ?= configs/scifact.yaml
DATASET = $(notdir $(basename $(CONFIG)))

.PHONY: install test ingest bm25 index retrieve stage1 clean-runs

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

ingest:
	$(PY) src/ingest.py --config $(CONFIG)

bm25:
	$(PY) src/bm25.py --config $(CONFIG)
	$(PY) src/evaluate.py --config $(CONFIG) \
		--run data/runs/$(DATASET)_bm25_test.trec \
		--model-name bm25_rank_bm25 --training-arm baseline \
		--index-type bm25 --query-prefix-used no \
		--notes "rank_bm25, not Anserini-comparable"

index:
	$(PY) src/build_index.py --config $(CONFIG) --tag base

retrieve:
	$(PY) src/retrieve.py --config $(CONFIG) --tag base
	$(PY) src/evaluate.py --config $(CONFIG) \
		--run data/runs/$(DATASET)_base_test.trec \
		--model-name BAAI/bge-small-en-v1.5 --training-arm baseline \
		--index-type flat --query-prefix-used yes \
		--notes "off-the-shelf"

stage1:
	bash scripts/run_stage1.sh $(CONFIG)

# Deliberately absent: any target that deletes data/frozen_splits or rewrites
# results/metrics.csv. Both are append-only contracts, and a Makefile target is
# far too easy a way to destroy a month of comparable numbers.
clean-runs:
	rm -f data/runs/*.trec
