"""Hyperparameter sweep, measured on DEV, against one frozen baseline config.

Every number in results/metrics.csv up to the epochs sweep came from running the
guide's defaults once. This script is what makes a choice a choice: train a
config, re-index, retrieve, and score it on the dev split, then move to the
next - always changing exactly one thing relative to BASELINE.

DEV, never test. A configuration chosen by watching the test score has been
tuned on the number you intend to report, and the number stops being an
estimate of unseen performance. Test is looked at once, at the end, with the
winner.

Usage:
    .venv/Scripts/python scripts/sweep.py --config configs/nfcorpus.yaml --plan epochs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, replace

# sys.executable, not a hardcoded path. On Windows, subprocess needs the .exe
# extension that a shell would have supplied, and ".venv/Scripts/python" fails
# with a bare WinError 2. This also guarantees every child runs under the same
# interpreter as the sweep itself.
PY = sys.executable


# =============================================================================
# ONE FROZEN BASELINE. Every run in every plan is BASELINE with exactly one
# field changed. This is what makes runs across different sweeps comparable to
# each other, not just to the row directly above them:
#
#   - same frozen split, same dev queries        (fixed by --config; never varies)
#   - same seed                                   (pinned here, not left to cfg)
#   - same evaluation split and evaluator         (dev; train.py's evaluator is
#                                                   identical across all runs)
#   - same arm and negative source, unless THAT is the axis under test
#   - same epochs/batch/lr, unless THAT is the axis under test
#
# Do not edit BASELINE to "try something". Add a new plan entry that overrides
# one field via `replace(BASELINE, field=...)` instead - that is the whole
# point of keeping one object as the reference everything else is a delta from.
# =============================================================================

@dataclass(frozen=True)
class RunSpec:
    name: str
    arm: str = "real_labels"
    negatives_tag: str = "base"
    epochs: int = 3          # winner of the epochs sweep, not the guide's 2
    batch_size: int = 64     # the config default; see LR_SCALING below
    lr: float = 2.0e-5       # the config default, paired with batch_size=64
    eval_steps: int = 150
    seed: int = 42


BASELINE = RunSpec(name="baseline")


# -----------------------------------------------------------------------------
# Batch size / learning rate: linear scaling, computed - not independently
# guessed, and not (yet) auto-tuned at runtime.
#
# The linear scaling rule (Goyal et al., 2017): when batch size scales by k,
# scale the learning rate by k too. The reasoning is mechanical, not folklore -
# a k-times-larger batch averages its gradient over k times more examples, which
# shrinks the gradient's variance by k without changing its expected direction,
# so the step size can grow by the same k without becoming a worse update.
#
# So this is FIXED, not DYNAMIC: a plain function you call while building a
# RunSpec, evaluated once, up front, from BASELINE's own (batch_size, lr) pair.
# It is not read at train time and does not adapt during a run. The natural
# next step - making train.py compute this itself from whatever --batch-size it
# is given, so a batch-size override never needs a matching --lr override typed
# alongside it - is exactly the "dynamic" version you asked NOT to build yet.
# scaled_lr() is written as the seam that upgrade will slot into.
# -----------------------------------------------------------------------------

def scaled_lr(batch_size: int, base: RunSpec = BASELINE) -> float:
    return base.lr * (batch_size / base.batch_size)


def at(**overrides) -> RunSpec:
    """BASELINE with exactly the given fields changed. Everything else pinned."""
    return replace(BASELINE, **overrides)


PLANS: dict[str, list[RunSpec]] = {
    # Direct evidence motivated this one: BASELINE's own epochs=3 came from
    # here. Kept for the record and for re-running if the negatives or data
    # ever change.
    "epochs": [
        at(name="armA_ep2", epochs=2),
        at(name="armA_ep3", epochs=3),
        at(name="armB_qwen_ep2", arm="synthetic",
           negatives_tag="base_nfcorpus_qwen", epochs=2),
        at(name="armB_qwen_ep3", arm="synthetic",
           negatives_tag="base_nfcorpus_qwen", epochs=3),
    ],
    # One run, evaluated at every epoch boundary, to see where the curve turns
    # rather than compare two more arbitrary points.
    #
    # Caveat: the LR schedule decays over all 5 epochs, so this run's epoch-3
    # checkpoint is NOT the same model as the "epochs" plan's epoch-3 run. The
    # intermediate points show the SHAPE of the curve, not interchangeable
    # models - do not average them together.
    #
    # 9,321 rows at batch 64 is 146 steps/epoch, so eval_steps=146 lands one
    # measurement on each boundary.
    "epochs5": [
        at(name="armA_ep5", epochs=5, eval_steps=146),
    ],
    # The guide calls batch size the single most important hyperparameter,
    # because in MultipleNegativesRankingLoss every other item in the batch
    # acts as an extra negative - so batch size IS the negative count.
    # --cached-loss exists precisely to make these fit a 4GB GPU.
    #
    # lr is DERIVED via scaled_lr(), not chosen. At batch 128 that is 4.0e-5;
    # at batch 256, 8.0e-5. If either number looks wrong for this model, that
    # is itself a finding about the scaling rule, not a typo to silently fix.
    "batch": [
        at(name="armA_bs128", batch_size=128, lr=scaled_lr(128)),
        at(name="armA_bs256", batch_size=256, lr=scaled_lr(256)),
    ],
    # range_min controls negative difficulty far more than range_max does - see
    # the range_max entry above. The mining loop starts scanning at range_min
    # and stops after collecting negatives_per_query candidates, so range_min is
    # what actually determines which documents get picked, for EVERY query, not
    # just the rare ones that exhaust the window. Lower = harder negatives but
    # more risk of unlabelled-relevant contamination (measured: at range_min=5,
    # 5,741 candidates were excluded as known-relevant vs 4,089 at 10 and 2,623
    # at 20 - confirms the contamination gradient is real).
    #
    # Requires re-mining first (see the two `mine_negatives.py --range-min ...`
    # commands this plan assumes have already been run), which is why this
    # references pre-built negatives-tag files rather than mining inline.
    "range_min": [
        at(name="rangemin5", negatives_tag="base_rangemin5"),
        at(name="rangemin20", negatives_tag="base_rangemin20"),
    ],
    # A learning-rate sweep AT THE BASELINE batch size, independent of the
    # scaling rule above - this tests whether 2e-5 itself was well chosen for
    # batch 64, which the "batch" plan assumes rather than checks.
    "lr": [
        at(name="armA_lr1e5", lr=1.0e-5),
        at(name="armA_lr4e5", lr=4.0e-5),
    ],
}


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd), flush=True)
    # Output is NOT captured: it streams straight into the sweep log so progress
    # is visible during a multi-hour run instead of appearing only at the end.
    # Capturing it also meant a child's progress-bar characters could not be
    # decoded by the Windows default codepage, which raised in the reader
    # thread rather than failing the command.
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--plan", required=True, choices=sorted(PLANS))
    ap.add_argument("--dataset", default="nfcorpus")
    args = ap.parse_args()

    print(f"BASELINE: {BASELINE}")
    print(f"Every run below is BASELINE with exactly the listed field(s) "
          f"overridden.\n")

    for spec in PLANS[args.plan]:
        tag = f"sweep_{spec.name}"
        model_dir = f"models/{args.dataset}_{tag}"
        t0 = time.perf_counter()
        print(f"\n=== {spec.name}  (seed={spec.seed}, arm={spec.arm}, "
              f"epochs={spec.epochs}, batch={spec.batch_size}, "
              f"lr={spec.lr:.2e}) ===", flush=True)

        run([PY, "-u", "src/train.py", "--config", args.config,
             "--arm", spec.arm,
             "--negatives-tag", spec.negatives_tag,
             "--cached-loss",
             "--epochs", str(spec.epochs),
             "--batch-size", str(spec.batch_size),
             "--lr", str(spec.lr),
             "--seed", str(spec.seed),
             "--eval-steps", str(spec.eval_steps),
             "--tag", tag])
        run([PY, "-u", "src/build_index.py", "--config", args.config,
             "--model", model_dir, "--tag", tag])
        # Same split every time: dev. Never test.
        run([PY, "-u", "src/retrieve.py", "--config", args.config,
             "--tag", tag, "--split", "dev"])
        run([PY, "-u", "src/evaluate.py", "--config", args.config,
             "--split", "dev",
             "--run", f"data/runs/{args.dataset}_{tag}_dev.trec",
             "--model-name", model_dir, "--trained-on", args.dataset,
             "--training-arm", spec.arm, "--index-type", "flat",
             "--query-prefix-used", "yes",
             "--notes",
             f"SWEEP {args.plan}: {spec.name} "
             f"(epochs={spec.epochs} batch={spec.batch_size} "
             f"lr={spec.lr:.2e} seed={spec.seed})"])

        print(f"=== {spec.name} done in {(time.perf_counter()-t0)/60:.0f} min ===",
              flush=True)

    print("\nCompare the [SPLIT=DEV] rows in results/metrics.csv.")
    print("Only the winner gets measured on test, and only once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
