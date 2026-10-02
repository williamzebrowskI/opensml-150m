# Document shuffle and larger validation continuation

Prepared 2026-09-23. Full-model execution is left to the user.

The lower-LR run stopped at step **70,710**, **7,530,332,160 tokens**, with
the LR cap/floor at **5e-6**. Its original validation best is **2.825756658**
at step 67,139. This experiment preserves that run and resumes its **latest**
checkpoint into `runs/full_15b_five_mac_shuffled_v1`.

## Changes

- Main validation increases from 16 to 256 batches per source: **4,194,304
  tokens total**, 1,048,576 for each of the four sources. Batch size 2, context
  2048, source weights and content-hash split rules remain unchanged.
- Training uses a deterministic **256 accepted-document shuffle buffer per
  source**. It starts from the saved HF positions and existing token leftovers.
  Pending documents, draw counters, dedup history and consumed token offsets
  are checkpointed, including when prefetch is ahead of training. There is no
  source recycling and no materialized dataset/token shards; bounded pending
  document text is included in resume checkpoints.
- LR remains **5e-6**, with no rewarm, additional cut, optimizer reset, tokenizer
  change, architecture change, mixture change or batch-size change.

The larger validation set is a different metric. On the first launch, the worker
evaluates the unchanged resumed model **before its first optimizer update**,
saves that baseline as the new run's initial best, and switches the scheduler's
comparison baseline while retaining its LR cap and lifetime counters. The old
scheduler history is retained in transition provenance. Subsequent resumes
restore this state without repeating the baseline reset.

The original validation set is independently rebuilt and its historical
fingerprint checked. Its score is logged as `[eval-legacy STEP]` and saved in
checkpoint metadata, so it remains comparable with the previous **2.825756658**
best. The new expanded `[eval STEP]` score selects checkpoints and drives the
scheduler. Do not splice these two different metrics into one loss curve.

Larger evaluation improves measurement; it does not itself train the model or
guarantee a lower loss. Bounded shuffling disrupts local ordering but is not a
global random permutation and does not establish that the old ordering caused
the plateau. Keep the final test split untouched.

## Commands

Preview the derived settings without loading a model, accessing peers or writing
checkpoints:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/scripts/launch_five_mac.py \
  --data-review
```

Start a bounded trial of **2,350 additional updates** (250,265,600 tokens):

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/scripts/launch_five_mac.py \
  --data-review --run --clear-stop --stop-after-steps 2350
```

It saves and stops after those updates. Repeating the command resumes the new
run's own latest checkpoint for another 2,350 updates. Omit `--stop-after-steps`
only when intentionally continuing toward the retained 15B total-token target.
Do not combine `--data-review` with `--lr-floor` or a batch/topology change.
The launcher stages the current worker package and verifies hashes on all five
Macs. Full-model workers are not started during development checks.

Expect a longer startup while the larger fixed held-out set is streamed into
RAM, and longer evaluation pauses. Progress prints every 16 collected batches.
The first launch also performs the pre-update baseline evaluation. Validation
still runs every 25M training tokens.

Assess roughly ten evaluations after this trial using the expanded baseline,
per-source scores, the unchanged legacy metric, and fixed text prompts. Small
score changes are evidence to investigate, not a promise of a new useful model.
Changing shuffle order and evaluation size together does not isolate a causal
shuffle benefit; a matched continuation without shuffling would be needed for
that comparison.

## Verification

`tests/test_data_review.py` covers exact interrupted/uninterrupted shuffled
resume with prefetch, cursor migration, split isolation, buffer exhaustion,
malformed state, strict recipe/data contracts, launcher output selection, and
a one-layer CPU fixture exercising baseline creation, checkpoint save and
restart. Existing continuation/scheduler/stream tests also run. The native
five-rank collective probe verifies communication, weighted gradients and
tiny BF16/FP32 checkpoint roundtrips for the current worker hashes.

Evidence is recorded in `diagnostics/data_review_20260923/`; production parent
checkpoints and stop markers remain intact.

Verified: **51 focused tests passed**, all five native RDMA probe receipts passed,
and a model-free live read collected the full 4,194,304 held-out tokens. Original
validation fingerprint: `99e7d4417061db5a70138701694ad5577d0f9ff0d0fc729ddc6beff06b3a7081`.
Expanded fingerprint: `740011448903ec5996b939ae24a3c6dc1b7e05772b0a6195d262304a681ad751`.
That live read took about five minutes. Its completed HF reader lingered during
interpreter shutdown and was terminated after its success receipt; no production
process was involved. The actual full-model expanded loss remains unmeasured
until the user launches the continuation.
