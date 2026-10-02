# Intact SmolTalk pilot from the V1 pretrained base

This separate pilot starts from the preserved Stage B best at **pretraining
step 73,243** (`step_0073243_7f3070237eea`). It loads weights only, upcasts them
to FP32 and initializes fresh AdamW state. Local SFT checkpoints are labeled
**0 through 128**; they are not continuations of any old SFT step with that number.
The original base, prior SFT models and playground stay unchanged.

## Start or resume

After preparation, sample review and readiness checks:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python -u \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/intact_smoltalk_base_pilot/launch.py \
  --run --clear-stop
```

Main Mac only. Output: `runs/sft_intact_smoltalk_base_pilot_v1`.
The maximum is **128 updates / 4,096 conversation exposures**, not a full
12,000-conversation pass. Baseline and updates **16, 32, 64 and 128** save and
evaluate. Every 16 updates also saves a checkpoint. A failed retention guard
requests a checkpointed stop. Read the saved answers before explicitly clearing
that stop. Ctrl-C saves after the current operation; repeat the command to resume
exact optimizer and conversation cursors. Completed pilots never extend.
Nothing selects a best checkpoint or promotes it automatically.

## Comparison being tested

The selector, pinned source, seed, intact histories, native tokenizer and
token-weighted assistant/EOS cross-entropy match the retained
`intact_smoltalk_2048` recipe. The source weights change from SFT 2048 to the
pretrained Stage B best. The production budget is capped and evaluation is earlier.
No replay, KL anchor, RL, DPO or external teacher is added.

The LR follows the **first 128 updates of the previous 750-update schedule**:
38-update warmup to `1e-5`, followed by cosine decay toward `1e-6` at update 750.
This pilot does not reach that LR floor; truncating the budget does not compress
the cosine schedule. Batch 32, clip 1.0, matrix weight decay 0.1, AdamW betas
0.9/0.95 and context 2,048 are unchanged. These are a controlled starting point,
not established optimal base-model settings.

The dataset packet contains 12,000 training, 500 development and 500 reserved
whole source conversations. Only the first 4,096 conversations in the existing
seeded first-pass order receive production updates. All assistant turns include
their complete history and EOS; user/system/history positions are masked. Long
conversations are rejected whole, never truncated. Source data is the same pinned
`HuggingFaceTB/smol-smoltalk` revision
`f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`: Magpie ultra-short, everyday
conversation, rewrites, summaries and constraints. Publisher-generated synthetic
answers and sampled source review do not guarantee correctness. Prior source
exposure and overlap screens do not establish semantic decontamination.

Preparation reads bounded ranged Parquet data and writes a selected-conversation
snapshot for reproducible/offline run and resume. It does not cache the complete
raw source files. Prepared data and readiness belong to this pilot only.

## Evaluation and readiness

All 500 development conversations receive reference-history likelihood scoring;
30 stratified conversations receive generated answers using both reference and
the model's own history. Reused reading, constraint, QA and prose checks remain
diagnostics. Retention thresholds compare to this **pretrained baseline**, not
the removed SFT run. Natural stopping, repetition and surface rules alone do
not measure correct, complete conversation.

Review actual answers at the early checkpoints. Extend only through a separately
authorized continuation if useful completion improves and existing abilities
survive. Preserve 1920 and 1472 as benchmark references. Reserved conversations
and public suites remain unused during preparation and training.

`--prepare` rebuilds deterministic selections; sample review must be recorded
before `--check`. The check verifies every assistant mask/EOS, native playground
history parity, the 4,096-example cursor, cached/full generation agreement,
independent CE, disposable real-model updates, exact saved-state restoration
and exact next-update model weights. Next-update Adam moments must agree within
`1e-9`; their measured maximum difference is recorded in readiness.
Disposable weights are removed and original inputs are hash-verified unchanged.
No production updates start without `--run`.
