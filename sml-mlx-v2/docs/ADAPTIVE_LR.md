# Validation-Aware Learning-Rate Experiment

Prepared 2026-09-21. This is an opt-in continuation, not evidence that a lower
learning rate will improve the model. Production training is left stopped for
the user to launch.

## Starting Point

- Original run: `runs/full_15b_v1`.
- Cleanly stopped checkpoint: step **38,056**, **4,052,811,776 total tokens**.
- Original validation best: **3.0306107052** at step **34,744**.
- Most recent validation: **3.0419** at step **38,030**.
- Fourteen evaluations since that best had not surpassed it. This motivates an
  experiment, but does not establish that LR caused the plateau.

Resume **latest**, not best, to preserve all completed updates and the consumed
HF stream position. The original best, latest, logs, and STOP marker are left
untouched. The launcher verifies the checkpoint's integrity before use.

## Policy

Configuration: `configs/plateau.json`.

| Setting | Value |
| --- | --- |
| Initial LR cap | **0.0002**, down from 0.0003 |
| Signal | Existing fixed, mixture-weighted validation loss |
| Significant improvement | Loss more than **0.002 below** the controller's best |
| Patience | **8 consecutive eligible evaluations** without such improvement |
| Reduction | Multiply the cap by **0.5** |
| Cooldown | **2 evaluations** after activation and after each reduction |
| Minimum cap | **0.00003** |
| Total token target | **15,000,000,000**, including all prior training |

Eight evaluations span about 200M tokens; cooldown adds about 50M. At recent
throughput, a full cooldown-plus-patience interval is roughly five hours.
The first evaluation remains aligned to the existing 25M-token boundaries.
Old failed evaluations are not replayed into the new controller: the lower
initial rate gets a fresh observation window. The saved global best is used as
the initial validation reference.

Every update uses:

```text
effective_lr = min(existing_token_schedule(total_tokens), saved_plateau_cap)
```

Thus the cap path, if the plateau persists, is 0.0002 -> 0.0001 -> 0.00005 ->
0.00003. There is no automatic increase or rewarm on restart. The old final
cosine schedule still starts at 12B tokens and reaches 0.00003 at 15B. It cannot
raise an already reduced rate; if the controller reaches the floor first, the
effective rate stays at that floor through the end.

All other settings remain unchanged: model, tokenizer, corpus and split policy,
batch 4/3/3/3, accumulation 4, 2,048-token context, BF16 compute, FP32 optimizer
and gradients, and AdamW moments. The scheduler is small Python control logic;
PyTorch is not added to the training pipeline.

## Checkpoints and Safety

New output: `runs/full_15b_plateau_v1`.

- The transition records the exact parent contract, replica hash, step, tokens,
  and validation best. Unrelated recipe changes are rejected.
- State includes the LR cap, significant-best loss, bad-evaluation count,
  cooldown, reduction count, and last observed evaluation. Missing or
  incompatible scheduler state fails rather than silently resetting.
- Every rank processes the same broadcast validation score. Scheduler states
  are checked for equality on resume and before committing checkpoints.
- The current weights and optimizer must still pass the existing replica-hash
  checks. Dataset position and held-out fingerprint are restored and verified.
- A checkpoint is saved at activation and after each validation, as well as the
  existing 10M-token saves. This preserves every controller decision on resume.
- Best-checkpoint selection still uses **any raw validation improvement**;
  the 0.002 threshold applies only to LR decisions.
- The new run inherits the original best-loss reference. It may have no local
  `best.json` until it beats that score. The original best remains in the parent
  run. Do not delete that run. After the adaptive branch produced new bests,
  the playground default was explicitly updated to follow this branch's
  `best.json`. Its badge refreshes every 15 seconds, and each new completion
  resolves the current best without changing an in-progress response.
- Code and inputs are staged, copied, and checksum-verified on all four Macs by
  the launcher. No separate manual update on the peer Macs is required.

## Commands

Preview only, without SSH, downloads, or model training:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/scripts/launch_adaptive.py
```

Start or resume the experiment yourself from Mac-1:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/scripts/launch_adaptive.py \
  --run --clear-stop
```

The first launch selects the parent's latest checkpoint. Later launches select
the experiment's own latest checkpoint, preserving scheduler state. Native ring
control is the default, matching the last production session. The existing
`--ring-control all-sum` fallback remains explicitly selectable.

Ctrl-C requests the existing coordinated save/stop. Wait for all `[finished]`
messages. The command never silently starts fresh or automatically restarts a
failed worker. Do not run the original and experimental launchers together.

## Interpreting Results

Watch `[lr-controller ...]` for the effective **next** update's LR, cooldown,
patience, cap, and reductions. A lower training loss alone is insufficient.
Compare the fixed validation scores and source breakdowns over 200-250M new
tokens, along with the unchanged greedy prompts. Improvements in loss do not
guarantee factual or coherent generations.

This is a single adaptive continuation, **not a matched A/B convergence test**.
If it improves, the saved parent permits a later comparison against the old
schedule with the same starting weights, optimizer, token budget and stream.
Do not claim the scheduler is superior without such evidence.

## Verification

Tests cover patience, noise thresholds, cooldown, LR floor, final cosine
composition, nonfinite inputs, duplicate evaluations, state corruption,
explicit migration, and refusal to silently reset the schedule. A synthetic
CPU streaming run compares uninterrupted versus interrupted/resumed training,
including exact weights, optimizer tensors, controller state and stream cursor.
These are correctness tests, not a production convergence trial or a new live
four-Mac performance benchmark.

```bash
<SOURCE_WORKSPACE>/.venv/bin/python -m unittest discover \
  -s <SOURCE_WORKSPACE>/sml-mlx-v2/tests -v
```
