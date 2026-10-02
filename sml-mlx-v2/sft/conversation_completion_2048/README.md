# Complete conversational replies from preserved 2048

This is a separate **noncommercial research branch**. It starts from exact
constraint-completion 2048 (`e4e8a0de28e2141f6a580b49cc950aa64492c9f94d0d3aa98e6a26a2f439764b`)
and writes only to `runs/sft_conversation_completion_2048_v1`.

The release conversation audit found frequent omissions, confused ownership,
and unsuccessful follow-ups despite almost universal EOS termination. Previous
broad conversation passes lowered likelihood without making replies useful.
This experiment tests complete natural demonstrations plus explicit updates
that preserve all unaffected information. It does not promise higher benchmarks.

## Data

- 1,536 human-written No Robots demonstrations: 896 generation, 192 brainstorm,
  128 rewrite, 192 summarize, 64 closed QA, 64 extraction. Single user/assistant
  conversations only; persona/system examples are excluded rather than stripped.
- 512 authored targets from **256 constructed two-turn dialogues**. Families:
  invitations, clarification, polite rewriting, full summaries, ownership,
  restrictions, tiny complete stories, list edits, and greetings/help offers.
  Template instances are not independent human conversations. Train/dev/test
  use different settings, names, and phrasing but share task families.
- 2,048 skill-replay exposures from examples already consumed by the 1920
  ancestor: 512 commonsense, 768 formatting, and 768 reading.

No Robots is pinned to revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`;
the train Parquet SHA256 and byte count are in `source.json`. Raw source bytes
are verified in RAM and discarded. Only selected training/dev/internal-test
rows are retained. Human tasks are split by whole prompt ID before training.
The official No Robots test file stays unused. Exact and long-phrase benchmark
overlap rejection is not semantic or pretraining decontamination. The release
audit prompts remain evaluation-only.

The publisher describes the data as human-written and licenses it
**CC-BY-NC-4.0**. This source restriction matters when considering a commercial
release of the trained derivative. The preserved 2048 does not acquire this
new dataset restriction from a separate experiment.
[Publisher card](https://huggingface.co/datasets/HuggingFaceH4/no_robots).

Length and role filters do not certify human targets as factually correct or
fully compliant. The preparation sample is reviewed for meaning, all requested
parts, length and unfinished answers. Accepted inspected IDs stay fixed during replacements; this is a curated inspection
sample, not an unbiased error-rate estimate. Sample review is not exhaustive validation
of all 1,536 demonstrations.

## Objective and budget

Standard full-parameter supervised assistant-only cross entropy including EOS,
with fresh FP32 AdamW. **Not RL or DPO.** Each update contains eight new replies
and eight skill-replay examples. Across the whole run, new replies are 75% human
and 25% constructed; all new targets are exposed once. Family weights are
50% conversational CE, 15% formatting CE, 20% reading CE, 15% correct-choice
ranking, plus 0.1 frozen-parent training-prose KL.

256 updates, warmup 16, peak LR `3e-6`, final LR `1e-6`, zero weight decay,
gradient clip 1.0, microbatch one. These are conservative experimental settings,
not an established optimal learning rate. Source weights and tokenizer are frozen.
Training uses the exact existing plain-SFT playground history serialization.

Checkpoint/evaluation updates: 0, 32, 64, 96, 128, 160, 192, 224, 256, corresponding
to steps **2048, 2080, 2112, 2144, 2176, 2208, 2240, 2272, 2304**. All nine bundles
are retained; no evaluated candidate is automatically deleted. Roughly 21 GiB
plus working reserve is required; the launcher requires 35 GiB free initially.
Interruptions save exact optimizer and exposure cursors. The same command resumes
but never silently extends a finished run.

## Evaluation

Greedy decoding, temperature 0, repetition penalty 1.0, no forced length, 192
new tokens. Store all 48 human development replies and 36 whole development
dialogues, including the model's **actual first answer** in the follow-up context.
Gold-history follow-ups are saved separately to expose sensitivity to mistakes
in the model's first reply. Surface coverage, obsolete facts, formatting and
reference exactness are diagnostics, never proof of truth or useful conversation.
Blank semantic review forms require correct content, relevance, full completion,
coherence, and retained skills. No automatic best checkpoint or playground promotion.

Compare candidates with unchanged 2048 and then freeze the choice before reserved
testing. Any derivative needs its own public benchmark scores; original 2048's
scores cannot be attributed to it. Public benchmarks in this project have already
influenced development and are not pristine final holdouts.

## Commands

Preparation and readiness have already been performed when this command is delivered:

```sh
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/sft/conversation_completion_2048/launch.py \
  --run --clear-stop
```

For re-preparation before a run starts: `--prepare`, review `review_samples.json`,
then `--check`. `--check` verifies masks/EOS, all batch cursors, cached/full-prefix
generation and a disposable optimizer update without saving a trained candidate.
Readiness receipts are bound to exact code, data, source and runtime identities.

After reviewing a completed run, benchmark an explicitly chosen evaluated step:

```sh
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/sft/conversation_completion_2048/benchmark.py \
  --step 2304 --suite all --check
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/sft/conversation_completion_2048/benchmark.py \
  --step 2304 --suite all --run
```

2304 is merely the final step, not automatically the best.
