# Clean replies: matched learning-rate comparison from curriculum 512

This is a controlled continuation, not a promised chat model. It tests whether
cleaner, simpler targets can be learned at all, and whether that learning transfers.
Both arms start independently from preserved
`runs/sft_base_curriculum_v1/step_0000512_fd767fb7545a` with fresh FP32 AdamW.
The high-rate arm does not continue from the low-rate arm.

Run both sequentially on the main Mac:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/clean_reply_ab/launch.py \
  --arm both --run --clear-stop
```

Use `--arm low` or `--arm high` instead to run just that arm. No flags prints a
plan. `--check` performs disposable updates and a resume check, not production
training. The same training command resumes the selected unfinished arm(s);
completed arms are skipped and never extended automatically. Ctrl-C saves the
current arm after the current update, and prevents the next arm from starting.

## Final recipe

- Low: warm up for 8 updates to 3e-6; cosine decay to 1e-6.
- High: warm up for 8 updates to 1e-5; cosine decay to 1e-6.
- Identical 384 unique examples, deterministic ordering, 4 passes, 128 updates
  (1,536 exposures) per arm. Absolute step 512 -> 640 in each separate directory.
- Every update: 8 original examples + 4 rehearsal examples; batch 12,
  microbatch 1. Context 1024; complete targets plus EOS, no target truncation.
- Equal-example assistant-only cross entropy; prompt and padding masked.
- Fresh optimizer moments, FP32 parameters; clip norm 1, zero weight decay.
- No RL, preference loss, frozen-teacher loss, or repetition penalty.
- No automatic selection, benchmark run, or playground promotion.

## What is actually in the data

`authored.json` contains 128 **distinct content groups**, with two training
phrasings each (256 examples), plus a third untrained phrasing. These are original
assistant-authored examples, not 256 human conversations or 256 independent tasks.
The eight families each have 16 content groups: everyday conversation, practical
help, explanations, sentence editing, summaries, reading, follow-ups, and simple
output formatting. Answers are concise but complete; training them may still
produce narrow or repetitive behavior. No coding or mathematical exercises.

The initial 256-row rehearsal selection was reduced after review to 128 parent
TRAIN examples: 32 reading, 32 unanswerable-passage questions, 32 commonsense,
and 32 choice questions with formatting. These preserve source wording and labels.
Rejected candidates included malformed targets, mismatched years, answerable
questions labeled unanswerable, and unjustified social assumptions. The final
replay was reviewed against the passages or supplied choices; this is not an
external fact check of every statement in those passages. Source IDs and reasons
are in `replay_review.json`.

Rehearsal is reconstructed in RAM from the exact parent selection on pinned public
streams. No raw corpus or selected replay text is cached on disk. Initial
preparation can take several minutes because the parent selection must be
reproduced. The manifest verifies exact data hashes before training.

Source information (including revisions) is saved in `selection.json`:
SQuAD v2 (CC-BY-SA-4.0), CommonsenseQA (MIT), SocialIQA (CC-BY-4.0).
Reconstruction also reads the parent's pinned Smol dataset (Apache-2.0); those
natural-dialogue rows do not enter this experiment's rehearsal gradients.
Retain source attribution and the applicable dataset license notices when sharing
source-derived data. The newly authored examples are stored locally for review.

## Interpreting the evaluations

At updates 0, 32, 64, 96, and 128 the script saves full greedy answers (160-token
limit), teacher-forced likelihood, reference exact-match, stopping and repetition.

- `trained`: 128 previously trained phrasings, one per original content group.
- `paraphrase`: third phrasing of those same tasks. This tests near transfer, not
  fresh knowledge or independent task families.
- `new_tasks`: 32 separate authored development tasks (4 per family).
- `retention`: 32 existing parent development questions, never gradient targets.
- Prose likelihood: the same evaluation-only passages used previously.
- `test.json`: 32 additional prompts, excluded from all training and automatic
  evaluation. Freeze an arm/checkpoint choice before evaluating them once.

Exact-match is not a valid general correctness score for open-ended replies.
Read the generated answers for factual correctness, task relevance, completion,
repetition and invented identity claims. Successful memorization without new-task
improvement is a negative transfer result, not grounds for another blind extension.
Falling NLL alone is not success. Public benchmarks remain outside training.

Outputs are separate:
`runs/sft_clean_reply_ab_v1/low` and `runs/sft_clean_reply_ab_v1/high`.
Each includes configuration, immutable input hashes, checkpoint bundles, metrics,
and full development responses. Source 512, base Stage B, and skill model 1920
are hashed and remain protected. Nothing selects or deletes them automatically.
