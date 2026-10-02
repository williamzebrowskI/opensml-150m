# Skill recovery from preserved Two-Turn 768

A separate, experimental continuation of `runs/sft_two_turn_640_v1/step_0000768_49364940d7ce`.
No previous failed continuation is used. Main Mac only. 256 updates to step 1024.

## Hypothesis and limits

The earlier CommonsenseQA/SocialIQA/CommonGen mix improved ARC-Easy but lost
instruction following and PIQA. Added response KL did not resolve that tradeoff.
This run separates learning to score answer choices from learning how to reply:

- 1,536 labeled CommonsenseQA/SocialIQA TRAIN questions (768 each) teach supervised
  choice ranking. All six questions in each update contribute. Their short answer
  labels are NOT assistant completion targets.
- 768 exact-content formatting views from 256 earlier TRAIN contexts replace
  CommonGen training. Three views per context; these are explicit text-preservation
  tasks, not independent conversations or new factual knowledge.
- 1,536 unchanged replay exposures (1,024 unique) protect complete answers and
  two-turn tasks, including their EOS targets.

Per update: 6 ranked questions, 3 formatting tasks, 6 replay examples. CE is
assistant-only, per-example normalized: 0.30 formatting, 0.50 replay. Supervised
choice-ranking weight 0.20; existing frozen-768 narrative KL weight 0.10 remains.
No response KL, teacher model, RL or DPO. Fresh FP32 AdamW, peak LR 2e-6 to 1e-6,
16 warmup updates, context 1024. Never truncates away targets.

This changes the mix, ranking exposure, LR and budget together. It is a new recipe,
not a single-variable ablation or an established optimal configuration. The research
precedent for combining ranking and supervised losses is RRHF
(https://arxiv.org/abs/2304.05302), but this is labeled-choice CE ranking, NOT RRHF.
Success on all benchmarks is a goal, not a predicted outcome.

## Data and evaluation

Pinned public files stream into bounded RAM, no raw corpus cache. The reviewed
previous training pools and historical split intersection are reused. Nine further
ambiguous/unsupported TRAIN labels were excluded during sample review. This is
not exhaustive human review. CommonGen remains a reused evaluation-only diagnostic;
its concept coverage score is not treated as semantic correctness.

Development runs at updates 0/64/128/192/256. New format grading requires both valid
format AND unchanged complete content, not merely keyword presence. Existing
reading, own-history, per-source commonsense, natural-answer, prose and stopping
checks remain. New format views have held-out settings but shared templates; other
probes were previously inspected. Review saved answers before choosing a checkpoint.
There is no automatic playground promotion. The run can retain baseline 768 if no
candidate passes its gates. Public benchmarks must stay out of training/selection;
freeze a development candidate before the reserved/public comparison.

```sh
<SOURCE_WORKSPACE>/.venv/bin/python <SOURCE_WORKSPACE>/sml-mlx-v1/sft/skill_recovery_768/launch.py --run --clear-stop
```

Output: `runs/sft_skill_recovery_768_v1`. Ctrl-C requests a checkpointed stop;
the same command resumes. Readiness checks perform disposable updates and verify
loss direction, masked prompts/padding, EOS targets, frozen anchor, complete data
batches, format-content grading and optimizer resume. No production run is started
by preparation/checks. Preserve original 768 for the final comparison.
