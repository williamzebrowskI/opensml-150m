# Two-turn continuation from High LR 640

Start from the preserved **Clean Reply High LR step 640**, not 1920, GRPO, or the deleted 1944 adapter. This is a targeted experiment, not a general-chat capability claim.

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/two_turn_640/launch.py \
  --run --clear-stop
```

Default invocation prints the plan. `--check` performs disposable GPU verification, without creating a production run. `--run` requires a matching readiness receipt. Ctrl-C finishes the current update and checkpoints; the same command resumes. A completed run does not automatically extend.

## Data and budget

- 256 original constructed scenarios, eight families, 32 scenarios per family.
- Each contributes the first reply and two **alternative** follow-ups from the same first reply. The alternatives are separate two-turn dialogues, not consecutive user requests. Their correct targets differ.
- 768 new targets, consumed once. Same underlying templates with varied content; these are not 256 independently collected human dialogues.
- 256 existing authored clean-reply TRAIN examples are consumed three times (768 rehearsal exposures). They were already used by High LR 640. Rehearsal covers conversation, practical advice, explanation, rewrite, summary, reading, follow-up and formatting. No prior test examples enter training.
- 128 updates, batch 12 (6 new + 6 rehearsal), microbatch one: 1,536 total exposures. Final cumulative step **768**, in the separate `runs/sft_two_turn_640_v1` directory. This is unrelated to the earlier deleted step 768 run.
- Full-parameter assistant-only cross-entropy including EOS; fresh FP32 AdamW, zero weight decay, clipping 1.0. LR warms for 8 updates to **3e-6**, then decays to **1e-6**. Context 1024, no truncation.
- Original data are generated deterministically into RAM. Rehearsal is read from existing local authored source. No download, public benchmark items, code/math training or external teacher service.

Task families: replace a detail, append information, remove content, recall context, change a material constraint, change answer format, distinguish missing from supplied information, and answer a clarification question with a completed message.

Example:

> Assistant: Thank you, Seth, for bringing the apron.
>
> Follow-up A: Add this sentence at the end: "I will see you on Tuesday." Return the whole note.
>
> Target A: Thank you, Seth, for bringing the apron. I will see you on Tuesday.
>
> Follow-up B: Add this sentence at the end: "I will meet you at the workshop." Return the whole note.
>
> Target B: Thank you, Seth, for bringing the apron. I will meet you at the workshop.

## Research and choice

[MT-Eval](https://github.com/KwanWaiChung/MT-Eval) distinguishes recollection, expansion, refinement and follow-up, and identifies context distance and error propagation as important problems. [Parrot](https://aclanthology.org/2024.acl-long.525/) motivates context-dependent multi-turn instruction data. These inform task design; their examples and preference-training algorithm are not copied. [Multi-IF](https://huggingface.co/datasets/facebook/Multi-IF) extends IFEval for evaluation and stays out of training.

Previously tried generic human-conversation data did not repair the observed operations. The original local curriculum makes the required edit and preserved content inspectable. Ordinary SFT gives an explicit correct continuation; RL/DPO would additionally need a reward/preference design that actually catches content failures. This is a reasoned experiment, not evidence that this dataset or learning rate is optimal. Data, method and budget differ from the retired adapter pilot, so it is not a single-variable comparison.

## Evaluation and limits

At updates 0/32/64/96/128, save full answers for 32 development scenarios (96 targets) and 32 reused clean-reply development tasks. Also generate both follow-ups using the model's own first reply, to expose error propagation. Record strict reference accuracy, both-branches correctness, repetition, stopping, likelihood and prose drift.

Whole scenarios are split, with separate event/object settings and some separate follow-up wordings for development/test. The same task families and template structure remain; this measures transfer within the constructed task distribution, not broad conversational generalization. Another 32 scenario groups are reserved and never evaluated during training. Exact-reference scores penalize valid alternative wording; actual answer review is required. Existing rehearsal/prose diagnostics have already been inspected and are not fresh tests.

There is **no automatic best selection, benchmark run or playground promotion**. Review early unseen-context answers against unchanged 640. Favor correct application of the current request while preserving relevant context, not merely lower loss or longer replies. The old 640 remains unchanged.
