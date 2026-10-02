# Balanced skills from Two-Turn 768

Starts only from `runs/sft_two_turn_640_v1/step_0000768_49364940d7ce`.
512 additional updates produce step 1280, main Mac only. Original 768, 640,
1920 and other runs remain unchanged. Nothing promotes automatically.

## Training

- 3,072 CommonsenseQA/SocialIQA TRAIN examples (1,536 each).
- 1,536 CommonGen TRAIN examples, one checked formatting view each.
- 1,536 exact replay exposures: 768 original two-turn targets once and 256
  authored clean-reply targets three times, all previously consumed by 768.
- Per update: six commonsense, three instruction, three replay examples.
- New instruction examples are CommonGen complete sentences with formatting;
  edits, explanations and conversational turns are supplied by replay, not
  a newly collected conversation dataset.
- Equal-example CE within each family; weights 0.40 commonsense, 0.20
  instruction, 0.25 replay, 0.15 correct-choice ranking. Additional KL 0.10
  against frozen 768 on train-only SocialIQA narratives/questions.
- LR 3e-6 peak, 1e-6 final, 32 warmup updates, fresh FP32 AdamW, context 1024.
- Pinned public files streamed into bounded RAM (each raw file <=3.5 MB),
  selected rows remain in RAM; no raw dataset disk cache or external teacher.
- Shared scope filters reject code/math. Historical source licensing is
  recorded in data.py (MIT and CC-BY-4.0); preserve attribution in distribution.

Some commonsense TRAIN rows can repeat base-SFT exposures. Their new dev/test
partitions intersect both prior source partition rules to keep previous
training groups out. CommonGen uses its established split; it was absent from
this parent's ancestry. Known benchmark text is used for overlap rejection
only, never training targets. Heuristic filters are not semantic decontamination.
Sample review does not certify all source annotations as correct.

## Evaluation and selection

Evaluate at 0,128,256,384,512 updates. Save full outputs for commonsense,
format/content coverage, two-turn reading and own-generated first replies,
natural probes and prose loss. Two-turn/prose diagnostics and some probes are
reused, not fresh tests. CommonGen format coverage is not semantic correctness.
Candidates must retain reading, own-history, format and instruction proxy scores
within configured limits. Review actual answers before testing/promotion.

Readiness runs only disposable updates, verifies frozen anchor, masks, ranking
loss direction, model and optimizer resume, data reproduction and decoding parity.

```sh
<SOURCE_WORKSPACE>/.venv/bin/python <SOURCE_WORKSPACE>/sml-mlx-v1/sft/skill_balance_768/launch.py --check
<SOURCE_WORKSPACE>/.venv/bin/python <SOURCE_WORKSPACE>/sml-mlx-v1/sft/skill_balance_768/launch.py --run --clear-stop
```

Ctrl-C saves after the current operation; the same run command resumes exact
optimizer state and data cursors. Completion never extends the budget.
Output: `runs/sft_skill_balance_768_v1`. After reviewing development answers,
`finalize.py --step CHOSEN_STEP --run` freezes a candidate before reserved
source-test comparison. Public benchmarks are separate and run only after
selection; IFEval is not used to select the training checkpoint.
