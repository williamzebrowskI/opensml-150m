# Response repair: step 384 → 512

This is a continuation of the selected **step-384 response-expansion SFT**,
not a restart from pretraining or step 256. It restores the existing weights,
FP32 AdamW master parameters, moments and optimizer counter. The user confirmed
step 384 after the initial message referred to a nonexistent step 284.

## Run or resume

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/sft/response_repair/launch.py \
  --run --clear-stop
```

This runs **128 additional updates**, from 384 to 512, on the main Mac only.
The constant LR is **3e-6**, batch size 8, microbatch size 2, AdamW betas
0.9/0.95, weight decay 0.01 and gradient clipping 1.0. Training remains
assistant-only with equal-example loss weighting. The original tokenizer,
2,048-token context and `User: {message}\nAssistant:` format are unchanged.
EOS ends the answer. No reference model or teacher is loaded.

The launcher first evaluates and saves the unchanged parent. This baseline
can take longer than a training update. It never repeats the completed run
automatically. Ctrl-C requests a save after the current update/evaluation;
wait for completion, then use the same command to resume the exact cursor.

Output: `runs/sft_response_repair_v1`. Prior SFT checkpoints, pretrained
weights, tokenizer, optimizer files and playground exports are preserved.
The playground continues to offer steps 256 and 384; this command does not
replace a served model or auto-promote the new result.

## What changes

The previous expansion learned some complete responses but still echoed
greetings, answered the wrong entity, missed negation and ignored answer-only
instructions. Repeating its whole dataset would not explicitly address those
distinctions. This repair changes the data and selection protocol; it retains
the optimizer and LR to avoid changing those at the same time.

There are **1,024 local examples**, one pass. Every update contains four
factual/format examples and four response examples, shuffled deterministically.

| Component | Examples |
| --- | ---: |
| Exact prior training-only rehearsal, 32 examples per earlier task family | 256 |
| New location, ownership, missing-information, negation and correction contrasts | 256 |
| Greetings, including lowercase and common typo forms | 64 |
| Reply to a greeting versus explicitly copy it | 64 |
| Polite requests, 32 different actions | 64 |
| Short factual answers and explanations, 32 different questions | 64 |
| Supplied-text reasons and summaries | 64 |
| Useful clarification questions | 64 |
| Invitations | 64 |
| Welcomes, thanks and congratulations | 64 |

The first two rows form the **50% factual/format portion**. The other eight
form the **50% response portion**. The new examples are original authored
cards and controlled variations; this is not 768 independent natural
conversations. There is no HF stream/download, external teacher, code or math
problem training. Examples are reconstructed in memory from the frozen code.

The observed `hello`, `hii` and `hello there` failures enter the training set.
Their future improvement is therefore a repair check, not proof of generalization.
Fresh greeting requests remain held out. Previous formal evaluation prompts
and the 32 playground-comparison prompts are excluded from training.

For each new factual scenario, the data changes which fact is true and asks
for either an answer alone or a complete sentence. Variants stay together in
one split. Names/objects/locations and authored topics are split before training.
Some underlying tasks and everyday vocabulary are intentionally shared.

## Evaluation and best selection

At additional updates 0, 32, 64 and 128 (steps 384, 416, 448 and 512):

- Evaluate **208 new development prompts**, including 80 factual/format
  prompts in 20 groups of four and 128 response prompts. Repeated variants
  are not statistically independent examples.
- Keep factual content and requested format separate. Factual content uses
  accepted complete/brief text answers; format can be correct even when the
  fact is wrong. A correct full sentence can have correct content but fail an
  explicit answer-only request. Open-ended tasks use lexical and form rules,
  not a semantic judge. These proxies can miss valid paraphrases and accept
  flawed wording; every generated answer is saved for inspection.
- Measure prompt echoing separately; an explicit copying task can legitimately
  reproduce the supplied text. Accidental echoing cannot pass a greeting task.
- Repeat the original 128-prompt, 64-pair development test with its original
  48-token limit. New tasks use 64 tokens. Generation is greedy.
- Measure drift on the same 64 prose passages, relative to step 384. This is
  a small diagnostic, not the full pretraining validation set.

The parent starts as best. **A candidate is eligible only if its factual-content,
factual-format and original paired scores are each at least the parent's.**
Among eligible candidates, prefer higher task-average rule coverage, then
factual content, format, earlier paired accuracy and finally lower assistant
loss. This fixed rule can retain step 384 even if some new capabilities improve;
the report and latest checkpoint still show the attempted repair. `selection_eligible`
means this measured non-regression condition, not approval as a general assistant.

After selection, compare the unchanged parent and selected best on **208 fresh
test prompts in 68 scenario/topic groups**. These never enter gradients or best
selection. The protocol is related to the curriculum and remains a small local
diagnostic; inspect the saved answers before deciding whether the repair worked.

Read `report.json`, `evaluations/step_*.json`, and `fresh_test.json`.
The earlier [playground comparison](../../reports/SFT_PLAYGROUND_COMPARISON_20260924.md)
documents the failures that motivated this run and is now a regression reference.

## Verification

```bash
cd <SOURCE_WORKSPACE>
PYTHONPATH=sml-mlx-v2 .venv/bin/python -m unittest sft.response_repair.test_repair -v
.venv/bin/python sml-mlx-v2/sft/response_repair/launch.py --check
```

The tests cover data separation, the 50/50 update mixture, literal copying,
content versus format, non-regressing selection, exact optimizer inheritance,
interrupted/uninterrupted equivalence, the update limit and parent preservation.
`--check` verifies the real step-384 bundle, restores its optimizer, tokenizes
every training/development/test example, and runs a forward pass plus a short
training-prompt probe without applying any gradient update. Its receipt is
`readiness.json`. Changing code/configuration requires repeating that check.
