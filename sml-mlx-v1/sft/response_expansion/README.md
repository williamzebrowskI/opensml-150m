# OpenSML response expansion — step 256 to 384

This continues the **best OpenSML SFT checkpoint at step 256**. It does not
restart from the pretrained model. The existing tokenizer, plain
`User: ...\nAssistant: ...` format, EOS and restored FP32 AdamW master weights,
moments and step counter are retained. Only the main Mac runs; SmolLM2 is not trained.

The previous second pass improved the unchanged development paired score from
65.6% to 71.9%, but its open-ended answers still often copied a keyword or the
question. This pass introduces short, complete responses and wider task wording.
It is an experiment, not evidence that the model is ready for general chat.

## Run or resume

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/response_expansion/launch.py \
  --run --clear-stop
```

The launcher prints its plan, evaluates the unchanged step-256 parent, then
runs **exactly 128 additional updates** at constant **3e-6**, finishing at step
384. Batch size is 8 with microbatches of 2, weight decay 0.01, betas 0.9/0.95,
gradient clipping 1.0 and assistant-only equal-example cross-entropy.

Press Ctrl-C once to request saving after the current update/evaluation. The
same command resumes the saved optimizer and exact next example. A completed
run does not repeat automatically. Output is `runs/sft_response_expansion_v1`;
the earlier SFT runs, reference comparison and pretrained checkpoints remain intact.

To print the plan, omit flags. `--check` checks all source hashes, loads the real
checkpoint and optimizer, tokenizes all examples, and runs a forward pass and
short training-prompt generation probe without applying any gradient updates.
After any recipe/code changes, repeat `--check` before `--run`.

## Data: 1,024 examples, one pass

These are **original locally authored examples and deterministic variations**,
generated in memory. There is no Hugging Face download or stronger teacher.
There are no math or programming tasks. This is a small controlled curriculum,
not 768 independently authored topics or a large general instruction corpus.

| New training examples | Count |
| --- | ---: |
| Full-sentence location and ownership answers | 128 |
| Updated facts and acknowledging missing information | 128 |
| Polite rewrites and precise name edits | 128 |
| Everyday explanations and supplied-text causal explanations | 128 |
| One-sentence summaries and factual descriptions | 128 |
| Friendly invitations | 64 |
| Greetings and clarification questions | 64 |
| **Total new examples** | **768** |
| Rehearsal from the original training set, 32 per prior task family | **256** |

Only old **training** examples enter rehearsal. Previous evaluation prompts
are excluded. New explanatory topics, polite requests and clarification
situations are partitioned before training, keeping their paraphrases together.
Names are separated across training/development/test. Controlled fact tasks
share objects and underlying task patterns across splits.

## Evaluation and checkpoint selection

- At additional updates 0, 32, 64 and 128: 104 new development prompts, the
  unchanged 128-prompt/64-pair factual development test, and the same 64 prose
  passages used for drift checks. Prose drift is relative to step 256, not the
  pretrained base.
- New tasks use explicit phrase, length, EOS, wrong-fact and exact-edit rules.
  The family-average `new_rules` score is **automated rubric coverage**, not a
  general chat accuracy score. Valid paraphrases can fail; a sentence with
  unsupported extras can pass. All answers are saved for inspection.
- Selection averages new task rule coverage and the old paired exact score
  with equal weight. Ties prefer old paired correctness, then new coverage,
  old exactness, and lower new assistant NLL. This can trade performance between
  the two groups, so inspect both. Prose drift is reported, not used as a gate.
  The unchanged parent participates in selection and stays best if nothing improves.
- New responses have a 64-token generation budget; old factual evaluation
  retains its original 48-token limit. All generation is greedy.
- After selection is complete, **156 fresh test prompts** compare the untouched
  step-256 parent with the selected best. These tests never enter gradients or
  checkpoint selection. Their repeated paraphrases are not independent topics;
  this remains a related-task transfer test, not a broad external benchmark.

Inspect `report.json`, `evaluations/step_*.json` and `fresh_test.json`. Latest
and best checkpoints are saved separately. No automatic playground promotion
is performed; its serving format must match this experiment before interactive use.

## Verification

```bash
cd <SOURCE_WORKSPACE>
PYTHONPATH=sml-mlx-v1 .venv/bin/python -m unittest sft.response_expansion.test_expansion -v
.venv/bin/python sml-mlx-v1/sft/response_expansion/launch.py --check
```

Tests cover split isolation, training-only rehearsal, reference grading,
prompt-echo/incorrect-answer rejection, exact optimizer inheritance,
interrupted versus uninterrupted continuation, the update limit, and retaining
the parent when later candidates score worse. The real-check receipt is
`readiness.json`; it records that production training was not started.
