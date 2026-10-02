# Conversation Foundation V2 — text-only SFT from V2 pretrained

This is a new post-training branch, not a new pretrained model. Starts from V2
pretrained step 73243, SHA256
`ca9bd5d82005f28e77f319d3a5a29f29fb4e4f86e7ceea049f01162fd8aae80f`.
It does not initialize from 512, 768, or DPO. Existing models remain unchanged.

## Start / resume

```sh
/Users/williamzebrowski/sml-mlx/.venv/bin/python -u \
  /Users/williamzebrowski/sml-mlx/sml-mlx-v2/sft/conversation_foundation_v2/launch.py \
  --run
```

Use the same command to resume the last complete checkpoint, optimizer and data
cursor. Run one GPU training or evaluation job at a time. A graceful Ctrl-C saves
after the current update/evaluation. A manual STOP marker requires `--clear-stop`.
There are no automatic quality-based stops, pruning, or best-model promotion.
Integrity errors, nonfinite updates and insufficient disk space still raise errors.

## Recipe

24,576 unique conversations, two shuffled passes, 16 conversations per update:
**3,072 updates / 49,152 conversation exposures**. Each batch mixes:

| Skill pool | Unique conversations | Per update |
| --- | ---: | ---: |
| Multi-turn conversation | 12,288 | 8 |
| Request completion / explanations | 3,072 | 2 |
| Rewriting and summarization | 3,072 | 2 |
| Passage-based answers | 1,536 | 1 |
| Checked fictional text exercises | 4,608 | 3 |

The first three pools select from the existing checksummed Smol-SmolTalk and
UltraChat candidate cache. Grounding uses human-annotated SQuAD v2 reading
examples: 1,152 with verified answer spans and 384 labeled unanswerable.
Article partitions stay separate, and numerical questions are excluded.
This replaces the weaker synthetic passage conversations found in sample review. Roles and contents are filtered, not just source
names. `self-oss-instruct` is entirely excluded. Code, math and JSON tasks,
obvious refusal boilerplate, unfinished placeholders, repetitive targets and
oversized conversations are filtered. Heuristic filters are imperfect and can
also reject suitable text. Public targets are sample-reviewed, not all fact-checked.

The checked exercises cover corrections, ownership, preferences, topic changes,
missing facts, direct messages, edits, casing, endings, bullets and paragraphs.
They are 18.75% of each batch. They use twelve templates with varied fictional
content, not 4,608 independently authored tasks. Their final targets pass explicit
checks. Broader public conversations provide most of the variety.

Whole conversations fit the existing 2,048-token native history format. Every
assistant answer is at most 384 tokens, with at most four assistant turns per
public conversation. No answer is cut to fit. All assistant replies and EOS are
supervised; prompt and history tokens are masked. Loss is balanced by conversation
and assistant turn, rather than dominated by long answers.

FP32 AdamW; peak LR 8e-6, warmup 128 updates, cosine decay to 8e-7, weight decay
0.01, gradient clip 1. This is a proposed recipe, not an established optimum.
It changes multiple factors relative to the earlier foundation, so it is not a
causal ablation identifying which earlier choice caused a weakness.

## Fixed evaluation from the beginning

Before training and at steps **64, 128, 256, 512, 768, 1024, 1536, 2048, 2560,
3072**, save/evaluate. Retain every checkpoint. Reserve at least 40 GiB free.

1. Validation loss on 128 held-out public conversations.
2. Own-history generation on 16 public conversations, up to two turns each,
   recording stopping, repetition, lengths and complete replies.
3. Forty fixed fictional behavior conversations: eight each for history,
   completion, grounding, text instructions and relevance/topic changes.
   Multi-turn prompts use the model's actual preceding responses.
4. Another 128 public conversations and forty behavior scenarios are reserved
   as test material and not evaluated during training.

Behavior categories report **proxy pass counts**, not an LLM judge or a human
quality score. Exact-answer checks can detect bounded retrieval errors. Keyword,
length and format checks can miss nonsense and reject valid paraphrases. Each
scenario includes an expected example and a review rubric; saved answers must be
read for relevance, coherent conversation, tone, support and actual completion.
There is no combined automatic best score. Development/test scenarios use a
shared evaluation design, separate from training exercise templates; this is not
proof of broad out-of-distribution generalization.

The development set is new. Its scores must not be compared directly with the
old foundation's development percentages. Use the same-suite reference results
for Foundation 512 and Text Follow-up 768:

```sh
/Users/williamzebrowski/sml-mlx/.venv/bin/python -u \
  /Users/williamzebrowski/sml-mlx/sml-mlx-v2/sft/conversation_foundation_v2/launch.py \
  --baselines
```

Reference evaluations are inference-only and resume completed models. Results:
`sml-mlx-v2/diagnostics/conversation_foundation_v2_references/`.
Public dev/test selection excludes previous foundation/text-follow-up training
records, and old held-out user phrases are excluded from this run. Public
benchmark prompts/answers are not training targets; exact/13-word overlap
filters are used, without claiming perfect semantic or pretraining decontamination.

## Public benchmarks after checkpoint review

Replace STEP with a saved step. Full benchmark scores use the existing protocols:

```sh
.venv/bin/python -u sml-mlx-v2/sft/conversation_foundation_v2/full_benchmarks.py --step STEP --run
.venv/bin/python -u sml-mlx-v2/sft/conversation_foundation_v2/benchmark.py --step STEP --run
```

The first command runs ARC-Easy, ARC-Challenge, PIQA, HellaSwag and IFEval.
The second uses MT-Bench questions with the existing local Prometheus judge;
its 1–5 scores are not official GPT-4 MT-Bench scores. The standard suite retains
math/coding questions for comparability even though training excludes those tasks.
Any text-only category summary must be labeled as a subset.

## Checks and provenance

`--inspect` checks full data, all answer masks/EOS, unique groups, scope and exact
two-pass schedules. `--check` additionally uses disposable models to test prompt
parity with the playground, cached decoding, masked loss, checkpoint optimizer
restore, exact next-update weights and the evaluation path. It never starts
production training or overwrites the pretrained model. Checks are recorded in
`readiness.json`; frozen inputs and selected-data hashes are checked on resume.

Sources retain per-row dataset/file/index information. Source-cache hashes and
pinned dataset revisions are in the preparation receipt. This reuses an earlier
bounded candidate pool, not a fresh census of the entire upstream datasets.

- [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), Apache-2.0.
- [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2), CC BY-SA 4.0; pinned revision `3ffb306f725f7d2ce8394bc1873b24868140c412`, train split only. Answers retain source annotations; prompts are wrapped as passage questions.
- [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), MIT; train_sft only.

No improvement in chat quality, IFEval, or multiple-choice benchmarks is promised.
