# Complete Transfer from Recovery 1024

This is a new SFT experiment from preserved `sft_skill_recovery_768_v1/step_0001024_4d34123910e4`, not a continuation of deleted control 1536. It does not promise to beat 1920 or 1024.

## Motivation and data

The previous run mostly practiced 12–16-token reading/formatting responses. Both models still answered IFEval with a median of eight tokens. This experiment changes the response distribution: 576 independently sourced natural replies of 40–240 words and 192 new source contexts reformatted as complete lists or paragraphs. All natural and grounded source contexts are disjoint. There is only one view per training source context and one training pass.

Source: [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), revision `8049631c405ae6576f93f445c6b8166f76f5505a`, `train_sft` only. The publisher lists MIT and describes the data as ChatGPT-generated. No local teacher is used. This is not human-written data. The initial broader selection was rejected after sample review exposed incomplete prompts, stale travel advice and unsupported factual claims. The final selection uses stricter self-contained prompt and topic filters; sample review is not exhaustive factual verification.

Every raw Parquet file is streamed to one bounded RAM buffer (~244 MB), SHA-256 verified, decoded in batches, and discarded. Selected examples stay in RAM. Production reconstructs the same pinned selection; it does not use the temporary setup cache. Three files (~732 MB total network transfer) are fetched per invocation. No raw dataset disk cache. Small reviewed samples and generated evaluation answers are saved as experiment records.

384 labeled commonsense questions from the historical TRAIN pool and 384 unique reading/two-turn replay examples protect earlier skills. Some historical QA rows were previously consumed; they are not advertised as fresh. This is general English, writing and everyday advice with code/math and specialist topics filtered, not broad knowledge pretraining.

## Training

192 updates, absolute steps 1024–1216, single main Mac. Each update uses 3 natural replies, 1 complete-source reply, 2 labeled choice-ranking examples and 2 reading replay examples. Fresh FP32 AdamW, peak LR 8e-6, final 2e-6, 16-update warmup, zero weight decay. Main loss: 60% new assistant-only CE, 20% reading CE, 20% labeled ranking; frozen parent prose KL weight 0.05. EOS is supervised; prompts and padding are masked. This is ordinary supervised training, not RL/DPO/gradient projection. Both data and learning rate change; this is a recipe test, not a causal ablation of either variable.

All scheduled checkpoints (every 32 updates) and interrupted saves are retained. Approximately 17 GB for seven full bundles; launcher requires 35 GiB free. The source checkpoint is never overwritten. No automatic restart, deletion, best selection or playground promotion.

## Evaluation and limits

New development and reserved test source groups each contain 32 natural replies and 16 complete-source tasks. Natural generation is saved for manual review of relevance, correctness and completion; NLL/length/stopping alone cannot establish quality. Complete-source tasks require every supplied sentence, in order and with the requested structure; missing facts, changed content, extra text or truncation fail. These are explicit copy-and-structure tasks, not tests of open-ended factual reasoning. Their small sample is a diagnostic. Historical commonsense, reading and prose checks are reused. Train/dev/test roots and exact answers are separated, but lexical benchmark exclusion is heuristic and does not prove semantic decontamination.

Public benchmark questions are not training data. Freeze one checkpoint after reviewing development answers before running reserved/public evaluations. Public benchmarks have been inspected repeatedly across prior experiments; they are diagnostics, not a pristine final test. Do not claim wins without matched scoring and actual answer review.

## Commands

Run after readiness checks have passed:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python <SOURCE_WORKSPACE>/sml-mlx-v2/sft/complete_transfer_1024/launch.py --run --clear-stop
```

`--check` reconstructs data, verifies masks and content grading, tests two real updates and exact optimizer resume on temporary copies, and checks cached/full-context generation. It does not start production training. The benchmark wrapper accepts `--step <evaluated-step> --suite all --check` followed by the same command with `--run`.
