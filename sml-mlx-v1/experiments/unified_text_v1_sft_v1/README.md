# Unified text SFT from pretrained v1

## Purpose

Test whether learning the strengths of Conversation Foundation 512 and Text Follow-up 768 together produces a better assistant. The preserved 768 is the reference to beat. No benchmark gain is guaranteed, and no checkpoint is automatically promoted.

The source is the original pretrained v1 bundle, `runs/stage_b_prose_v1/step_0073243_7f3070237eea`. This experiment has its own data, checkpoints, logs, evaluations, and benchmark results under this directory.

## Frozen mixture

| Category | Conversations | Share |
|---|---:|---:|
| Conversation | 7,680 | 62.50% |
| Writing and summarization | 1,984 | 16.15% |
| Recovery and ordinary follow-ups | 768 | 6.25% |
| Requested text formatting | 768 | 6.25% |
| Science questions | 768 | 6.25% |
| Passage reading | 320 | 2.60% |
| **Total** | **12,288** | **100%** |

Public examples come from the same frozen source pools used by the successful recipes: Smol-SmolTalk (Magpie short chats, Everyday Conversations, rewrite, summarize), UltraChat, answerable SQuAD v2 passages, and ARC **training** questions with gold option text. Coding-oriented Self-OSS examples are excluded. Public conversations and answers are retained intact after automatic scope, length, duplicate, overlap, and repetition filters. They have not received exhaustive factual review.

Smol-SmolTalk revision: `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`.
UltraChat revision: `8049631c405ae6576f93f445c6b8166f76f5505a`.
Full lineage and source-pool byte hashes are saved in `data/prepared.json`.

Recovery examples use fictional supplied facts with balanced history, correction, preference, unknown information, grounded cause, task completion, shortening, and topic-switch tasks. There are 96 training conversations per recovery family. Formatting has 96 per family, with two assistant turns per example; plain prose is the first target and a requested transformation is the second. No incorrect student draft is used as a training target. No teacher model is required.

The mixture reuses some examples seen by the old 512/768; it also selects additional examples from their frozen TRAIN pools. This is appropriate for training a fresh branch from the pretrained model. It does not load 512/768 weights or optimizer states.

## Training

- One pass, 768 updates, 16 whole conversations per update.
- Every batch: 10 ordinary conversations, 1 formatting follow-up, 1 recovery conversation, and 4 supplemental reading/science/writing records. Supplemental categories are spread across the entire run.
- Equal weight per conversation and equal weight among that conversation's assistant turns. The CE within each turn averages its assistant tokens.
- Native plain User/Assistant format. User/system/history positions are masked; complete current assistant targets include EOS.
- No answer truncation. Targets longer than 384 tokens or conversations that cannot fit the 2,048-token context are excluded.
- Full parameter SFT with one fresh AdamW optimizer. Betas 0.9/0.95, weight decay 0.01, gradient clipping 1.
- One continuous schedule: 64-update warmup to 2e-5, cosine decay to 2e-6 at update 768. No phase or optimizer reset.
- Evaluation/checkpoints at 64, 128, 256, 384, 512, 640, 768; initial evaluation at 0.
- Ctrl-C finishes the current update/evaluation, saves, and exits. Running the same command resumes the committed optimizer and data cursor. All intermediate checkpoints are retained.

## Evaluation and the goal of beating 768

Retained 768 and this run use the **same legacy retention prompts, greedy decoder, EOS rule, repetition detector, and response limits**. The run reports changes in follow-up joint pass proxy, chat stopping, chat repetition, and validation conversation NLL.

New development checks use 176 held-out conversations (32 each for conversation, formatting, recovery, science, writing; 16 reading). Reserved test data are not evaluated. Recovery and format checks are literal content/style proxies, not comprehensive semantic grades. Saved generated answers support qualitative review. Reading exact matches and short gold science answers need careful interpretation; likelihood is not factual accuracy.

Full public benchmark scoring uses the unchanged local runner: all 15,428 ARC Easy/Challenge, PIQA and HellaSwag items and all 541 IFEval prompts (834 instructions), with the same protocol as retained 768. Benchmark questions/overlapping prompt spans are excluded from training; corpus split groups and passage article boundaries remain separated. Template wording is intentionally shared in authored development checks. This is not a guarantee of complete semantic decontamination.

A useful replacement should show public instruction gains while retaining multiple-choice knowledge and conversation usefulness. Formatting gains accompanied by irrelevant answers, repetition, unwanted lists, or worse history handling do not establish an overall improvement. No public benchmark is used for automatic training termination or checkpoint selection.

## Commands

Start or resume training (production training is never started by preparation/check commands):

```sh
<SOURCE_WORKSPACE>/.venv/bin/python -u \
  <SOURCE_WORKSPACE>/sml-mlx-v1/experiments/unified_text_v1_sft_v1/scripts/launch.py \
  --run
```

Optional readiness or read-only baseline rerun: replace `--run` with `--check` or `--baselines`.

Benchmark a saved checkpoint (example update 512; use this experiment's step, not the old 768 parent step):

```sh
<SOURCE_WORKSPACE>/.venv/bin/python -u \
  <SOURCE_WORKSPACE>/sml-mlx-v1/experiments/unified_text_v1_sft_v1/scripts/benchmark.py \
  --step 512 --suite all --run
```

Print a side-by-side benchmark comparison after both suites finish:

```sh
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/experiments/unified_text_v1_sft_v1/scripts/compare.py \
  --step 512
```

New checkpoint numbers are SFT updates from the pretrained v1 source. A new step 768 is a different model from the preserved legacy 512→768 model.
