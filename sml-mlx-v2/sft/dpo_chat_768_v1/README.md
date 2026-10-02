# Text-only DPO from Text Follow-up 768

A separate, longer preference-training experiment. The retained parent is
`runs/sft_text_followup_512_v1/step_0000768_10f5ea2a3354`.
It is never overwritten. This is not the older preference-2688 branch.

## Run or resume

```sh
<SOURCE_WORKSPACE>/.venv/bin/python -u \
  <SOURCE_WORKSPACE>/sml-mlx-v2/sft/dpo_chat_768_v1/launch.py \
  --run
```

The same command resumes the saved optimizer and data cursor. Run one GPU
training/evaluation job at a time. A graceful interrupt saves after the current
update/evaluation. An interrupted unsaved update is replayed from the last saved
checkpoint. A manually created `runs/dpo_chat_768_v1/STOP` marker requires
`--clear-stop` to resume. There are no automatic quality-based stops or pruning.
Invalid data, changed inputs, nonfinite gradients, or insufficient disk space
remain errors.

## Length and data

- 1,024 optimizer updates, cumulative step 768 to 1,792.
- 1,024 unique preference pairs: 1,009 filtered HelpSteer2 pairs and 15 reviewed
  pairs containing actual frozen-768 responses and authored better answers.
- Two passes through those pairs, two pairs per update. This is **2,048 pair
  exposures**, not 2,048 distinct pairs.
- 2,048 unique previously seen training conversations, also replayed twice:
  three multi-turn chat conversations and one reading/writing/science text
  conversation per update. Complete assistant replies and EOS are supervised.
- No coding, math, or JSON tasks are intended. Scope filters cover prompts and
  both candidate answers. Public answers are capped at 512 tokens; replay
  assistant replies at 384. Full sequences must fit 2,048 tokens; no silent
  truncation. One reviewed repetitive model negative was generation-capped;
  its provenance records that it was incomplete.

HelpSteer2 supplies human preferences and quality ratings. Chosen replies need
at least 3/4 for helpfulness, correctness and coherence. Length-ratio, duplicate,
scope and benchmark-overlap filters also apply. A sample was inspected and
unsuitable rows removed. These are **not independently verified labels for
every pair**. See `review.json`, `model_review.json`, `rejections.json` and the
frozen preparation receipt. UltraFeedback was inspected and rejected during
setup; it is not in this run.

## Objective and conversation retention

The policy and frozen reference both start at the identical parent768 weights.
Standard reference DPO uses summed answer log probabilities, including EOS,
with beta 0.1. Prompt and padding tokens are excluded.

The combined objective uses weights 0.25 for DPO, 0.10 for preferred-answer
cross entropy, 0.65 for replay cross entropy, and 0.10 for forward KL from
frozen768 on replay answer positions. These are loss coefficients, not data
percentages. Replay is balanced by conversation and assistant turn so long
replies do not automatically dominate. KL covers all vocabulary probabilities
at those supervised positions.

Learning rate warms up for 64 updates to 3e-7 and decays to 3e-8. The frozen
reference and replay reduce pressure to forget prior behavior, but do not
guarantee retention. Compare actual conversations as well as metrics.

## Checkpoints and evaluation

Saved additional updates: 64, 128, 256, 384, 512, 640, 768, 896, 1024.
Cumulative steps: 832, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792.
Every checkpoint is retained; reserve at least 30 GiB free.

An evaluation runs before training and at each save point. It uses **the same
old Text Follow-up 768 development data, prompt format and generation settings**.
This preserves comparability with 768's earlier development results. It also
scores 64 held-out preference pairs; another 64 remain reserved and unused.
The preference relative win rate measures likelihood-ratio changes versus
frozen768, not a human or MT-Bench conversational win rate.

The printed summary includes stopping, repetition, follow-up checks, chat-only
checks, validation loss and preference loss. Saved JSON and Markdown include
generated responses. No automatic best selection. Full IFEval, multiple-choice
and MT-Bench are separate post-training evaluations, not run every checkpoint.

Existing parent development checks use templates and literal content matching;
they do not establish semantic correctness. Source/benchmark exclusion uses
exact matches and 13-word overlap heuristics, not proof of zero contamination.

## Validation

`--inspect` checks all prepared sequences and exposure schedules without training.
`--check` uses disposable model copies and temporary checkpoints to verify DPO
gradient direction, masked likelihoods and EOS, native conversation prompt
parity, cached decoding, zero initial KL, unchanged reference weights, exact
optimizer restoration and exact next-update weights. It smoke-tests evaluation
and verifies protected files stayed unchanged. Results are in `readiness.json`.
These checks do not run production training.

## Sources and attribution

- NVIDIA [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2),
  CC BY 4.0, pinned revision `990b2711a36180dd19d9c94b8627844866f8982a`.
  Original train and preference files are fingerprinted in `_sources`.
  This preparation selects, filters and wraps original text into chat roles.
- Replay comes from the existing parent lineage's Smol-SmolTalk, UltraChat,
  reading and science TRAIN material. Original per-row provenance is retained.
- [DPO documentation](https://huggingface.co/docs/trl/dpo_trainer).
- [SmolLM2 SFT/DPO recipe](https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/README.md).
  This custom native-MLX recipe is not a reproduction of that model's training.
