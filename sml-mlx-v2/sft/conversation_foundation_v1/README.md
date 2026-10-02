# Fresh V2 conversation foundation

Starts from preserved V2 pretrained step 73243 (SHA256 `ca9bd5d82005f28e77f319d3a5a29f29fb4e4f86e7ceea049f01162fd8aae80f`). No earlier post-trained weights are used. This is an experimental recipe, not a promise of a high MT-Bench score.

## Why this experiment

Inspection of excerpts from checkpoint 384's saved MT-Bench responses found missed requests, repeated passages, weak follow-up edits, and unsupported claims. The previous 384 result alone cannot isolate dataset, optimization, model-capacity, or pretraining limitations. This run changes data coverage and loss weighting together; it is not a causal ablation. Public benchmark prompts are used only for exclusion and final evaluation, never as training targets or the in-training development set.

## Recipe

- 32,000 distinct conversations, one shuffled pass; batch 16, 2,000 updates.
- 24,000 Smol-SmolTalk conversations: 19,600 Magpie, 400 everyday, 2,000 rewriting, 1,000 summarization, 1,000 coding.
- 8,000 UltraChat conversations from `train_sft` only.
- Complete conversations must fit the existing 2,048-token native history format. Whole rows are rejected if any answer exceeds 768 tokens or its context overflows. Answers/history are not cut to fit. The selection therefore favors shorter conversations and is not the entire publisher distribution.
- Supervise all assistant replies and EOS; mask instructions, user messages and historical replies. Equal conversation weight and equal turn weight within a conversation, rather than giving long replies greater weight through their token counts.
- Fresh FP32 AdamW; peak LR 2e-5, 100-update warmup, cosine decay to 2e-6, weight decay .01, gradient clipping 1. No preference loss, teacher generation, or ranking loss.
- Held out 384 development and 384 reserved test conversations, using deterministic root grouping and cross-split prompt overlap checks. The reserved test is not used by the training evaluator.
- Exact normalized and 13-word overlap filtering against MT-Bench, IFEval, ARC, PIQA and HellaSwag prompts, including training answers. This does not establish semantic or pretraining decontamination.
- Raw Parquet sources are pinned to revisions, fully SHA256-verified in memory, then discarded. A checksummed candidate pool and selected messages are retained locally for reproducibility and inexpensive refiltering.

Source cards: [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) (Apache-2.0) and [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (MIT). The mixture is a hypothesis for this model, not a published optimal ratio. Synthetic responses can contain factual, reasoning, and instruction-following mistakes; quality filters and limited manual sample review do not verify every target.

## Use

From the project root:

```bash
.venv/bin/python -u sml-mlx-v2/sft/conversation_foundation_v1/launch.py --run
```

Preparation and readiness checks are separate setup stages (`--prepare`, then sample review, then `--check`). `--run` requires the prepared dataset and matching passed readiness receipt. The setup check performs disposable optimizer updates in memory and in a temporary directory; it never changes the pretrained source or starts production training.

Ctrl-C requests a checkpoint after the current update/evaluation. Rerun the same command to resume with the saved optimizer and conversation cursor. Repetition warnings do not automatically stop training. `--run --clear-stop` clears the STOP file left by the earlier policy; manual Ctrl-C remains available. No checkpoint is automatically promoted or deleted.

## Evaluation and checkpoints

Evaluate at updates 0, 64, 128, 256, 512, 1000, 1500 and 2000. Checkpoints are retained at those nonzero updates and graceful interruptions. Plan approximately 17 GiB for the seven scheduled bundles, plus room for temporary writes; the launcher requires 35 GiB free for a new run. Repeated interruptions can add bundles.

Development evaluation measures assistant NLL on 128 held-out conversations, then generates up to two replies using the model's own previous response on 24 source-balanced held-out conversations. It records EOS/length/context stops, repetition, empty replies, and all generated text. These are diagnostics, **not correctness, relevance or MT-Bench scores**. A majority of repetitive responses at two consecutive nonzero evaluations prints a warning and continues. This was changed at the user’s request after step 256; the previous contract and readiness checks are archived under the run’s `policy_changes/` directory. Lower NLL or longer replies do not automatically select a model.

After reviewing development responses, select a checkpoint for the same full MT-Bench/local-Prometheus protocol used previously, plus full knowledge/IFEval retention checks. No public benchmark is run during training. Local Prometheus scores are 1–5 and are not official GPT-4 MT-Bench scores.

Output: `sml-mlx-v2/runs/sft_conversation_foundation_v1/`. Checkpoint metadata includes its pretrained parent, training format, tokenizer and frozen input contract. Training, playground serving and evaluation use the same existing `plain-user-assistant-eos-v1` serializer.

Once you choose an evaluated checkpoint, run the existing MT-Bench protocol with (replace 512 with the selected step):

```bash
.venv/bin/python -u sml-mlx-v2/sft/conversation_foundation_v1/benchmark.py --step 512 --run
```

It reuses the original judge and questions; it does not silently alter the scoring protocol. As in earlier evaluations, an invalid judge score is saved and reported as an error, never converted to a model score. The same command resumes completed answers/judgments. A deterministic invalid judgment can require the explicit missing-score recovery used in prior evaluations.
