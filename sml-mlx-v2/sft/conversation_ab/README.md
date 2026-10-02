# Conversational A/B pilot — branch A

Branch A starts from preserved balanced-skills checkpoint 1920. It uses ordinary supervised fine-tuning with assistant-only cross entropy, including the end-of-answer token. There is no unlikelihood loss, RL, DPO, teacher-model inference, or automatic playground promotion.

Run on the main Mac:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/sft/conversation_ab/launch.py \
  --run --clear-stop
```

The default invocation without `--run` prints the plan. After changing training inputs or code, run the same launcher with `--check` to rebuild readiness evidence. Checks reconstruct the data and perform disposable update/resume tests in a temporary directory; they do not create the production run.

## Frozen experiment

| Setting | Value |
|---|---|
| Parent | `runs/sft_skill_balance_v1/step_0001920_628a0a65099b` |
| Branch A output | `runs/sft_conversation_ab_v1/A` |
| Unique training examples | 160 |
| Public-source conversations | 96 reviewed OASST2 examples |
| New original examples | 24 assistant-authored examples |
| Rehearsal | 40 examples already consumed before step 1920 |
| Passes | 3, shuffled separately and deterministically |
| Updates / batch | 120 / 4; final model step 2040 |
| Assistant targets per pass | 7,573, including EOS |
| Context | 1,024; no example truncation |
| Learning rate | 12-update warmup to 8e-6, cosine decay to 2e-6 |
| Optimizer | Fresh FP32 AdamW, no weight decay, gradient clipping at 1 |
| Development evaluations | Before training and after updates 40, 80, 120 |

Training averages each example's assistant-token loss, then averages the four examples equally. Reported development NLL is token-weighted, so its weighting differs from the training objective. Prompt and conversation-history tokens receive no supervised loss. Each answer includes EOS. Follow-ups use the playground's existing `Previous messages` / `Current message` format.

## Data and scope

Public data: [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2), revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`, declared Apache-2.0. The immutable 63.5 MB Parquet is streamed into RAM, hash-checked, decoded, and discarded after selection. Selected rows remain in RAM for all three passes. Previously consumed rehearsal sources are also reconstructed through the existing streaming loaders. No raw corpus is cached by this recipe on disk.

Source filtering requires English, reviewed non-synthetic rows, highly rated rank-zero answers, complete ancestry, and bounded answer length. Code and math content are excluded. The approved public rows and original examples received assistant review for relevance, completeness and obvious factual errors; this is not independent human verification or a guarantee of correctness. Public source metadata does not guarantee that every contributor avoided AI assistance. The 24 new examples were authored by the coding assistant; no separate teacher model is run during preparation or training.

Whole conversation trees are assigned to splits before selection, with at most one selected answer per tree. There are 24 development and 24 reserved test examples each: eight public-source conversations and 16 original prompts. Previously used benchmark and audit prompts are excluded where available. These small evaluations cannot establish general conversational competence or statistical superiority.

This is a small curated pilot, not a broad conversational training corpus. Three passes may overfit; inspect candidates after each pass. The size, mix and learning rate are experimental choices, not demonstrated optima. Rehearsal preserves exposure to reading, commonsense and formatting tasks but does not guarantee retention.

## Review and resume

The run saves complete development answers plus repetition, stopping, likelihood and retention diagnostics. Lower loss alone does not select a winner. Review relevance, correctness, completeness, and repetition against the unchanged parent. Reserved test generations remain untouched until a candidate is frozen; there is no automatic best checkpoint or promotion.

Ctrl-C requests stopping at an update boundary and saving a complete checkpoint. Re-run the command above to resume the saved optimizer and exact example cursor. Input hashes and runtime versions must match. A completed run will not extend automatically.

Checkpoint 1920 and all prior runs are preserved. Branch B is not launched or implemented by this command. A later B comparison must start from unchanged 1920 with this same data, ordering and update budget, adding only the proposed repetition-aware training term. It must not continue from branch A's final checkpoint.
