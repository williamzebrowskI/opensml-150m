# Plain-text follow-up from Conversation Foundation 512

A separate 256-update SFT experiment. It preserves the original foundation checkpoint and every new checkpoint. No automatic quality stop, pruning, model selection, or public benchmark execution.

## Start or resume

```bash
<SOURCE_WORKSPACE>/.venv/bin/python -u <SOURCE_WORKSPACE>/sml-mlx-v1/sft/text_followup_512_v1/launch.py --run
```

Wait for other training/benchmarks to finish. The shared experiment lock prevents concurrent runs. On first launch, disposable setup checks run before production training. The baseline evaluation at update zero follows these checks. The same command resumes from the latest saved checkpoint with its optimizer and exact data cursor. Ctrl-C requests a save after the current update/evaluation. No checkpoint is deleted.

## Frozen recipe

- Source: Conversation Foundation 512, weight SHA256 `9f9574d7ba0cae82720e139f7889c1ea3a4b8ac81e9664fd59883839c669c5f2`.
- 4,096 conversations, one pass, 16 conversations per update.
- Every batch contains 8 replay conversations and 4 plain-text constraint conversations; the remaining 4 are shuffled reading, science, or writing tasks.
- Total: 2,048 conversation replay, 1,024 authored constraint conversations, 320 passage questions, 320 science questions, 384 rewriting/summarization examples.
- Replay uses only rows within the first 8,192 training conversations consumed by source 512. Source labels are preserved in each row's origin. Reading is answerable SQuAD data; science is from ARC **training** splits, using direct text answers and no answer-choice ranking loss.
- No intentional JSON, coding, or mathematical tasks; filters reject structured syntax, code terminology, and numerical science questions. Natural-language bullet and numbered lists remain supported.
- Plain-text constraints: bullets, numbered lists, paragraphs, case, required ending, sentence count, and quotation marks. Two-turn fictional tasks include invitations, thank-you messages, borrowing requests, apologies, birthday messages, and lost-property notices, followed by a requested text transformation. Invitations also require applying a date correction. These cover a narrow set of text transformations, not all of IFEval.
- FP32, assistant-only complete-reply/EOS loss, equal conversation/turn weights, native existing prompt format.
- Fresh optimizer, peak LR 3e-6, 16 warmup updates, cosine decay to 3e-7, gradient clipping 1.
- Local update 0 evaluates the unchanged parent. Saves/evaluations after 64, 128, 192, 256 updates correspond to cumulative steps **576, 640, 704, 768 in this new branch**. They are unrelated to historical checkpoints with the same numbers.
- Output: `sml-mlx-v1/runs/sft_text_followup_512_v1`.

## Evaluation and limitations

A held-out development set includes conversation, writing, reading, science, and text constraints. Each evaluation writes full generated responses and teacher references to JSON and a readable Markdown review. NLL, stopping, repetition, and literal text-constraint/content proxies are reported. It also reports changes from the unchanged 512 baseline. None is an automatic best criterion. Conversational usefulness still needs response review. The reserved test split is not evaluated during training.

Data separation uses the original source splits plus exact and 13-word overlap exclusion against public MT-Bench, IFEval, ARC, PIQA, and HellaSwag prompts. Authored constraints share task templates across split-disjoint fictional scenarios: their scores are narrow diagnostics, not proof of broad generalization. Source replay/writing contain synthetic answers; automated checks and sample review are not comprehensive factual or semantic validation. No method guarantees that all benchmarks improve or that every prior behavior is preserved.

Public benchmark results are kept out of gradient updates. Broad IFEval contains JSON and other out-of-scope instructions; this text-only recipe does not attempt to cover all of them.
