# Reading repair from OpenSML SFT step 448

This is a focused experiment in answering short questions from a supplied passage. It is not a general chat training recipe. It starts directly from the preserved response-repair checkpoint **448**, not the deleted response-diversity checkpoint 832.

## Run

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/reading_repair/launch.py \
  --run --clear-stop
```

Main Mac only. No reference model, teacher, or five-Mac launch. One pass: **512 additional optimizer updates**, cumulative steps **449–960**, batch 8 / microbatch 2. Constant **3e-6** learning rate. FP32 AdamW master weights, moments, and step counter are restored exactly from 448. Existing tokenizer, model, assistant-only loss masking, and EOS stay unchanged. Training is not started by preparation or readiness checks.

Output: `sml-mlx-v1/runs/sft_reading_repair_v1`. Ctrl-C requests a checkpoint after the current update/evaluation. Wait for `[stopped]`; the same command resumes the exact example position. A completed run does not automatically repeat.

## What the model sees

| Training component | Examples | Purpose |
| --- | ---: | --- |
| Original synthetic reading scenarios | 2,688 | Roles/ownership, contents, negation/availability, changed facts, sender/recipient, missing facts, false premises |
| SQuAD v2 human-written questions | 1,152 | 864 answerable and 288 genuinely labelled unanswerable questions from diverse Wikipedia passages |
| Previously consumed rehearsal examples | 256 | Preserve factual, greeting, clarification, and response skills learned before 448 |
| Total | 4,096 | One pass, deterministic mixed order |

**This revisits SQuAD.** Earlier SQuAD training did not produce a useful general assistant. Here it supplies 28.1% of examples, alongside targeted relational examples and rehearsal, starting from 448 at 3e-6 rather than the earlier 1e-4 peak. This is a hypothesis to test, not a claim that a different mix guarantees success.

Each original scenario changes the facts and the question independently. Train, development, and test use different entity lists and sentence/question renderings. These remain synthetic templates, so success on them alone is insufficient. An additional **28 independently written development questions** cover more natural situations. They are never supervised training examples.

SQuAD uses only the pinned official **train** split. Article titles determine the train/development/test assignment. Selection also checks paragraph separation, source answer spans, lengths, unique questions, and limits concentration by article and paragraph. There is no training on the official validation split or on our benchmark questions. Six ambiguous or malformed training rows found in a 32-row manual sample were excluded. Most selected examples have not received individual human review.

Dedicated math/programming and quantitative exercises are filtered. Factual dates and ordinary English or science passages can still appear; this is not a claim that every source topic is nontechnical. The lexical filter is inspectable in `source.py` and cannot guarantee perfect semantic filtering.

The pinned 16.37 MB source is streamed for hash verification and streamed again for record selection. At most 4,096 candidates per split/answerability pool are retained while selecting. The final small training/evaluation subset is held in RAM. No raw corpus file or token shard is saved. Dataset metadata may be cached by the streaming library. `selection.json` stores IDs, hashes, order and statistics; source review notes retain IDs and decisions, not a corpus cache.

## How we decide whether it helped

Development checks every 64 updates:

- 448 generated reading questions, including four-way changed-fact/question groups;
- 128 human questions from separate articles, reporting exact answer and token F1;
- 28 separately worded development challenges;
- the existing 64-pair transfer check, existing repair diagnostics, and fixed prose-loss diagnostic;
- repetition and output-length failures, with full generated answers saved for inspection.

A new best must improve generated family-macro exact accuracy by at least **2.5 percentage points**, answer **at least 2 more** independent challenges correctly, lose no more than **3 points** of human F1 or **2/64** legacy pairs, and keep prose drift within **+0.08**. Repetition/length-limit rates may not exceed the larger of the parent's rate and 1%. These are experimental selection thresholds, not statistical significance guarantees. Selection then prioritizes independent challenge correctness, reading exact accuracy, human F1, paired reading accuracy, and legacy retention. **Lower loss alone never selects a checkpoint.**

Reference matching is deliberately strict: it accepts listed answers after punctuation/article normalization, not arbitrary plausible paraphrases. Token F1 is a separate overlap measure, not semantic correctness. Review raw answers before promotion. The challenge is small; a two-answer change is noisy.

At the end, selection is frozen. **Parent 448, selected best, and final 960** are all evaluated on 576 separate test questions (448 original and 128 human), even if the selected best remains 448. Identical weights are evaluated once. Final test answers do not influence checkpoint selection. These private checks are not official SQuAD results and are not directly comparable to published scores.

No model is automatically promoted in the playground. The current playground remains on 448. Public benchmark results are preserved and excluded from training/selection; run benchmarks again only after a useful candidate emerges.

## Verification

`--prepare` rebuilds the deterministic selection. `--check` verifies the pinned data, parent/checkpoint hashes, exact FP32 optimizer restoration, all encodings, reference grading, finite real-model forward/backward computation, and cached/uncached generation agreement. It applies **zero real optimizer updates**. A changed recipe or code fails closed until rechecked.

`test_reading.py` checks wrong-answer rejection, split separation, source span validation, prompt/EOS masks, generation shapes, and identical weights/optimizer after interrupted versus uninterrupted tiny-model training. It also checks that the final checkpoint is included in testing when the parent remains best.

The original 448 weights and optimizer are protected by hashes. Earlier pretraining and retained SFT checkpoints are not rewritten. Data attribution is in `DATA_LICENSE.md`.
