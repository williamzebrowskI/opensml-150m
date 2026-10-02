# Complete answers with constraints, starting from 1920

This experiment continues the **preserved Balanced Skills 1920** checkpoint,
not the 512/640/768/1024/1472 branch. It targets the tendency to emit a short
fragment when a request needs several complete points or sentences. It is an
experiment in instruction completion; it does not promise higher scores on
every benchmark or a ready conversational assistant.

## Run

The dataset selection and disposable readiness tests are prepared separately.
Once readiness passes, run:

```sh
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/sft/constraint_completion_1920/launch.py \
  --run --clear-stop
```

The same command resumes saved weights, FP32 optimizer state and exact data
cursors after interruption. A completed run cannot silently extend itself.
Ctrl-C requests a save after the current operation. No remote Mac is used.

Output is `runs/sft_constraint_completion_1920_v1`. Existing weights are read-only.
128 additional updates end at step **2048**. This is a different checkpoint from
older experiments whose final step was also 2048; always use the full run name.

All nine scheduled checkpoints are retained: **1920, 1936, 1952, 1968, 1984,
2000, 2016, 2032, 2048**. A stop between these also saves the current state.
Complete development evaluations occur at 1920, 1936, 1952, 1984, 2016 and 2048.
The launcher requires 35 GiB free for checkpoint storage plus transaction reserve.

## Data and objective

- 1,024 distinct selected SmolTalk `smol-constraints` TRAIN rows, one pass.
- 1,024 replay exposures already consumed by the original 1920: CommonsenseQA,
  SocialIQA, formatted CommonGen and reading/reply examples. Replay is restricted
  using the parent's exact saved exposure cursors.
- Each update: eight new complete responses, two commonsense choices, three
  formatted examples and three reading examples.
- Loss weights: 50% new assistant-only CE, 15% format CE, 20% reading CE,
  15% answer-choice ranking. QA answer fragments receive no conversational CE.
  A separate 0.1 frozen-parent prose KL term discourages drift. Every assistant
  target includes EOS. Long responses do not dominate simply by token count.
- Fresh FP32 AdamW, 16-update warmup, peak LR `5e-6`, cosine decay to `1e-6`.
  This is a conservative pilot setting, not a claim of optimal learning rate.

The public source is pinned at revision
`5feaf2fd3ffca7c237fc38d1861bc30365d48ffa` and byte-verified before use. Its
16.9 MB TRAIN file is decoded in bounded RAM. No raw dataset is cached on disk;
only IDs/hashes, a small explicit review sample and generated evaluation answers
are saved. The publisher created these targets synthetically; no teacher model
is run locally.

The initial 2,000–4,000-example aim was reduced after screening. We do not fill
the budget by repeating near-duplicate templates. Screening rejects missing
tasks, unfilled placeholders, copied/embedded answers, obvious truncation,
unsupported constraint syntax, and targets that fail the implemented checks.
Targets are 30–150 words. Supported constraints include sentence/word/bullet
counts, sections, paragraphs, case, keywords, titles and postscripts. This is
not full coverage of every IFEval requirement (for example arbitrary JSON).

Topic-word similarity and 13-word answer overlap remove near duplicates before
group-based splitting. Public benchmark text and existing probes are used only
for overlap rejection. These filters do **not** prove semantic independence,
absence of pretraining overlap, or correctness of every source answer. A spread
of 32 TRAIN examples was read in full, alongside exploratory samples; specific
bad targets were excluded. Many remaining examples are generic practical or
business explanations. Review limitations are recorded in `review.json`.

Sources:
- https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- https://arxiv.org/html/2502.02737v1 (SmolTalk/Smol-Constraint construction)
- https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/README.md

## Evaluation and selection

64 new development source groups and eight authored development situations
check complete responses; 64 separate source groups plus eight authored
situations are reserved for one frozen comparison. Existing commonsense,
reading, formatting, natural-answer and prose diagnostics measure retention.
The existing retention/public suites are reused, not untouched tests.

`all_rules` and `rules_and_surface` are deliberately labeled **proxies**. A long,
well-formatted but wrong answer fails semantic review. Read the full saved
answers and mark correctness, relevance and completion, including all requested
parts. Compare parent and candidate reviews; inspect retention answers too.
The script never chooses a best model automatically and never promotes one to
the playground.

After reviewing a development candidate, `finalize.py --step STEP --run` freezes
it before generating reserved parent/candidate answers. It requires completed
semantic reviews, improvement over the parent's semantic score and passing
retention guards. `benchmark.py --step STEP --suite all --check`, then `--run`,
uses the same existing public benchmark implementation and settings. Do not
select another checkpoint using the reserved answers.

## Readiness

`launch.py --prepare` rebuilds and pins the selected data. `launch.py --check`
verifies all target masks and EOS boundaries, independent verifier fixtures,
cached/full-context generation parity, and the full evaluator on a small
TRAIN-only new-task fixture. It runs eight disposable updates on a fixed TRAIN
batch, verifies that its target loss falls and parent weights remain unchanged,
then checks exact save/resume equivalence and both benchmark contracts. These
temporary weights are deleted; no production run or public benchmark is started.

The precise successful results, source hashes and implementation hashes are
recorded in `readiness.json`. Changed code/config/data requires a fresh check.
