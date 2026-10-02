# Balanced skills continuation from 896

This is a new OpenSML-only experiment, prepared for the user to launch on the main
Mac. Its purpose is to improve instruction compliance and commonsense while
preserving reading. It is **not** a demonstrated SmolLM2-Instruct winner.
See [the target comparison and plan](../../reports/SMOLLM2_TARGET_PLAN_20260925.md).

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/skill_balance/launch.py \
  --run --clear-stop
```

Default without `--run` prints the plan. `--prepare` rebuilds the pinned data
manifest; `--check` validates the objective and disposable resume. Preparation
must be reviewed before starting a new recipe. Once a production contract exists,
data/config/code changes are refused. Rerun the same command after an interruption
to resume the exact saved family cursors and optimizer state; completed jobs do
not extend automatically. Ctrl-C finishes the current operation, saves and stops.

## Recipe

- Parent: saved 896 weights upcast to FP32; fresh AdamW, not restored old moments.
- 1,536 new optimizer updates; 12,288 new task rows plus 3,072 reading replays.
  There are 9,216 distinct new source examples: each of the 3,072 CommonGen
  sentences is used with two different formatting requests. The other 6,144
  new examples are distinct QA questions. This is intentional paired-format
  exposure, not 12,288 independent new source sentences.
- Per update: 4 commonsense QA + 4 sentence/format instructions + 2 reading tasks.
- Supervised loss shares: QA 30%, instruction 30%, reading 25%, correct-choice
  ranking 15%. CE is averaged per example within each family, then weighted.
  Ranking compares token-normalized conditional answer likelihood at temperature
  1.0, without chat wrappers or EOS. It is an auxiliary labeled objective.
- Frozen-896 KL replay: weight 0.10, full vocabulary, training-only prose windows
  of at most 256 tokens. It encourages retention without inventing new answers.
- LR warms to 8e-6 in 96 updates and decays to 2e-6; zero weight decay; clip 1.0.
- Native 32K tokenizer and plain `User: ...\nAssistant:` format; assistant-only
  masks include EOS. Context 1,024, no silent example truncation. MLX FP32 explicit
  vanilla attention and reference FFN match the existing public evaluator.

## Data and scope

Immutable revision/file hashes are in `data.py` and `selection.json`. The three new
Parquet files stream into explicitly bounded buffers (largest 3.5 MB), are verified,
decoded in batches, then discarded. Accepted rows remain in RAM. Existing SQuAD
replay is reconstructed from its pinned stream. No raw corpus disk cache is created.
Small audit samples and row IDs/hashes are saved; this is not disk dataset sharding.

- [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa): card declares MIT.
- [SocialIQA](https://huggingface.co/datasets/allenai/social_i_qa): card declares CC BY 4.0.
- [CommonGen](https://huggingface.co/datasets/allenai/common_gen): card declares MIT.
- Existing [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) reading
  subset and original training scenarios, restricted to rows already consumed by 896.

Retain source attribution/license notices with downstream distribution. These
declared licenses permit commercial use subject to their terms; this recipe does
not assert an independent provenance audit of every upstream sentence.

No dedicated math/code sources or live teacher are used. Lexical filtering removes
detected math/code/specialist requests. CommonGen includes caption fragments;
selection requires capitalization, a finite-clause POS heuristic and concept
coverage, and rejects repeated/missing-content targets. POS tagging is a small
fixed linguistic filter, not an LLM teacher. Its pinned resource is `_tagger/`;
NLTK comes from the existing benchmark runtime. Surface/word-stem checks can miss
bad grammar or incorrect relationships. QA labels can also be ambiguous. Inspect
`review_samples.json`; neither “human” nor “verified formatting” means every fact
was independently checked.

The selected curriculum does not include long-form reasoning, multi-turn dialogue,
math, coding, broad science instruction or an all-purpose conversational corpus.
Eight output styles cannot teach every IFEval requirement; success on these private
formatting tasks is only evidence to investigate transfer.

## Evaluation and retention

New dev/test sources use separate concept sets, situations and questions; train,
dev and test transformations never share a source group. Public ARC/PIQA/HellaSwag/
IFEval prompts and choices are used only as rejection filters. No public labels
or official verifier outcomes enter the optimizer. Existing reading/prose checks
are reused and clearly labeled as retention diagnostics.

All development candidates and the two latest saves are retained (about 14–19 GiB
of weights/state during this recipe). `candidate.json` is an automatic **review
candidate**, not an approved best. Case/concept coverage is not semantic correctness.
The playground is never changed by this launcher. Original 896 and 906 stay intact.

After reviewing saved answers, freeze the chosen step and evaluate it once:

```bash
# Replace STEP with the reviewed global candidate step, such as 1152.
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/skill_balance/finalize.py \
  --step STEP --run
```

The finalizer records the choice before testing. Changing the choice after seeing
test results is refused. Public suites are then available through `benchmark.py`:
run `--suite multiple-choice --check`, then `--suite multiple-choice --run`, and
the analogous commands for `--suite ifeval`. It uses the existing frozen public
protocol and a separate results directory; it does not alter prior benchmark results.

Before declaring all SmolLM2 scores beaten, compare the released Instruct model
under a matched documented setup, inspect actual answers, and report uncertainty.

Checkpoint 1920 is retained with its benchmark results. Checkpoint 2432 and its
specific evaluation/benchmark artifacts were removed at the user's request on
2026-09-25. `latest.json` points to the surviving 1920 bundle; the original 896
remains preserved in its own run.

## Preparation verified on 2026-09-25

`readiness.json` records eight passing unit tests, a full deterministic rebuild
of the pinned selection, cached/full-context generation agreement on three
prompts, a small evaluation-path check, and three disposable real-model updates.
The uninterrupted and resumed model weights were identical; optimizer state
differed by at most 1.17e-10 in FP32. The frozen reference and protected inputs
were unchanged. These are correctness checks, not evidence of benchmark gains.
Production updates: **zero**. The production output directory was not created.

The selection contains 155,081 assistant target tokens including EOS: 24,500
commonsense, 109,738 instruction and 20,843 reading. Additional ranking and
prose-KL scoring are separate from that count. Three inspected CommonGen targets
were explicitly excluded for incorrect grammar or unhelpful niche slang; their
IDs and reasons are recorded in `selection.json`. The remaining source pool has
heuristic filtering and sample review, not an exhaustive manual quality audit.
