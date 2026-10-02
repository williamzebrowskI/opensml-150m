# Full public evaluation of saved OpenSML checkpoints

Two inference-only commands run on the main Mac. They do not train, load an
optimizer, change checkpoint weights, or change the playground's selected model.
The default is the preserved **SFT step 448**, not latest step 512.

## Reading-repair checkpoint 896

Use the explicit `sft-896` model ID for the selected reading-repair checkpoint.
This pins step 896, rather than the final step 960. Run these sequentially:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/evaluation/full_benchmarks/launch.py \
  --model sft-896 --suite multiple-choice --run
```

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/evaluation/full_benchmarks/launch.py \
  --model sft-896 --suite ifeval --run
```

The protocol is unchanged from 448. Results go to
`diagnostics/full_benchmarks_v1/sft-896/`, preserving the existing 448 results.
Each suite needs its own readiness receipt; replace `--run` with `--check` if
the code or inputs change after preparation. Completed historical results retain
their original manifests; registering another model does not rewrite them.

## Run

Full multiple-choice suite:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/evaluation/full_benchmarks/launch.py \
  --suite multiple-choice --run
```

Then run IFEval separately:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/evaluation/full_benchmarks/launch.py \
  --suite ifeval --run
```

Run one command at a time. The shared experiment lock prevents overlap with
other launchers that use the same lock. The existing playground may stay open;
using it during evaluation competes for the same GPU and can affect duration.

Ctrl-C saves completed items; use **the same command** to resume. An interrupted
IFEval response restarts that one prompt. Already completed evaluations are not
repeated. The launcher rejects resumes if the model, data, runtime or evaluation
code changed. No continuous job or automatic restart is installed.

## What is evaluated

| Task | Official split | Items | Purpose |
|---|---|---:|---|
| ARC-Easy | test | 2,376 | Science question answering |
| ARC-Challenge | test | 1,172 | Harder science question answering |
| PIQA | validation | 1,838 | Everyday physical commonsense |
| HellaSwag | validation | 10,042 | Plausible text continuation |
| IFEval | upstream split named `train` | 541 prompts, 834 instructions | Verifiable instruction following |

The four multiple-choice tasks total **15,428 questions / 58,032 candidate
answers**. These are full official splits; no duplicates or difficult rows are
removed. The revisions and source-byte hashes match the pinned PetitGPT v1
protocol. IFEval's split is named `train` upstream, but this runner uses every
row for evaluation only; nothing enters the training pipeline.

Our pretraining used general/educational prose, without a dedicated math/code
curriculum. Stage B also applied lexical math/code filters, which do not prove
perfect exclusion. Science was not categorically excluded. ARC therefore measures
broader scientific knowledge and should be reported separately, not treated as
the sole quality gate for a short prose assistant. Official IFEval prompts can
also contain topics, languages or constraints beyond our intended use. We keep
the complete official suite rather than quietly filtering it to improve scores.
No coding or math training job is introduced by this evaluation.

## Protocol

- Load the original OpenSML model and tokenizer; verify bundle, source model SHA,
  base architecture metadata and frozen tokenizer identity. No HF model conversion
  is needed. Upcast saved parameters to FP32 in memory.
- Multiple-choice: zero-shot official prompts, no SFT wrapper, BOS or EOS score.
  ARC and PIQA use `Question: ...\nAnswer:`. HellaSwag uses the pinned harness
  preprocessing. Joint-tokenize prompt + one space + answer, verify the prompt
  prefix is unchanged, and score only causally shifted candidate tokens.
- Report both **acc** (summed answer log probability) and **acc_norm** (divide
  that sum by candidate Unicode character count, excluding the added separator).
  First maximum wins ties. No combined score across tasks is invented.
- Batch size 1, no padding, no candidate KV cache. FP32 weights/logits/likelihood
  reductions, reference SwiGLU and explicit vanilla attention in MLX.
- IFEval: unchanged official user prompt in this model's trained plain format
  `User: {prompt}\nAssistant:`. Greedy generation, batch size 1, KV cache,
  **1,280 new-token maximum**, repetition penalty 1, existing EOS only, no other
  stopping strings or token suppression. All 541 prompts fit the full budget
  within context 2,048; the longest is 375 tokens. No input is truncated.
- Preserve generated whitespace. The vendored official task's strict/loose
  checks determine any scoring normalization. Record raw generated token IDs,
  full response, EOS/length stop, per-instruction flags, prompt accuracy and
  instruction accuracy. Cap-hit outputs are scored as generated.
- Verifier files are byte-for-byte from EleutherAI's IFEval implementation at
  `b954108c9baaaa934b4ad842033b31a97ee30816`, adapted upstream from Google Research;
  our wrapper calls them unchanged. Dependencies and NLTK tables are local to
  this directory and pinned. Random defaults and language detection are seeded.

This is a **native MLX evaluation following the pinned harness protocol**, not
an installed `lm_eval` run. PetitGPT used native PyTorch FP32 MATH attention;
we do not claim bit-identical cross-framework numerical results. Dataset and
metric alignment enables a more meaningful comparison, not a controlled
experiment isolating architecture, data or compute. Pretraining contamination
cannot be ruled out. The multiple-choice tasks measure candidate selection,
not reliable conversation; IFEval scores formal constraints, not full factual
correctness. Preserve generated-answer review alongside these scores.

## Outputs and provenance

Results are written separately:

```text
sml-mlx-v2/diagnostics/full_benchmarks_v1/sft-448/multiple-choice/
sml-mlx-v2/diagnostics/full_benchmarks_v1/sft-448/ifeval/
```

Each contains `manifest.json`, resumable `records.jsonl`, `summary.json` and an
integrity receipt. A partial summary explicitly says `partial`; each task keeps
its expected denominator. The manifest freezes checkpoint/input/code hashes and
runtime versions. Any incomplete final journal write is retained in
`records.incomplete-tail` before resuming from the preceding committed item.

The prepared question cache stays under `_cache/`, independent of all training
loaders. It is evaluation data, not an SFT dataset. Keep it and the questions out
of future training. Third-party data remains subject to its original terms; this
setup does not relicense or publish the question sets.

## Checks and optional controls

Readiness checks load the actual checkpoint, validate all candidate boundaries
or all official instruction types, then use synthetic fixtures for numerical
likelihood and cached-generation checks. They do **not** run full official model
evaluations. Nine tests cover scoring/masking/normalization, first-maximum ties,
context rejection, official strict/loose grading, partial summaries, and an
interrupted/resumed runner. Test outputs use a temporary directory.

```bash
cd <SOURCE_WORKSPACE>
.venv/bin/python sml-mlx-v2/evaluation/full_benchmarks/test_evaluation.py
.venv/bin/python sml-mlx-v2/evaluation/full_benchmarks/launch.py --suite multiple-choice --check
.venv/bin/python sml-mlx-v2/evaluation/full_benchmarks/launch.py --suite ifeval --check
```

Optional controls: add `--model sft-384` to either command, or `--model pretrained`
to multiple-choice only. Each control needs its own `--check` first. Outputs stay
separate by model. The pretrained model has no trained instruction template, so
IFEval deliberately rejects it rather than silently choosing a chat format.

If this setup is copied to another checkout, first install the pinned IFEval
requirements into `_runtime` (using a package manager targeting this Python),
then run `prepare.py` and both readiness checks. On this Mac those downloads and
local dependency installation have already been completed. The training virtual
environment's installed packages were not modified.

## Sources

- [PetitGPT evaluation protocol](https://github.com/yangqi0/petitgpt/blob/53a5bb33052fa8092987eacc1047e040a7c5ce30/docs/petitgpt-v1/provenance/EVALUATION_PROTOCOLS.json)
- [Pinned ARC task](https://github.com/EleutherAI/lm-evaluation-harness/blob/b954108c9baaaa934b4ad842033b31a97ee30816/lm_eval/tasks/arc/arc_easy.yaml)
- [Pinned HellaSwag preprocessing](https://github.com/EleutherAI/lm-evaluation-harness/blob/b954108c9baaaa934b4ad842033b31a97ee30816/lm_eval/tasks/hellaswag/utils.py)
- [Pinned IFEval verifier](https://github.com/EleutherAI/lm-evaluation-harness/tree/b954108c9baaaa934b4ad842033b31a97ee30816/lm_eval/tasks/ifeval)
- [Original Google IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)

`vendor_sources.json` records exact downloaded verifier URLs and SHA-256 hashes.
Original copyright headers and the harness license are preserved under `vendor/`.
