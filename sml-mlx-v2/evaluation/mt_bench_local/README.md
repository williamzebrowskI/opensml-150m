# MT-Bench with a local Prometheus judge

An inference-only evaluation for preserved OpenSML v2 chat checkpoints. No checkpoint is selected by default. No training, paid API, automatic promotion, or checkpoint changes.

## Ready-to-use commands

From the repository root, list the registered checkpoints:

```bash
.venv/bin/python sml-mlx-v2/evaluation/mt_bench_local/run.py --list
```

Once a checkpoint is chosen, replace `2688` with `640`, `768` (Two-Turn High LR 640 → Step 768), `1920`, or `2048` as desired:

```bash
.venv/bin/python -u sml-mlx-v2/evaluation/mt_bench_local/run.py --model 2688 --run
```

The same command resumes interrupted work. Each completed conversation and each judge result is committed separately. Optional `--stage answers` generates answers only; `--stage judge` judges an already complete answer file. `--model 2688 --summary` rebuilds the report from saved results.

To verify local assets and run three synthetic judge sanity checks (no checkpoint selection or benchmark run):

```bash
.venv/bin/python -u sml-mlx-v2/evaluation/mt_bench_local/run.py --check
```

The approximately 7.2 GiB MLX judge is downloaded in `judge_model/`. Runtime loading is local. To restore missing assets on another checkout, run `prepare.py`, then `run.py --check`. The model download needs Internet access; generation and judging do not.

## What this measures

- The unchanged official 80 MT-Bench questions, two turns each, across eight categories.
- Local Prometheus 2 7B **helpfulness/relevance scores from 1 to 5**, using its published helpfulness rubric. Higher is better.
- Average score by category and by turn; the overall mean is shown only after all 160 judgments complete.
- Responses, judge explanations, and stopping/context-limit statistics, saved for manual inspection.

**This is MT-Bench questions with a local Prometheus scoring protocol. It is not the standard GPT-4-judged MT-Bench 1–10 leaderboard score.** Do not rescale the numbers and claim equivalence. Model judgments can be wrong, including on correctness and on instructions embedded in responses. Inspect the saved conversations; three sanity checks verify wiring, not evaluation accuracy. The benchmark is challenging for a 150M model and does not comprehensively measure greetings or everyday social chat.

## Generation protocol

- Exact registered checkpoint weight hashes; FP32 native OpenSML inference with cached decoding.
- Native `User: ...\nAssistant:` template, no injected system instruction. The second turn uses the actual first generated answer through the same `Previous messages` / `Current message` formatter as the playground.
- One answer choice; category temperatures from FastChat (0.7 writing/roleplay, 0.1 STEM/humanities, 0 otherwise).
- Top-k 50 (the usual Transformers sampling default used by the upstream generation path), top-p 1, repetition penalty 1. MLX random seed 0 reset for every conversation. Random draws are not bit-identical to PyTorch.
- Up to 1,024 generated tokens per turn, with the model's 2,048-token context limit. Never silently drop history. Reduced budgets/overflow are recorded as `context_limit`; an overflowing prompt gets an empty answer and is still included in scoring.
- Stop at the native EOS token. Visible structural tokens, repetition, and irrelevant text are not silently repaired.

## Judge protocol

Pinned `mlx-community/prometheus-7b-v2.0-8bit`, an MLX conversion of `prometheus-eval/prometheus-7b-v2.0`. The model's Mistral chat template wraps the upstream absolute-grading system prompt and helpfulness rubric. Greedy judging, up to 1,024 tokens; no prompt truncation.

For math, coding, and reasoning, include the upstream GPT-4 reference answers (already published static data; no API call). Other categories use the upstream no-reference prompt. For the second turn, the judge sees the model's actual conversation history and scores only the latest response. Reference history is presented separately.

Invalid/unparseable judge output halts the run and saves `invalid_judgment.json`; no made-up score, zero substitution, or silently excluded item. A deterministic invalid output may need inspection rather than repeated retries.

## Files and reproducibility

Results: `sml-mlx-v2/diagnostics/mt_bench_local_v1/<model-id>/`

- `contract.json`: selected checkpoint, protocol, asset/code hashes and runtime versions.
- `answers.jsonl`: raw responses, prompts, token IDs and stopping reasons.
- `model_answer/<model-id>.jsonl`: standard FastChat answer format, suitable for separate judging.
- `judgments.jsonl`: per-turn local scores and explanations.
- `summary.json`: counts and averages.
- `conversations.md`: readable questions, answers and judgments.

Results cannot resume across a changed protocol. All runs use the project's shared experiment lock. Run checkpoints sequentially, with the same judge and protocol for comparisons. Readiness checks do not start the benchmark.

Sources and licenses are pinned in `sources.json`, `assets.json`, and `upstream/`:

- https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
- https://github.com/prometheus-eval/prometheus-eval
- https://huggingface.co/mlx-community/prometheus-7b-v2.0-8bit

The upstream Apache-2.0 license files are included. No benchmark examples are added to training data.
