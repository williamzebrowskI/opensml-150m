# Matched natural-response SFT comparison

**Closed 25 September 2026:** all four fits completed; none was promoted. The
checkpoint bundles and latest pointers were pruned at the user's request. Code,
logs and evaluation answers remain; these pruned fits cannot resume or finalize.
See the [completed review](../../reports/NATURAL_CONTROL_REVIEW_20260925.md).
The preparation instructions below describe the historical experiment.

This experiment asks whether a broader, fully exposed natural-response recipe
works on OpenSML, on a separate SmolLM2 base control, on both, or on neither.
It is prepared for the user to run. It is **not** a continuation of checkpoint
896, a teacher/distillation job, or a model already approved for the playground.

## Why this differs from earlier attempts

The small synthetic runs learned narrow templates; natural conversation and
factual reading often failed even when loss improved. LoRA and GRPO did not
produce a reliable general improvement. We also tried broad SFT from the base,
so merely resetting to the base or reusing the SmolTalk name is not a new idea.

This comparison uses a measured budget of **20,480 unique accepted examples per
fit**, **4,160,629 supervised answer tokens for OpenSML** and **4,095,832 for
SmolLM2** in the frozen selection, a
full pass, two learning rates, and the same data on a separate base-model control.
It uses complete answers, masks all prompt/padding positions, includes EOS, and
weights the objective by assistant tokens. Earlier very small short-answer
curricula are not mixed into training. Their reading tasks remain diagnostics.

Neither more data nor either learning rate is known to be optimal. These are
explicit hypotheses to test. The reference has a different architecture,
tokenizer and much larger pretraining exposure; differences cannot identify
pretraining token count as the unique cause.

## What runs

| Fit | Base | Peak LR | End LR | Updates | Examples |
|---|---|---:|---:|---:|---:|
| opensml-lr5e-05 | OpenSML Stage B | 5e-5 | 5e-6 | 1,280 | 20,480 |
| opensml-lr2e-04 | OpenSML Stage B | 2e-4 | 2e-5 | 1,280 | 20,480 |
| reference-lr5e-05 | SmolLM2-135M base | 5e-5 | 5e-6 | 1,280 | 20,480 |
| reference-lr2e-04 | SmolLM2-135M base | 2e-4 | 2e-5 | 1,280 | 20,480 |

All four run sequentially on the main Mac: **5,120 updates total**. Each fit
starts independently from its base with fresh FP32 AdamW; no fit continues
another fit. Batch 16, microbatch 1, context limit 1,024, 128-update warmup and
cosine decay. The architecture and native tokenizer are preserved. Checkpoints
896 and 906, pretraining checkpoints, other Macs and the playground are unchanged.

## Data and limitations

Pinned source: [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, dataset card license Apache-2.0.
This is synthetic source data. There is no new teacher generation.
The OpenHermes portion is excluded: its mixed original source permissions were
not clear enough for this user's commercial-use preference.

| Source subset | Examples | Share |
|---|---:|---:|
| smol-magpie-ultra-short, filtered conversation | 14,848 | 72.5% |
| smollm-rewrite-30k | 4,096 | 20% |
| smol-summarize-20k | 512 | 2.5% |
| everyday-conversations | 1,024 | 5% |

The summary quota is limited to 512 after inspecting unsupported names and
overstated commitments in preliminary samples. Ordinary conversation receives
the remaining budget; the summary screens were strengthened, not relaxed.

The script streams the pinned Parquet files using bounded ranged HTTP reads.
It holds a bounded candidate pool and the selected examples in RAM. It does not
save a raw corpus/token cache. Streaming **does use RAM**; it does not mean zero
buffering. The same command rebuilds and verifies the exact selection on resume.
`selection.json` stores row IDs, source locations, hashes and aggregate counts,
not raw training text. Logs/checkpoints contain provenance and evaluation answers.

Only a self-contained first user/assistant pair is used. For everyday dialogues,
the repeated opening greeting is skipped in favor of the first substantive pair.
Actual rewriting/summarization instructions from system messages are preserved
inside the visible task prompt. No target is silently truncated. Complete email
signatures are allowed. A 192-token preliminary cap biased the selection toward editing and persona
responses. The final cap allows complete ordinary explanations, up to several
short paragraphs, instead of cutting answers to hit an arbitrary brevity target.
Answers must fit both native tokenizers' 16–384 answer-token bounds (including EOS),
with at most 256 words and a 1,024-token full context.

Code/math, roleplay, specialist medical/current information, speaker-tag leakage,
obvious echoes, incomplete endings and repetitive responses are filtered.
Summaries also undergo conservative word-support, number and requested-format screens.
These are heuristic filters plus sampled manual inspection, **not a guarantee
that every synthetic response is factually correct or every out-of-scope phrase
is detected**. The selection is not claimed to be fully human verified.

Conversation text hashes assign 80%/10%/10% source pools to train/development/test
before selecting the bounded per-source quotas. Exact prompts and
13-word input overlaps are excluded across those splits, as are detected overlaps
with existing benchmark prompts and the new evaluation probes. Shared task
instructions are not treated as shared source passages. This is **not guaranteed
semantic topic isolation**; the development/test probes test similar skills with
different wording and facts. The test is reserved for this experiment, not a
new broad independent benchmark or a claim of zero pretraining contamination.

The official [small-model SFT recipe](https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/sft/config_smol.yaml)
uses much more data. We are not claiming to reproduce its results.

## Evaluation and selection

At updates 0, 320, 640, 960 and 1,280 the script saves:

- 40 manually authored development prompts with explicit meaning/completion rubrics.
- 16 generated responses to held-out natural source examples, equally across sources.
- 28 retained-reading diagnostics, four per family across seven families.
- Assistant NLL on 128 development examples and a fixed 64-passage prose diagnostic.

Every response is saved for review. There is **no keyword-based best checkpoint,
no automatic quality score, no automatic promotion, and no automatic extension**.
Low loss is not considered proof of useful answers. NLL is compared within a
model; NLL across different tokenizers is not directly comparable.

After all fits finish, `review_blinded.json` provides shuffled candidate codes
and empty fields for correctness, faithfulness and completion. The identity key
is saved separately. We should compare paired answers against each base, inspect
reading regressions and repetition, and report both gains and failures. Forty
new probes and the source examples are still a limited sample, not proof of
general chat quality. Existing public benchmark test sets must not be turned
into training examples.

The test split is not generated during training. Only after reviewing development
answers should we select one candidate per model and call `finalize.py` with
explicit `--opensml ARM:STEP --reference ARM:STEP` arguments. That command freezes
the choices before evaluating parent and selected weights on the test split.
It refuses to change the frozen selection on later calls.

If only the reference improves, inspect OpenSML's base capabilities/pretraining
before another small post-training patch. If neither improves, investigate the
shared recipe/data instead of blaming OpenSML alone. If OpenSML improves with
acceptable retained skills, benchmark and try that candidate interactively before
calling it the new preferred model.

## Start and resume

From any directory:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/natural_control/launch.py \
  --run --clear-stop
```

Without flags the command only prints the plan. It verifies code, dependencies,
source selection, model/tokenizer hashes and readiness evidence before training.
It requires 90 GiB free for the four fits and a checkpoint reserve. Files are
written under `runs/sft_natural_control_v1`.

Ctrl-C requests a stop after the current operation and saves the completed update.
No next fit starts. Rerun the same command to resume the saved FP32 model, optimizer,
and next example. Completed fits are skipped. Intermediate evaluation candidates
and the two most recent saves are retained; older non-candidate saves from this
experiment are pruned. An error retains the last complete checkpoint.

`--prepare` rebuilds the selection before an experiment exists. `--check` runs
masking, numerical and exact next-update resume checks on disposable model copies;
those temporary models are deleted. It does not train or overwrite production
checkpoints. If inputs change, readiness must be regenerated honestly; it is not
bypassed by changing a hash by hand.

The completed preparation checks and sampled data limitations are recorded in
[PREPARATION.md](PREPARATION.md).
