# Matched SFT transfer experiment

Prepared 24 September 2026 after the failed Natural Instructions full pass.
**This is a diagnostic, not a promised general-assistant training recipe.**
The previous model/optimizer, logs, source audit and dedicated recipe were deleted.
The measurements remain in [the lessons note](../../reports/SFT_LESSONS_20260924.md).

## Why this experiment

Several mixtures lowered assistant loss without producing reliable generated
answers. The last model chose Entailed on 60/64 reserved entailment examples.
The [base audit](../../reports/BASE_CAPABILITY_20260924.md) already proposed a
reference-base control if another audited full pass failed. This implements
that follow-up instead of treating a new dataset name as a likely solution.

Both models receive the **same 1,024 examples, order, 128 optimizer updates and
learning-rate schedule**. OpenSML starts from preserved Stage B best step 73,243.
The reference is the existing, unquantized SmolLM2-135M **base**, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. It generates no training answers and
does not teach or replace OpenSML. Its different architecture/tokenizer and
pretraining exposure make this a diagnostic control, not an architecture or
compute-matched causal experiment. A single schedule can also favor one model.

## What changes, explicitly

- Original authored templates with deterministic fact substitutions replace
  the third-party Natural Instructions mixture. These are synthetic task
  examples, not teacher-generated answers or a natural conversation corpus.
- Eight equally represented families: locations, ownership, later corrections,
  negation, record extraction, exact name replacement, short factual summaries,
  and missing information. No code or math tasks. A label such as Entailed is
  not a training target. Missing-information examples remain balanced within
  their family; the constant answer Not stated is only 1/16 of all rows.
- Each factual group has two variants with different correct answers. We
  measure whether **both** are correct and how often the answer stays unchanged.
  Sentence/field order varies, and location, owner and summary requests target
  either entity. Always extracting from the first sentence cannot solve them.
- Both use ordinary `User: ...\nAssistant: ...` text plus each tokenizer's
  existing EOS token. This intentionally differs from the previous custom
  turn-marker format, avoiding a new-role-token learning requirement. It is
  not a controlled ablation against that old format. No tokenizer is modified.
- Primary development comparison includes held-out facts under familiar and
  unseen wording. A final reserved test uses different named people/groups
  and another instruction phrasing. Related task structures remain: success
  would not prove broad instruction following.
- Open-ended prompts are saved for human review. Greetings, general factual
  explanations and invitations are not training categories in this diagnostic.

## Run

The agent runs tests and `--check` only. Start the actual comparison yourself:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/transfer_control/launch.py \
  --run --clear-stop
```

This trains OpenSML, evaluates it, then trains the reference on the main Mac.
The other four Macs are not involved. No Git push is needed. Omit flags to
print the plan. `--check` verifies real forward passes, generation, tokenization,
padding and protected files without updating either pretrained model.

Ctrl+C requests checkpointing after the current update/evaluation. Wait for
the command to return. The second arm will not start after a stop request.
The same run command restores a stopped arm's optimizer and exact next
example, skipping a completed arm. Config/code/runtime/data changes reject
resume. A completed comparison is not extended or repeated automatically.

Output: `runs/sft_transfer_control_v1/{opensml,reference}`. All new weights
live there. The pretrained sources, their optimizers/cursors and tokenizers
remain protected by hashes. The playground remains on its existing base model;
these plain-format experimental checkpoints are not automatically integrated.

## Budget and evaluation

One pass/model; batch 8, microbatch 2, BF16 computation with FP32-master AdamW,
betas 0.9/0.95, weight decay 0.01, gradient clipping 1.0. Equal-example
assistant-only loss includes EOS, masks input/padding and never truncates an
answer. Context limit remains 2,048. Warm up 8 updates to **3e-5**, cosine decay
to **3e-6** at 128. This reuses the smaller-task LR with prior positive evidence;
it is not an assertion of the optimal schedule for either base.

Data is generated on demand locally. There is **no live Hugging Face dataset
stream in this diagnostic**, no local teacher and no full downloaded corpus.
The 1,024 lightweight prompt/answer records are held in RAM to guarantee exactly
matched ordering. Tokenization runs per training batch. The existing pinned
reference download is reused. Checkpoint metadata stores only the step and
frozen recipe needed to reconstruct the next example.

At 0/32/64/128 updates, evaluate 128 held-out examples (64 changed-fact pairs)
in familiar and unseen wording, and 8 qualitative open prompts. Save raw answers,
per-family prediction frequencies, both-correct pairs, strict accuracy, stopping,
and a lenient fact-mention diagnostic. Fact mention is not semantic correctness
and does not select best. Strict matching has paraphrase limitations; review
raw outputs alongside the scores. All these are development measurements.

Development best is selected among trained candidates by unseen-wording
both-correct pairs, then exact accuracy, stopping and assistant NLL. Baseline
is reported separately: the existence of best.json never establishes improvement.
Evaluate selected best once on **192 reserved examples / 96 pairs** with new
wording. This does not change checkpoint selection. Latest and best are retained.

A fixed 64-passage sample from the preserved evaluation-only BoolQ passages
measures prose drift, not general knowledge. It never enters training or
checkpoint selection. Compare its loss only within each tokenizer, never
against the old 2.8186 prose baseline or across the two different tokenizers.
This smaller diagnostic is not a replacement for the full pretraining evaluation.

## How to interpret the result

- If both learn familiar forms but fail unseen wording/fact swaps, the curriculum
  has not established transfer. Rework examples/evaluation before scaling up.
- If the reference transfers substantially better while OpenSML fits training
  forms, investigate base-model/data/optimization limitations. That observation
  alone cannot identify which architectural or pretraining difference caused it.
- If OpenSML transfers and preserves prose reasonably, expand into varied,
  naturally phrased short responses. Passing templates alone is insufficient;
  inspect ordinary requests before considering a playground candidate.
- If neither learns even these forms, inspect the common objective/format and
  exposure before another large SFT. Do not infer irreparable base limitations
  from one learning rate or this small diagnostic.

No score gate will quietly stop or promote a run. No experimental results have
been claimed before running it. Source: [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
(Apache 2.0; original model card/license retained with the reference download).
The new curriculum has no third-party dataset source or dataset license terms.
