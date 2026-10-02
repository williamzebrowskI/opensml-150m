# Stage B: bounded educational-prose pilot

Prepared 2026-09-23. This is continued next-token pretraining from the best
evaluated base checkpoint. Production execution is left to the user.

## Start or resume on Mac-1

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/stage_b/launch.py \
  --run --clear-stop
```

Omit both flags to print the plan without loading a model or accessing the network.
The launcher stages and verifies current code, tokenizer, configuration and
checkpoint inputs on all five Macs. A Git push is not required. It refuses stale
data-review or five-rank correctness evidence and checks that workers are idle.

The same command resumes this pilot's own latest checkpoint after interruption.
It refuses to restart a completed pilot or silently extend its budget. Ctrl-C
requests a coordinated save and stop; wait for all five `[finished]` lines and
the launcher's verified-stop message.

## Starting point and budget

| Setting | Value |
| --- | --- |
| Frozen best parent | Step 73,008; 7,775,059,968 lifetime tokens |
| Expanded reference loss | 2.767114285717253 |
| Learning rate | 0.000005, retaining the settled parent floor and scheduler state |
| Optimizer | Existing FP32 master AdamW state, including moments |
| Model/tokenizer/context | Existing 150,439,168-parameter model; frozen 32K BPE; 2,048 tokens |
| Physical rank order | Mac-1, Mac-2, Mac-3, Mac-5, Mac-4 |
| Microbatches/accumulation | 4/3/2/2/2; accumulation 4 |
| Tokens per update | 106,496 |
| Additional pilot budget | 250,052,608 tokens, rounded up from 250M to 2,348 updates |
| Automatic endpoint | Step 75,356; 8,025,112,576 lifetime tokens |
| Output | `runs/stage_b_prose_v1` |

The independent parent snapshot is in
`experiments/stage_b_v1/parent/step_0073008_58d174e2c177`. Original pretraining
runs and their best/latest pointers are preserved. The pilot begins with a copy
of that evaluated parent as its best, so a worse experiment cannot replace it.

The higher-LR experiment did not establish a new best. This pilot changes data
selection and mixture while retaining the best checkpoint's LR; it does not
combine a data change with another LR experiment.

## Data decision

| Pinned source | Stage B token share | Existing reference share |
| --- | ---: | ---: |
| SmolLM corpus: deduplicated FineWeb-Edu | 55% | 55% |
| DCLM-Edu, score at least 3 | 20% | 25% |
| FineWiki English | 15% | 10% |
| Cosmopedia-v2 educational prose | 10% | 10% |

Exact repository revisions and license/provenance notes are in `corpus.json`.
The original proposal included 7% FinePhrase tutorials and 3% FinePhrase FAQ.
Spot review found incorrect grammar, factual inconsistencies and malformed
answers, so **neither FinePhrase source is enabled** in this pilot. We also kept
Cosmopedia at its existing 10% instead of increasing synthetic content. This
revision is a conservative first experiment, not an established optimal mixture.

All four training sources pass deterministic math/code topic, boilerplate,
length, repetition and text-encoding screens. Sources are streamed from Hugging
Face, with 256 accepted documents per source shuffled in RAM and four batches
prefetched. The full dataset is never loaded into RAM or materialized as token
shards. Pending documents, token tails, cursor positions and selection/dedup
state are saved in checkpoints for exact resume. Selection history grows only
within this pilot and fails explicitly at its memory-budget limit.

At the first Stage B boundary, raw source positions are retained. Unconsumed
documents and token tails from the previous, less-filtered buffers are explicitly
discarded and counted. Already consumed rows are not replayed. Per-stage mixture
counters start at zero while lifetime token counters continue from the parent.

The live audit scanned 300 raw rows per source at those saved positions and
verified reopening each HF cursor. Accepted counts were 189 web, 51 DCLM,
210 Wiki and 136 Cosmopedia. The selected mixture uses the same source definitions
and selection rules as those audit receipts; only weights and source membership
changed after reviewing the candidate mix. `data_audit.json` records the decision
and hashes the receipts.

This is a small streaming/format/topic review, **not a guarantee of factual
accuracy or perfect semantic math/code exclusion**. Existing web and Cosmopedia
content also contain noise. Exact text, URL and ID matching protects the rebuilt
reference documents; it does not certify absence of semantic near-duplicates
across the whole corpus or every historical training document.

## Evaluate the pilot before extending it

The expanded and legacy validation tokens, fingerprints and original 55/25/10/10
reference weights remain unchanged. Validation runs every 25M lifetime-token
boundary and at the pilot endpoint; checkpoints are saved every 10M boundary,
at evaluations and on stop. Best is selected by the original expanded validation
metric, never by the new training mixture's loss.

Eight fixed, authored passage/question cases run at the baseline, 50M sample
boundaries and endpoint. Raw answers and a strict exact-match diagnostic are
saved under `assessments/`; these cases never enter training. This tiny diagnostic
can reject correct paraphrases and is not a general benchmark.

Before choosing another run, compare sustained expanded validation improvement,
legacy and per-source loss, and the raw reading/completion outputs. Do not treat
a single tiny new best as proof of better language quality. No automatic longer
run, SFT phase, checkpoint promotion or playground change is scheduled.

Verification: 85 focused CPU fixture tests passed, including exact interrupted
optimizer/data resume and frozen reference validation. All five ranks passed
native RDMA collectives, unequal-batch gradient and tiny BF16/FP32 checkpoint
roundtrip checks. Production model execution was not part of setup.
