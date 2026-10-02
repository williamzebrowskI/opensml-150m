# Unified concise curriculum, V1

This is a new mixed SFT experiment from the preserved Stage B pretrained base at
step 73,243. It does not continue failed curriculum step 2048 or preserved SFT
1920. The user starts production training; preparation performs only disposable
checks.

## Run

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/base_curriculum_v1/launch.py \
  --run --clear-stop
```

The same command resumes an interrupted run. Startup rebuilds the pinned RAM
selection, then evaluates the base before the first update. Ctrl-C saves after
the current update. A completed run cannot automatically extend.

Output: `runs/sft_base_curriculum_v1`. Nothing is automatically promoted to the
playground. All local steps start at zero.

## Intended size

The revised selection targets **8,192 distinct examples**, one pass, **512 updates**.
Short complete conversational replies are much scarcer after filtering. We reduced
the size rather than filling the run with long replies or mostly multiple-choice
items. This is smaller than V1's 32,768 examples, not a scale-up. Final family
counts are frozen in `config.json` and verified against `selection.json`. The
final selection contains **317,982 supervised assistant targets**, compared with
3,771,110 in V1. This is a much smaller token budget as well as fewer examples.

| Group | Examples |
| --- | ---: |
| Everyday conversation, follow-ups, explanations and creative prompts | 2,656 |
| Rewriting and summaries | 2,176 |
| Grounded reading and missing information | 1,696 |
| Commonsense | 640 |
| Answer formatting applied to commonsense tasks | 1,024 |
| Total | 8,192 |

Development and reserved test each contain 273 examples; creative coverage is
only one item per split, and follow-up/explanation coverage is eight each. Treat
those family scores as qualitative diagnostics, not reliable benchmark estimates.
Twenty additional authored prompts are development-only manual review checks.

## Why this experiment

The first unified curriculum lowered development answer NLL from 2.173 to 1.588,
but repetitive and factually wrong generations remained. IFEval strict prompt
accuracy was 8.13%, below preserved 1920's 14.42%; normalized ARC-Easy was 50.93%,
below 57.07%. That run and its benchmark artifacts were deleted at the user's
request. A compact record remains in `reports/base_curriculum_v1_lessons.json`.

This experiment tests shorter complete targets and stricter data selection with a
lower learning rate. It changes several factors together and cannot establish
which factor caused an outcome. It does not assume that another dataset name,
more epochs, or lower teacher-forced loss will repair generation.

- Complete answers at most **128 tokens**, compared with 320 previously. Long
  examples are rejected, not truncated; requests must still receive complete
  replies.
- Reject repeated five-word passages, edit-commentary boilerplate, and certain
  false claims about the assistant learning from interactions.
- Dedicated rewrites provide the rewritten text rather than long explanations of
  edits. Plain numeric strings in the source must remain in rewrite targets.
- Contextual follow-ups keep real earlier messages; overly long histories and
  meta-editing discussions are excluded.
- Broad skills remain interleaved throughout the run. Target distributions are
  controlled per 256-example block; every selected source item occurs once.
- Fresh FP32 AdamW, ordinary assistant-only cross entropy including EOS, equal
  weight per example. Peak LR **1e-5**, final **2e-6**. Batch 16, microbatch 1,
  context 1,024. Main Mac only; no RL, DPO, KL term, or teacher generation.

The data still comes from selected pinned **smol-smoltalk**, **SQuAD v2**,
**CommonsenseQA**, and **SocialIQA** TRAIN sources. This is a revised subset and
mixture of known sources, not a claim that they are new or guaranteed to work.
Smol-smoltalk is synthetic. The other tasks use their published annotations.
See `selection.json` for source revisions, selected-content hashes, and final
counts; `config.json` freezes all run settings.

The source cards declare Apache-2.0, CC-BY-SA-4.0, MIT, and CC-BY-4.0 respectively.
Source attribution and provenance are retained. Topic filters exclude coding and
math exercises; general scientific explanations are allowed. Lexical filters are
imperfect, and sampled review cannot certify all facts or full compliance with
all instructions. A passing repetition screen does not imply a useful answer.

Sources:
- https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- https://huggingface.co/datasets/rajpurkar/squad_v2
- https://huggingface.co/datasets/tau/commonsense_qa
- https://huggingface.co/datasets/allenai/social_i_qa

## Evaluation and review

Development and reserved test use separate source groups. Public benchmark
prompts and known evaluation material are excluded using exact/13-word-overlap
screens; this does not prove absence of semantic overlap or pretraining exposure.
Data streams into bounded RAM, with no raw corpus disk cache. Train/dev/test
selection hashes must reproduce before any production update.

Development includes each skill family. Twenty additional authored review prompts
cover greetings, corrections, follow-ups, concise factual answers, practical
advice, summaries, missing information, and stories. These are **development**
checks, not a fresh reserved test or training targets. Their reference answers
are illustrative for open tasks, not a unique correct answer. They must never be
added to the training curriculum in response to observed failures.

Every evaluation saves actual greedy generations as well as reference likelihood,
EOS stopping, repeated five-word passages, and per-family metrics. This repetition
heuristic is stricter than V1's, so raw repetition counts are not directly
comparable. Development generation retains the same 320-token limit as V1 to
expose runaway answers rather than hiding them with a shorter generation budget.
Exact matching is meaningful for closed tasks, not a general quality score.

Review correctness, request completion, use of changed context, preservation of
facts, and repetition before selecting a candidate. Reference loss alone cannot
select a useful assistant. Training probes are a fixed subset, not whole-training
accuracy. Prose loss is measured on the same fixed existing evaluation passages;
these passages never receive gradients. Reserved test is not model-evaluated
during checks or training. Freeze a candidate before reserved/public testing.

## Verification

Readiness checks cover full-corpus masks/EOS/lengths, disjoint groups, fixed mix,
exact resume cursor, cached/full decoding parity, padding-invariant objective,
and disposable real-model updates followed by optimizer/weight save-restore.
The original base and preserved 1920 bundle are protected by hashes. A changed
code/config/data contract requires new checks. No production training starts
without `--run`.
