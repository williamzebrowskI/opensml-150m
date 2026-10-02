# Longer preference continuation from grounded/ranking 640

A new experimental branch, 2,048 additional updates (640 → 2688), starting from
`runs/sft_grounded_rank_384_v1/step_0000640_7b477e0e5eea`. The parent is read-only.
The objective is a testable proposal for broad improvement, not a guarantee that
every public benchmark will rise. No production training is started by setup.

## Why this experiment

The preceding corrective SFT round raised the four-task normalized MC average
from 45.26% to 45.55%, but strict IFEval prompts fell from 18.11% to 16.82% and
length-capped replies rose from 20 to 35. It was removed at the user's request.
The earlier 24-pair DPO pilot had little complete-response gain and allowed
preferred-answer likelihood to worsen. This run uses a substantially larger
preference set, direct preferred-answer supervision, and explicit chat retention.

The published SmolLM2 135M/360M recipe uses UltraFeedback DPO after SFT, beta 0.5,
LR 1e-6, and two epochs. That supports trying the method at this model size; it
does not establish the outcome for OpenSML. This implementation adds preferred
CE and retention objectives and uses different filtering/batching/precision.

- https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/README.md
- https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/dpo/config_smol.yaml
- https://arxiv.org/abs/2305.18290
- https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized

## Objective and data

Each update contains four preference pairs, one instruction, one whole replay
conversation and one labeled choice example each from ARC-Easy, ARC-Challenge,
PIQA and HellaSwag TRAIN. The native plain-user/assistant format is preserved.

| Term | Weight |
| --- | ---: |
| DPO relative to frozen 640, beta 0.5 | 0.15 |
| Preferred-answer mean token CE | 0.45 |
| Instruction rehearsal CE | 0.10 |
| Conversation rehearsal CE, every assistant turn | 0.10 |
| Character-normalized choice ranking, temperature 0.25 | 0.20 |
| Additional KL on replay assistant positions, including EOS | 0.10 |
| Additional KL on TRAIN prose | 0.10 |

DPO uses summed answer-plus-EOS log probabilities for both responses, with user
and padding tokens masked. CE is averaged per example, conversation turns are
averaged within each conversation, then family weights are applied. Frozen 640
is the DPO reference and KL anchor; there is no sampled reward or reward model.

Preferences: 4,096 unique original controlled pairs across 16 task families,
two passes (8,192 pair exposures). The first pass orders simpler tasks before
combinations; the second pass is shuffled. Each pair has a correct complete
answer and a deliberately incorrect alternative: wrong person/date, ignored
correction, missing required text, bad ordering, wrong conditional branch, wrong
arithmetic result, or a violated supplied ownership rule. Facts are fictional;
small integer multiplication is computed exactly. Natural rewrite targets are
one valid phrasing, not the only valid phrasing. We never label model samples as
wrong merely because they differ from that canonical wording.

Tasks include JSON, sorting, bullets, numbered lists, complete messages,
summaries, polite requests, ownership, conditional selection, filtering, simple
arithmetic, required endings, word counts, CSV, case/prefix rules and follow-up
corrections using the native chat format. Every selected pair is checked for the
constructed distinction and complete context; prompts/padding are never targets.
All preferred and rejected answers include EOS. No answer is truncated.

Grouping uses the actual task content, not incidental unused random fields.
96 development and 96 reserved test pairs have separate content groups. They
share task families and vocabulary, so their scores measure near transfer rather
than proving general instruction-following gains. Public benchmarks and earlier
held-out prompts are exclusion-only inputs via exact/13-word overlap screening.
This is heuristic; semantic/pretraining contamination remains unknown.

UltraFeedback was inspected during setup but rejected because sample review
found incorrect preferred answers and weak preference distinctions. None of its
rows enter this run. The downloaded source cache was removed. Review notes are
retained as a record of that decision. The method is inspired by reference DPO;
it is not a reproduction of SmolLM2's public-data recipe.

Rehearsal repeats the parent recipe's 768 instruction examples and 512 source
conversations; it is deliberately familiar retention data. Ranking has 8,192
exposures across 6,725 unique TRAIN questions. Small ARC pools repeat after
shuffling. Source partitions and HellaSwag origin-group exclusions are retained.
Public benchmark/test data never supplies gradients. Source attribution and
license records are in selection.json (PIQA license recorded as unknown).

## Training, monitoring and recovery

Fresh FP32 AdamW, betas 0.9/0.95, no weight decay, clip norm 1. LR warms for 128
updates to 1e-6 and cosine-decays to 1e-7 at update 2048. Context 1024; architecture
context remains 2048. Development generation cap is 384 tokens, distinct from
the later full IFEval protocol. The same deterministic schedule resumes from a
saved optimizer and exposure cursor.

Checkpoints/evaluations: baseline, local 32/64/128, then every 128 through 2048.
All checkpoints are retained. Allow at least 60 GiB free; the normal 19 bundles
use roughly 43 GiB. Ctrl-C/SIGTERM finishes the current update and saves.

The run stops and retains its last checkpoint after two consecutive development
evaluations have any material regression relative to baseline: any MC source
falls more than 6.25 percentage points; instruction completion falls more than
8.33 points; reading falls more than 12.5 points; stopping falls more than 5
points; repetition rises more than 5 points; or prose NLL rises more than 0.08.
These are coarse development guards, not statistical guarantees. Natural
preference generations also receive stopping/repetition checks. Reports include
raw answers. Loss, relative preference win, and stopping alone do not establish
response correctness or justify promotion. There is no automatic best model.

```bash
<SOURCE_WORKSPACE>/.venv/bin/python -u \
  <SOURCE_WORKSPACE>/sml-mlx-v2/sft/preference_long_640/launch.py --run
```

For intentional resume after reviewing a STOP (including a regression stop),
add `--clear-stop`. This acknowledges the stop and resets the consecutive-failure
counter; it does not change the data, optimizer or learning-rate schedule.
Production output: `runs/sft_preference_long_640_v1`.

Setup: `--prepare` regenerates selection, a current sample-review receipt is
required, and `--check` validates all selected target masks, the analytic DPO
gradient, independent tokenwise likelihood, frozen reference, two disposable
updates, bitwise save/load, numerical equivalence of the next resumed update,
and cached/full-context decoding. Disposable test checkpoints are removed.
Full public benchmarks and reserved test are not evaluated during training.
