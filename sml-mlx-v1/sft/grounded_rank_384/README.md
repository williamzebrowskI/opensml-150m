# Grounded and ranked continuation from intact-base-384

256 **additional optimizer updates** from the preserved intact SmolTalk checkpoint
384. The new branch ends at cumulative step 640; its local update number starts
at zero. This is a hypothesis-driven experiment, not a proven best recipe.

## Objective and data

Each update uses 13 examples. CE is averaged **within each example**, then within
each family, before applying the weights:

| Family | Examples/update | Total exposures | Weight |
|---|---:|---:|---:|
| Grounded answers | 4 | 1,024 | 40% |
| Executable instruction tasks | 3 | 768 | 25% |
| Complete conversation/rewrite/summary replay | 2 | 512 | 15% |
| Labeled candidate ranking | 4 | 1,024 | 20% |

Grounded examples alternate two human SQuAD v2 TRAIN examples and two authored
fictional relationship examples. Human targets include 384 answerable and 128
unanswerable questions. Article and paragraph groups are separated across the
selected training, development and reserved test partitions.

Instruction data covers JSON with exact keys, sorting, bullets, sentences, updated
invitations, supported summaries, uppercase, numbered steps, prefix/suffix rules,
word counts, extraction and line formatting. Development scenes are independent,
but share task templates: these diagnostics are not proof of broad generalization.

Ranking uses one science question (alternating ARC-Easy/Challenge TRAIN), one PIQA
TRAIN item, one HellaSwag TRAIN continuation and one CommonsenseQA TRAIN question
per update. Gold labels supervise a softmax over **answer log-likelihood divided
by Unicode character count**, matching the public normalized metric. Ranking has
no assistant prefix or EOS target; it does not teach a generic one-word assistant.
Temperature is 0.25. Official test/validation benchmark items and all IFEval rows
remain evaluation-only. Exact and 13-word overlap screens are heuristic.

Replay comes only from the earlier prepared TRAIN conversation pool. Native
instruction text and full gold history are preserved; development follow-ups are
also generated with the model's actual earlier answers. Long answers are bounded
by selection, never cut into incomplete targets. Sources are human annotated or
synthetic as recorded in selection.json; labels and sampled review are not an
exhaustive factual audit.

An additional KL term of weight 0.10 anchors the policy to frozen **checkpoint
384** on HellaSwag TRAIN prose. This is a retention anchor, not a stronger teacher.
Evaluation prose and held-out examples never supply training gradients.

Fresh FP32 AdamW is intentional because the objective changes. Peak LR 3e-6,
16-update warmup, cosine decay to 1e-6 at update 256, betas 0.9/0.95, no weight
decay, gradient clipping 1.0. All training examples fit 1,024 tokens without
truncation; the original model's 2,048-token architecture is unchanged.

## Running

From <SOURCE_WORKSPACE>:

```bash
.venv/bin/python -u sml-mlx-v1/sft/grounded_rank_384/launch.py --run
```

Preparation and readiness are separate modes, --prepare and --check. Preparation
pins each entire public source to its revision, byte count and SHA256, freezes
selected rows and samples; readiness requires a recorded sample review. It checks
all sequence masks and boundaries, independent scalar candidate likelihoods,
zero initial parent KL, a disposable update, bitwise checkpoint restoration and
next-update weights/masters (with bounded FP32 rounding for optimizer moments),
and a cached/full-context generation check. Disposable checks do not train the
preserved source or leave production checkpoints.

Development evaluation and retained checkpoints occur at local updates
0/32/64/96/128/160/192/224/256 (cumulative steps 384 through 640). No pruning,
automatic best selection or release promotion. Saved content must be reviewed;
retention_gate is only a diagnostic, not an automatic stop or promotion rule.
Official benchmarks are rerun only after a candidate is chosen. The reserved test
is frozen but is not evaluated during training.

Ctrl-C finishes the current update, saves the exact optimizer and cursor, and
leaves STOP. Resume deliberately with the same command plus --clear-stop.
The original 384 bundle remains unchanged. Run files are written to
runs/sft_grounded_rank_384_v1. No other experiment or benchmark may hold the shared
experiment lock concurrently.

Dataset revisions, file hashes, licenses, counts, overlap-screen rejections and
partition limitations are recorded in selection.json. A distinct checkpoint 640
in older runs is a different model; always identify this branch by its directory.
