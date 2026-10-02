# Protected updates from Skill Recovery 1024

Two independent fits start from the preserved Recovery 1024 weights, never the
rejected Context Reasoning checkpoints. The previous Context Reasoning run and
public results were deleted at the user's request (26 September 2026). Its 1536
candidate gained ARC-Challenge but lost instruction accuracy; it did not beat
Balanced Skills 1920 broadly. Original 1024 and 1920 remain untouched.

## Hypothesis

Earlier loss mixing and response KL did not reliably preserve old skills. This
experiment tests whether the optimizer's actual proposed parameter step conflicts
with old-task training gradients. It does **not** assume that all regressions are
caused by such conflicts or that preserving training losses preserves benchmarks.

- **control**: ordinary labeled-choice ranking + assistant-only format/reply SFT
  + frozen-parent training-prose KL.
- **protected**: exactly the same data, initialization, seed, schedule and losses,
  but project Adam's actual proposed parameter change against two separate
  halfspaces: formatting CE and reply/two-turn CE on the current replay batch.

The constraint is gradient dot parameter-change <= 0. It is a local, first-order
constraint on sampled training losses, not a non-increasing actual-loss guarantee.
The projection is applied after Adam, because projecting raw gradients does not
preserve that condition after Adam rescaling. The two-constraint quadratic problem
is solved by enumerating active sets. Corrected FP32 master weights stay in sync
with the model; Adam moments retain the raw objective history. Material constraint
violations from rounding skip the parameter move (and are logged). This is an
experimental extension inspired by GEM/A-GEM, **not** an exact paper reproduction,
RL, DPO, or a proven optimal recipe. No stronger teacher is downloaded or queried.

Research: [A-GEM](https://arxiv.org/abs/1812.00420),
[Gradient Surgery](https://arxiv.org/abs/2001.06782).
Research on [self-distillation](https://arxiv.org/abs/2601.19897) was considered,
but [its reported limitations](https://arxiv.org/abs/2607.01763) and our earlier
KL experiments argue against presenting another distillation pass as a reliable fix.
None of these papers establishes an expected gain for this 150M model.

## Data, budget, and limits

Per fit: 512 updates, final step 1536, fresh FP32 AdamW, peak LR 3e-6 to 1e-6,
32 warmup updates. Each update: four labeled QA choices (two per source), three
format examples, six reply/rehearsal examples. Weights: ranking .30, format .25,
reply .45, plus .10 frozen-1024 training-prose KL. Questions teach ranking rather
than one-word assistant completions. Both arms use length-normalized choice scores
as in the parent recipe; this is not calibrated to a public test set.

2,048 unique human-labeled QA rows (1,024/source), 1,536 format exposures (768
unique views of 256 contexts), 3,072 replay exposures (1,024 unique targets). Some
QA examples were consumed in previous training. Replay and formatting intentionally
repeat; this is not a newly collected conversation corpus. No coding or math tasks.

Sources stream from existing immutable, SHA-verified TRAIN files into bounded RAM,
with selected rows retained in RAM and no raw corpus disk cache:
- [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa/blob/main/README.md): MIT per publisher card.
- [SocialIQA](https://huggingface.co/datasets/allenai/social_i_qa/blob/main/README.md): CC-BY-4.0 per publisher card.
- Existing original local formatting, complete-response and two-turn replay.

Keep source attribution and pinned revision/selection metadata when distributing.
CommonGen is evaluation-only. Training rows are sample-reviewed; ambiguous labels
found in review are rejected, but remaining labels are not certified error-free.
Historical split intersections and heuristic public-prompt overlap exclusions are
retained. They do not prove complete semantic decontamination.

All development diagnostics are reused. Only 32 QA items per source survive the
historical disjoint-split constraints, so single-item score changes matter. Every
64 updates is saved and evaluated; **all checkpoints**, including interruption
saves, are retained. No automatic promotion. Review raw answers and compare paired
results; freeze arm/step selection before reserved testing. The public benchmarks
have informed many earlier experiments and should not be described as an untouched
final test. There is no promise this will beat 1024 or 1920.

## Run

Both arms sequentially on the main Mac (1024 updates total):

```sh
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/protected_update_1024/launch.py \
  --run --clear-stop
```

Optional `--arm control` or `--arm protected` runs just that independent fit.
The same command resumes from saved optimizer/step/data cursors. Ctrl-C finishes
the current operation and saves. Both arms load original 1024 independently.
Production files live in `runs/sft_protected_update_1024_v1/{control,protected}`.
Numerical checks use disposable tiny and real-model copies, never production.

After answer review, freeze a choice with `finalize.py --arm ARM --step STEP --run`.
Benchmark later with `benchmark.py --arm ARM --step STEP --suite all --check`, then
`--run`. Benchmarking is inference-only and does not promote a checkpoint.
