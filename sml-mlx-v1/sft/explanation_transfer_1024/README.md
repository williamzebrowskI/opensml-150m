# Explanation transfer from Skill Recovery 1024

This experiment starts from the preserved **Skill Recovery 1024**, not a later
failed continuation. It tests whether short human explanations improve transfer
beyond the repeatedly used choice-ranking, short-reading and formatting targets.
It is supervised fine-tuning, not RL, DPO or a reproduction of a rationale paper.
No improvement over 1024 or Balanced Skills 1920 is promised.

## Why this experiment

1024's published local results are ARC-Easy 56.99% raw / 56.31% normalized,
ARC-Challenge 25.60% / 27.56%, PIQA 64.91% / 63.49%, HellaSwag 30.90% / 35.16%,
and IFEval 92/541 strict prompts and 254/834 instructions. Balanced Skills 1920
has stronger normalized ARC-Challenge (30.03%) and PIQA (64.31%), but weaker
IFEval (78/541 and 220/834). Recent 1024 continuations moved small numbers of
questions in opposite directions. This suggests limited transfer, not failed
optimization and not proof of a fixed model ceiling.

The testable hypothesis is that supplying complete, evidence-linked replies
will teach something different from optimizing the correct option's score.
The new objective still trains the complete answer including EOS. It is not
merely a new ranking dataset or additional repeated copies of the same contexts.
Research on human explanations motivates the hypothesis but does not establish
that it improves a 150M decoder or these particular public benchmarks.

## Data and quality limits

- **e-SNLI**, original `esnli_train_1.csv`, pinned to author repository revision
  `7b585a3f077fdea899780eb0473940522ae44a2e` and SHA-256
  `7311c7bc16ad9f6a9adcd116a62ef991e1803dc0d71e253ce66b975c2aba8ee5`.
- 3,072 new training scenes, balanced across entailment, contradiction and
  neutral. Each target gives a requested verdict and a short human explanation.
  192 development and 192 reserved scenes are drawn from this same TRAIN file.
  Official e-SNLI development/test files are not consumed.
- Captions from the same image, and identical premises across different images,
  are grouped before splitting. Only one selected example per connected scene
  group. Exact explanation duplicates are excluded across selected rows.
- Benchmark prompts/options and existing probes are exclusion-only inputs,
  using exact normalized matches and 13-word overlap screening. This is a
  heuristic contamination screen; original pretraining overlap is unknown.
- Length, role text, repetition and obvious label/reason conflicts are filtered.
  `rejections.json` records mistakes found in sample inspection. **This is not
  a fully verified corpus.** Annotation errors, awkward English, visual-caption
  assumptions and similar situations can remain. Label accuracy alone cannot
  validate a generated reason.
- Rebuild downloads approximately 90 MB into bounded RAM. It does not save the
  source CSV, a Hugging Face cache, or a generated teacher corpus. Small review
  samples, selection digests, evaluation answers and checkpoint inputs are saved.

ECQA and StrategyQA were also inspected. ECQA's explanations included incorrect
causal claims and StrategyQA's sample included wrong/outdated facts. They are
not used in this run. e-SNLI also needed exclusions; its narrower task allows
conclusions to be checked against supplied descriptions, with the limitations
above. No dataset choice here establishes a guaranteed benchmark improvement.

## Training

512 updates, one exposure per new explanation. Per update:

| Component | Rows | Main loss weight |
|---|---:|---:|
| New human explanation answers | 6, two per class | 0.35 |
| Previously consumed CommonsenseQA/SocialIQA ranking | 2, one each | 0.15 |
| Exact-content formatting replay | 3 | 0.20 |
| Complete-reply and two-turn replay | 4 | 0.30 |

Each family is normalized per example; long explanations do not silently
override the stated weights. Frozen-1024 prose KL adds weight 0.10 on existing
TRAIN prose, not public evaluation passages. Fresh FP32 AdamW, no weight decay,
gradient clipping 1.0, learning rate 3e-6 peak to 1e-6 final after 32 warmup updates.
Native `User:` / `Assistant:` format and the existing tokenizer remain unchanged.

Main Mac only. All checkpoints are retained at steps **1024, 1088, 1152, 1216,
1280, 1344, 1408, 1472 and 1536**, plus any interrupted checkpoint. No pruning or
automatic promotion. Allow about 22 GiB for these bundles plus a save reserve.
Resume verifies the source, code, selections, optimizer moments, master weights
and exact data cursor. Re-running a completed job does not extend training.

## Evaluation and decision

Every checkpoint saves generated development answers, conclusion accuracy by
class, surface completion, repetition, stopping, assistant loss, independent
manual probes and the previous commonsense/reading/format/two-turn diagnostics.
Surface completion is explicitly **not semantic explanation accuracy**.

Review whether each explanation (1) reaches the right conclusion, (2) identifies
the relevant evidence, (3) avoids invented or reversed claims, and (4) finishes
the requested answer. Also compare the independent natural probes. Do not choose
a model solely because it has learned the three verdict words or punctuation.
The retention gate is a review signal, not statistical proof of non-regression.

Freeze one development choice with `finalize.py --step STEP --run` before
comparing it with unchanged 1024 on reserved scenes. The older retention test
diagnostics are reused and are not fresh. Public benchmarks should compare the
same scoring rules and inspect paired changes, not cherry-pick raw versus
normalized scores. Their repeated use across earlier runs limits independence.

## Commands

The setup has a separate preparation and disposable validation path:

```sh
/Users/williamzebrowski/sml-mlx/.venv/bin/python /Users/williamzebrowski/sml-mlx/sml-mlx-v1/sft/explanation_transfer_1024/launch.py --check
```

Start or resume the reviewed job:

```sh
/Users/williamzebrowski/sml-mlx/.venv/bin/python \
  /Users/williamzebrowski/sml-mlx/sml-mlx-v1/sft/explanation_transfer_1024/launch.py \
  --run --clear-stop
```

After reviewing answers, a chosen evaluated step can be benchmarked using
`benchmark.py --step STEP --suite all --check`, then the same arguments with
`--run`. No training or playground changes occur during benchmarking.

## Sources and attribution

- Camburu et al., [e-SNLI: Natural Language Inference with Natural Language Explanations](https://proceedings.neurips.cc/paper/2018/hash/4c7a167bb329bd92580a99ce422d6fa6-Abstract.html).
- [Author dataset repository](https://github.com/OanaMariaCamburu/e-SNLI), repository MIT license; retain its copyright/license notice. Underlying [SNLI](https://nlp.stanford.edu/projects/snli/) is distributed under CC BY-SA 4.0. These are separate provenance layers, not a claim that all data is solely MIT.
- [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301) studies rationale supervision with larger teachers and different models. This experiment uses existing human explanations, no local teacher, and does not claim the paper's results will transfer.
- Rehearsal keeps prior pinned CommonsenseQA (MIT), SocialIQA (CC BY 4.0) and
  original local reading/format/two-turn examples. See their existing source
  manifests. CommonGen is only part of the reused evaluation data here.
