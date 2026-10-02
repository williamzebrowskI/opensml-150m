---
license: apache-2.0
datasets:
- HuggingFaceTB/smollm-corpus
- HuggingFaceTB/dclm-edu
- HuggingFaceFW/finewiki
- HuggingFaceTB/smol-smoltalk
- HuggingFaceH4/ultrachat_200k
- rajpurkar/squad_v2
- allenai/ai2_arc
- databricks/databricks-dolly-15k
- allenai/tulu-3-sft-personas-instruction-following
- HuggingFaceTB/smoltalk
- allenai/sciq
language:
- en
pipeline_tag: text-generation
tags:
- text-generation
- mlx
- pretraining
- from-scratch
- small-language-model
- work-in-progress
- sft
---

# OpenSML-150M

**OpenSML-150M has exactly 150,439,188 parameters (150.44M).**

**Author:** William Zebrowski · **Checkpoint:** SFT step 768 · **Status:** research preview

[Technical report](../sml-mlx-v2/docs/TECHNICAL_REPORT.md) · [Results and provenance](../results/OPENSMl_150M_RESULTS.json) · [Tokenizer](../sml-mlx-v2/tokenizer/bytebpe32k_v1/tokenizer.json)

An English-first language model trained from scratch with Apple's MLX framework.
Pretraining processed **7.800B tokens** across Stage A and Stage B on a four-Mac
cluster, later expanded to five Macs using Thunderbolt RDMA. Three supervised
fine-tuning stages produced the selected OpenSML-150M checkpoint.

## Availability and Inference

This repository provides documentation, the tokenizer, and historical configuration
files. **Model weights and a tested standalone inference bundle are not yet
published**, so a download-and-generate quick start is pending. Local inference
uses native MLX on Apple hardware; other loading and export formats remain unverified.

## Model Specification

| Field | Value |
| --- | --- |
| Parameters (model-card label) | 150,439,188 (150.44M) |
| Layers / hidden width / FFN width | 20 / 768 / 2,048 |
| Attention | GQA: 12 query heads / 4 KV heads; head dimension 64 |
| Position encoding | RoPE, base 10,000 |
| Normalization / MLP | RMSNorm, Q/K normalization, SwiGLU |
| Vocabulary / context | 32,000 / 2,048 |
| Embeddings | Tied input and output |
| Linear biases / dropout | Neither used |
| Pretraining precision | BF16 compute; FP32 master weights and optimizer state |
| Benchmark precision | FP32 scoring, explicit vanilla MLX attention |

The independently fitted tokenizer is a 32,000-token byte-level BPE. Its recorded
held-out audit had zero round-trip failures. Full configuration, parameter
accounting, and tokenizer checks are in the [technical report](../sml-mlx-v2/docs/TECHNICAL_REPORT.md).

## Training and Model Lineage

The selected pretrained base is **Stage B step 73,243**, after **7,800,086,528
lifetime pretraining tokens**. Stage B continued Stage A with the adjusted mixture below.
Percentages are configured token shares.

| Source | Initial token share | Stage B token share | Selection |
| --- | ---: | ---: | --- |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 55% | 55% | `fineweb-edu-dedup` |
| [DCLM-Edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu) | 25% | 20% | `edu_int_score >= 3` |
| [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) | 10% | 15% | English |
| [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 10% | 10% | Textbook/tutorial/blog/educational format filter |

FineWeb-Edu and Cosmopedia v2 come from two subsets of the same SmolLM Corpus
repository. Pretraining used no dedicated code or math dataset.

```text
V2 pretrained base → Unified384 → Repair512 → OpenSML-150M (SFT step 768)
                      +384         +128       +256 updates
```

Unified384 used Smol-SmolTalk, UltraChat, SQuAD v2, ARC training questions, and
locally authored follow-ups. Repair512 added Dolly and Tulu Persona instruction
examples. The final stage used SmolTalk constraints, SQuAD v2, SciQ, and Repair512
replay. All model parameters were updated; the selected checkpoint is a direct
continuation, without LoRA or parameter interpolation. Dataset exposure, source
pins, optimizer schedules, and checkpoint hashes are in the technical report.

### Pretraining Loss Curves

![Recorded pretraining validation loss for Stage A and Stage B](../docs/figures/pretraining-overview.svg)

Recorded held-out loss without smoothing. The star marks the selected base;
later points are outside its training exposure. The late-training panel uses a
magnified scale. This chart uses the same original validation set throughout.

<details>
<summary>Stage B detail and per-source curves</summary>

![Stage B validation loss on the expanded held-out set](../docs/figures/pretraining-phase-b.svg)

This panel uses a different, expanded validation set; its absolute losses should
not be merged with the overview. The selected base has original-set loss
2.81855024 and expanded-set loss 2.76700753.

![Original-set validation loss by data source](../docs/figures/pretraining-sources.svg)

Per-source panels use separate vertical scales. Full validation accounting and
historical measurements are in the technical report.

</details>

## Evaluation

### Zero-shot Likelihood Benchmarks

Existing full-split evaluations cover **15,428 multiple-choice examples**.
The pretrained base and selected SFT model are reported separately.

| Benchmark | Split | Examples | Pretrained base acc / acc_norm | OpenSML-150M acc / acc_norm |
| --- | --- | ---: | ---: | ---: |
| ARC-Easy | Test | 2,376 | 54.67% / 48.53% | 56.65% / 55.43% |
| ARC-Challenge | Test | 1,172 | 23.38% / 26.88% | 26.02% / 29.52% |
| PIQA | Validation | 1,838 | 65.23% / 64.36% | 64.53% / 64.09% |
| HellaSwag | Validation | 10,042 | 30.21% / 34.44% | 30.55% / 33.94% |

SFT improves both ARC metrics. PIQA declines slightly; HellaSwag raw accuracy
rises slightly while normalized accuracy declines.

<details>
<summary>Likelihood prompts and scoring</summary>

The native evaluator uses zero-shot raw completion prompts, FP32 scoring,
reference MLX attention, batch size 1, and no chat template or added BOS/EOS.
`acc` ranks summed candidate log-likelihood; `acc_norm` divides by candidate
Unicode character count, excluding the leading delimiter. No generation or LLM
judge is involved. The evaluator follows pinned harness conventions but is not
an installed lm-evaluation-harness run. Dataset pins and integrity receipts are
in the technical report and results record.

</details>

### Comparison with Small Language Models

External scores below are **previously published measurements**. Each cell shows
`acc / acc_norm`; **—** means not reported in the selected source. Bold marks the
highest available reported value.

| Model | ARC-Easy acc / acc_norm | ARC-Challenge acc / acc_norm | PIQA acc / acc_norm | HellaSwag acc / acc_norm |
| --- | ---: | ---: | ---: | ---: |
| **OpenSML-150M (SFT)** | **56.65% / 55.43%** | **26.02% / 29.52%** | **64.53% / 64.09%** | **30.55% / 33.94%** |
| [GPT-2 (124M, base)](https://huggingface.co/openai-community/gpt2) | — / 39.48% | — / — | — / 62.51% | 28.92% / 31.14% |
| [OPT-125M (base)](https://huggingface.co/facebook/opt-125m) | 43.52% / 39.98% | 18.94% / 22.78% | 63.00% / 62.02% | — / — |
| [Pythia-160M (base)](https://huggingface.co/EleutherAI/pythia-160m) | 43.52% / 39.65% | 18.77% / 23.29% | 62.73% / 61.64% | — / — |

OpenSML's largest reported advantage is normalized ARC-Easy: +15.45 percentage
points over OPT-125M and +15.78 over Pythia-160M. These are cross-source
comparisons, not a controlled ranking: prompts, normalization, numerical settings,
and dataset revisions are not verified identical. OpenSML is SFT-trained; the
references are base models. Source records are in the technical report.

### Instruction Following: IFEval

| Checkpoint | Prompt strict | Instruction strict | Prompt loose | Instruction loose | 1,280-token cap hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenSML-150M | 15.16% (82/541) | 25.30% (211/834) | 15.71% (85/541) | 25.78% (215/834) | 33/541 |

These results apply to OpenSML-150M, not the pretrained base. All 541 prompts and
834 instructions were scored using programmatic strict/loose verifiers, without
an LLM judge.

<details>
<summary>IFEval formatting, generation, and scoring</summary>

Generation is zero-shot and greedy, with plain `User: {prompt}\nAssistant:`
formatting, no added system message, and a 1,280-new-token cap. The context is
2,048 tokens; document-end EOS is ID 1. No additional stop strings or repetition
penalty are used. All responses are scored as produced: 508 stop at EOS and 33
reach the generation cap; none hit the context limit. The dataset's only split
is named `train`, but these prompts are evaluation inputs, not supervised records.

</details>

### Complete-answer Diagnostics

Saved development checks report a follow-up joint proxy of 22/32 and named-constraint
passes on 5/32 public cases. Repetition was detected on 4/63 legacy turns and
18/73 public-development turns. These are mechanical development proxies;
a systematic independent review of complete-answer correctness is still pending.
Likelihood and IFEval do not establish reliable factual answers. No MT-Bench or
coding-success score is reported for the selected model.

## Intended Use and Limitations

Intended for small-language-model research and controlled experimentation.
Responses may be incorrect, repetitive, biased, or unsafe. Production use,
safety behavior, tool use, multilingual ability, and extended dialogue remain
unvalidated. Benchmark reuse during selection limits evaluation independence;
a comprehensive contamination audit has not been established.

## Checkpoint Verification

Saved hashes and integrity receipts identify the selected weights and completed
evaluations. They do not establish cross-framework inference parity. A released
weight export and reproducible inference guide remain pending. See the
[results and provenance record](../results/OPENSMl_150M_RESULTS.json) and technical report.

## Licensing and Data Provenance

Author-controlled project contributions use [Apache-2.0](../LICENSE). Upstream
materials retain their own terms. SFT includes SQuAD v2 (CC-BY-SA-4.0) and SciQ
(CC-BY-NC-3.0); repository metadata does not grant blanket commercial rights to
all training materials. Weight-release licensing review remains pending.
