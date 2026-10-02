# Training and reproducibility

This guide describes the OpenSML-150M source archive, the selected training path, and the prerequisites for running its code. The [technical report](sml-mlx-v1/docs/TECHNICAL_REPORT.md) contains the detailed historical account; the [run guide](RUN_GUIDE.md) is the compact entry-point index.

## Selected model and scope

```text
Stage A → Stage B base (step 73,243) → Unified384 → Repair512 → OpenSML-150M
                                       +384        +128        +256 updates
                                                               SFT step 768
```

The pretrained base consumed 7,800,086,528 tokens. SFT uses full-parameter updates, without LoRA or parameter interpolation. The final retained checkpoint is the midpoint of a longer configured experiment. Running the complete configured budgets does not reproduce the selected step 768.

This repository includes code, configurations, frozen tokenizer artifacts, source pins, selected checkpoint metadata, charts, and recorded results. It excludes model/optimizer tensors, raw datasets, prepared training pools, evaluation caches, and optional judge weights. Exact historical replay is not currently a verified one-command workflow.

## Environment and installation

Use Python 3.11 or newer on Apple Silicon. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_snapshot.py
```

The production environment used a custom MLX/JACCL build for distributed training. Dependency pins supply a baseline, not a guarantee of numerical or transport equivalence. Consult the [MLX runtime notes](sml-mlx-v1/docs/MLX_NATIVE_BUILD.md) and [recorded runtime](provenance/runtime.json) before setting up a cluster.

Archived network tools describe the original site's hardware and interfaces. Configure and verify the intended hosts, paths, networking, storage, and interpreter locations before any actual distributed launch. Reading `--help` does not configure a cluster.

## Tokenizer contract

The selected tokenizer is the frozen [32K byte-BPE bundle](sml-mlx-v1/tokenizer/bytebpe32k_v1/), including its manifest, configuration, and audit. Preserve it when loading historical checkpoints. Retraining the tokenizer changes token IDs and the checkpoint contract.

Inspect tokenizer preparation and training options from the project directory:

```bash
cd sml-mlx-v1
python -m sml_v1.train_tokenizer --help
```

The dedicated source package is named `sml_v1`. Serialized tokenizer/checkpoint identifiers and historical provenance retain original names for compatibility. Renaming those identifiers or editing frozen files would change their hashes.

## Pretraining: Stage A and Stage B

| Stage | Included recipe and entry point | Data sources |
| --- | --- | --- |
| Stage A | [configs/corpus.json](sml-mlx-v1/configs/corpus.json), [configs/pilot.json](sml-mlx-v1/configs/pilot.json), `python -m sml_v1.launch` | FineWeb-Edu, DCLM-Edu, English FineWiki, Cosmopedia v2 |
| Stage B | [stage_b/corpus.json](sml-mlx-v1/stage_b/corpus.json), `python stage_b/launch.py` | Continuation with an adjusted mixture of the same sources |

The initial mixture assigns 55%, 25%, 10%, and 10% configured token shares respectively; Stage B assigns 55%, 20%, 15%, and 10%. These are configured token shares, not a claim that each complete upstream dataset was consumed.

From `sml-mlx-v1/`, inspect the available controls:

```bash
python -m sml_v1.launch --help
python -m sml_v1.continuation --help
python -m sml_v1.data_review --help
python stage_b/launch.py --help
python scripts/launch_five_mac.py --help
```

Stage A began with the initial recipe and subsequently underwent documented continuation, learning-rate, topology, and data transitions. The pilot JSON alone is not the complete historical pretraining schedule. Use the technical report and recorded receipts to determine the relevant boundary. Resuming the selected Stage B base requires its original model, optimizer, and cursor state.

## Supervised fine-tuning

| Stage | Retained updates | Included implementation | Sources |
| --- | ---: | --- | --- |
| Unified384 | 384 | [Unified experiment](sml-mlx-v1/experiments/unified_text_v1_sft_v1/) | Smol-SmolTalk, UltraChat, SQuAD v2, ARC training questions, locally authored follow-ups |
| Repair512 | 128 additional | [Public repair experiment](sml-mlx-v1/experiments/public_repair_384_v1/) | Dolly and Tulu Persona examples, with the recipe's earlier-data mixture |
| Final continuation | 256 additional | [Retained candidate](sml-mlx-v1/experiments/repair512_search_v1/candidates/028-broad-public-long-stable/) | SmolTalk constraints, SQuAD v2, SciQ, Repair512 replay |

From `sml-mlx-v1/`:

```bash
python experiments/unified_text_v1_sft_v1/scripts/launch.py --help
python experiments/public_repair_384_v1/scripts/launch.py --help
python experiments/repair512_search_v1/scripts/experiment.py --help
```

The final candidate's full data/config describe a 512-update continuation; the retained +256 midpoint consumes the first 4,096 training records. Preserve the recorded ordering and boundary when studying that checkpoint. Supporting recipes in `sft/` are implementation dependencies, not all selected-model ancestors.

## Required external assets

| Asset | Included record | Needed for |
| --- | --- | --- |
| Pretrained base, Unified384, Repair512, and selected weights | [Checkpoint metadata](provenance/checkpoints/) | Loading, stage continuation, or re-evaluation |
| Original optimizer and streaming cursor state | Historical report and bundle manifests | Exact pretraining continuation |
| Prepared earlier source pools | Archived preparation code and run guide | Reconstructing Unified384 and replay inputs |
| Unified/repair prepared examples, final-candidate data and raw pinned inputs | Recipe configurations, selections, and source pins | Replaying the SFT stages |
| Protected evaluation/control bundles and benchmark exclusion inputs | Recipe integrity guards | Readiness checks and overlap filtering |
| Benchmark caches and optional judge weights | Evaluation preparation code and pinned sources | New evaluation runs |

Historical launchers retain integrity guards. Some original readiness records use source-workspace paths or hashes predating this extraction and rename. Review and regenerate readiness manifests against the intended inputs; do not bypass those checks or silently substitute another dataset. Deterministic preparation code and pins do not establish byte-identical reconstruction without the required inputs.

## Evaluation and native inference

From `sml-mlx-v1/`:

```bash
python evaluation/full_benchmarks/prepare.py --help
python evaluation/full_benchmarks/launch.py --help
python evaluation/mt_bench_local/run.py --help
python scripts/completion_playground.py --help
```

Recorded likelihood evaluation covers 15,428 examples across ARC-Easy, ARC-Challenge, PIQA, and HellaSwag. IFEval covers 541 prompts and 834 instructions. Exact prompts, normalization, precision, generation caps, split pins, and receipt boundaries are in the technical report and [selected results](results/OPENSMl_150M_RESULTS.json).

MT-Bench tooling is included; no MT-Bench score is claimed for the selected checkpoint. External-model comparisons use published scores. No new external-model evaluation is required to read or reproduce those comparison tables.

The unchanged selected weights and frozen tokenizer are available in the Hugging Face native inference bundle (authorized access is required while that repository is private). Loading and short generation were verified on the recorded local MLX runtime. See the root README and inference_native/INFERENCE_VERIFICATION.json. Earlier checkpoints and optimizer state remain external; cross-framework inference parity is unverified.

## Verification status

Archive preparation verified 27 existing pipeline tests, eight selected `--help` entry points, frozen tokenizer loading, network-helper imports, and a successful wheel build. These checks exercise small test models and source packaging; they do not establish production cluster readiness, exact historical replay, or benchmark equivalence after relocation.

To run the existing offline pipeline checks from the repository root:

```bash
python -m unittest discover -s sml-mlx-v1/tests -p test_pipeline.py -v
```

[Verification receipts](provenance/VERIFICATION.json) record the performed checks. The [snapshot manifest](provenance/SNAPSHOT_MANIFEST.json) records packaged-file hashes and original source provenance. The verifier detects file changes; it does not train or benchmark a model.

Different data, ordering, tokenizer, schedules, runtime, or post-training boundaries produce a new experiment rather than the documented selected model. Follow [third-party notices](THIRD_PARTY_NOTICES.md) for component attribution and source terms; use [CITATION.cff](CITATION.cff) to cite this project.

A fresh public download has now been tested in a separate PyPI environment with the pinned inference requirements; see [clean-download verification](provenance/CLEAN_DOWNLOAD_VERIFICATION.json). This improves the inference validation status, not exact historical training replay.
