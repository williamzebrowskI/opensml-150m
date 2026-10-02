# OpenSML-150M

Native MLX small-language-model training and research on Apple Silicon.

**Private source snapshot.** This repository contains the V1 model implementation,
pretraining pipeline, tokenizer artifacts and fitting code, retained SFT recipes,
evaluation implementation, recorded results, and technical documentation.

[Technical report](sml-mlx-v1/docs/TECHNICAL_REPORT.md) · [Model card](https://huggingface.co/wzebrowski/OpenSML-150M) · [Run guide](RUN_GUIDE.md) · [Source notice](SOURCE_NOTICE.md) · [Snapshot manifest](provenance/SNAPSHOT_MANIFEST.json)

## Selected model

The V1 pretrained base is Stage B step 73,243, after 7,800,086,528 tokens.
Its selected fine-tuned descendant is total SFT step 768:

```text
V1 Stage A → Stage B base → Unified384 → Repair512 → OpenSML-150M
                            +384         +128       +256 SFT updates
```

The current model-card label uses 150,439,188 parameters; existing component
accounting totals 150,439,168. The report records this unresolved 20-parameter
difference. Both round to 150.44M.

## Repository contents

| Path | Purpose |
| --- | --- |
| `sml-mlx-v1/sml_v1/` | Architecture, tokenizer fitting, streaming, optimizer and distributed pretraining |
| `sml-mlx-v1/configs/`, `stage_b/` | Stage A and B recipes and launch code |
| `sml-mlx-v1/tokenizer/bytebpe32k_v1/` | Frozen tokenizer, configuration and audit |
| `sml-mlx-v1/experiments/` | Selected SFT lineage code and source pins; only the retained final candidate metadata |
| `sml-mlx-v1/sft/` | Dependencies used by retained recipes; supporting branches are not model ancestors |
| `sml-mlx-v1/evaluation/` | Multiple-choice and IFEval evaluators; MT-Bench implementation retained |
| `sml-mlx-v1/docs/` | Current technical report and historical operational guides |
| `docs/figures/`, `results/` | Pretraining charts and selected benchmark/provenance record |
| `sml-mlx-v1/sml_v1/cluster/`, `sml-mlx-v1/scripts/` | V1 cluster and inference utilities |
| `provenance/` | Snapshot source hashes, runtime versions and selected checkpoint metadata |

## Setup

Python 3.11+ on Apple Silicon:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_snapshot.py
```

The production runtime used a custom MLX/JACCL build. The PyPI dependency pins
are a baseline, not a claim of numerical or distributed parity with that build.
See [MLX runtime notes](sml-mlx-v1/docs/MLX_NATIVE_BUILD.md).

This snapshot does not contain model/optimizer tensors, raw datasets, or prepared
training pools. Exact replay requires those pinned external/local artifacts and
site-specific cluster setup. Original launchers retain their integrity checks;
this is a complete code-and-recipe archive, not a verified one-command reproduction.
See the run guide for stage entry points and prerequisite assets.

## License

Author-controlled contributions are Apache-2.0. Third-party code and training data
retain their own terms; see SOURCE_NOTICE.md. Keep this repository private until
release documentation, portability checks, and upstream attribution are reviewed.

## Release naming

This dedicated repository calls its training pipeline V1. Saved tokenizer and checkpoint
contracts retain their original identifiers for compatibility, and provenance receipts
retain historical source paths. This naming change does not change the model lineage.
