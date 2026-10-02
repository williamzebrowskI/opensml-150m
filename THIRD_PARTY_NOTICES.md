# Third-party notices

This index records third-party components and data sources associated with OpenSML-150M. Author-controlled contributions use the root [Apache-2.0 license](LICENSE). Upstream components retain their own notices and terms. Code, installed dependencies, evaluation inputs, and training datasets have distinct provenance.

## Included evaluation components

| Component | Source and revision | Notices retained in this repository |
| --- | --- | --- |
| IFEval task implementation | EleutherAI lm-evaluation-harness, commit `b954108c9baaaa934b4ad842033b31a97ee30816` | [Harness MIT license](sml-mlx-v1/evaluation/full_benchmarks/vendor/HARNESS_LICENSE.md), [vendoring notice](sml-mlx-v1/evaluation/full_benchmarks/vendor/NOTICE.md), and [source URLs/checksums](sml-mlx-v1/evaluation/full_benchmarks/vendor_sources.json). Individual files retain Google Research Apache-2.0 notices where present. |
| MT-Bench questions and reference answers | LMSYS FastChat, revision `587d5cfa1609a43d192cedb8441cac3c17db105d` | [Pinned sources](sml-mlx-v1/evaluation/mt_bench_local/sources.json) and [upstream inputs](sml-mlx-v1/evaluation/mt_bench_local/upstream/). These inputs are evaluation material, not a claimed training source. |
| Prometheus judge integration | Prometheus revision `dcfb44272d5d0414832f5dbb8c2a05ebc2614234`; optional MLX judge `mlx-community/prometheus-7b-v1.0-8bit` at revision `b92ca38184118ec29626b902fad7664a0f127c26` | [Pinned sources](sml-mlx-v1/evaluation/mt_bench_local/sources.json). Judge weights are not included. Consult the pinned upstream component for its terms before obtaining it. No MT-Bench result is claimed for the selected model. |

## Training data sources

The selected lineage is Stage A → Stage B → Unified384 → Repair512 → OpenSML-150M at SFT step 768. Replay reuses earlier records; it is not an additional upstream dataset. The table identifies repositories rather than claiming that their entire contents were consumed.

| Source | Role in selected lineage | Attribution and records |
| --- | --- | --- |
| [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | FineWeb-Edu and Cosmopedia v2 pretraining subsets | [Stage A corpus](sml-mlx-v1/configs/corpus.json), [Stage B corpus](sml-mlx-v1/stage_b/corpus.json) |
| [HuggingFaceTB/dclm-edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu) | Pretraining | Corpus files and technical report |
| [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) | English pretraining | Corpus files and technical report |
| [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | Unified384 instruction examples | Unified preparation code and technical report |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | Unified384 instruction examples | Unified preparation code and technical report |
| [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2) | Reading-comprehension SFT, including final stage | Pranav Rajpurkar, Robin Jia, and Percy Liang. Recorded declaration: CC-BY-SA-4.0. [Detailed attribution and adaptations](sml-mlx-v1/sft/reading_repair/DATA_LICENSE.md). |
| [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | Unified384 training-split questions | Unified preparation code; benchmark splits are separately evaluated |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | Repair512 instruction examples | Recorded declaration: CC-BY-SA-3.0; repair source pins |
| [allenai/tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) | Repair512 instruction examples | Recorded declaration: ODC-BY; repair source pins |
| [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | Final-stage constraint examples | Recorded declaration: Apache-2.0; final-stage source pins |
| [allenai/sciq](https://huggingface.co/datasets/allenai/sciq) | Final-stage science examples | Recorded declaration: CC-BY-NC-3.0; final-stage source pins |

These declarations are recorded project metadata, not a fresh determination of all underlying material's terms. For pinned revisions, source-file hashes, filtering and exposure, see the [repair pins](sml-mlx-v1/experiments/public_repair_384_v1/source_pins.json), [final-stage pins](sml-mlx-v1/experiments/repair512_search_v1/candidates/028-broad-public-long-stable/source_pins.json), and [technical report](sml-mlx-v1/docs/TECHNICAL_REPORT.md). Raw and prepared dataset pools are not distributed in this repository.

## Supporting research components

The [e-SNLI license notice](sml-mlx-v1/sft/explanation_transfer_1024/ESNLI_LICENSE.txt) records the MIT notice attributed to Oana-Maria Camburu. Its supporting experiment implementation does not make e-SNLI an ancestor of the selected checkpoint.

Installed dependencies are specified in [requirements.txt](requirements.txt) and [pyproject.toml](sml-mlx-v1/pyproject.toml). They include MLX, MLX-LM, NumPy, Hugging Face dataset/tokenizer tools, PyArrow, Requests, FastAPI, Pydantic, and Uvicorn. Their upstream distributions carry their respective licenses; this archive does not vendor their installed implementations.

Some license-only records from the original evaluation environment are retained under `sml-mlx-v1/evaluation/full_benchmarks/_runtime/` for absl-py, immutabledict, langdetect, NLTK, packaging, and six. The corresponding installed packages are excluded.

## Scope

No repository-level license statement replaces upstream terms or establishes blanket commercial rights to training materials. A public model-weight release remains pending. See [SOURCE_NOTICE.md](SOURCE_NOTICE.md) for the archive's scope and [CITATION.cff](CITATION.cff) for citation metadata.
