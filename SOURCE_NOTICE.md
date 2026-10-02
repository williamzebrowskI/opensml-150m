# Source and data notice

This repository is extracted from the owner's sml-mlx workspace. It preserves
V2 source and supporting imported helper modules, not a fresh training run.
Selected checkpoint lineage: V2 Stage B → Unified384 → Repair512 → selected step 768.
Other helper recipe names identify code dependencies, not additional ancestors.

Author-controlled contributions use the included Apache-2.0 license.
The vendored IFEval implementation retains its harness license and notice under
sml-mlx-v2/evaluation/full_benchmarks/vendor/. MT-Bench source attribution is
preserved in evaluation/mt_bench_local/sources.json and upstream files.

Pretraining dataset repositories: HuggingFaceTB/smollm-corpus (FineWeb-Edu and
Cosmopedia v2), HuggingFaceTB/dclm-edu, HuggingFaceFW/finewiki.
SFT dataset repositories: HuggingFaceTB/smol-smoltalk, HuggingFaceH4/ultrachat_200k,
rajpurkar/squad_v2, allenai/ai2_arc, databricks/databricks-dolly-15k,
allenai/tulu-3-sft-personas-instruction-following, HuggingFaceTB/smoltalk, allenai/sciq.
Locally authored follow-up and recovery examples are separately described in the report.

Upstream data retain their own terms. The selected source records include SQuAD
CC-BY-SA-4.0, Dolly CC-BY-SA-3.0, Tulu Persona ODC-BY and SciQ CC-BY-NC-3.0.
This notice is not a completed public weight-release licensing determination.
Raw and prepared dataset pools and model/optimizer weights are excluded.
No credentials, SSH private keys, or existing network-plan files are packaged.
