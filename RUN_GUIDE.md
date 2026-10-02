# Run guide

Commands below identify the archived pipeline entry points. They do not start
training unless an explicit run flag/subcommand is supplied. Run commands from
the repository root with the virtual environment active. Frozen tokenizers,
configuration JSONs, and source code are included; weights and data caches are not.

## Pretraining and tokenizer

```bash
cd sml-mlx-v1
python -m sml_v1.train_tokenizer --help
python -m sml_v1.launch --help
python -m sml_v1.continuation --help
python -m sml_v1.data_review --help
python stage_b/launch.py --help
python scripts/launch_five_mac.py --help
```

The selected tokenizer is already fitted; retraining it changes the model contract.
Stage A began with configs/corpus.json and configs/pilot.json; historical continuation
and later LR/data transitions are described in docs/TECHNICAL_REPORT.md.
Stage B uses stage_b/corpus.json. Configure your own cluster identities and networking;
operational guides describe the source hardware and are historical.

## Selected SFT lineage

```bash
python experiments/unified_text_v1_sft_v1/scripts/launch.py --help
python experiments/public_repair_384_v1/scripts/launch.py --help
python experiments/repair512_search_v1/scripts/experiment.py --help
```

Retained boundaries are 384, 128 additional, and 256 additional updates.
The original plans extend beyond those boundaries; do not run their full configured
budgets and call the resulting weights the selected model. The final candidate is
experiments/repair512_search_v1/candidates/028-broad-public-long-stable.

Required asset families:

- V1 Stage B base, Unified384 and Repair512 checkpoint bundles with hashes in provenance/checkpoints.
- Frozen prepared source pools under sft/conversation_foundation_original, text_followup_512_v1, grounded_rank_384 and preference_long_640.
- Unified and repair data/prepared.json, final-candidate data.json and pinned raw source inputs.
- Legacy evaluation/control checkpoint bundles protected by the original launchers.
- Benchmark exclusion inputs and native tokenizer contract.

Preparation scripts and source pins are included. Some recipe guards reference
original workspace paths and original file hashes. Relocation changes those contracts;
review and regenerate readiness manifests rather than bypassing hash checks.
No production training or model evaluation was started while preparing this repository.

## Evaluation

```bash
python evaluation/full_benchmarks/prepare.py --help
python evaluation/full_benchmarks/launch.py --help
python evaluation/mt_bench_local/run.py --help
```

Inspect the archived selected candidate benchmark wrappers for its exact model key.
Evaluation caches and judge weights must be prepared separately. MT-Bench code is
included, but no MT-Bench score is claimed for the selected checkpoint. External
baseline numbers in the card are published scores; the abandoned external-baseline
runner is not packaged and its local results are not used.

## Documentation

Current technical report: sml-mlx-v1/docs/TECHNICAL_REPORT.md.
Original September 18 material is a historical appendix. Other dated operational
documents are historical guides and may describe earlier branches or absent assets.
The snapshot verifier checks packaged file integrity without training or model inference.
