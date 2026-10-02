# Balanced transfer from Explanation Transfer 1472

This 512-update continuation starts from the preserved
`runs/sft_explanation_transfer_1024_v1/step_0001472_901be558458f` weights. It
does not modify that checkpoint or borrow 1920 or 2048 weights. Final step is
1984. It is SFT with a supervised answer-choice ranking term and a frozen-parent
prose KL anchor; it is not RL.

The new portion uses 1,024 CosmosQA TRAIN contexts with complete-answer targets
and 1,024 WinoGrande XL TRAIN situations with labeled choice ranking. These
come from the existing pinned, screened, sample-reviewed source partitions.
The other 4,608 exposures rehearse previously consumed e-SNLI explanations,
instruction formatting, reading/two-turn answers, and CommonsenseQA/SocialIQA
choices. They are replay, not independent new examples. OpenBookQA was considered
but excluded after its own top-rated training rows showed unsupported or wrong
answers in sampled review. No public benchmark questions are training targets.

The objective allocates 25% to complete contextual answers, 20% each to prior
instruction, reading, and explanation targets, and 15% to gold choice ranking.
It also uses a 0.10 prose KL term against frozen 1472. The fresh FP32 AdamW
schedule warms up to 3e-6 and decays to 1e-6; no existing optimizer is reused.

All development checkpoints are retained: baseline 1472 and every 64 updates
through 1984. Each records actual outputs and separate content, stopping,
instruction, reading, explanation-verdict, and prose metrics. A correct verdict
or surface-complete explanation does not establish semantic correctness; read
the answers before choosing a checkpoint. The run does not select a best model,
run reserved tests, benchmark itself, or promote anything automatically.

Preparation and checks were completed before giving the run command. To start or
resume on the main Mac:

```sh
/Users/williamzebrowski/sml-mlx/.venv/bin/python \
  /Users/williamzebrowski/sml-mlx/sml-mlx-v1/sft/balanced_transfer_1472/launch.py \
  --run --clear-stop
```

`--prepare` builds and pins the selected derivative data locally after verifying
the upstream pinned source files in RAM. `--check` performs a disposable update.
The run checks hashes of the parent checkpoint, selection, training data, code,
tokenizer, and dependencies; it resumes the saved optimizer and data cursor.
Stop with Ctrl-C, then rerun the same command. A completed run will not restart.

The new-source labels are screened and only sample-reviewed, not certified one by
one. The old explanations also contain known annotation noise. Context grouping
and public-overlap screens are heuristic. The repeated public benchmark suites
are development diagnostics, not a pristine final test.

Sources: [CosmosQA](https://huggingface.co/datasets/allenai/cosmos_qa),
[WinoGrande](https://huggingface.co/datasets/allenai/winogrande),
[e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI). Pinned revisions, bytes,
hashes, license notes, and selection decisions are in the inherited source modules
and frozen preparation records.
