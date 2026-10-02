# Context reasoning from Skill Recovery 1024

Starts from preserved Recovery 1024, not 1920 or the rejected content-preserving
checkpoints. Output is separate: `runs/sft_context_reasoning_1024_v1/main`.
Final step 2048 belongs to this lineage; it is not the old two-turn 1920→2048.

## Experiment

The new ingredient is contextual commonsense, not more formatting templates:

- 2,048 WinoGrande XL TRAIN situations teach choice ranking and reference resolution.
  They do not teach one-word assistant replies or copying the whole prompt.
- 2,048 Cosmos QA TRAIN situations teach labeled choice ranking and short natural
  answers to questions about the supplied everyday context.
- 2,048 old CommonsenseQA/SocialIQA ranking exposures retain earlier skills.
- 6,144 old answer replay exposures: three natural/two-turn answers and three
  exact-content formatting targets per update. These are 1,792 unique rows,
  intentionally repeated; they are not new independent examples.

One pass over 4,096 new source examples; 1,024 updates; fresh FP32 AdamW;
64-update warmup to 5e-6, then cosine decay to 1e-6; no weight decay.
Loss weights: 0.30 new-answer CE, 0.50 replay CE, 0.20 labeled choice ranking.
Additional 0.10 prose KL against frozen original 1024. Per-example averaging
prevents longer answers dominating. No stronger teacher, RL, DPO, or generated
rationales. The main Mac runs this experiment.

This may improve reasoning transfer while protecting instruction behavior. It
is not a promise of beating 1920, especially on science questions: these sources
are everyday reasoning, not a new science curriculum. QASC and OpenBookQA were
investigated but not selected after sampled annotation-quality concerns. No
QASC, OpenBookQA or SWAG examples are used here.

## Data checks and limitations

Public TRAIN only. Revisions and SHA256 checksums pinned in data.py. Raw files
stream into bounded RAM (largest 16.7 MB); selected rows remain in RAM. No raw
corpus disk cache is written. Small review samples/manifests are saved.

Development and reserved test each contain 128 examples per new source, from
TRAIN subpartitions. Cosmos contexts stay together; WinoGrande normalized
prefixes (names replaced) stay together to catch typical counterfactual pairs.
Grouping is heuristic, not proof of semantic independence. Old retention probes
are reused. Exact/13-gram matches to locally available public benchmark prompts
are rejected. This is not complete semantic decontamination.

Source annotations undergo scope/length/schema/answer-shape checks and sampled
manual review; known bad rows are explicitly rejected. **Not every example is
manually verified.** Cosmos QA includes plausible-inference judgments rather
than only entailed facts. Whitespace and punctuation are normalized; no teacher
rewrites the source labels. Sample review does not certify the whole corpus.

## Run

```sh
/Users/williamzebrowski/sml-mlx/.venv/bin/python \
  /Users/williamzebrowski/sml-mlx/sml-mlx-v2/sft/context_reasoning_1024/launch.py \
  --run --clear-stop
```

Repeating this command resumes exact optimizer/data cursors. Completed runs do
not restart. Original 1024 and 1920 remain preserved. There is no automatic
extension or playground promotion. Shared experiment locks prevent simultaneous
training jobs. Keep at least 45 GiB free for checkpoints and optimizer states.

Development checkpoints: 1024, 1152, 1280, 1536, 1792, 2048. Candidate selection
requires retention gates relative to 1024 and improvement on the new private
reasoning diagnostic. Review the saved actual answers before selecting. Once a
candidate is selected, freeze it before reserved evaluation:

```sh
# Replace STEP with the reviewed absolute checkpoint step.
/Users/williamzebrowski/sml-mlx/.venv/bin/python \
  /Users/williamzebrowski/sml-mlx/sml-mlx-v2/sft/context_reasoning_1024/finalize.py \
  --step STEP --run
```

Then benchmark that frozen candidate against unchanged 1024 and 1920 under the
same settings. Public benchmarks have already been inspected many times; they
are an iterative development comparison, not a pristine one-shot test. Use
paired question outcomes to judge small changes. Better benchmark scores alone
do not establish better everyday conversation.

## Attribution

- Sakaguchi, Le Bras, Bhagavatula, Choi. *WinoGrande: An Adversarial Winograd Schema
  Challenge at Scale* (2019). https://github.com/allenai/winogrande
  Official README licenses the **dataset** under CC-BY; code is separately
  Apache-2.0. https://github.com/allenai/winogrande#license
- Huang, Le Bras, Bhagavatula, Choi. *Cosmos QA: Machine Reading Comprehension with
  Contextual Commonsense Reasoning* (2019). https://github.com/wilburOne/cosmosqa
  CC-BY-4.0, per publisher dataset card:
  https://huggingface.co/datasets/allenai/cosmos_qa#licensing-information

Both are attribution licenses allowing commercial use under their terms.
Preserve notices when distributing data derivatives. Inherited replay-source
attribution remains in its existing experiment directories.
