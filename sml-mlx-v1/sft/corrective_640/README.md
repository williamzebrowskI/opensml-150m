# Correction branch from grounded/ranking checkpoint 640

256 additional updates, ending at cumulative step 896. Parent weights and all
existing experiments are preserved. Run output: `runs/sft_corrective_640_v1`.

## Training

Fresh FP32 AdamW, peak LR 1e-6, final LR 3e-7, 16-update warmup, clipping 1,
no weight decay. Example losses are averaged per example and then per family;
every original assistant reply receives content and EOS supervision, using its
complete native instruction/history prefix. No example or answer is truncated.

| Family | Examples/update | Loss weight | Selected training examples |
| --- | ---: | ---: | ---: |
| Grounded reading and evidence explanations | 4 | 35% | 1,024 |
| Combinations of output/content instructions | 5 | 25% | 1,280 |
| Full conversation retention and new summary/rewrite dialogues | 2 | 20% | 512 |
| Labeled candidate ranking | 4 | 20% | 1,024 |

Reading uses 696 answerable and 72 unanswerable SQuAD TRAIN examples from
previously unused source paragraphs, plus 128 new relationship problems and 128
fictional experiment problems. Explanations state only supplied observations;
they are not invented explanations for ARC annotations. Unknown-answer targets
are retained so the model can still abstain when information is absent.

Instruction tasks combine sorting, letter case, ordering, JSON structure,
updated facts, exact word counts and bullet/line/sentence formatting. TRAIN uses
presentation variants 0/1, development uses 2, reserved test uses 3. Scene groups
are disjoint; shared task families still limit the generalization claim.

Ranking uses two ARC-Challenge TRAIN questions, one PIQA TRAIN question and one
HellaSwag TRAIN continuation per update. It uses answer log likelihood divided
by Unicode character count, without a chat prefix or EOS target. Existing public
test/validation questions and all IFEval prompts are exclusion-only inputs.

Replay alternates 256 original whole everyday conversations and 256 newly
authored two-turn dialogues. Every assistant turn is supervised. Original
conversations are retention replay, may have trained ancestor 384, and are newly
selected for this round rather than entirely unseen data. All selected groups
are disjoint from the 384-to-640 round. Authored dialogues preserve named people,
current schedules, locations, requests and confirmation questions.

An extra weight-0.10 KL anchor uses frozen 640 on HellaSwag TRAIN prose only.
There is no stronger teacher, reward model, DPO, model blend or reinforcement
learning in this experiment. The objective is a testable hypothesis.

## Run

```bash
<SOURCE_WORKSPACE>/.venv/bin/python -u \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/corrective_640/launch.py --run
```

Preparation and readiness checks use `--prepare` and `--check`, respectively.
Setup requires a recorded review of stratified TRAIN samples and freezes source
weights, selection, code and runtime. Readiness checks every selected example,
all assistant/EOS masks, candidate token boundaries, scalar likelihoods, a
disposable update, checkpoint restoration and the next update, and cached versus
full-context decoding. Checks do not leave production training checkpoints.

Save and evaluate every 32 updates: 640, 672, 704, 736, 768, 800, 832, 864, 896.
All checkpoints are retained; there is no automatic best selection or promotion.
Development reports answerable accuracy, incorrect abstention, composed
instruction passes, evidence explanations, authored dialogue exact matches,
candidate scoring and prose retention. Natural conversation correctness still
requires reading saved outputs. Stopping alone is not success. Development
stopping/repetition include intermediate generated conversation turns.

Ctrl-C finishes the current update and saves optimizer/task cursors. Resume with
the same command plus `--clear-stop`. Public benchmarks are run only after a
candidate is chosen; the reserved test is frozen and unused during training.

Source revisions, hashes, selection counts and overlap limitations are in
`selection.json`. License status is inherited from pinned source records;
PIQA's dataset license is recorded as unknown. No release of these sources is
implied. Exact/13-word overlap screening does not prove absence of semantic or
pretraining contamination.
