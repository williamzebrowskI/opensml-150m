> RETIRED: This experiment failed answer-quality review. Its run artifacts, selection manifest and readiness receipt were removed at user request. Do not run this recipe again unchanged. Parent 512 is preserved. See reports/answer_completion_v1_lessons.json.

# Answer-completion continuation from curriculum 512

This experiment continues the preserved concise unified-curriculum checkpoint
`runs/sft_base_curriculum_v1/step_0000512_fd767fb7545a`. It does not restart from
the pretrained base or continue the separate 1920 lineage. Production training
is started by the user; nothing is automatically promoted to the playground.

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/sft/answer_completion/launch.py \
  --run --clear-stop
```

The same command resumes an interrupted run. Ctrl-C saves after the current
update. Completion is 256 additional updates, absolute step 768, with no automatic
extension. Outputs are separate: `runs/sft_answer_completion_v1`.

## Hypothesis and limits

512 passed 215/834 IFEval instructions strictly but 103/541 responses hit the
1,280-token generation limit. All 103 repeated at least one five-word phrase
three or more times. Inspection also found incorrect answers and generic setup
phrases. Stopping alone is not correctness. Lower target likelihood loss did not
predict useful open-ended responses in earlier experiments.

This is ordinary SFT on complete replies, with skill rehearsal every update.
It is not RL, DPO, unlikelihood training, an EOS-only objective, or a new teacher.
It tests whether broader contextual supervision at a low learning rate improves
actual responses without sacrificing the prior curriculum's skills. Success is
not guaranteed by this data recipe or by a falling loss.

## Frozen recipe

- 4,096 examples, one pass, batch 16, microbatch 1, main Mac only.
- Every update has eight new examples and eight previously consumed examples.
- New: 768 everyday/conversational/explanatory, 640 rewrite/summary/formatting, and
  640 contextual follow-up/missing-information examples.
  New editing/formatting is 256 rewrites + 192 summaries + 192 formatting tasks;
  it changed from the proposed all-editing subset because too few unused,
  nonoverlapping editing examples passed. Natural replies are 380 conversation,
  384 explanation and four distinct greeting examples; substantial everyday
  dialogue is also represented by follow-ups and rehearsal. Exact family counts
  and source identities are frozen in config and selection manifests.
- Rehearsal: 2,048 examples drawn across all 11 prior families, at one quarter
  of each family's previous size. No saved evaluation examples become training.
- Fresh FP32 AdamW moments initialized from 512 weights; no parent optimizer
  continuation. LR warms over 16 updates to 3e-6, then decays to 1e-6.
- Equal-example assistant-only cross entropy, including exactly one EOS.
  Context 1,024. Entire targets must fit; no answer truncation.
- New answers may be up to 192 tokens / 120 words, compared with the prior
  128-token cap. This admits complete slightly longer replies while rejecting
  long essays. Complete replies under eight words are now eligible too.
- Reuse the same pinned TRAIN datasets: smol-smoltalk (Apache-2.0), SQuAD v2
  (CC-BY-SA-4.0), CommonsenseQA (MIT), SocialIQA (CC-BY-4.0).
  Smol-smoltalk contains synthetic examples. No teacher model is run locally.
- Smol first turns and up to three genuine follow-up turns are considered;
  earlier messages stay in the prompt. Only one example per source group is
  selected. New examples exclude *all* source groups selected in the 512 run.
- Math exercises and coding remain excluded by inherited topic screens;
  general science is allowed. Heuristics and sampled review cannot guarantee
  every fact, instruction, or topic boundary is correct.

Preparation reconstructs the parent's exact selections and selects new rows in
one pinned streaming pass. Raw Parquet blocks are discarded. Only bounded
candidate pools and selected conversations are held in RAM. The selection
manifest stores identifiers/hashes/counts, not a raw dataset cache. Source cards,
revisions, and attribution are retained in the manifest. CC-BY/CC-BY-SA
attribution and other license obligations remain applicable.

## Selection and evaluation

Checkpoint/evaluate every 64 updates: 512, 576, 640, 704, 768. All are review
candidates, never an automatically selected best.

The 273 source development examples and 273 reserved source test examples are
reused from the parent and remain excluded from training. Their scores provide a
within-lineage comparison, not a fresh independent test after repeated review.
There are also 24 newly authored development prompts and 24 separately reserved
manual prompts. They test familiar categories; they do not establish general
benchmark coverage. Public benchmark prompts, earlier review prompts, and prose
diagnostic passages are excluded from gradients using exact and overlap checks.
These screens do not certify the absence of all semantic or pretraining overlap.

Saved answers must be reviewed for relevance, correctness, complete task
execution, stopping and repetition. Do not select by target NLL or exact match
alone: many open answers have several valid wordings. Generation allows 320 tokens
in development so the 192-token training limit does not conceal loops. Public
benchmark settings remain unchanged when a candidate is later evaluated.

Keep 512 and the separate preserved 1920 reference. Choose a candidate using only
development answers, then freeze that choice before reserved/public testing.

## Checks

The readiness receipt covers masks, EOS, full-corpus lengths, exact source-group
isolation, new-versus-rehearsal membership, 8+8 batches, target loss/padding,
cached/full decoding parity, and disposable model updates with save/restore
checks. These disposable updates are discarded and never modify source weights.
Checkpoint 512, the pretrained base, and 1920 are checksum-protected.
