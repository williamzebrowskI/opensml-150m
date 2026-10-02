# Teacher-assisted conversation SFT from Text Follow-up 768

This is offline teacher-assisted supervised training. It is not token-distribution
GKD, DPO, or reinforcement learning. Starts from the retained Text Follow-up 768;
512 and 768 are read-only inputs. The teacher is Qwen3.8-27B, MLX 4-bit,
pinned to 10c35caafbb80f7dc6a7a432cdd11af10a6d4818. Teacher inference is local;
no API key or paid inference service is used. Thinking is explicitly disabled through
the pinned chat template; only final response text can become a student target. The initial download is about 16.1 GB.

## Run the preparation and matched comparison

```sh
/Users/williamzebrowski/sml-mlx/.venv/bin/python -u \
  /Users/williamzebrowski/sml-mlx/sml-mlx-v1/sft/teacher_chat_768_v1/launch.py \
  --run
```

This command resumes existing work and performs these stages in order:

1. Freeze fresh source groups selected from the unused portion of the original
   Smol-SmolTalk/UltraChat selection. Exclude the retained models' consumed rows,
   their holdouts, existing transfer probes, and public benchmark overlaps using
   normalized text and word ngrams. Reserve whole groups before any generation.
2. Ask the unchanged 768 model for an initial answer. A local teacher writes a
   corrected answer and a natural follow-up involving revision, history,
   completion, passage grounding, text instructions or a topic switch.
3. Let 768 answer that follow-up using its OWN earlier answer. The teacher writes
   a complete correction to the follow-up and reviews both proposed targets in
   a separate call. Reject malformed, out-of-scope, incomplete, over-context,
   repetitive, or failed-review items. Never truncate an answer to make it fit.
4. Prepare 1,024 accepted training groups (two supervised records each), plus
   32 new development and 32 reserved-test groups. Add 2,048 records of familiar
   conversation, text formatting, reading, science and writing replay from 768.
   Code, calculation/math and structured-output tasks are filtered out. JSON is
   used only for files and teacher metadata, never as a student training task.
5. Compare three learning rates: 3e-7, 1e-6, 3e-6. Each trial starts from 768,
   uses identical batches and 64 optimizer updates, and follows the FIRST 64
   steps of its planned 512-update schedule. The sweep is not a shortened cosine
   run whose final learning rate would differ when resumed.
6. Generate fixed, unseen development conversations with each model's own
   history, score them with the teacher, and evaluate the original retention
   suite. Write comparison.json / comparison.md and readable transcripts.

This command intentionally ends at the comparison boundary. It does not choose a
winner automatically. Same-teacher ratings are a diagnostic, not independent
validation or official MT-Bench. Review the actual transcripts along with scores,
original content/formatting checks and repetition. There is no invented score
for a failed judgment; the valid and expected counts are reported separately.

## Continue the selected trial to the longer run

After reviewing the comparison, use the selected name (conservative, moderate or
faster). For example, if moderate is selected:

```sh
/Users/williamzebrowski/sml-mlx/.venv/bin/python -u \
  /Users/williamzebrowski/sml-mlx/sml-mlx-v1/sft/teacher_chat_768_v1/launch.py \
  --continue moderate
```

This restores the exact optimizer and data cursor from that trial and continues
to **512 total additional updates** (step 1280), not 512 more after the comparison.
There are two passes over the 4,096-record mix. Each batch has 8 teacher records
and 8 replay records. Every full target includes EOS. In a teacher follow-up, the
student's flawed earlier answer is context only and receives NO supervised loss.
Checkpoints/evaluations are retained at additional updates 64, 128, 256, 384, 512.
No quality-based automatic stopping or pruning; invalid numerical updates still
raise an error. Original 512/768 and all three comparison branches remain intact.

Teacher generation, data receipts, and training are resumable with the same
command. Ctrl-C during generation preserves completed group receipts. Ctrl-C
during training saves after the current update/evaluation. An unexpected crash
resumes from the most recent complete checkpoint with its optimizer. Only one
experiment process can hold the shared project experiment lock.

## Outputs

`runs/teacher_chat_768_v1/` contains:

- `data/groups/`: accepted and rejected group receipts, original student drafts,
  teacher targets, and review decisions.
- `data/prepared.json`: frozen train/development/reserved-test data.
- `data/review_samples.md`: readable samples for inspection.
- `conservative/`, `moderate/`, `faster/`: independent checkpoints and logs.
- `baseline.json`: unchanged 768 evaluated on exactly the same new prompts.
- `comparison.md`: scored comparison and retention metrics.
- `*/evaluations/*.md`: model-generated development transcripts.

Preparation may take substantially longer than the short training comparisons:
it involves thousands of local teacher generations. If the reserved candidate
pool cannot supply enough quality-approved groups, preparation stops BEFORE
training and reports the deficit. Filters are never silently relaxed.

## Evidence and limits

The source datasets already contain synthetic assistant answers. What changes in
this experiment is that teacher corrections are conditioned on THIS model's
actual behavior and generated history. The teacher also authors the follow-up;
this is a curated simulated conversation distribution, not real user traffic.
Teacher self-review can miss mistakes and teacher scoring can favor its own
style. Neither the targets nor evaluations are described as human-verified.
Normalized/ngram exclusion does not prove semantic decontamination.
A strict text-only filter may reject some harmless text; this is recorded rather
than replaced with coding/math examples. Retention replay reduces risk but does
not guarantee preservation, and no benchmark gains are promised.

`--check` runs teacher correction/review, prompt/mask checks, an actual disposable
student update and exact restored-next-update comparison, evaluation and teacher
judging. It never saves a production trained model. Unit tests:

```sh
cd /Users/williamzebrowski/sml-mlx
PYTHONPATH=sml-mlx-v1 .venv/bin/python -m unittest sft.teacher_chat_768_v1.test_pipeline -v
```

## Teacher replacement — September 29

The user requested Qwen3.8-27B-4bit instead of the previous 14B teacher. The
untrained 14B preparation receipts were removed (32 test, 32 dev and 151 training
groups accepted). The original source-group/replay plan is preserved, but all
teacher-generated examples and reviews will be rebuilt by 27B. None of those
14B examples are mixed into the new preparation. Student training had not started.
The same --run command starts/resumes the replacement experiment.
