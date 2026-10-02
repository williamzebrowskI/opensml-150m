Public repair pilot from unified checkpoint 384
=============================================

Parent:
experiments/unified_text_v1_sft_v1/runs/sft/step_0000384_807628c9dbdc
Pinned weights: fcc23cd8a204762f05ee43bd84c88846383dccfda40eec06cd390a59666656db

Purpose
-------
Test whether short, diverse public answers improve general instruction following,
grounding, and conversational output without amplifying repetition. This is a
separate continuation of unified 384. Retained legacy 768 is a comparison model.
No claim that this dataset automatically repairs the model or improves benchmarks.

Data
----
4096 training conversations; 128 development and 128 reserved test conversations.
Each training batch contains four conversations from each of these groups:
  1024 short public chats: Smol-SmolTalk and UltraChat-200k.
  1024 human-written ordinary answers: Dolly-15k.
  1024 context-grounded QA, extraction, and summarization: Dolly-15k.
  1024 public instruction examples: Tulu 3 Persona Instruction Following.

The short chat pool reuses pinned public TRAIN source files cached by the earlier
foundation preparation. Conversations consumed during unified updates 1..384 are
excluded; these are not replay examples from the parent. Existing dev/test groups
are excluded. All public benchmark prompts are exclusion inputs only.

Selection preserves the original wording and whole answers. No answer truncation.
Assistant target limits, including EOS: 192 chat/instruction; 160 Dolly.
Reject malformed conversations, out-of-scope code/math, repeated long spans,
boilerplate, context overflow, and benchmark/holdout prompt overlap. Tulu examples
must have supported, unambiguous named constraints and pass mechanical checks.
Supported checks include case, counts, punctuation, quotations, titles, and endings.
Conservative parsing does not prove all implicit requirements or semantic quality.
Human-written Dolly answers are not exhaustively fact checked.

The instruction subset is mainly single constraints. Its rare paragraph/title and
combined-constraint examples are limited; this pilot does not cover all IFEval tasks.
See data/prepared.json for exact source counts, families, rejections, provenance,
split fingerprints, input hashes, and limitations. Raw source revisions/hashes are
recorded in source_pins.json. Holdouts are independent conversations, not a promise
of independence from every task style in public data.

Training
--------
One pass, 256 optimizer updates, 16 conversations per update. Assistant-only
conversation-balanced cross entropy with EOS. User/system tokens are masked.
Each conversation has equal loss weight; its assistant turns share that weight.
Full parameter SFT with a fresh optimizer; parent weights load unchanged initially.
Learning rate: 16-update warmup to 2e-6, cosine decay to 2e-7.
This conservative rate is an experiment, not a proven optimal rate.

Development evaluations at 0,64,128,192,256 additional updates. Same legacy
retention prompts/decoder/limits as legacy 768. Public evaluation generates every
turn of eight chats, eight ordinary Dolly, eight grounded Dolly, and all 32 Tulu
development conversations. Every output is saved next to its reference. Formal
constraint checks are separate from content correctness. Test remains reserved.
Greedy evaluation uses 384 tokens so long/repetitive failures remain visible;
outputs are not cleaned or forced into constraints at inference time.

Checkpoint labels are parent step plus additional updates: 448,512,576,640.
All artifacts live in this experiment, not the legacy 768 or unified run folders.
All four pilot checkpoints are retained. No automatic best selection/promotion.
SIGINT saves after the current update/evaluation; rerun the same command to resume
from saved model, optimizer, and data cursor. A filesystem STOP file can pause it;
use --clear-stop --run only when intentionally restarting after that STOP file.

Run training
------------
<SOURCE_WORKSPACE>/.venv/bin/python -u \
  <SOURCE_WORKSPACE>/sml-mlx-v1/experiments/public_repair_384_v1/scripts/launch.py \
  --run

Readiness is checked with --check and performs zero optimizer updates. Running
without a mode prints the plan. --prepare rebuilds data before a run has started.
Scripts, configuration, source bytes, tokenizer, parent weights, architecture,
prepared data, and shared training code are checked for changes.

Benchmark a saved pilot after reviewing development output (example: +64)
-------------------------------------------------------------------------
<SOURCE_WORKSPACE>/.venv/bin/python -u \
  <SOURCE_WORKSPACE>/sml-mlx-v1/experiments/public_repair_384_v1/scripts/benchmark.py \
  --updates 64 --suite all --run

Allowed additional updates: 64,128,192,256. The wrapper uses the unchanged full
ARC-Easy, ARC-Challenge, PIQA, HellaSwag, and official IFEval protocols. Benchmarks
do inference only and save results under this experiment's benchmarks directory.

Sources and attribution
-----------------------
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
https://huggingface.co/datasets/databricks/databricks-dolly-15k
https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following

Observe each source's license/attribution conditions if redistributing its data.
Dolly-15k: CC BY-SA 3.0; Databricks authors and contributors.
Tulu Persona Instruction Following: ODC-BY; Allen Institute for AI.
Smol-SmolTalk: Apache-2.0; Hugging Face contributors.
UltraChat-200k: MIT; HuggingFaceH4 and UltraChat contributors.
