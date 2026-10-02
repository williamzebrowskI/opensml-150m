# Branch B: supervised fine-tuning plus sequence repetition unlikelihood

Run after the readiness checks pass:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/sft/conversation_b/launch.py \
  --run --clear-stop
```

The main Mac trains from **unchanged step 1920**, with fresh FP32 AdamW. It does not continue from A's 2040 checkpoint. The output is `runs/sft_conversation_ab_v1/B`, separate from A. Nothing is automatically promoted to the playground.

## Matched experiment

All ordinary SFT settings match completed branch A: the same 160 examples (96 reviewed OASST2, 24 assistant-authored, 40 previously consumed rehearsal), deterministic ordering, three passes, batch four, 120 optimizer updates, 12-step warmup to 8e-6 and cosine decay to 2e-6. The final step number is 2040 **within the B directory**. Data are streamed and reconstructed in bounded RAM using the unchanged A loader. The run checks that A's saved input contract still matches.

Only the extra training objective and the compute needed for it differ. This is a matched update/data-budget comparison, not an equal-wall-time comparison. B will be slower because it generates continuations before each update.

## Added loss

This is a bounded adaptation of sequence-level unlikelihood from [Welleck et al., Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319), not an exact reproduction of their experiments.

For each non-rehearsal **training** prompt in the current batch, the current policy greedily generates up to 96 tokens, with ordinary EOS stopping. Tokens are detached discrete observations. Tokens belonging to second or subsequent occurrences of a generated four-token n-gram become negative targets; the first occurrence is not marked unless it overlaps a later repeated span. Prompt/history tokens and special tokens including EOS are excluded. Reference answers never become negative labels. Rehearsal receives normal supervised CE only.

The extra objective is `-log(1 - p(repeated_token | generated_prefix))`, averaged over marked positions per continuation and then over all collected continuations. Continuations without marked repetitions contribute zero. Total loss is the same equal-example assistant-only CE used by A plus **0.1 × unlikelihood**. Gradients are combined before clipping and a single optimizer update. Probabilities at one are clamped for finite FP32 calculations; the numerical epsilon is 1e-6. The weight, rollout limit and n-gram rule are fixed experimental choices.

The log reports CE, UL and the number of negative tokens. UL can be zero, especially initially, because the parent often gives short answers. No artificial minimum answer length is imposed. The training term targets exact repeated token sequences; it can miss semantic repetition and can penalize some legitimate repeated phrasing. It is not a factual-correctness objective, RL or DPO.

## Evaluation and resume

The unchanged A evaluator saves complete development replies at updates 0, 40, 80 and 120, alongside likelihood, stopping, repetition and retention diagnostics. No development/test answers enter the training objective. Reserved test generations remain unused. Compare actual relevance, correctness and completeness as well as looping: less repetition alone is insufficient for promotion.

Ctrl-C requests saving at the next update boundary. The same command resumes the exact optimizer and data cursor. Completed runs do not automatically extend. Readiness checks (`--check`) test the loss direction and masking, reconstruct the same selections, verify cached generation and perform temporary update/save/resume checks without starting the production run.

Keep A's artifacts and original step 1920 for the comparison. Shared files must remain unchanged after readiness verification; stale inputs fail closed.
