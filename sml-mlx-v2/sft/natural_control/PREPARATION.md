# Preparation receipt — 2026-09-25

Status: prepared and checked; production training has not started.

## Frozen selection

20,480 training examples, with 128 development and 128 test examples. The
training mixture is 14,848 conversation, 4,096 rewriting, 512 summarization,
and 1,024 everyday-conversation examples.

Supervised targets, including EOS: 4,160,629 for OpenSML and 4,095,832 for
SmolLM2. Each independent fit sees this selection once.

- Training row hash: `bcd718df92cc10902f8faf4fa566f888f818a9f303273ec93409a2f73c224fb5`
- Development row hash: `89a2ad13e9033f2798e6d2c07ddfab99ec896f4641900511505555723c7576ca`
- Test row hash: `8afce17b24ac0bde88de83de5e45021d3984597d89d1034141239c13a10fbbde`

`selection.json` records exact source locations and identities. The launcher
rebuilds the selection from the pinned source and rejects a mismatch.

## Sampled inspection

During preparation, sampled reviews led to stronger scope, completeness,
roleplay, numeric-table, and summary-support exclusions. The final selection
was then inspected using 24 complete samples: 12 conversation and four from
each other source. This is a small convenience sample, not a statistical
quality estimate. No explicit coding or computational-math task appeared in
those 24 samples.

Remaining limitations are real:

- Some natural responses add unsolicited editing suggestions or use a weak,
  meta-conversational prompt. Creative writing is still present.
- The solar-power explanation oversimplifies battery storage. A rewriting
  example preserves an imprecise historical claim supplied in its input.
- Time-sensitive entertainment suggestions can survive the current-information
  filters. These examples do not establish that the model has current knowledge.
- Summary support is checked with lexical heuristics. This does not establish
  factual correctness, semantic entailment, or perfect scope filtering.

This remains imperfect synthetic source data. It is a controlled experiment
with reviewed held-out answers, not a fully fact-verified training corpus.
The named examples remain in the frozen selection; no claim is made that
every defect identified in inspection has been removed.

## Checks completed

Nine protocol tests passed, covering filtering regressions, selection JSON
round trips, checkpoint retention, planning, stop/resume, and one-pass exposure.

Disposable checks passed for both model implementations:

- Prompt and padding targets are masked; assistant EOS is supervised.
- Token-weighted batch loss agrees with separately scored examples.
- Changing future padding tokens leaves the masked loss unchanged.
- Microbatch gradient relative L2 difference: OpenSML 0.000629;
  SmolLM2 0.001022.
- Saved/reloaded next-update model weights match exactly. Maximum optimizer
  tensor differences are below 5e-10.

These are targeted numerical checks, not proof that every model path or dataset
example is correct. Six disposable optimizer updates were executed across the
two models; zero production updates were executed. Temporary model copies were
removed. The current code, configuration, source selection, tokenizer/model
hashes, and dependency versions are bound in `readiness.json`.

## Decision after training

Review the saved, blinded development answers before choosing a candidate.
Compare useful response quality and retained reading skills against each base.
Freeze one selection per model before generating reserved test answers.
Neither falling loss nor success on templated tasks alone makes a candidate
the preferred playground model.
