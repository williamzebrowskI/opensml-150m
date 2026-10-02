# Evidence and Reporting

**Documentation foundation | 2026-09-18**

[Overview](../README.md) · [Report](TECHNICAL_REPORT.md) · [Results](RESULTS.md)

## Claim Sources

| Claim family | Primary evidence |
| --- | --- |
| Architecture | [Model](../sml_v1/model.py), [pilot recipe](../configs/pilot.json), tests |
| Optimizer, reduction, evaluation | [Trainer](../sml_v1/pretrain.py), staged job code |
| Source revisions and filters | [Corpus](../configs/corpus.json), staged corpus |
| Tokenizer and audit | [Manifest](../tokenizer/bytebpe32k_v1/manifest.json), [report](../tokenizer/bytebpe32k_v1/report.json) |
| Inherited implementation | [Provenance](../provenance.json) |
| Rationale, not completed experiments | [Original plan](../sml-mlx-v1.md) |

An executed job's **staged recipe/code and committed metadata take precedence
over current defaults**. Source files can change after launch. Documentation is
not independent proof that a test was executed.

## Snapshot Evidence

Paths below are relative to `sml-mlx-v1/`. These local artifacts may not accompany
a future public repository. Retention can remove checkpoint bundles; preserve
reviewed scalar exports before relying on them as permanent public citations.

```text
Pilot endpoint:
runs/pilot_v1/step_0004696_2c67c3baa1ed/model.safetensors.json

Continuation recipe:
runs/full_15b_v1/jobs/20260918_085811_b4e5c849/input/recipe.json

Continuation log:
runs/full_15b_v1/jobs/20260918_085811_b4e5c849/train.log

Snapshot metadata:
runs/full_15b_v1/step_0007982_5c3feb7c8148/model.safetensors.json
```

Step 7,982 records aggregate best loss `3.315954786539078` at `850051072` tokens.
Its bundle is `step_0007982_5c3feb7c8148`; Results rounds for readability.

```text
Recorded contract identities at step 7,982:
code      fdaa5eceee48990f8e20adef266d9357348288e83f4c287f35256d5e31690d80
data      d9bfbfd69b065eafe0254fc36f96354461704f9bde0cbd6c0fd9f60383083614
recipe    63e71785eadf1964db0f3397d577084d1e426fc571307fbb73a6bd2feeccbccf
tokenizer 1194292c6d906ac19e6f01c6ad2b42825143fad426a22aa8856d903776701ecb
```

These identify recorded artifacts; this documentation update does not independently
rerun every checksum or correctness test.

## Update Rules

1. Date snapshots and identify checkpoint, run, and evaluation protocol.
2. Separate proposed, configured, executed, measured, and independently verified claims.
3. Distinguish consumed tokens, source-byte quotas, and planned budgets.
4. Label resumes, schedule changes, rewinds, and replay in learning curves.
5. Include all evaluations in curves, not just improving milestones.
6. Preserve full generations and decoding settings beside aggregate scores.
7. Link future figures to machine-readable values and plotting methods.
8. Keep V1 and alternate branches separate from V1's parameter ancestry.

## Before Publication

Freeze a reviewed evidence package containing configuration, tokenizer identity,
evaluation tables, full selected generation suites, environment versions, and
checkpoint provenance. Provide inference and evaluation instructions that do not
require our private paths or cluster configuration.

Review source attribution and code licensing explicitly. Do not inherit another
project's license by assumption. Exclude credentials, private SSH configuration,
machine identifiers, raw samples, and unnecessary optimizer states from releases.

Transparent research reports such as [PetitGPT's technical report](https://github.com/yangqi0/petitgpt/blob/53a5bb33052fa8092987eacc1047e040a7c5ce30/docs/petitgpt-v1/TECHNICAL_REPORT.md)
informed the documentation goal, not our claimed results. These pages describe
our implementation and evidence, not a reproduction of that project's model.
