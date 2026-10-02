# Data and Tokenizer

**Implemented recipe and completed tokenizer audit | 2026-09-18**

[Overview](../README.md) · [Report](TECHNICAL_REPORT.md) · [Evidence](EVIDENCE.md)

## Corpus Contract

| Source | Training token share | Selection |
| --- | ---: | --- |
| FineWeb-Edu | 55% | SmolLM corpus, `fineweb-edu-dedup` |
| DCLM-Edu | 25% | `edu_int_score >= 3` |
| FineWiki | 10% | English |
| Cosmopedia v2 | 10% | Format matching textbook/tutorial/blog/educational |

Full repository IDs, pinned revisions, fields, limits, filters, and recorded
license labels live in [the corpus configuration](../configs/corpus.json).
Tokenizer fitting uses these shares **by UTF-8 bytes**; model training schedules
them **by tokens**. No dedicated code/math source is included, but educational
prose can contain equations, code, and incidental non-English content.

Live training follows source order without a document shuffle buffer. Tokenizer
sampling used seeded shuffling. This distinction supports consumed-cursor resume
without relying on an unrestored live shuffle buffer.

## Splits, Boundaries, and Storage

Normalized document hashes assign 97% to training and 1% each to tokenizer
development, validation, and a reserved final split. Model input retains original
text; normalization is for identity/split decisions.

A cross-source window of 10,000 recent accepted hashes suppresses exact
duplicates. It is not exhaustive corpus deduplication. Each document ends in one
explicit EOS; causal windows may cross documents without resetting attention.

Mac-1 streams, tokenizes, and prefetches into RAM. Checkpoints retain consumed
cursors, pending tokens, and recent hashes. Unconsumed queued batches are
recreated on resume. Repeated remote reads while restoring a cursor do not
necessarily mean examples are trained on again.

Training does not save corpus/token shards. Tokenizer samples, logs, checkpoints,
staged inputs, and small HF metadata caches are separate disk uses. No result
from the reserved final split is reported here.

## Frozen ByteBPE32K v1

| Audit item | Recorded result |
| --- | --- |
| Vocabulary | 32,000 IDs including structural tokens |
| Fitting sample | 500,048,186 UTF-8 bytes; 113,898 documents |
| Held-out audit | 4,010,605 bytes; 970 documents |
| Round-trip failures | 0 in the held-out audit |
| Regression fixtures | 9 passed |
| Text handling | Full byte alphabet; no normalization or implicit BOS/EOS |
| Structural IDs | Padding 0; document-end 1; turn-start 2; turn-end 3 |

Literal control-token spellings are ordinary content, not inserted control IDs.
Reserved turn tokens do not make this base model an instruction-tuned assistant.

| Held-out source | Bytes/token |
| --- | ---: |
| Web | 4.6255 |
| DCLM | 4.4532 |
| Wiki | 4.0593 |
| Cosmopedia | 5.0610 |

These checks establish audited encoding properties, not downstream quality.
Sources: [audit](../tokenizer/bytebpe32k_v1/report.json) and
[manifest](../tokenizer/bytebpe32k_v1/manifest.json).

```text
Artifact fingerprint:
1194292c6d906ac19e6f01c6ad2b42825143fad426a22aa8856d903776701ecb
tokenizer.json SHA-256:
e095f0413a11fa91fbb58f37d00e4c6bcda59beb89b28f6f680db1436a7e8891
```

## Attribution

Preserve source revisions and attribution in published recipes. Dataset license
labels are not blanket rights to every underlying document. Review source cards
before redistribution: [SmolLM corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus),
[DCLM-Edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu),
[FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki).
Do not publish raw tokenizer samples or training text by default.
