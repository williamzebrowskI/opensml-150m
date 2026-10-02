# Data provenance and attribution

Human-written source: **SQuAD 2.0**, by Pranav Rajpurkar, Robin Jia, and Percy Liang; Stanford Question Answering Dataset.

- [Official project](https://rajpurkar.github.io/SQuAD-explorer/)
- [Dataset card](https://huggingface.co/datasets/rajpurkar/squad_v2)
- [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/)
- Repository: `rajpurkar/squad_v2`
- Revision: `3ffb306f725f7d2ce8394bc1873b24868140c412`
- File: `squad_v2/train-00000-of-00001.parquet`
- SHA-256: `f6da32ffb482ff463ad056477740d1bb284b96a45db3a08bee6a225ca6abf291`

The dataset card identifies CC BY-SA 4.0. This is not a noncommercial-only source. Keep attribution, the license link, and change notices with redistributed adapted examples and follow applicable share-alike terms. Underlying passages originate from Wikipedia. This note records provenance rather than guaranteeing downstream model licensing.

Changes: topic/length/overlap and manual exclusion filters; deterministic article-based subdivision of the official training split; prompt wrapper; answer-span extraction; unanswerable labels rendered as “Not stated.” No generated teacher responses. Original example IDs and article titles remain available in generated evaluation records. Selection IDs and hashes are retained in `selection.json`.

Original synthetic training examples and independently written development questions are local project material. Rehearsal consists solely of local original examples already consumed before checkpoint 448. Prior reviewed prompts in `excluded_previous_prompts.json` are exclusion records, never training examples.
