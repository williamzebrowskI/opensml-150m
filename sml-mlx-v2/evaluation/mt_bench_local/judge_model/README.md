---
datasets:
- prometheus-eval/Feedback-Collection
- prometheus-eval/Preference-Collection
language:
- en
library_name: transformers
license: apache-2.0
metrics:
- pearsonr
- spearmanr
- kendall-tau
- accuracy
pipeline_tag: text2text-generation
tags:
- text2text-generation
- mlx
---

# mlx-community/prometheus-7b-v2.0-8bit

The Model [mlx-community/prometheus-7b-v2.0-8bit](https://huggingface.co/mlx-community/prometheus-7b-v2.0-8bit) was converted to MLX format from [prometheus-eval/prometheus-7b-v2.0](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) using mlx-lm version **0.15.2**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/prometheus-7b-v2.0-8bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)
```
