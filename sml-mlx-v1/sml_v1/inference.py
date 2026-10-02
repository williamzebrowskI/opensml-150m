"""Native MLX inference utilities."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

def _cast_model_floats(model: nn.Module, dtype):
    float_dtypes = {mx.float16, mx.bfloat16, mx.float32}
    casted = tree_map(
        lambda x: x.astype(dtype)
        if isinstance(x, mx.array) and x.dtype in float_dtypes
        else x,
        model.parameters(),
    )
    model.update(casted)

def _sample_next_id(
    logits: mx.array,
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    if temperature <= 0.0 or top_k == 1:
        mx.eval(logits)
        return int(mx.argmax(logits, axis=-1).item())

    scaled = logits / max(temperature, 1e-6)
    if top_k > 0:
        k = min(top_k, scaled.shape[-1])
        indices = mx.argsort(scaled, axis=-1)[:, -k:]
        values = mx.take_along_axis(scaled, indices, axis=-1)
        probabilities = mx.softmax(values.astype(mx.float32), axis=-1)
        mx.eval(probabilities, indices)
        probabilities_np = np.asarray(probabilities[0])
        indices_np = np.asarray(indices[0], dtype=np.int64)
        choice = rng.choice(len(indices_np), p=probabilities_np / probabilities_np.sum())
        return int(indices_np[choice])

    probabilities = mx.softmax(scaled.astype(mx.float32), axis=-1)
    mx.eval(probabilities)
    probabilities_np = np.asarray(probabilities[0])
    return int(rng.choice(len(probabilities_np), p=probabilities_np / probabilities_np.sum()))
