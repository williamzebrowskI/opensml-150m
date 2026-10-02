"""Explicit BF16 compute with FP32 AdamW state and master parameters."""

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map


class MasterAdamW(optim.AdamW):
    """Keep small updates in FP32 without promoting the compute model."""
    def apply_gradients(self, gradients, parameters):
        # Base MLX lazily initializes from gradients (sufficient for zero moments,
        # but incorrect for master weights). Use the matching actual parameters.
        if not self._initialized:
            self.init(tree_map(lambda g, p: p, gradients, parameters))
        return super().apply_gradients(gradients, parameters)

    def init_single(self, parameter, state):
        master = parameter.astype(mx.float32)
        super().init_single(master, state)
        state["master"] = master

    def apply_single(self, gradient, parameter, state):
        gradient = gradient.astype(mx.float32)
        master = state["master"]
        # Norm scales should not receive decoupled weight decay.
        if parameter.ndim < 2:
            updated = optim.Adam.apply_single(self, gradient, master, state)
        else:
            updated = super().apply_single(gradient, master, state)
        state["master"] = updated
        return updated.astype(parameter.dtype)
