"""Integer-target cross entropy without full-vocabulary FP32 intermediates.

Metal forward and reverse-mode kernels reduce/compute in FP32; only the
incoming logits and outgoing logits gradient use the model's dtype.
"""

from functools import lru_cache
import mlx.core as mx
import mlx.nn.losses as losses


@lru_cache(maxsize=1)
def _kernels():
    forward = mx.fast.metal_kernel(
        name="sml_ce_forward", input_names=["logits", "targets"],
        output_names=["loss", "stats"],
        source=r"""
        uint row = threadgroup_position_in_grid.y;
        uint tid = thread_index_in_threadgroup;
        uint lane = tid % 32;
        uint simd = tid / 32;
        threadgroup float partial[8];
        float maximum = -INFINITY;
        for (uint col = tid; col < V; col += 256)
            maximum = metal::max(maximum, float(logits[row * V + col]));
        maximum = metal::simd_max(maximum);
        if (lane == 0) partial[simd] = maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum = metal::simd_max(lane < 8 ? partial[lane] : -INFINITY);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint col = tid; col < V; col += 256)
            total += metal::exp(float(logits[row * V + col]) - maximum);
        total = metal::simd_sum(total);
        if (lane == 0) partial[simd] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = metal::simd_sum(lane < 8 ? partial[lane] : 0.0f);
        if (tid == 0) {
            float logtotal = metal::log(total);
            stats[row * 2] = maximum;
            stats[row * 2 + 1] = logtotal;
            loss[row] = logtotal - (float(logits[row * V + targets[row]]) - maximum);
        }
        """,
    )
    backward = mx.fast.metal_kernel(
        name="sml_ce_backward", input_names=["logits", "targets", "stats", "cot"],
        output_names=["gradient"],
        source=r"""
        uint col = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        if (col < V) {
            float p = metal::exp((float(logits[row * V + col]) - stats[row * 2])
                                - stats[row * 2 + 1]);
            gradient[row * V + col] = T((p - float(col == targets[row])) * cot[row]);
        }
        """,
    )
    return forward, backward


@mx.custom_function
def _loss_and_stats(logits, targets):
    rows, vocab = logits.shape
    forward, _ = _kernels()
    return tuple(forward(
        inputs=[logits, targets], template=[("T", logits.dtype), ("V", vocab)],
        grid=(256, rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows,), (rows, 2)], output_dtypes=[mx.float32, mx.float32],
    ))


@_loss_and_stats.vjp
def _loss_vjp(primals, cotangents, outputs):
    logits, targets = primals
    rows, vocab = logits.shape
    _, backward = _kernels()
    gradient, = backward(
        inputs=[logits, targets, outputs[1], cotangents[0]],
        template=[("T", logits.dtype), ("V", vocab)],
        grid=(vocab, rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[logits.shape], output_dtypes=[logits.dtype],
    )
    return gradient, mx.zeros_like(targets)


def cross_entropy(logits, targets):
    """Unreduced CE; targets must already be valid vocabulary indices."""
    if mx.default_device() == mx.cpu:
        return losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    if logits.shape[:-1] != targets.shape or targets.dtype != mx.int32:
        raise ValueError("Metal CE requires shape-matched int32 target indices")
    loss, _ = _loss_and_stats(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss.reshape(targets.shape)
