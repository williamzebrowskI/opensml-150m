#!/usr/bin/env python3
"""MLX-only transformer model for small causal language modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int = 1024
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    n_layers: int = 12
    mlp_ratio: float = 4.0
    mlp_multiple_of: int = 256
    rope_base: float = 10000.0
    bias: bool = False
    qk_norm: bool = True
    attention_impl: str = "fast"
    loss_dtype: str = "float32"
    ce_impl: str = "reference"
    ffn_impl: str = "reference"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("n_heads must divide d_model")
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_kv_heads must be positive and divide n_heads")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if self.mlp_ratio <= 1.0:
            raise ValueError("mlp_ratio must be > 1.0")
        if self.mlp_multiple_of <= 0:
            raise ValueError("mlp_multiple_of must be > 0")
        if self.attention_impl not in {"fast", "vanilla"}:
            raise ValueError("attention_impl must be one of: fast, vanilla")
        if self.loss_dtype not in {"float32", "model"}:
            raise ValueError("loss_dtype must be one of: float32, model")
        if self.ce_impl not in {"reference", "metal"}:
            raise ValueError("ce_impl must be reference or metal")
        if self.ce_impl == "metal" and self.loss_dtype != "float32":
            raise ValueError("Metal CE uses FP32 reductions")
        if self.ffn_impl not in {"reference", "packed-metal"}:
            raise ValueError("ffn_impl must be reference or packed-metal")


def _tree_leaves(tree):
    if isinstance(tree, dict):
        for v in tree.values():
            yield from _tree_leaves(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _tree_leaves(v)
    else:
        yield tree


def count_parameters(model: nn.Module) -> int:
    total = 0
    for leaf in _tree_leaves(model.parameters()):
        if isinstance(leaf, mx.array):
            n = 1
            for d in leaf.shape:
                n *= int(d)
            total += n
    return total


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_dim = cfg.n_heads * self.head_dim
        self.kv_dim = cfg.n_kv_heads * self.head_dim
        self.max_seq_len = cfg.max_seq_len

        self.qkv_proj = nn.Linear(cfg.d_model, self.q_dim + 2 * self.kv_dim, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.q_norm = nn.RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.rope = nn.RoPE(self.head_dim, base=cfg.rope_base)
        self.attention_impl = cfg.attention_impl

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)

        def split_heads(t: mx.array, n_heads: int) -> mx.array:
            t = t.reshape(bsz, seqlen, n_heads, self.head_dim)
            return t.transpose(0, 2, 1, 3)

        q = split_heads(q, self.n_heads)
        k = split_heads(k, self.n_kv_heads)
        v = split_heads(v, self.n_kv_heads)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cache_offset = 0
        if cache is not None and cache[0] is not None:
            cache_offset = int(cache[0].shape[2])
        q = self.rope(q, offset=cache_offset)
        k = self.rope(k, offset=cache_offset)

        if cache is not None:
            k_cache, v_cache = cache
            if k_cache is not None and v_cache is not None:
                k = mx.concatenate([k_cache, k], axis=2)
                v = mx.concatenate([v_cache, v], axis=2)

        q_len = q.shape[2]
        k_len = k.shape[2]
        if q_len > self.max_seq_len or k_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {q_len}/{k_len} exceeds max_seq_len={self.max_seq_len}"
            )
        scale = 1.0 / math.sqrt(self.head_dim)
        # Cache compact KV heads, not the expanded heads used by vanilla GQA.
        next_cache = (k, v)
        if self.attention_impl == "fast":
            attn = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask="causal",
            )
        else:
            repeats = self.n_heads // self.n_kv_heads
            if repeats > 1:
                k = mx.repeat(k, repeats, axis=1)
                v = mx.repeat(v, repeats, axis=1)
            full_mask = nn.MultiHeadAttention.create_additive_causal_mask(k_len)
            mask = full_mask[k_len - q_len : k_len, :k_len]
            mask = mx.expand_dims(mx.expand_dims(mask, axis=0), axis=0).astype(q.dtype)
            scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
            scores = scores + mask
            probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
            attn = mx.matmul(probs, v)
        out = attn.transpose(0, 2, 1, 3).reshape(bsz, q_len, d_model)
        out = self.proj(out)
        return out, next_cache


class FeedForward(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        # SwiGLU uses three projections, so 2/3 keeps parameter count comparable
        # to a conventional 4x two-projection MLP.
        hidden_dim = int((2.0 / 3.0) * cfg.d_model * cfg.mlp_ratio)
        hidden_dim = cfg.mlp_multiple_of * math.ceil(hidden_dim / cfg.mlp_multiple_of)
        self.hidden_dim = hidden_dim
        self.ffn_impl = cfg.ffn_impl
        self.gate_up = nn.Linear(cfg.d_model, 2 * hidden_dim, bias=cfg.bias)
        self.down = nn.Linear(hidden_dim, cfg.d_model, bias=cfg.bias)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up(x)
        if self.ffn_impl == "packed-metal":
            try:
                from .fast_swiglu import packed_swiglu
            except ImportError:
                from fast_swiglu import packed_swiglu
            return self.down(packed_swiglu(gate_up))
        gate, up = mx.split(gate_up, [self.hidden_dim], axis=-1)
        return self.down(nn.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        attn_out, new_cache = self.attn(self.norm1(x), cache=cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        # MLX >= 0.32.2 supplies the memory-reduced Metal RMSNorm backward
        # for every nn.RMSNorm here, including block and Q/K norms. Keep the
        # native primitive so compilation/autodiff select that implementation.
        self.norm = nn.RMSNorm(cfg.d_model)

    def logits(
        self,
        input_ids: mx.array,
        caches: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        x = self.embed(input_ids)
        if caches is None:
            caches = [None] * len(self.blocks)

        new_caches = []
        for block, block_cache in zip(self.blocks, caches):
            x, next_cache = block(x, cache=block_cache)
            new_caches.append(next_cache)

        x = self.norm(x)
        weight = self.embed.weight
        logits = mx.matmul(x, weight.transpose(1, 0))
        return logits, new_caches

    def __call__(
        self,
        input_ids: mx.array,
        targets: Optional[mx.array] = None,
        ignore_index: int = -100,
    ) -> dict:
        logits, _ = self.logits(input_ids, caches=None)
        out = {"logits": logits}
        if targets is None:
            return out

        loss_logits = logits.astype(mx.float32) if self.cfg.loss_dtype == "float32" else logits
        mask = targets != ignore_index
        safe_targets = mx.where(mask, targets, mx.zeros_like(targets))
        if self.cfg.ce_impl == "metal":
            try:
                from .fast_loss import cross_entropy
            except ImportError:
                from fast_loss import cross_entropy
            per_token = cross_entropy(logits, safe_targets)
        else:
            per_token = losses.cross_entropy(loss_logits, safe_targets, reduction="none")
        mask = mask.astype(per_token.dtype)
        denom = mx.maximum(mask.sum(), mx.array(1.0, dtype=per_token.dtype))
        loss = (per_token * mask).sum() / denom
        out["loss"] = loss
        out["token_loss"] = per_token
        return out

    def step(
        self,
        token_ids: mx.array,
        caches: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        # token_ids shape: [B, 1]
        logits, new_caches = self.logits(token_ids, caches=caches)
        return logits[:, -1, :], new_caches
