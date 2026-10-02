#!/usr/bin/env python3
"""MLX pretraining runner for a small transformer LM.

Use train/launch_pretrain_wide.sh for the active single-host model. Resume with
the current run's latest.json pointer. Transactional bundle checkpoints and the
compiled training step currently require a single host.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

try:
    from .model import TransformerConfig, TransformerLM, count_parameters
    from .precision import MasterAdamW
    from .tokenization import load_tokenizer
    from .checkpoint_bundle import resolve_bundle, save_bundle
    from .performance import PerformanceTracker
    from .data import (
        HFStreamingBatcher,
        StreamingDatasetAdapter,
        data_state_path,
        load_data_state,
        parse_source_configs,
        save_data_state,
    )
except ImportError:
    from model import TransformerConfig, TransformerLM, count_parameters
    from precision import MasterAdamW
    from tokenization import load_tokenizer
    from checkpoint_bundle import resolve_bundle, save_bundle
    from performance import PerformanceTracker
    from data import (
        HFStreamingBatcher,
        StreamingDatasetAdapter,
        data_state_path,
        load_data_state,
        parse_source_configs,
        save_data_state,
    )


def _tree_leaves(tree: Any):
    if isinstance(tree, dict):
        for v in tree.values():
            yield from _tree_leaves(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _tree_leaves(v)
    else:
        yield tree


def _tree_add(a: Any, b: Any):
    if isinstance(a, dict):
        return {k: _tree_add(a[k], b[k]) for k in a}
    if isinstance(a, list):
        return [_tree_add(x, y) for x, y in zip(a, b)]
    if isinstance(a, tuple):
        return tuple(_tree_add(x, y) for x, y in zip(a, b))
    if isinstance(a, mx.array):
        return a + b
    return a


def _tree_scale(tree: Any, scale: float):
    if isinstance(tree, dict):
        return {k: _tree_scale(v, scale) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_scale(v, scale) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_scale(v, scale) for v in tree)
    if isinstance(tree, mx.array):
        return tree * scale
    return tree


def _flatten_for_safetensors(tree: Any, prefix: str = "", out: Optional[dict] = None):
    if out is None:
        out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten_for_safetensors(v, key, out)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            key = f"{prefix}.{i}" if prefix else str(i)
            _flatten_for_safetensors(v, key, out)
    elif isinstance(tree, tuple):
        for i, v in enumerate(tree):
            key = f"{prefix}.{i}" if prefix else str(i)
            _flatten_for_safetensors(v, key, out)
    elif isinstance(tree, mx.array):
        key = prefix if prefix else "param"
        out[key] = tree
    return out


def _all_sum(x: mx.array, stream_mode: str = "cpu") -> mx.array:
    if stream_mode == "cpu":
        try:
            return mx.distributed.all_sum(x, stream=mx.cpu)
        except TypeError:
            return mx.distributed.all_sum(x)
    try:
        return mx.distributed.all_sum(x)
    except TypeError:
        return mx.distributed.all_sum(x, stream=mx.cpu)


def _allreduce_tree(tree: Any, world: int, stream_mode: str = "cpu"):
    if world == 1:
        return tree

    def reduce_leaf(v):
        if isinstance(v, mx.array):
            return _all_sum(v, stream_mode=stream_mode) / world
        return v

    return tree_map(reduce_leaf, tree)


def _grad_norm(tree: Any) -> float:
    sq = mx.array(0.0, dtype=mx.float32)
    for leaf in _tree_leaves(tree):
        if isinstance(leaf, mx.array):
            x = leaf.astype(mx.float32)
            sq = sq + (x * x).sum()
    norm = mx.sqrt(sq + 1e-12)
    mx.eval(norm)
    return float(norm.item())


def _clip_grads(tree: Any, max_norm: float):
    if max_norm <= 0:
        return tree, 0.0
    norm = _grad_norm(tree)
    if norm <= max_norm:
        return tree, norm
    scale = max_norm / (norm + 1e-6)
    return _tree_scale(tree, scale), norm


def _grad_norm_array(tree: Any) -> mx.array:
    sq = mx.array(0.0, dtype=mx.float32)
    for leaf in _tree_leaves(tree):
        if isinstance(leaf, mx.array):
            x = leaf.astype(mx.float32)
            sq = sq + (x * x).sum()
    return mx.sqrt(sq + 1e-12)


def _clip_grads_for_compile(tree: Any, max_norm: float):
    if max_norm <= 0:
        return tree, mx.array(0.0, dtype=mx.float32)
    norm = _grad_norm_array(tree)
    limit = mx.array(max_norm, dtype=mx.float32)
    scale = mx.minimum(mx.array(1.0, dtype=mx.float32), limit / (norm + 1e-6))
    return _tree_scale(tree, scale), norm


def _build_local_compiled_train_step(
    *,
    model: TransformerLM,
    optimizer: optim.AdamW,
    grad_accum: int,
    grad_clip: float,
    ignore_index: int,
):
    def loss_fn(x, y):
        return model(x, targets=y, ignore_index=ignore_index)["loss"]

    step_and_grad = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=[model.state, optimizer.state], outputs=[model.state, optimizer.state])
    def compiled_train_step(xs, ys, lr):
        total_loss = mx.array(0.0, dtype=mx.float32)
        grads_acc = None
        for micro in range(grad_accum):
            loss, grads = step_and_grad(xs[micro], ys[micro])
            if isinstance(optimizer, MasterAdamW):
                grads = tree_map(lambda g: g.astype(mx.float32), grads)
            total_loss = total_loss + loss.astype(mx.float32)
            grads_acc = grads if grads_acc is None else _tree_add(grads_acc, grads)
        grads_acc = _tree_scale(grads_acc, 1.0 / float(grad_accum))
        grads_acc, grad_norm = _clip_grads_for_compile(grads_acc, grad_clip)
        optimizer.learning_rate = lr
        optimizer.update(model, grads_acc)
        return total_loss / float(grad_accum), grad_norm

    return compiled_train_step


def _cast_model_floats(model: nn.Module, dtype):
    float_dtypes = {mx.float16, mx.bfloat16, mx.float32}
    casted = tree_map(
        lambda x: x.astype(dtype)
        if isinstance(x, mx.array) and x.dtype in float_dtypes
        else x,
        model.parameters(),
    )
    model.update(casted)


def _atomic_save_safetensors(path: str, tensors: dict[str, mx.array]):
    tmp_path = path + ".tmp.safetensors"
    mx.save_safetensors(tmp_path, tensors)
    os.replace(tmp_path, path)


def _assign_flat_state(template: Any, flat: dict[str, mx.array], prefix: str = ""):
    if isinstance(template, dict):
        return {
            k: _assign_flat_state(v, flat, f"{prefix}.{k}" if prefix else k)
            for k, v in template.items()
        }
    if isinstance(template, list):
        return [
            _assign_flat_state(v, flat, f"{prefix}.{i}" if prefix else str(i))
            for i, v in enumerate(template)
        ]
    if isinstance(template, tuple):
        return tuple(
            _assign_flat_state(v, flat, f"{prefix}.{i}" if prefix else str(i))
            for i, v in enumerate(template)
        )
    if isinstance(template, mx.array):
        key = prefix if prefix else "param"
        if key not in flat:
            raise KeyError(f"Missing tensor in checkpoint: {key}")
        value = flat[key].astype(template.dtype)
        if value.shape != template.shape:
            raise ValueError(
                f"Checkpoint shape mismatch for {key}: file={value.shape}, template={template.shape}"
            )
        return value
    return template


def _save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, metadata: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model_tensors = _flatten_for_safetensors(model.parameters())
    optimizer_tensors = _flatten_for_safetensors(optimizer.state)
    _atomic_save_safetensors(path, model_tensors)
    _atomic_save_safetensors(path + ".optimizer.safetensors", optimizer_tensors)
    meta_path = path + ".json"
    meta_tmp = meta_path + ".tmp"
    with open(meta_tmp, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    os.replace(meta_tmp, meta_path)


def _load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer) -> tuple[bool, bool]:
    if not os.path.exists(path):
        return False, False

    flat = mx.load(path)
    model.update(_assign_flat_state(model.parameters(), flat))
    mx.eval(model.parameters())

    optimizer_path = path + ".optimizer.safetensors"
    optimizer_loaded = False
    if os.path.exists(optimizer_path):
        optimizer_flat = mx.load(optimizer_path)
        optimizer.state = _assign_flat_state(optimizer.state, optimizer_flat)
        mx.eval(optimizer.state)
        optimizer_loaded = True
    return True, optimizer_loaded


def _infer_resume_step(resume_path: str) -> int:
    meta_path = resume_path + ".json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return int(meta.get("step", 0))
        except Exception:
            pass

    stem = Path(resume_path).stem
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return 0


def _broadcast_tree_from_source(
    tree: Any,
    rank: int,
    world: int,
    source_rank: int = 0,
    stream_mode: str = "cpu",
):
    if world == 1:
        return tree

    def bcast(v):
        if not isinstance(v, mx.array):
            return v
        src = v if rank == source_rank else mx.zeros_like(v)
        return _all_sum(src, stream_mode=stream_mode)

    return tree_map(bcast, tree)


def _broadcast_model(
    model: nn.Module,
    rank: int,
    world: int,
    source_rank: int = 0,
    stream_mode: str = "cpu",
):
    params = _broadcast_tree_from_source(
        model.parameters(), rank, world, source_rank=source_rank, stream_mode=stream_mode
    )
    model.update(params)
    mx.eval(model.parameters())


class TokenDataset:
    def __init__(self, path: str, token_dtype: str):
        self.path = path
        if path.endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
        else:
            arr = np.memmap(path, dtype=np.dtype(token_dtype), mode="r")
        self.tokens = arr.reshape(-1)
        self.n_tokens = int(self.tokens.shape[0])
        if self.n_tokens < 2:
            raise ValueError(f"Dataset at {path} has too few tokens: {self.n_tokens}")

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        seed: int,
        step: int,
        rank: int,
        stream: int = 0,
    ):
        max_start = self.n_tokens - seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset at {self.path} has {self.n_tokens} tokens, need > seq_len+1 ({seq_len + 1})"
            )

        step_seed = seed + (step * 1_000_003) + (rank * 100_003) + (stream * 9_973)
        rng = np.random.default_rng(step_seed)
        starts = rng.integers(0, max_start, size=batch_size)

        x = np.empty((batch_size, seq_len), dtype=np.int32)
        y = np.empty((batch_size, seq_len), dtype=np.int32)
        for i, s in enumerate(starts.tolist()):
            chunk = self.tokens[s : s + seq_len + 1]
            x[i] = np.asarray(chunk[:-1], dtype=np.int32)
            y[i] = np.asarray(chunk[1:], dtype=np.int32)

        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)

def _resolve_dtype(name: str):
    table = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def _build_lr_schedule(
    base_lr: float,
    min_lr_ratio: float,
    warmup_steps: int,
    max_steps: int,
):
    min_lr = base_lr * min_lr_ratio

    def lr_for_step(step: int) -> float:
        if step < warmup_steps:
            return base_lr * float(step + 1) / float(max(1, warmup_steps))
        if step >= max_steps:
            return min_lr
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    return lr_for_step


@dataclass
class PlateauLrController:
    patience: int
    factor: float
    cooldown_evals: int
    min_lr: float
    threshold: float
    lr_scale: float = 1.0
    bad_evals: int = 0
    cooldown_remaining: int = 0
    reductions: int = 0
    best_metric: float = float("inf")
    warmup_steps: int = 0
    start_step: int = 0

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def effective_lr(
        self,
        scheduled_lr: float,
        *,
        manual_multiplier: float = 1.0,
        manual_lr: Optional[float] = None,
        step: Optional[int] = None,
    ) -> float:
        if manual_lr is not None:
            return manual_lr
        effective = scheduled_lr * self.lr_scale * manual_multiplier
        # The post-warmup floor must not turn tiny warmup updates into full-size ones.
        if step is not None and step < self.warmup_steps:
            return effective
        return max(self.min_lr, effective)

    def observe(
        self,
        metric: float,
        scheduled_lr: float,
        *,
        allow_reduction: bool = True,
        completed_step: Optional[int] = None,
    ) -> Optional[tuple[float, float]]:
        improved = metric < self.best_metric - self.threshold
        if improved:
            self.best_metric = metric
            self.bad_evals = 0

        protected = completed_step is not None and completed_step < max(
            self.warmup_steps, self.start_step
        )
        if not self.enabled or not allow_reduction or protected:
            # Manual control and startup must not build up a delayed automatic cut.
            self.bad_evals = 0
            return None
        if improved:
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            return None
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return None

        self.bad_evals += 1
        if self.bad_evals < self.patience:
            return None

        old_scale = self.lr_scale
        current_lr = self.effective_lr(scheduled_lr)
        if current_lr <= self.min_lr * (1.0 + 1e-9):
            self.bad_evals = 0
            return None

        new_scale = old_scale * self.factor
        if scheduled_lr * new_scale < self.min_lr:
            new_scale = self.min_lr / max(scheduled_lr, 1e-30)
        self.lr_scale = new_scale
        self.bad_evals = 0
        self.cooldown_remaining = self.cooldown_evals
        self.reductions += 1
        return old_scale, new_scale

    def state_dict(self) -> dict[str, Any]:
        return {
            "lr_scale": self.lr_scale,
            "bad_evals": self.bad_evals,
            "cooldown_remaining": self.cooldown_remaining,
            "reductions": self.reductions,
            "best_metric": self.best_metric,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lr_scale = float(state.get("lr_scale", self.lr_scale))
        self.bad_evals = int(state.get("bad_evals", self.bad_evals))
        self.cooldown_remaining = int(
            state.get("cooldown_remaining", self.cooldown_remaining)
        )
        self.reductions = int(state.get("reductions", self.reductions))
        self.best_metric = float(state.get("best_metric", self.best_metric))


def _read_lr_control_file(path: str) -> Optional[dict[str, float]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("LR control file must contain a JSON object")
    if "mode" in raw:
        if raw == {"mode": "auto"}:
            return None
        raise ValueError("Automatic LR control must be exactly {'mode': 'auto'}")

    has_multiplier = "lr_multiplier" in raw
    has_lr = "lr" in raw
    if has_multiplier == has_lr:
        raise ValueError("LR control file must define exactly one of 'lr_multiplier' or 'lr'")
    if has_multiplier:
        value = float(raw["lr_multiplier"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("lr_multiplier must be a finite value greater than zero")
        return {"lr_multiplier": value}

    value = float(raw["lr"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("lr must be a finite value greater than zero")
    return {"lr": value}


def _build_fixed_eval_batches(
    dataset: Any,
    eval_steps: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    rank: int,
):
    batches = []
    for i in range(eval_steps):
        x, y = dataset.sample_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            seed=seed,
            step=i,
            rank=rank,
            stream=99,
        )
        mx.eval(x, y)
        batches.append((x, y))
    return batches


def _evaluate(
    model: TransformerLM,
    batches: list[tuple[mx.array, mx.array]],
    world: int,
    ignore_index: int,
    collective_stream: str,
) -> float:
    losses = []
    for x, y in batches:
        loss = model(x, targets=y, ignore_index=ignore_index)["loss"]
        if world > 1:
            loss = _all_sum(loss, stream_mode=collective_stream) / world
        mx.eval(loss)
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


def _evaluate_by_source(model, batches, row_sources, labels, ignore_index):
    totals = np.zeros(len(labels), dtype=np.float64)
    counts = np.zeros(len(labels), dtype=np.int64)
    if len(batches) != len(row_sources):
        raise ValueError("Evaluation source labels must match the fixed batches")
    for (x, y), sources in zip(batches, row_sources):
        output = model(x, targets=y, ignore_index=ignore_index)
        mask = y != ignore_index
        row_loss = (output["token_loss"] * mask).sum(axis=-1)
        row_count = mask.sum(axis=-1)
        mx.eval(row_loss, row_count)
        np.add.at(totals, sources, np.asarray(row_loss))
        np.add.at(counts, sources, np.asarray(row_count))
    breakdown = {
        label: {"loss": float(totals[i] / counts[i]), "tokens": int(counts[i])}
        for i, label in enumerate(labels) if counts[i] > 0
    }
    return float(totals.sum() / max(1, counts.sum())), breakdown


def _parse_sample_prompts(value: Any) -> list[str]:
    if not value:
        return []
    payload = value
    if isinstance(value, str):
        if value.lstrip().startswith("["):
            payload = json.loads(value)
        elif Path(value).exists():
            with open(value, "r") as f:
                payload = json.load(f)
        else:
            payload = json.loads(value)
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("sample_prompts must be a JSON list of strings or a JSON file path")
    return [item for item in payload if item.strip()]


def _select_sample_prompts(
    prompts: list[str],
    count: int,
    event_index: int,
    seed: int,
) -> list[tuple[int, str]]:
    """Select a reproducible random subset independently for each event."""
    if not prompts:
        return []
    if count <= 0 or count >= len(prompts):
        return list(enumerate(prompts))
    rng = np.random.default_rng(seed + event_index)
    indices = rng.choice(len(prompts), size=count, replace=False)
    return [(int(index), prompts[int(index)]) for index in np.atleast_1d(indices)]


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


def _generate_completion(
    model: TransformerLM,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    add_bos: bool = True,
) -> str:
    prompt_ids: list[int] = []
    if add_bos and tokenizer.bos_id() >= 0:
        prompt_ids.append(int(tokenizer.bos_id()))
    prompt_ids.extend(tokenizer.encode(prompt, out_type=int))
    if not prompt_ids:
        raise ValueError("Cannot sample from an empty prompt")
    if len(prompt_ids) >= max_seq_len:
        prompt_ids = prompt_ids[-(max_seq_len - 1) :]

    logits, caches = model.logits(mx.array([prompt_ids], dtype=mx.int32), caches=None)
    next_logits = logits[:, -1, :]
    generated: list[int] = []
    rng = np.random.default_rng(seed)
    eos_id = int(tokenizer.eos_id())

    for _ in range(max_new_tokens):
        next_id = _sample_next_id(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        if eos_id >= 0 and next_id == eos_id:
            break
        generated.append(next_id)
        if len(prompt_ids) + len(generated) >= max_seq_len:
            break
        token = mx.array([[next_id]], dtype=mx.int32)
        next_logits, caches = model.step(token, caches=caches)

    return tokenizer.decode(generated)


def _load_config_defaults(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config JSON must be an object (key/value pairs).")
    normalized = {}
    for k, v in raw.items():
        normalized[k.replace("-", "_")] = v
    return normalized


def _infer_spm_vocab_size(spm_model: str) -> int:
    proc = spm.SentencePieceProcessor(model_file=spm_model)
    vocab_size = int(proc.vocab_size())
    if vocab_size <= 0:
        raise ValueError(f"Invalid SentencePiece vocab size from {spm_model}: {vocab_size}")
    return vocab_size


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, remaining = pre.parse_known_args(argv)

    parser = argparse.ArgumentParser(
        parents=[pre],
        description="Small distributed MLX pretraining runner (36 GB worker-friendly defaults)"
    )
    parser.add_argument("--data-mode", type=str, default="tokens", choices=["tokens", "hf_stream"])
    parser.add_argument("--train-tokens", type=str, default="", help="Path to train tokens (.npy or flat binary)")
    parser.add_argument("--val-tokens", type=str, default="", help="Optional validation tokens (.npy or flat binary)")
    parser.add_argument("--token-dtype", type=str, default="uint16", choices=["uint16", "uint32", "int32"])
    parser.add_argument(
        "--spm-model",
        type=str,
        default="/Users/williamzebrowski/sml-mlx/sml-mlx-v1/tokenizer/fineweb_spm/spm.model",
        help="SentencePiece model used for HF text streaming mode.",
    )
    parser.add_argument(
        "--train-sources",
        type=str,
        default="",
        help="HF source list (JSON string or path) for data-mode=hf_stream.",
    )
    parser.add_argument(
        "--val-sources",
        type=str,
        default="",
        help="Optional HF source list (JSON string or path) for eval in hf_stream mode.",
    )
    parser.set_defaults(add_bos=True, add_eos=True)
    parser.add_argument("--add-bos", action="store_true", dest="add_bos")
    parser.add_argument("--no-add-bos", action="store_false", dest="add_bos")
    parser.add_argument("--add-eos", action="store_true", dest="add_eos")
    parser.add_argument("--no-add-eos", action="store_false", dest="add_eos")
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--tokenizer-path", default="", help="Pinned byte-BPE tokenizer.json; overrides SentencePiece")
    parser.add_argument("--shuffle-schedule", action="store_true", help="Interleave the weighted source schedule")
    parser.add_argument("--precision", choices=["legacy", "mixed"], default="legacy")
    parser.add_argument("--ce-impl", choices=["reference", "metal"], default="reference")
    parser.add_argument("--ffn-impl", choices=["reference", "packed-metal"], default="reference")

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-kv-heads", type=int, default=0, help="KV heads for GQA; 0 uses n_heads")
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--mlp-multiple-of", type=int, default=256)
    parser.set_defaults(qk_norm=True)
    parser.add_argument("--qk-norm", action="store_true", dest="qk_norm")
    parser.add_argument("--no-qk-norm", action="store_false", dest="qk_norm")
    parser.add_argument("--attention-impl", type=str, default="fast", choices=["fast", "vanilla"])
    parser.add_argument("--loss-dtype", type=str, default="float32", choices=["float32", "model"])
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--stop-after-steps", type=int, default=0, help="Short smoke run; does not change the LR schedule")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-rank micro-batch")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument(
        "--plateau-lr-patience",
        type=int,
        default=0,
        help="Reduce LR after this many non-improving evaluations; 0 disables.",
    )
    parser.add_argument("--plateau-lr-factor", type=float, default=0.5)
    parser.add_argument("--plateau-lr-cooldown", type=int, default=2)
    parser.add_argument(
        "--plateau-lr-start-step", type=int, default=0,
        help="Do not count plateau evaluations before this absolute step or the end of warmup.",
    )
    parser.add_argument(
        "--plateau-lr-min",
        type=float,
        default=0.0,
        help="Absolute floor for the effective LR after plateau reductions.",
    )
    parser.add_argument(
        "--plateau-lr-threshold",
        type=float,
        default=0.0,
        help="Minimum validation-loss decrease counted as an improvement.",
    )
    parser.add_argument(
        "--reset-lr-controller",
        action="store_true",
        help="Ignore LR-controller state stored in a resume checkpoint.",
    )
    parser.add_argument(
        "--lr-control-file",
        type=str,
        default="",
        help="Optional live JSON file containing either lr_multiplier or lr.",
    )
    parser.add_argument(
        "--lr-control-every",
        type=int,
        default=10,
        help="Poll the live LR control file every N steps.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--optimizer-beta1", type=float, default=0.9)
    parser.add_argument("--optimizer-beta2", type=float, default=0.95)
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.set_defaults(optimizer_rank0_only=False)
    parser.add_argument(
        "--optimizer-rank0-only",
        action="store_true",
        dest="optimizer_rank0_only",
        help="Only rank 0 applies optimizer updates; other ranks receive weights via broadcast each step.",
    )
    parser.add_argument(
        "--no-optimizer-rank0-only",
        action="store_false",
        dest="optimizer_rank0_only",
    )

    parser.add_argument("--backend", type=str, default=os.getenv("MLX_BACKEND", "jaccl"))
    parser.add_argument(
        "--collective-stream",
        type=str,
        default="cpu",
        choices=["cpu", "default"],
        help="Stream for distributed collectives. cpu is safer on some macOS/Metal setups.",
    )
    parser.add_argument("--expected-world", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--profile-every", type=int, default=0, help="Write wall-clock component timings every N updates; 0 disables")
    parser.add_argument(
        "--mix-log-every",
        type=int,
        default=0,
        help="Print cumulative per-source token percentages every N steps; 0 disables.",
    )
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--eval-by-source", action="store_true", help="Log fixed held-out loss separately for each data source")
    parser.add_argument(
        "--sample-prompts",
        default="",
        help="JSON list of fixed completion prompts, or a path to a JSON list.",
    )
    parser.add_argument("--sample-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=0,
        help="Generate completions every N steps; 0 follows eval_every for compatibility.",
    )
    parser.add_argument(
        "--sample-prompts-per-event",
        type=int,
        default=0,
        help="Number of prompts generated per sampling event; 0 generates every prompt.",
    )
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--sample-top-k", type=int, default=40)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument(
        "--comparison-prompts", default="",
        help="JSON completion prompts (or file) for fixed greedy comparisons, separate from random samples.",
    )
    parser.add_argument("--comparison-every", type=int, default=0, help="Fixed greedy comparison interval; 0 disables")
    parser.add_argument("--comparison-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=4,
        help="CPU batches prepared ahead by the HF streaming worker (0 disables).",
    )
    parser.add_argument(
        "--trace-first-step",
        action="store_true",
        help="Print detailed stage markers for the first training step on each rank.",
    )
    parser.set_defaults(compile_train_step=True)
    parser.add_argument("--compile-train-step", action="store_true", dest="compile_train_step")
    parser.add_argument("--no-compile-train-step", action="store_false", dest="compile_train_step")

    parser.add_argument("--save-dir", type=str, default="/Users/williamzebrowski/sml-mlx/sml-mlx-v1/train/checkpoints")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-format", choices=["legacy", "bundle"], default="legacy")
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--checkpoint-reserve-gib", type=float, default=5.0)
    parser.add_argument(
        "--checkpoint-rank",
        type=int,
        default=0,
        help="Only this rank writes model checkpoints. Rank 0 keeps checkpoints on the home host.",
    )
    parser.set_defaults(save_stream_state=True)
    parser.add_argument(
        "--save-stream-state",
        action="store_true",
        dest="save_stream_state",
        help="Write per-rank HF stream cursor state next to checkpoints.",
    )
    parser.add_argument(
        "--no-save-stream-state",
        action="store_false",
        dest="save_stream_state",
        help="Disable per-rank HF stream cursor checkpoints on worker hosts.",
    )
    parser.set_defaults(resume_stream_state=True)
    parser.add_argument(
        "--resume-stream-state",
        action="store_true",
        dest="resume_stream_state",
        help="Restore the HF stream cursor stored beside a resume checkpoint.",
    )
    parser.add_argument(
        "--no-resume-stream-state",
        action="store_false",
        dest="resume_stream_state",
        help="Resume model and optimizer but intentionally start a new data stream.",
    )
    parser.add_argument(
        "--reset-best-val-loss",
        action="store_true",
        help="Reset best validation tracking when the validation mixture changes.",
    )
    parser.add_argument("--resume", type=str, default="")

    if pre_args.config:
        cfg_path = Path(pre_args.config).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        cfg_defaults = _load_config_defaults(str(cfg_path))
        unknown = sorted(set(cfg_defaults.keys()) - set(vars(parser.parse_args([])).keys()))
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        parser.set_defaults(**cfg_defaults)
        parser.set_defaults(config=str(cfg_path))

    return parser.parse_args(remaining)


def main():
    args = parse_args()
    if min(args.batch_size, args.grad_accum, args.max_steps, args.log_every) <= 0:
        raise ValueError("Batch size, accumulation, max steps and logging interval must be positive")
    if args.keep_checkpoints < 1 or args.checkpoint_reserve_gib < 0:
        raise ValueError("Invalid checkpoint retention or disk reserve")
    if args.stop_after_steps < 0 or (args.stop_after_steps and args.checkpoint_format != "bundle"):
        raise ValueError("stop_after_steps requires bundle mode and must be nonnegative")
    if args.checkpoint_format == "bundle":
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        run_lock = open(Path(args.save_dir) / ".run.lock", "a+")
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another training process is using {args.save_dir}") from None
        if args.precision != "mixed" or not args.save_stream_state or args.data_mode != "hf_stream":
            raise ValueError("Bundle checkpoints require mixed precision and saved HF stream state")
        if args.resume:
            args.resume = resolve_bundle(args.resume)
        elif any(Path(args.save_dir).glob("step_*/manifest.json")):
            raise ValueError("Save directory already contains checkpoints; resume latest.json or use a new directory")
    args.sample_prompts = _parse_sample_prompts(args.sample_prompts)
    args.comparison_prompts = _parse_sample_prompts(args.comparison_prompts)
    if args.comparison_every < 0 or args.comparison_max_new_tokens <= 0:
        raise ValueError("Comparison interval must be nonnegative and token limit positive")
    if args.comparison_every and not args.comparison_prompts:
        raise ValueError("comparison_every requires comparison_prompts")
    if args.plateau_lr_patience < 0:
        raise ValueError("plateau_lr_patience must be >= 0")
    if not 0.0 < args.plateau_lr_factor < 1.0:
        raise ValueError("plateau_lr_factor must be between 0 and 1")
    if args.plateau_lr_cooldown < 0:
        raise ValueError("plateau_lr_cooldown must be >= 0")
    if args.plateau_lr_start_step < 0 or args.warmup_steps < 0:
        raise ValueError("Plateau start step and warmup must be nonnegative")
    if not math.isfinite(args.plateau_lr_min) or not math.isfinite(args.plateau_lr_threshold):
        raise ValueError("Plateau LR minimum and threshold must be finite")
    if args.plateau_lr_min < 0 or args.plateau_lr_threshold < 0:
        raise ValueError("plateau LR minimum and threshold must be >= 0")
    if args.plateau_lr_min > args.lr:
        raise ValueError("Plateau LR floor cannot exceed the peak learning rate")
    if args.lr_control_every <= 0:
        raise ValueError("lr_control_every must be > 0")
    if args.data_mode == "hf_stream":
        tokenizer = load_tokenizer(args.spm_model, args.tokenizer_path)
        spm_vocab_size = tokenizer.vocab_size()
        if args.tokenizer_path and args.vocab_size not in (0, spm_vocab_size):
            raise ValueError("Configured vocabulary does not match the pinned tokenizer")
        if args.vocab_size > 0 and args.vocab_size != spm_vocab_size:
            print(
                f"[config] overriding vocab_size={args.vocab_size} with spm_vocab_size={spm_vocab_size} "
                f"from {args.spm_model}",
                flush=True,
            )
        args.vocab_size = spm_vocab_size
        args.tokenizer_fingerprint = getattr(tokenizer, "fingerprint", "legacy")
    elif args.vocab_size <= 0:
        raise ValueError("Missing/invalid --vocab-size (or vocab_size in --config).")
    if args.data_mode == "tokens" and not args.train_tokens:
        raise ValueError("Missing --train-tokens (or train_tokens in --config).")
    if args.data_mode == "hf_stream" and not args.train_sources:
        raise ValueError("Missing --train-sources (or train_sources in --config) for hf_stream mode.")

    try:
        group = mx.distributed.init(backend=args.backend, strict=True)
    except TypeError:
        group = mx.distributed.init(backend=args.backend)
    except RuntimeError as e:
        # Single-host ring launches can provide an empty MLX_HOSTFILE.
        # Fall back to a non-strict singleton group when explicitly testing world=1.
        if args.expected_world == 1:
            try:
                group = mx.distributed.init(backend="any", strict=False)
            except TypeError:
                group = mx.distributed.init(backend="any")
        else:
            raise e

    rank = int(group.rank() if callable(getattr(group, "rank", None)) else group.rank)
    world = int(group.size() if callable(getattr(group, "size", None)) else group.size)
    print(f"[rank {rank}] host={socket.gethostname()} world={world}", flush=True)
    if args.expected_world is not None and world != args.expected_world:
        raise RuntimeError(f"Expected world={args.expected_world}, got {world}")
    if args.checkpoint_rank < 0 or args.checkpoint_rank >= world:
        raise RuntimeError(f"checkpoint_rank must be in [0, {world - 1}], got {args.checkpoint_rank}")
    if args.checkpoint_format == "bundle" and world != 1:
        raise ValueError("Transactional bundle mode is currently single-host only")

    model_dtype = _resolve_dtype(args.dtype)
    mx.random.seed(args.seed)
    cfg = TransformerConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads or args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        mlp_multiple_of=args.mlp_multiple_of,
        qk_norm=args.qk_norm,
        attention_impl=args.attention_impl,
        loss_dtype=args.loss_dtype,
        ce_impl=args.ce_impl,
        ffn_impl=args.ffn_impl,
    )
    model = TransformerLM(cfg)
    _cast_model_floats(model, model_dtype)
    mx.eval(model.parameters())

    num_params = count_parameters(model)
    tokens_per_update = args.batch_size * args.grad_accum * args.max_seq_len * world
    planned_tokens = tokens_per_update * args.max_steps
    dtype_bytes = 2 if model_dtype in (mx.float16, mx.bfloat16) else 4
    param_gib = (num_params * dtype_bytes) / (1024**3)
    if rank == 0:
        print(
            f"[model] params={num_params/1e6:.2f}M dtype={args.dtype} "
            f"approx_param_mem={param_gib:.2f} GiB/rank",
            flush=True,
        )
        print(
            f"[budget] tokens/update={tokens_per_update:,} planned_tokens={planned_tokens:,} "
            f"tokens/param={planned_tokens/max(1, num_params):.1f}",
            flush=True,
        )
        print(
                f"[train] world={world} backend={args.backend} "
                f"seq={args.max_seq_len} batch={args.batch_size} accum={args.grad_accum} "
                f"opt_mode={'rank0_only' if args.optimizer_rank0_only else 'all_ranks'} "
                f"collective_stream={args.collective_stream} "
                f"checkpoint_rank={args.checkpoint_rank} "
                f"save_stream_state={args.save_stream_state}",
                flush=True,
            )

    optimizer_cls = MasterAdamW if args.precision == "mixed" else optim.AdamW
    optimizer = optimizer_cls(
        args.lr,
        betas=[args.optimizer_beta1, args.optimizer_beta2],
        eps=args.optimizer_eps,
        weight_decay=args.weight_decay,
        bias_correction=True,
    )
    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)

    start_step = 0
    best_val_loss = float("inf")
    loaded = False
    optimizer_loaded = False
    resume_lr_controller_state = None
    if args.resume and rank == args.checkpoint_rank:
        if args.checkpoint_format == "bundle":
            with open(args.resume + ".json") as stream:
                resume_metadata = json.load(stream)
            if resume_metadata.get("distributed_training"):
                raise ValueError("This checkpoint uses token-based JACCL accounting; resume with launch_pretrain_jaccl.sh")
            resume_config = dict(resume_metadata["config"])
            resume_config.setdefault("ffn_impl", "reference")
            if resume_config != asdict(cfg):
                raise ValueError("Resume model configuration mismatch")
            for key in ("tokenizer_fingerprint", "precision", "dtype", "batch_size", "grad_accum"):
                if resume_metadata["args"].get(key) != getattr(args, key, None):
                    raise ValueError(f"Resume configuration mismatch: {key}")
        loaded, optimizer_loaded = _load_checkpoint(args.resume, model, optimizer)
        if not loaded:
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        start_step = _infer_resume_step(args.resume)
        if args.checkpoint_format == "bundle" and not optimizer_loaded:
            raise ValueError("Bundle resume requires complete optimizer state")
        if args.checkpoint_format == "bundle" and int(optimizer.step.item()) != start_step:
            raise ValueError("Optimizer step does not match checkpoint metadata")
        meta_path = args.resume + ".json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                resume_meta = json.load(f)
            if not args.reset_best_val_loss:
                best_val_loss = float(resume_meta.get("best_val_loss", best_val_loss))
            if not args.reset_lr_controller:
                state = resume_meta.get("lr_controller")
                if isinstance(state, dict):
                    resume_lr_controller_state = state

    if world > 1:
        step_value = mx.array(start_step if rank == args.checkpoint_rank else 0, dtype=mx.int32)
        start_step = int(_all_sum(step_value, stream_mode=args.collective_stream).item())
        best_value = mx.array(
            best_val_loss if rank == args.checkpoint_rank else 0.0,
            dtype=mx.float32,
        )
        best_val_loss = float(_all_sum(best_value, stream_mode=args.collective_stream).item())
    if start_step >= args.max_steps:
        raise ValueError(f"Checkpoint step {start_step} already reaches max_steps={args.max_steps}")

    lr_controller = PlateauLrController(
        patience=args.plateau_lr_patience,
        factor=args.plateau_lr_factor,
        cooldown_evals=args.plateau_lr_cooldown,
        min_lr=args.plateau_lr_min,
        threshold=args.plateau_lr_threshold,
        best_metric=best_val_loss,
        warmup_steps=args.warmup_steps,
        start_step=args.plateau_lr_start_step,
    )
    if resume_lr_controller_state is not None:
        lr_controller.load_state_dict(resume_lr_controller_state)
    if world > 1:
        controller_values = mx.array(
            [
                lr_controller.lr_scale,
                float(lr_controller.bad_evals),
                float(lr_controller.cooldown_remaining),
                float(lr_controller.reductions),
                lr_controller.best_metric,
            ]
            if rank == args.checkpoint_rank
            else [0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=mx.float32,
        )
        controller_values = _all_sum(
            controller_values, stream_mode=args.collective_stream
        )
        mx.eval(controller_values)
        lr_controller.load_state_dict(
            {
                "lr_scale": float(controller_values[0].item()),
                "bad_evals": int(controller_values[1].item()),
                "cooldown_remaining": int(controller_values[2].item()),
                "reductions": int(controller_values[3].item()),
                "best_metric": float(controller_values[4].item()),
            }
        )

    _broadcast_model(
        model,
        rank,
        world,
        source_rank=args.checkpoint_rank,
        stream_mode=args.collective_stream,
    )
    optimizer.state = _broadcast_tree_from_source(
        optimizer.state,
        rank,
        world,
        source_rank=args.checkpoint_rank,
        stream_mode=args.collective_stream,
    )
    mx.eval(optimizer.state)
    if rank == 0 and args.resume:
        print(
            f"[ckpt] resume={args.resume} loaded={loaded or world > 1} "
            f"optimizer_loaded={optimizer_loaded or world > 1} start_step={start_step}",
            flush=True,
        )
    if rank == 0 and lr_controller.enabled:
        print(
            f"[lr-controller] patience={lr_controller.patience} "
            f"factor={lr_controller.factor:.3f} cooldown={lr_controller.cooldown_evals} "
            f"threshold={lr_controller.threshold:.2e} min_lr={lr_controller.min_lr:.3e} "
            f"start_step={max(lr_controller.start_step, lr_controller.warmup_steps)} "
            f"scale={lr_controller.lr_scale:.6f} reductions={lr_controller.reductions}",
            flush=True,
        )
    if rank == 0 and args.lr_control_file:
        print(
            f"[lr-control] file={args.lr_control_file} poll_every={args.lr_control_every}",
            flush=True,
        )

    train_stream = None
    val_stream = None
    if args.data_mode == "tokens":
        train_data = TokenDataset(args.train_tokens, args.token_dtype)
        val_data = TokenDataset(args.val_tokens, args.token_dtype) if args.val_tokens else None
        if rank == 0:
            print(
                f"[data] mode=tokens train_tokens={train_data.n_tokens:,}"
                + (f" val_tokens={val_data.n_tokens:,}" if val_data else ""),
                flush=True,
            )
    else:
        train_sources = parse_source_configs(args.train_sources)
        val_sources = parse_source_configs(args.val_sources) if args.val_sources else []
        train_stream = HFStreamingBatcher(
            sources=train_sources,
            spm_model=args.spm_model,
            world_size=world,
            rank=rank,
            seed=args.seed,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
            tokenizer_path=args.tokenizer_path,
            shuffle_schedule=args.shuffle_schedule,
        )
        train_data = StreamingDatasetAdapter(train_stream, prefetch_batches=args.prefetch_batches)

        if val_sources:
            val_stream = HFStreamingBatcher(
                sources=val_sources,
                spm_model=args.spm_model,
                world_size=world,
                rank=rank,
                seed=args.seed + 99991,
                add_bos=args.add_bos,
                add_eos=args.add_eos,
                tokenizer_path=args.tokenizer_path,
                shuffle_schedule=args.shuffle_schedule,
            )
            val_data = StreamingDatasetAdapter(val_stream, prefetch_batches=0)
            args.validation_fingerprint = val_stream.fingerprint
            if args.checkpoint_format == "bundle" and args.resume and not args.reset_best_val_loss:
                saved_fingerprint = resume_metadata["args"].get("validation_fingerprint")
                if saved_fingerprint != args.validation_fingerprint or resume_metadata["args"]["eval_steps"] != args.eval_steps:
                    raise ValueError("Validation data changed; use --reset-best-val-loss intentionally")
        else:
            val_data = None

        if rank == 0:
            mix_weight_total = sum(s.weight for s in train_sources)
            configured_mix = " ".join(
                f"{s.label or f'source_{i}'}={100.0 * s.weight / mix_weight_total:.1f}%"
                for i, s in enumerate(train_sources)
            )
            print(
                f"[data] mode=hf_stream train_sources={len(train_sources)} "
                f"val_sources={len(val_sources)} tokenizer={args.tokenizer_path or args.spm_model}",
                flush=True,
            )
            print(f"[mix] configured {configured_mix}", flush=True)
            if any(s.shuffle_buffer > 0 for s in train_sources):
                print(
                    "[warn] shuffle_buffer>0 reduces exact data-cursor resume fidelity. "
                    "Use shuffle_buffer=0 for exact-ish resume.",
                    flush=True,
                )

        if args.resume and args.resume_stream_state:
            ds_path = data_state_path(args.resume, rank)
            payload = load_data_state(ds_path)
            if payload is not None:
                state = payload.get("stream_state", payload)
                train_data.load_state_dict(state)
                if "step" in payload:
                    data_step = int(payload["step"])
                    if data_step != start_step:
                        raise RuntimeError(
                            f"Model/optimizer step ({start_step}) does not match data-state step ({data_step})"
                        )
                print(
                    f"[rank {rank}] loaded data state from {ds_path} (step={payload.get('step', 'n/a')})",
                    flush=True,
                )
            else:
                if args.checkpoint_format == "bundle":
                    raise ValueError(f"Bundle resume requires matching data state: {ds_path}")
                print(
                    f"[rank {rank}] data state not found for resume: {ds_path}",
                    flush=True,
                )
        elif args.resume:
            print(
                f"[rank {rank}] intentionally starting a new data stream; "
                "model and optimizer resume is unchanged",
                flush=True,
            )

    lr_for_step = _build_lr_schedule(
        base_lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    eval_batches = []
    if val_data is not None and args.eval_every > 0:
        eval_batches = _build_fixed_eval_batches(
            dataset=val_data,
            eval_steps=args.eval_steps,
            batch_size=args.batch_size,
            seq_len=args.max_seq_len,
            seed=args.seed + 777_777,
            rank=rank,
        )
        if rank == 0:
            print(
                f"[eval] materialized {len(eval_batches)} fixed held-out batches per rank",
                flush=True,
            )
    elif rank == 0 and args.eval_every > 0:
        print("[warn] eval_every>0 but no validation source is configured", flush=True)

    eval_row_sources = None
    if args.eval_by_source and eval_batches:
        if world != 1 or val_stream is None:
            raise ValueError("Per-source evaluation requires a local HF streaming validation set")
        eval_row_sources = [
            [val_stream.schedule[(i * args.batch_size + j) % len(val_stream.schedule)]
             for j in range(args.batch_size)]
            for i in range(len(eval_batches))
        ]
    if args.checkpoint_format == "bundle" and eval_batches:
        digest = hashlib.sha256()
        for x, y in eval_batches:
            digest.update(np.asarray(x).tobytes())
            digest.update(np.asarray(y).tobytes())
        args.eval_fingerprint = digest.hexdigest()
        if args.resume and not args.reset_best_val_loss:
            if resume_metadata["args"].get("eval_fingerprint") != args.eval_fingerprint:
                raise ValueError("Fixed evaluation tokens differ from the checkpoint; use --reset-best-val-loss intentionally")

    sample_tokenizer = None
    sample_interval = args.sample_every if args.sample_every > 0 else args.eval_every
    if (args.sample_prompts and args.sample_max_new_tokens > 0) or args.comparison_every:
        sample_tokenizer = load_tokenizer(args.spm_model, args.tokenizer_path)
    if args.sample_prompts and args.sample_max_new_tokens > 0:
        if rank == 0:
            prompts_per_event = (
                len(args.sample_prompts)
                if args.sample_prompts_per_event <= 0
                else min(args.sample_prompts_per_event, len(args.sample_prompts))
            )
            print(
                f"[sample] prompts={len(args.sample_prompts)} "
                f"every={sample_interval} per_event={prompts_per_event} "
                f"max_new_tokens={args.sample_max_new_tokens} "
                f"temperature={args.sample_temperature} top_k={args.sample_top_k}",
                flush=True,
            )

    comparison_settings = {
        "prompts": args.comparison_prompts, "temperature": 0.0, "top_k": 1,
        "max_new_tokens": args.comparison_max_new_tokens,
        "max_seq_len": args.max_seq_len, "add_bos": args.add_bos,
        "tokenizer_fingerprint": getattr(args, "tokenizer_fingerprint", None),
    }
    comparison_fingerprint = hashlib.sha256(
        json.dumps(comparison_settings, sort_keys=True).encode()
    ).hexdigest()
    if rank == 0 and args.comparison_every:
        print(f"[comparison] prompts={len(args.comparison_prompts)} every={args.comparison_every} "
              f"temperature=0 top_k=1 max_new_tokens={args.comparison_max_new_tokens} "
              f"file={args.save_dir}/comparison_samples.jsonl", flush=True)

    def loss_fn(x, y):
        return model(x, targets=y, ignore_index=args.ignore_index)["loss"]

    step_and_grad = nn.value_and_grad(model, loss_fn)
    use_compiled_local_step = bool(args.compile_train_step) and world == 1 and not args.trace_first_step
    if args.profile_every < 0 or (args.profile_every and (
        not use_compiled_local_step or args.checkpoint_format != "bundle"
    )):
        raise ValueError("Profiling requires compiled single-host bundle training and a nonnegative interval")
    compiled_local_step = None
    if use_compiled_local_step:
        compiled_local_step = _build_local_compiled_train_step(
            model=model,
            optimizer=optimizer,
            grad_accum=int(args.grad_accum),
            grad_clip=float(args.grad_clip),
            ignore_index=int(args.ignore_index),
        )
    if rank == 0 and world == 1:
        print(
            f"[compile] train_step={'enabled' if use_compiled_local_step else 'disabled'} (local single-host path)",
            flush=True,
        )
    ema_loss = None
    t_loop = time.perf_counter()
    manual_lr_multiplier = 1.0
    manual_lr_override = None
    last_control_key = (manual_lr_multiplier, manual_lr_override)
    completed_step = start_step
    last_bundle_step = None
    stop_requested = False
    interrupt_requested = False
    stop_announced = False
    previous_sigint = None
    performance = PerformanceTracker(
        Path(args.save_dir) / "performance.jsonl", tokens_per_update, enabled=args.profile_every > 0,
    )

    def request_stop(signum, frame):
        nonlocal stop_requested, interrupt_requested
        # Keep the handler idempotent and free of I/O, including during checkpoint cleanup.
        stop_requested = True
        interrupt_requested = True

    if args.checkpoint_format == "bundle":
        previous_sigint = signal.signal(signal.SIGINT, request_stop)

    def commit_bundle(at_step, is_best=False):
        nonlocal stop_announced
        if interrupt_requested and not stop_announced:
            print("[stop] Saving model, optimizer and consumed data position. "
                  "Repeated Ctrl-C will not interrupt the save; please wait for [done].", flush=True)
            stop_announced = True
        save_started = time.perf_counter()
        metadata = {
            "step": at_step, "args": vars(args), "config": asdict(cfg),
            "world": world, "backend": args.backend, "best_val_loss": best_val_loss,
            "lr_controller": lr_controller.state_dict(), "timestamp": time.time(),
            "tokens_processed": at_step * tokens_per_update,
        }
        destination = save_bundle(
            args.save_dir, model, optimizer, metadata, train_data.state_dict(),
            best=is_best, keep=args.keep_checkpoints, reserve_gib=args.checkpoint_reserve_gib,
        )
        print(f"[ckpt] committed {'best + latest' if is_best else 'latest'} {destination}", flush=True)
        performance.record("checkpoint", time.perf_counter() - save_started)

    for step in range(start_step, args.max_steps):
        is_new_best = False
        t0 = time.perf_counter()
        if args.lr_control_file and (
            step == start_step or (step + 1) % args.lr_control_every == 0
        ):
            control_valid = True
            control_present = False
            control_multiplier = 1.0
            control_lr = -1.0
            if rank == args.checkpoint_rank:
                try:
                    control = _read_lr_control_file(args.lr_control_file)
                    if control is not None:
                        control_present = True
                        control_multiplier = float(control.get("lr_multiplier", 1.0))
                        control_lr = float(control.get("lr", -1.0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    control_valid = False
                    print(f"[warn] ignoring invalid LR control file: {exc}", flush=True)

            if world > 1:
                control_values = mx.array(
                    [
                        1.0 if control_valid else 0.0,
                        1.0 if control_present else 0.0,
                        control_multiplier,
                        control_lr,
                    ]
                    if rank == args.checkpoint_rank
                    else [0.0, 0.0, 0.0, 0.0],
                    dtype=mx.float32,
                )
                control_values = _all_sum(
                    control_values, stream_mode=args.collective_stream
                )
                mx.eval(control_values)
                control_valid = bool(float(control_values[0].item()) > 0.5)
                control_present = bool(float(control_values[1].item()) > 0.5)
                control_multiplier = float(control_values[2].item())
                control_lr = float(control_values[3].item())

            if control_valid:
                if control_present:
                    manual_lr_multiplier = control_multiplier
                    manual_lr_override = control_lr if control_lr > 0 else None
                else:
                    manual_lr_multiplier = 1.0
                    manual_lr_override = None
                control_key = (manual_lr_multiplier, manual_lr_override)
                if rank == 0 and control_key != last_control_key:
                    if manual_lr_override is not None:
                        detail = f"lr={manual_lr_override:.3e}"
                    elif manual_lr_multiplier != 1.0:
                        detail = f"lr_multiplier={manual_lr_multiplier:.6f}"
                    else:
                        detail = "automatic schedule restored"
                    print(f"[lr-control {step+1:6d}] {detail}", flush=True)
                last_control_key = control_key

        scheduled_lr_t = lr_for_step(step)
        lr_t = lr_controller.effective_lr(
            scheduled_lr_t,
            manual_multiplier=manual_lr_multiplier,
            manual_lr=manual_lr_override,
            step=step,
        )
        if use_compiled_local_step:
            data_started = time.perf_counter()
            micro_x = []
            micro_y = []
            for micro in range(args.grad_accum):
                x, y = train_data.sample_batch(
                    batch_size=args.batch_size,
                    seq_len=args.max_seq_len,
                    seed=args.seed,
                    step=step,
                    rank=rank,
                    stream=micro,
                )
                micro_x.append(x)
                micro_y.append(y)
            x_batch = mx.stack(micro_x, axis=0)
            y_batch = mx.stack(micro_y, axis=0)
            lr_arr = mx.array(lr_t, dtype=mx.float32)
            compute_started = time.perf_counter()
            performance.record("data_wait_and_batch", compute_started - data_started)
            step_loss, grad_norm_arr = compiled_local_step(x_batch, y_batch, lr_arr)
            mx.eval(step_loss, grad_norm_arr, model.state, optimizer.state)
            performance.record("train_compute", time.perf_counter() - compute_started)
            step_loss_value = float(step_loss.item())
            grad_norm = float(grad_norm_arr.item())
        else:
            total_loss_local = 0.0
            grads_acc = None

            for micro in range(args.grad_accum):
                if args.trace_first_step and step == start_step:
                    print(f"[rank {rank}] trace step={step+1} micro={micro} stage=sample_batch", flush=True)
                x, y = train_data.sample_batch(
                    batch_size=args.batch_size,
                    seq_len=args.max_seq_len,
                    seed=args.seed,
                    step=step,
                    rank=rank,
                    stream=micro,
                )
                if args.data_mode == "hf_stream" and world > 1:
                    # Keep ranks aligned when live streaming can have per-host jitter.
                    if args.trace_first_step and step == start_step:
                        print(f"[rank {rank}] trace step={step+1} micro={micro} stage=sample_sync", flush=True)
                    ready = mx.array(1.0, dtype=mx.float32)
                    ready = _all_sum(ready, stream_mode=args.collective_stream)
                    mx.eval(ready)
                if args.trace_first_step and step == start_step:
                    print(f"[rank {rank}] trace step={step+1} micro={micro} stage=fwd_bwd", flush=True)
                loss, grads = step_and_grad(x, y)
                if isinstance(optimizer, MasterAdamW):
                    grads = tree_map(lambda g: g.astype(mx.float32), grads)
                mx.eval(loss)
                total_loss_local += float(loss.item())
                grads_acc = grads if grads_acc is None else _tree_add(grads_acc, grads)

            if args.trace_first_step and step == start_step:
                print(f"[rank {rank}] trace step={step+1} stage=grad_accum_done", flush=True)
            grads_acc = _tree_scale(grads_acc, 1.0 / float(args.grad_accum))
            if world > 1:
                if args.trace_first_step and step == start_step:
                    print(f"[rank {rank}] trace step={step+1} stage=allreduce_grads", flush=True)
                grads_acc = _allreduce_tree(grads_acc, world, stream_mode=args.collective_stream)

            if args.trace_first_step and step == start_step:
                print(f"[rank {rank}] trace step={step+1} stage=clip_grads", flush=True)
            grads_acc, grad_norm = _clip_grads(grads_acc, args.grad_clip)
            if args.optimizer_rank0_only and world > 1:
                if rank == 0:
                    if args.trace_first_step and step == start_step:
                        print(f"[rank {rank}] trace step={step+1} stage=optimizer_update_rank0", flush=True)
                    optimizer.learning_rate = lr_t
                    optimizer.update(model, grads_acc)
                    mx.eval(model.parameters(), optimizer.state)
                if args.trace_first_step and step == start_step:
                    print(f"[rank {rank}] trace step={step+1} stage=broadcast_model", flush=True)
                _broadcast_model(model, rank, world, source_rank=0, stream_mode=args.collective_stream)
            else:
                if args.trace_first_step and step == start_step:
                    print(f"[rank {rank}] trace step={step+1} stage=optimizer_update_all", flush=True)
                optimizer.learning_rate = lr_t
                optimizer.update(model, grads_acc)
                mx.eval(model.parameters(), optimizer.state)

            if args.trace_first_step and step == start_step:
                print(f"[rank {rank}] trace step={step+1} stage=reduce_step_loss", flush=True)
            step_loss = mx.array(total_loss_local / float(args.grad_accum), dtype=mx.float32)
            if world > 1:
                step_loss = _all_sum(step_loss, stream_mode=args.collective_stream) / world
            mx.eval(step_loss)
            step_loss_value = float(step_loss.item())
        if not math.isfinite(step_loss_value) or not math.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite at step {step+1}: loss={step_loss_value}, grad_norm={grad_norm}. "
                "Try --dtype bfloat16 and/or lower --lr."
            )
        if args.precision == "mixed" and (step == start_step or (step + 1) % args.log_every == 0):
            dtypes = {v.dtype for v in _tree_leaves(model.parameters()) if isinstance(v, mx.array)}
            if dtypes != {model_dtype}:
                raise RuntimeError(f"Compute dtype drift detected: {dtypes}")
            if step == start_step:
                print(f"[precision] verified compute={args.dtype} master_weights=float32 "
                      "moments=float32 gradient_accumulation=float32", flush=True)
        completed_step = step + 1
        performance.update()
        if args.stop_after_steps > 0 and completed_step - start_step >= args.stop_after_steps:
            stop_requested = True

        ema_loss = step_loss_value if ema_loss is None else (0.98 * ema_loss + 0.02 * step_loss_value)
        dt = time.perf_counter() - t0
        toks_per_step = args.batch_size * args.grad_accum * args.max_seq_len * world
        toks_per_sec = toks_per_step / max(dt, 1e-9)

        if rank == 0 and ((step + 1) % args.log_every == 0 or step == 0):
            elapsed = time.perf_counter() - t_loop
            effective_tps = (completed_step - start_step) * tokens_per_update / max(elapsed, 1e-9)
            print(
                f"[step {step+1:6d}] loss={step_loss_value:.4f} ema={ema_loss:.4f} "
                f"lr={lr_t:.3e} grad_norm={grad_norm:.3f} "
                f"tok/s={toks_per_sec:,.0f} run_tok/s={effective_tps:,.0f} "
                f"tokens={completed_step * tokens_per_update:,}",
                flush=True,
            )

        if (
            rank == 0
            and train_stream is not None
            and args.mix_log_every > 0
            and ((step + 1) % args.mix_log_every == 0)
        ):
            stream_state = train_data.state_dict()
            labels = stream_state.get("source_labels", [])
            counts = stream_state.get("source_tokens_emitted", [])
            stream_total = sum(int(v) for v in counts)
            if labels and stream_total > 0:
                actual_mix = " ".join(
                    f"{label}={100.0 * int(count) / stream_total:.2f}%"
                    for label, count in zip(labels, counts)
                )
                absolute_tokens = (step + 1) * toks_per_step
                print(
                    f"[mix {step+1:6d}] new_stream_tokens={stream_total:,} "
                    f"absolute_tokens={absolute_tokens:,} {actual_mix}",
                    flush=True,
                )

        if not interrupt_requested and eval_batches and args.eval_every > 0 and ((step + 1) % args.eval_every == 0):
            eval_started = time.perf_counter()
            source_metrics = {}
            if eval_row_sources is not None:
                val_loss, source_metrics = _evaluate_by_source(
                    model, eval_batches, eval_row_sources,
                    [s.cfg.label or f"source_{i}" for i, s in enumerate(val_stream.sources)],
                    args.ignore_index,
                )
            else:
                val_loss = _evaluate(
                    model=model, batches=eval_batches, world=world,
                    ignore_index=args.ignore_index, collective_stream=args.collective_stream,
                )
            if not math.isfinite(val_loss):
                raise FloatingPointError("Non-finite validation loss; preserving the last complete checkpoint")
            performance.record("evaluation", time.perf_counter() - eval_started)
            if rank == 0:
                val_ppl = math.exp(min(20.0, val_loss))
                print(
                    f"[eval {step+1:6d}] val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}",
                    flush=True,
                )
                if source_metrics:
                    details = " ".join(f"{name}={item['loss']:.4f}" for name, item in source_metrics.items())
                    print(f"[eval-sources {step+1:6d}] {details}", flush=True)
            reduction = lr_controller.observe(
                val_loss,
                lr_for_step(completed_step),
                completed_step=completed_step,
                allow_reduction=(
                    manual_lr_override is None and manual_lr_multiplier == 1.0
                ),
            )
            if reduction is not None and rank == 0:
                old_scale, new_scale = reduction
                old_lr = max(lr_controller.min_lr, lr_for_step(completed_step) * old_scale)
                new_lr = max(lr_controller.min_lr, lr_for_step(completed_step) * new_scale)
                print(
                    f"[lr-controller {step+1:6d}] plateau detected; "
                    f"scale={old_scale:.6f}->{new_scale:.6f} "
                    f"effective_lr={old_lr:.3e}->{new_lr:.3e} "
                    f"reductions={lr_controller.reductions}",
                    flush=True,
                )
            if rank == 0:
                next_lr = lr_controller.effective_lr(
                    lr_for_step(completed_step), step=completed_step,
                    manual_multiplier=manual_lr_multiplier, manual_lr=manual_lr_override,
                )
                if lr_controller.enabled:
                    if completed_step < max(lr_controller.start_step, lr_controller.warmup_steps):
                        controller_status = "startup_protected"
                    elif manual_lr_override is not None or manual_lr_multiplier != 1.0:
                        controller_status = "manual_control"
                    elif next_lr <= lr_controller.min_lr * (1 + 1e-9):
                        controller_status = "at_floor"
                    else:
                        controller_status = "monitoring"
                    print(f"[lr-controller {completed_step:6d}] {controller_status} "
                          f"bad_evals={lr_controller.bad_evals}/{lr_controller.patience} "
                          f"cooldown={lr_controller.cooldown_remaining} next_lr={next_lr:.3e}", flush=True)
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                with open(Path(args.save_dir) / "metrics.jsonl", "a") as stream:
                    stream.write(json.dumps({
                        "step": completed_step, "tokens": completed_step * tokens_per_update,
                        "val_loss": val_loss, "val_by_source": source_metrics,
                        "eval_fingerprint": getattr(args, "eval_fingerprint", None),
                        "lr": lr_t, "next_lr": next_lr, "lr_controller": lr_controller.state_dict(),
                        "run_tokens_per_second": ((completed_step - start_step) * tokens_per_update /
                                                  max(time.perf_counter() - t_loop, 1e-9)),
                    }) + "\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_new_best = True
                best_path = os.path.join(args.save_dir, "best.safetensors")
                best_metadata = {
                    "step": step + 1,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                    "config": asdict(cfg),
                    "world": world,
                    "backend": args.backend,
                    "lr_controller": lr_controller.state_dict(),
                    "timestamp": time.time(),
                }
                if rank == args.checkpoint_rank and args.checkpoint_format == "legacy":
                    _save_checkpoint(best_path, model, optimizer, best_metadata)
                    print(
                        f"[ckpt] updated best {best_path} val_loss={best_val_loss:.4f}",
                        flush=True,
                    )
                if args.checkpoint_format == "legacy" and args.data_mode == "hf_stream" and train_stream is not None and args.save_stream_state:
                    save_data_state(
                        data_state_path(best_path, rank),
                        {"step": step + 1, "stream_state": train_data.state_dict()},
                    )

        if (
            rank == 0
            and not interrupt_requested
            and sample_tokenizer is not None
            and args.sample_prompts
            and args.sample_max_new_tokens > 0
            and sample_interval > 0
            and ((step + 1) % sample_interval == 0)
        ):
            sample_started = time.perf_counter()
            event_index = (step + 1) // sample_interval - 1
            selected_prompts = _select_sample_prompts(
                prompts=args.sample_prompts,
                count=args.sample_prompts_per_event,
                event_index=event_index,
                seed=args.sample_seed,
            )
            for prompt_index, prompt in selected_prompts:
                if interrupt_requested:
                    break
                completion = _generate_completion(
                    model=model,
                    tokenizer=sample_tokenizer,
                    prompt=prompt,
                    max_seq_len=args.max_seq_len,
                    max_new_tokens=args.sample_max_new_tokens,
                    temperature=args.sample_temperature,
                    top_k=args.sample_top_k,
                    seed=args.sample_seed + (step + 1) * 1_009 + prompt_index,
                    add_bos=args.add_bos,
                )
                print(f"[sample {step+1:6d}] prompt={prompt!r}", flush=True)
                print(completion, flush=True)
            performance.record("sampling", time.perf_counter() - sample_started)

        if rank == 0 and not interrupt_requested and args.comparison_every and completed_step % args.comparison_every == 0:
            sample_started = time.perf_counter()
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(args.save_dir) / "comparison_samples.jsonl", "a") as stream:
                for prompt in args.comparison_prompts:
                    if interrupt_requested:
                        break
                    completion = _generate_completion(
                        model=model, tokenizer=sample_tokenizer, prompt=prompt,
                        max_seq_len=args.max_seq_len, max_new_tokens=args.comparison_max_new_tokens,
                        temperature=0.0, top_k=1, seed=args.sample_seed, add_bos=args.add_bos,
                    )
                    print(f"[comparison {completed_step:6d}] prompt={prompt!r}", flush=True)
                    print(completion, flush=True)
                    stream.write(json.dumps({
                        "step": completed_step, "prompt": prompt, "completion": completion,
                        "settings": comparison_settings, "fingerprint": comparison_fingerprint,
                    }) + "\n")
                    stream.flush()
            performance.record("sampling", time.perf_counter() - sample_started)

        if args.checkpoint_format == "bundle":
            if is_new_best or stop_requested or (args.save_every > 0 and completed_step % args.save_every == 0):
                commit_bundle(completed_step, is_best=is_new_best)
                last_bundle_step = completed_step
            if stop_requested:
                break
        elif args.save_every > 0 and ((step + 1) % args.save_every == 0):
            ckpt_path = os.path.join(args.save_dir, f"step_{step+1:07d}.safetensors")
            metadata = {
                "step": step + 1,
                "args": vars(args),
                "config": asdict(cfg),
                "world": world,
                "backend": args.backend,
                "best_val_loss": best_val_loss,
                "lr_controller": lr_controller.state_dict(),
                "timestamp": time.time(),
            }
            if rank == args.checkpoint_rank:
                _save_checkpoint(ckpt_path, model, optimizer, metadata)
                print(f"[ckpt] saved {ckpt_path}", flush=True)
            if args.data_mode == "hf_stream" and train_stream is not None and args.save_stream_state:
                ds_payload = {"step": step + 1, "stream_state": train_data.state_dict()}
                save_data_state(data_state_path(ckpt_path, rank), ds_payload)
                if rank == args.checkpoint_rank:
                    print(f"[ckpt] saved data-state for all ranks at step {step+1}", flush=True)

        if args.profile_every and completed_step % args.profile_every == 0:
            performance.report(completed_step)

    if args.checkpoint_format == "bundle":
        if last_bundle_step != completed_step:
            commit_bundle(completed_step)
        performance.report(completed_step, final=True)
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        print(f"[done] stopped at step={completed_step}; resume with {args.save_dir}/latest.json", flush=True)
        return

    final_path = os.path.join(args.save_dir, "final.safetensors")
    if rank == args.checkpoint_rank:
        metadata = {
            "step": args.max_steps,
            "args": vars(args),
            "config": asdict(cfg),
            "world": world,
            "backend": args.backend,
            "best_val_loss": best_val_loss,
            "lr_controller": lr_controller.state_dict(),
            "duration_sec": time.perf_counter() - t_loop,
            "timestamp": time.time(),
        }
        _save_checkpoint(final_path, model, optimizer, metadata)
        print(f"[done] saved {final_path}", flush=True)
    if args.data_mode == "hf_stream" and train_stream is not None and args.save_stream_state:
        save_data_state(
            data_state_path(final_path, rank),
            {"step": args.max_steps, "stream_state": train_data.state_dict()},
        )


if __name__ == "__main__":
    main()
