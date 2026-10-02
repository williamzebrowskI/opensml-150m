"""Transactional single-host checkpoints with integrity checks and retention."""

import gzip
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import mlx.core as mx
from mlx.utils import tree_flatten

try:
    from .tokenization import file_sha256
except ImportError:
    from tokenization import file_sha256

FORMAT = "sml-pretrain-bundle-v1"


def _write_json(path, payload):
    with open(path, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _pointer(root, name, bundle):
    path = root / f"{name}.json"
    temp = root / f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_json(temp, {"format": FORMAT, "bundle": bundle.name})
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _pointer_target(path):
    payload = json.loads(path.read_text())
    name = payload.get("bundle", "")
    if payload.get("format") != FORMAT or not name.startswith("step_") or Path(name).name != name:
        raise ValueError(f"Invalid checkpoint pointer: {path}")
    target = path.parent / name
    if target.is_symlink():
        raise ValueError("Checkpoint bundle must not be a symlink")
    return target


def resolve_bundle(path):
    path = Path(path).absolute()
    if path.suffix == ".json":
        path = _pointer_target(path)
    elif path.name == "model.safetensors":
        path = path.parent
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"Not a complete checkpoint bundle: {path}")
    required = {"model.safetensors", "model.safetensors.optimizer.safetensors", "model.safetensors.json"}
    if not required <= manifest["files"].keys():
        raise ValueError("Incomplete checkpoint bundle manifest")
    for name, info in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe checkpoint filename")
        file = path / name
        if file.is_symlink() or file.stat().st_size != info["bytes"] or file_sha256(file) != info["sha256"]:
            raise ValueError(f"Checkpoint integrity check failed: {file}")
    return str(path / "model.safetensors")


def _prune(root, keep):
    protected = set()
    for name in ("best", "latest"):
        pointer = root / f"{name}.json"
        if pointer.exists():
            protected.add(_pointer_target(pointer).name)
    bundles = []
    for path in root.glob("step_*_*"):
        if path.is_symlink() or not path.is_dir():
            continue
        try:
            manifest = json.loads((path / "manifest.json").read_text())
        except (OSError, ValueError):
            continue
        if manifest.get("format") == FORMAT:
            bundles.append((manifest["created_ns"], path))
    bundles.sort(key=lambda item: item[0], reverse=True)
    protected.update(path.name for _, path in bundles[:keep])
    for _, path in bundles:
        if path.name not in protected:
            shutil.rmtree(path)


def save_bundle(root, model, optimizer, metadata, stream_state, *, best=False, keep=3, reserve_gib=5.0):
    """Commit weights, optimizer, metadata and cursor before publishing pointers.

    A failed write never replaces a prior checkpoint. Retention runs only after
    commit and never touches legacy files, logs, reports or unrecognized folders.
    """
    if keep < 1 or reserve_gib < 0:
        raise ValueError("keep must be positive and disk reserve nonnegative")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tensors = dict(tree_flatten(model.parameters()))
    states = dict(tree_flatten(optimizer.state))
    needed = int(sum(x.nbytes for x in [*tensors.values(), *states.values()]) * 1.1)
    needed += int(reserve_gib * 1024**3)
    if shutil.disk_usage(root).free < needed:
        raise OSError(f"Insufficient disk space: need {needed / 1024**3:.2f} GiB including reserve; previous checkpoints are intact")
    name = f"step_{metadata['step']:07d}_{uuid.uuid4().hex[:12]}"
    staging = root / ("." + name + ".tmp")
    destination = root / name
    staging.mkdir()
    try:
        mx.save_safetensors(str(staging / "model.safetensors"), tensors)
        mx.save_safetensors(str(staging / "model.safetensors.optimizer.safetensors"), states)
        _write_json(staging / "model.safetensors.json", metadata)
        if stream_state is not None:
            with gzip.open(staging / "model.safetensors.rank0.data_state.json.gz", "wt") as stream:
                json.dump({"step": metadata["step"], "stream_state": stream_state}, stream)
        files = {}
        for file in staging.iterdir():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
            files[file.name] = {"bytes": file.stat().st_size, "sha256": file_sha256(file)}
        _write_json(staging / "manifest.json", {
            "format": FORMAT, "step": metadata["step"], "created_ns": time.time_ns(), "files": files,
        })
        os.replace(staging, destination)
        if best:
            _pointer(root, "best", destination)
        _pointer(root, "latest", destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    try:
        _prune(root, keep)
    except OSError as exc:
        print(f"[warn] checkpoint saved but retention failed: {exc}", flush=True)
    return str(destination)
