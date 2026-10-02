"""Deterministic token-weighted local pools; no silent recycling or stream resets."""

from pathlib import Path
import numpy as np
from .common import file_sha256, read_json


def verify_data(directory, tokenizer):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest["format"] != "sml-v2-data-v1" or manifest["tokenizer"] != tokenizer.fingerprint:
        raise ValueError("Dataset/tokenizer mismatch")
    if manifest["corpus"] != tokenizer.manifest["corpus_fingerprint"]:
        raise ValueError("Tokenizer and model corpus recipes differ")
    for record in manifest["files"]:
        for key, sha in (("file", "sha256"), ("index", "index_sha256")):
            name = record[key]
            if Path(name).name != name or file_sha256(directory / name) != record[sha]:
                raise ValueError(f"Corrupt prepared data: {name}")
        if (directory / record["file"]).stat().st_size != record["tokens"] * 2:
            raise ValueError("Token count and file size disagree")
    return manifest


class TokenPools:
    def __init__(self, directory, manifest, split, seq_len, state=None):
        self.fingerprint = file_sha256(Path(directory) / "manifest.json")
        self.seq_len = seq_len
        self.split = split
        self.weights = manifest["weights"]
        self.arrays = {r["source"]: np.memmap(Path(directory) / r["file"], mode="r", dtype="<u2")
                       for r in manifest["files"] if r["split"] == split}
        if set(self.arrays) != set(self.weights) or seq_len < 1:
            raise ValueError("Missing sources or invalid context")
        self.offsets = {s: 0 for s in self.weights}
        if state is not None:
            if (state["fingerprint"], state["split"], state["seq_len"]) != (self.fingerprint, split, seq_len):
                raise ValueError("Saved data cursor is incompatible")
            offsets = state["offsets"]
            if set(offsets) != set(self.weights):
                raise ValueError("Cursor source mismatch")
            if any(type(v) is not int or v < 0 or v % seq_len or v >= len(self.arrays[s])
                   for s, v in offsets.items()):
                raise ValueError("Invalid saved token offsets")
            self.offsets = dict(offsets)

    def state_dict(self):
        return dict(fingerprint=self.fingerprint, split=self.split, seq_len=self.seq_len,
                    offsets=dict(self.offsets))

    def row(self, source=None):
        # Fixed-length rows make source weighting a TOKEN budget, not document counts.
        source = source or min(self.weights, key=lambda s: self.offsets[s] / self.weights[s])
        start = self.offsets[source]
        stop = start + self.seq_len + 1
        if stop > len(self.arrays[source]):
            raise RuntimeError(f"{source} pool exhausted; prepare more data in a new experiment, not a silent repeat")
        row = np.asarray(self.arrays[source][start:stop], dtype=np.int32)
        self.offsets[source] += self.seq_len
        return row[:-1], row[1:]

    def batch(self, count, source=None):
        pairs = [self.row(source) for _ in range(count)]
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])

    def global_batch(self, accum, batches):
        pairs = [self.batch(sum(batches)) for _ in range(accum)]
        return np.stack([[p[axis] for p in pairs] for axis in (0, 1)])
