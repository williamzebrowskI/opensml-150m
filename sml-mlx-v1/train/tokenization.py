"""Pretraining tokenizer interface; legacy SentencePiece remains supported."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ByteBPETokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer

        path = Path(path)
        manifest = json.loads(path.with_name("manifest.json").read_text())
        for name, expected in manifest["sha256"].items():
            if Path(name).name != name or file_sha256(path.with_name(name)) != expected:
                raise ValueError(f"Tokenizer integrity check failed: {name}")
        self.backend = Tokenizer.from_file(str(path))
        self.backend.no_padding()
        self.backend.no_truncation()
        self.special_ids = manifest["special_ids"]
        self.fingerprint = file_sha256(path.with_name("manifest.json"))
        if self.vocab_size() != manifest["vocab_size"]:
            raise ValueError("Tokenizer vocabulary does not match manifest")

    def encode(self, text, out_type=int):
        if out_type is not int:
            raise ValueError("Only integer token IDs are supported")
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.backend.decode([int(i) for i in ids], skip_special_tokens=False)

    def vocab_size(self):
        return self.backend.get_vocab_size(with_added_tokens=True)

    def bos_id(self):
        return self.special_ids.get("bos", -1)

    def eos_id(self):
        return self.special_ids["eos"]

    def pad_id(self):
        return self.special_ids.get("pad", -1)


def load_tokenizer(spm_model="", tokenizer_path=""):
    if tokenizer_path:
        return ByteBPETokenizer(tokenizer_path)
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=spm_model)
