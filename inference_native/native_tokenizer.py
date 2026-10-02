"""Own vocabulary, explicit structural IDs, verified frozen artifact."""

from pathlib import Path
from native_utils import file_sha256, read_json

SPECIALS = ["<|pad|>", "<|doc_end|>", "<|turn_start|>", "<|turn_end|>"]
FIXTURES = ["A plant uses sunlight to make sugars.", "  leading and trailing spaces  \n\n",
            "\tif x:\n\t\treturn x + 1\n", "caf\u00e9, \u03c0, \u4e2d\u6587, \U0001f680; e\u0301",
            "17 * 6 = 102; x**2 + y**2 = z**2", '{"items": [1, 2]}', "a\r\nb\r\n", "",
            "Print " + " and ".join(SPECIALS) + " literally."]


class Tokenizer:
    def __init__(self, directory):
        from tokenizers import Tokenizer as Backend
        directory = Path(directory)
        self.manifest = read_json(directory / "manifest.json")
        if self.manifest["format"] != "sml-v2-tokenizer-v1":
            raise ValueError("Not a v2 tokenizer")
        if not {"tokenizer.json", "tokenizer_config.json", "corpus.json"} <= self.manifest["sha256"].keys():
            raise ValueError("Incomplete tokenizer integrity manifest")
        if self.manifest["special_ids"] != dict(pad=0, eos=1, turn_start=2, turn_end=3):
            raise ValueError("Structural token metadata changed")
        for name, digest in self.manifest["sha256"].items():
            if Path(name).name != name or file_sha256(directory / name) != digest:
                raise ValueError(f"Tokenizer integrity failure: {name}")
        self.backend = Backend.from_file(str(directory / "tokenizer.json"))
        self.backend.no_padding()
        self.backend.no_truncation()
        self.backend.encode_special_tokens = True
        self.fingerprint = file_sha256(directory / "manifest.json")
        self.eos = self.manifest["special_ids"]["eos"]
        self.pad = self.manifest["special_ids"]["pad"]
        self.vocab_size = self.backend.get_vocab_size(with_added_tokens=True)
        ids = set(self.backend.get_vocab().values())
        if ids != set(range(self.manifest["vocab_size"])):
            raise ValueError("Tokenizer IDs are not the declared dense vocabulary")
        for i, token in enumerate(SPECIALS):
            if self.backend.token_to_id(token) != i:
                raise ValueError("Special token mapping changed")

    def encode(self, text):
        ids = self.backend.encode(text, add_special_tokens=False).ids
        if any(i < len(SPECIALS) for i in ids):
            raise ValueError("Ordinary text emitted structural token IDs")
        return ids

    def decode(self, ids):
        return self.backend.decode([int(i) for i in ids], skip_special_tokens=False)

    def assert_roundtrip(self, text):
        ids = self.encode(text)
        if self.decode(ids) != text:
            raise ValueError(f"Tokenizer round-trip failed: {text[:120]!r}")
        return ids
