"""One pinned source/filter/split contract for tokenizer and model data."""

import hashlib
import json
import unicodedata
from .common import fingerprint

POLICY = "sml-v2-content-splits-v1"


def content_key(text):
    # Used for splitting/dedup only; never normalize the actual training text.
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def partition(key):
    bucket = int(key[:16], 16) % 10000
    if bucket < 100:
        return "final"
    if bucket < 200:
        return "validation"
    if bucket < 300:
        return "tokenizer_dev"
    return "train"


def identity(config):
    return fingerprint({"config": config, "policy": POLICY})


def accepted(row, source, config):
    text = row.get(source["text_field"])
    if not isinstance(text, str):
        raise ValueError(f"{source['label']}: missing string field {source['text_field']}")
    if source.get("min_score"):
        score = source["min_score"]
        if score["field"] not in row:
            raise ValueError(f"{source['label']}: missing quality score {score['field']}")
        if float(row[score["field"]]) < score["value"]:
            return None
    if source.get("format_contains"):
        if not isinstance(row.get("format"), str):
            raise ValueError("Missing Cosmopedia format metadata")
        if not any(v in row["format"].lower() for v in source["format_contains"]):
            return None
    if len(text) < config["min_chars"] or len(text.encode("utf-8")) > config["max_bytes"]:
        return None
    if "\x00" in text or text.count("\ufffd") > 2:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.5:
        return None
    return text


def documents(source, config, wanted):
    if source.get("local_jsonl"):
        def rows():
            with open(source["local_jsonl"]) as f:
                for line in f:
                    yield json.loads(line)
        dataset = rows()
    else:
        from datasets import load_dataset
        dataset = load_dataset(source["repo"], source.get("config"), revision=source["revision"],
                               split=source["split"], streaming=True)
        # Streaming shuffle permutes source shards as well as a bounded row buffer.
        dataset = dataset.shuffle(seed=config["seed"], buffer_size=config["shuffle_buffer"])
    for number, row in enumerate(dataset, 1):
        if number > config["max_rows_per_source"]:
            raise RuntimeError(f"{source['label']}: scan limit reached; inspect filters/budget")
        text = accepted(row, source, config)
        if text is None:
            continue
        key = content_key(text)
        if partition(key) != wanted:
            continue
        yield {"text": text, "key": key, "source": source["label"],
               "source_row": str(row.get("id", row.get("url", row.get("seed_data", key))))[:1000]}
