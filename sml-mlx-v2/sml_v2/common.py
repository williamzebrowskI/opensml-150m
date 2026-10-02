import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with tmp.open("w") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"Another process owns {path}") from e
        yield


def load_corpus(path):
    value = read_json(path)
    labels = [s["label"] for s in value["sources"]]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Source labels must be nonempty and unique")
    for s in value["sources"]:
        if not s["label"].isalnum() or type(s["weight"]) is not int or s["weight"] <= 0:
            raise ValueError("Invalid source label/weight")
        if not s.get("local_jsonl") and (len(s["revision"]) != 40 or not all(c in "0123456789abcdef" for c in s["revision"])):
            raise ValueError("Pin each Hugging Face source to a full commit SHA")
    for key in ("shuffle_buffer", "min_chars", "max_bytes", "max_rows_per_source"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError(f"Invalid corpus setting: {key}")
    if value.get("split_buckets", 10000) != 10000:
        raise ValueError("The fixed split policy uses exactly 10000 buckets")
    return value


def quotas(total, sources):
    weight = sum(s["weight"] for s in sources)
    result = {s["label"]: total * s["weight"] // weight for s in sources}
    for s in sources[:total - sum(result.values())]:
        result[s["label"]] += 1
    return result
