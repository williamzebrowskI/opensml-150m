"""Prepare a bounded corpus-matched sample, then fit an independent CPU BPE."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from .common import ROOT, atomic_json, file_sha256, fingerprint, load_corpus, lock, quotas, read_json
from .corpus import documents, identity
from .tokenization import FIXTURES, SPECIALS, Tokenizer


def read_rows(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def prepare_samples(config, cache, train_bytes, dev_bytes):
    spec = {"corpus": identity(config), "train_bytes": train_bytes, "dev_bytes": dev_bytes,
            "version": 1}
    marker = cache / "spec.json"
    if marker.exists() and read_json(marker) != spec:
        raise ValueError("Sample settings changed; use a new --sample-dir")
    atomic_json(marker, spec)
    records, seen = [], set()
    for split, budget in [("train", train_bytes), ("tokenizer_dev", dev_bytes)]:
        target = quotas(budget, config["sources"])
        for source in config["sources"]:
            name = f"{split}-{source['label']}.jsonl"
            path = cache / name
            record_path = cache / (name + ".meta.json")
            if record_path.exists():
                record = read_json(record_path)
                if file_sha256(path) != record["sha256"]:
                    raise ValueError(f"Corrupt cached sample: {path}")
                for row in read_rows(path):
                    if row["key"] in seen:
                        raise ValueError("Duplicate in completed sample")
                    seen.add(row["key"])
                print(f"[reuse] {name}: {record['bytes']:,} text bytes", flush=True)
                records.append(record)
                continue
            partial = path.with_suffix(".partial")
            total, count, next_log = 0, 0, 5000000
            print(f"[sample] {source['label']} {split}: target={target[source['label']]:,} UTF-8 bytes", flush=True)
            with partial.open("w") as f:
                for row in documents(source, config, split):
                    if row["key"] in seen:
                        continue
                    seen.add(row["key"])
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")
                    total += len(row["text"].encode("utf-8"))
                    count += 1
                    if total >= next_log:
                        print(f"[sample] {source['label']} {split}: {total:,} bytes, {count:,} documents", flush=True)
                        next_log += 5000000
                    if total >= target[source['label']]:
                        break
                f.flush()
                os.fsync(f.fileno())
            if total < target[source['label']]:
                raise RuntimeError(f"{source['label']} exhausted before quota; no silent source substitution")
            os.replace(partial, path)
            record = dict(file=name, split=split, source=source['label'], bytes=total,
                          documents=count, sha256=file_sha256(path), requested_bytes=target[source['label']])
            atomic_json(record_path, record)
            records.append(record)
    atomic_json(cache / "manifest.json", dict(spec=spec, files=records,
                note="Whole documents; per-source byte quotas can overshoot by one document. Exact-content dedup only."))
    return records


def fit(config, cache, output, records, vocab_size=32000):
    from tokenizers import Tokenizer as Backend, models, trainers, pre_tokenizers, decoders
    recipe = dict(corpus=identity(config), sample_manifest=file_sha256(cache / "manifest.json"),
                  vocab_size=vocab_size, specials=SPECIALS, min_frequency=2,
                  tokenizers=importlib.metadata.version("tokenizers"), version=1)
    if output.exists():
        tokenizer = Tokenizer(output)
        if tokenizer.manifest["recipe"] != recipe:
            raise ValueError("Existing tokenizer differs; use a new --output, never overwrite it")
        print(f"[ready] Existing verified tokenizer: {output}", flush=True)
        return tokenizer
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name("." + output.name + "." + uuid.uuid4().hex)
    staging.mkdir()
    try:
        backend = Backend(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
            special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
        def texts():
            for record in records:
                if record["split"] == "train":
                    for row in read_rows(cache / record["file"]):
                        yield row["text"]
        count = sum(r['documents'] for r in records if r['split'] == 'train')
        print(f"[fit] CPU-only BPE: {count:,} documents -> {vocab_size:,} total IDs; no model training", flush=True)
        backend.train_from_iterator(texts(), trainer=trainer, length=count)
        if backend.get_vocab_size() != vocab_size:
            raise ValueError("Sample produced fewer vocabulary entries than requested; increase sample/diversity")
        backend.encode_special_tokens = True
        backend.save(str(staging / "tokenizer.json"))
        atomic_json(staging / "tokenizer_config.json", dict(tokenizer_class="PreTrainedTokenizerFast",
                    eos_token=SPECIALS[1], pad_token=SPECIALS[0], add_bos_token=False,
                    add_eos_token=False, clean_up_tokenization_spaces=False))
        atomic_json(staging / "corpus.json", config)
        atomic_json(staging / "manifest.json", dict(format="sml-v2-tokenizer-v1", recipe=recipe,
            vocab_size=vocab_size, special_ids=dict(pad=0, eos=1, turn_start=2, turn_end=3),
            corpus_fingerprint=identity(config), boundary_policy="raw-text-then-one-document-EOS-v1",
            sha256={p.name: file_sha256(p) for p in staging.iterdir() if p.is_file()}))
        tokenizer = Tokenizer(staging)
        for text in FIXTURES:
            tokenizer.assert_roundtrip(text)
        report = {"fixtures_passed": len(FIXTURES), "sources": {}, "vocab_size": vocab_size}
        for record in records:
            if record["split"] != "tokenizer_dev":
                continue
            size, tokens, docs = 0, 0, 0
            started = time.monotonic()
            for row in read_rows(cache / record["file"]):
                ids = tokenizer.assert_roundtrip(row["text"])
                size += len(row["text"].encode("utf-8")); tokens += len(ids); docs += 1
            report["sources"][record["source"]] = dict(bytes=size, tokens=tokens, documents=docs,
                bytes_per_token=size / max(1, tokens), seconds=time.monotonic()-started,
                roundtrip_failures=0)
        atomic_json(staging / "report.json", report)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"[ready] New independent tokenizer: {output}\n[report] {output / 'report.json'}", flush=True)
    print("No model training started. Review the report before preparing the pilot data.", flush=True)
    return Tokenizer(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="Explicitly download bounded text samples and train CPU tokenizer")
    p.add_argument("--config", type=Path, default=ROOT / "configs/corpus.json")
    p.add_argument("--sample-dir", type=Path, default=ROOT / "data/tokenizer_sample_v1")
    p.add_argument("--output", type=Path, default=ROOT / "tokenizer/bytebpe32k_v1")
    p.add_argument("--sample-bytes", type=int, default=500000000)
    p.add_argument("--dev-bytes", type=int, default=4000000)
    args = p.parse_args()
    config = load_corpus(args.config)
    if min(args.sample_bytes, args.dev_bytes) < 1000:
        p.error("Sample budgets must be at least 1000 bytes")
    print(json.dumps(dict(corpus_fingerprint=identity(config), weights={s['label']:s['weight'] for s in config['sources']},
        train_text_bytes=args.sample_bytes, development_text_bytes=args.dev_bytes,
        tokenizer_vocab=32000, output=str(args.output), run=args.run), indent=2))
    if not args.run:
        print("Plan only. Add --run to prepare samples and train the tokenizer; no GPU/SSH is used.")
        return
    if shutil.disk_usage(ROOT).free < 3 * (args.sample_bytes + args.dev_bytes) + 2 * 1024**3:
        raise OSError("Need sample storage plus 2 GiB disk reserve")
    with lock(args.output.parent / ".tokenizer.lock"), lock(args.sample_dir / ".sample.lock"):
        records = prepare_samples(config, args.sample_dir, args.sample_bytes, args.dev_bytes)
        fit(config, args.sample_dir, args.output, records)


if __name__ == "__main__":
    main()

