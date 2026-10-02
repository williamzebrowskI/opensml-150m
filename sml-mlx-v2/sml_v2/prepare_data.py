"""Stream pinned sources once into verifiable, resume-safe local token pools."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import numpy as np

from .common import ROOT, atomic_json, file_sha256, load_corpus, lock, quotas, read_json
from .corpus import documents, identity
from .tokenization import Tokenizer
from .data import verify_data


def prepare(config, tokenizer, output, train_tokens, val_tokens, guard_tokens):
    spec = dict(corpus=identity(config), tokenizer=tokenizer.fingerprint,
                train_tokens=train_tokens, val_tokens=val_tokens, guard_tokens=guard_tokens)
    if tokenizer.manifest["corpus_fingerprint"] != spec["corpus"]:
        raise ValueError("Corpus changed since tokenizer fitting; choose the recipe before fitting")
    if tokenizer.vocab_size > 65536:
        raise ValueError("uint16 pools require vocabulary <=65536")
    marker = output / "spec.json"
    if marker.exists() and read_json(marker) != spec:
        raise ValueError("Data settings changed; use a new --output")
    atomic_json(marker, spec)
    records = []
    db = sqlite3.connect(output / "dedup.sqlite")
    try:
        db.execute("CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY)")
        db.execute("DELETE FROM seen")
        db.commit()
        for split, budget in (("train", train_tokens), ("validation", val_tokens)):
            targets = quotas(budget, config["sources"])
            for source in config["sources"]:
                label = source["label"]
                target = targets[label] + guard_tokens
                path = output / f"{split}-{label}.bin"
                index = path.with_suffix(".index.jsonl")
                meta = path.with_suffix(".meta.json")
                if meta.exists():
                    record = read_json(meta)
                    if file_sha256(path) != record["sha256"] or file_sha256(index) != record["index_sha256"]:
                        raise ValueError("Completed pool changed")
                    with index.open() as f:
                        for line in f:
                            db.execute("INSERT INTO seen VALUES (?)", (json.loads(line)["key"],))
                    db.commit()
                    records.append(record)
                    print(f"[reuse] {path.name}: {record['tokens']:,} tokens", flush=True)
                    continue
                partial = path.with_suffix(".bin.partial")
                index_partial = path.with_suffix(".index.partial")
                total, count, next_log = 0, 0, 1000000
                print(f"[prepare] {split}/{label}: target={target:,} tokens", flush=True)
                with partial.open("wb") as f, index_partial.open("w") as idx:
                    for row in documents(source, config, split):
                        added = db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (row["key"],)).rowcount
                        if not added:
                            continue
                        ids = tokenizer.encode(row["text"]) + [tokenizer.eos]
                        f.write(np.asarray(ids, dtype="<u2").tobytes())
                        idx.write(json.dumps(dict(key=row["key"], start=total, tokens=len(ids),
                                                  source_row=row["source_row"])) + "\n")
                        total += len(ids); count += 1
                        if total >= next_log:
                            print(f"[prepare] {split}/{label}: {total:,} tokens", flush=True)
                            next_log += 1000000
                            db.commit()
                        if total >= target:
                            break
                    for stream in (f, idx):
                        stream.flush(); os.fsync(stream.fileno())
                if total < target:
                    raise RuntimeError(f"Source {label} exhausted or filtering too restrictive")
                os.replace(partial, path); os.replace(index_partial, index)
                record = dict(file=path.name, index=index.name, source=label, split=split,
                              tokens=total, documents=count, sha256=file_sha256(path),
                              index_sha256=file_sha256(index))
                atomic_json(meta, record)
                db.commit()
                records.append(record)
    finally:
        db.close()
    atomic_json(output / "manifest.json", dict(format="sml-v2-data-v1", **spec,
        weights={s['label']: s['weight'] for s in config['sources']}, files=records,
        boundaries="EOS between documents; causal windows may cross documents; no padding",
        dedup="Exact whitespace/NFC-normalized document hashes, not near-duplicate families"))
    verify_data(output, tokenizer)
    print(f"[ready] Verified token pools: {output}. No model training started.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--config", type=Path, default=ROOT / "configs/corpus.json")
    p.add_argument("--pilot", type=Path, default=ROOT / "configs/pilot.json")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer/bytebpe32k_v1")
    p.add_argument("--output", type=Path, default=ROOT / "data/pilot_v1")
    args = p.parse_args()
    recipe = read_json(args.pilot)
    if recipe.get('data_mode') == 'hf_stream':
        message = 'This recipe uses live HF streaming. Do not prepare/save token pools; run scripts/launch_pretrain.py --run instead.'
        if args.run:
            p.error(message)
        print(message)
        return
    config = load_corpus(args.config)
    update = sum(recipe['batches']) * recipe['grad_accum'] * recipe['model']['max_seq_len']
    validation = max(1000000, 100 * recipe['eval_batches_per_source'] * recipe['eval_batch_size'] * recipe['model']['max_seq_len'])
    print(f"Pilot: {recipe['target_tokens']:,} tokens; source guards={2*update:,}; validation budget={validation:,}; output={args.output}")
    if not args.run:
        print("Plan only. Fit/review the tokenizer first, then add --run. No GPU/SSH is used.")
        return
    needed = 3 * (recipe['target_tokens'] + validation + 8 * update) + 4 * 1024**3
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < needed:
        raise OSError("Insufficient disk space for token pools, index and 4 GiB reserve")
    with lock(args.output / ".prepare.lock"):
        prepare(config, Tokenizer(args.tokenizer), args.output, recipe['target_tokens'], validation, 2 * update)


if __name__ == "__main__":
    main()
