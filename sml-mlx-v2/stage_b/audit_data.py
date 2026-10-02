#!/usr/bin/env python3
"""Bounded CPU-only live selection audit; never loads a model or starts training."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v2.common import atomic_json, file_sha256, load_corpus
from sml_v2.stage_b_data import select_document
from sml_v2.stream import SourceCursor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=int, default=500)
    p.add_argument('--source')
    args = p.parse_args()
    os.nice(10)
    corpus = ROOT/'stage_b/corpus.json'
    cfg = load_corpus(corpus)
    selection_hash = file_sha256(ROOT/'sml_v2/stage_b_data.py')
    parent = ROOT/'experiments/stage_b_v1/parent/step_0073008_58d174e2c177'
    with gzip.open(parent/'model.safetensors.rank0.data_state.json.gz','rt') as f:
        saved = json.load(f)['stream_state']
    out = ROOT/'diagnostics/stage_b_20260923'
    out.mkdir(exist_ok=True)
    for source in cfg['sources']:
        label = source['label']
        if args.source and args.source != label:
            continue
        cursor = SourceCursor(source, 3, saved['sources'].get(label))
        print(f'[audit] {label}: opening at parent row {cursor.rows}',flush=True)
        counts, candidates = Counter(), []
        try:
            for i in range(args.rows):
                row = cursor.next()
                text, keys, reason = select_document(row,source,cfg)
                counts[reason or 'accepted'] += 1
                if text:
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    candidates.append(dict(sha256=digest, id=row.get('id') or row.get('url'), text=text))
            # Reopening from the saved HF position must return the same next row.
            state = cursor.state_dict()
            first = cursor.next()
            retry = SourceCursor(source,3,state)
            try:
                assert first == retry.next(), 'HF cursor restart differs'
            finally:
                retry.close()
        finally:
            cursor.close()
        samples = sorted(candidates, key=lambda x:x['sha256'])[:8]
        record = dict(source=label, scanned=args.rows, accepted=counts['accepted'],
                      counts=dict(counts), hf_cursor_restart=True,
                      corpus_sha256=file_sha256(corpus),
                      selection_sha256=selection_hash,
                      samples=samples)
        atomic_json(out/(label+'.json'),record)
        print(json.dumps({k:v for k,v in record.items() if k!='samples'}),flush=True)
    if file_sha256(ROOT/'sml_v2/stage_b_data.py') != selection_hash:
        raise RuntimeError('Selection implementation changed during the audit; rerun it')


if __name__ == '__main__':
    main()
    # All audit output is synchronously committed. Third-party HF reader pools
    # may otherwise wait indefinitely at interpreter shutdown after early close.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
