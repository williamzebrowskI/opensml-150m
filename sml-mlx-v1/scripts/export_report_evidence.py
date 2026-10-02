#!/usr/bin/env python3
"""Export bounded, read-only training-log evidence for the technical report."""

import argparse
import ast
import csv
import hashlib
import io
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB = '20260918_085811_b4e5c849'
EVAL = re.compile(
    r'^\[eval\s+(\d+)\] val_loss=([\d.]+) val_ppl=([\d.]+) sources=(\{.*\})$'
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through-step', type=int, required=True)
    parser.add_argument('--job', default=DEFAULT_JOB)
    args = parser.parse_args()
    job = ROOT / 'runs/full_15b_v1/jobs' / args.job
    recipe_path = job / 'input/recipe.json'
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    corpus = json.loads((job / 'input/corpus.json').read_text())
    weights = {s['label']: s['weight'] for s in corpus['sources']}
    tokens_per_update = (
        sum(recipe['batches']) * recipe['grad_accum'] * recipe['model']['max_seq_len']
    )
    initial_recipe_path = ROOT / 'runs/pilot_v1/jobs/20260917_222411_0c7bcdb4/input/recipe.json'
    initial_recipe = json.loads(initial_recipe_path.read_text())
    initial_update = (
        sum(initial_recipe['batches']) * initial_recipe['grad_accum']
        * initial_recipe['model']['max_seq_len']
    )
    if initial_update != tokens_per_update:
        raise ValueError('Token reconstruction requires unchanged update geometry')
    records, logs = [], []
    for phase, path in [
        ('initial', ROOT / 'runs/pilot_v1/train.log'),
        ('continuation', job / 'train.log'),
    ]:
        data = path.read_bytes()
        count = 0
        # Ignore a partial last line if the active log is being appended to.
        for line_number, line in enumerate(data.rsplit(b'\n', 1)[0].decode().splitlines(), 1):
            match = EVAL.fullmatch(line)
            if not match or int(match[1]) > args.through_step:
                continue
            step = int(match[1])
            values = ast.literal_eval(match[4])
            if values.keys() != weights.keys():
                raise ValueError(f'Unexpected validation sources: {path}:{line_number}')
            weighted = sum(values[k] * weights[k] for k in weights) / sum(weights.values())
            if abs(weighted - float(match[2])) > 0.000051:
                raise ValueError(f'Aggregate mismatch: {path}:{line_number}')
            records.append({
                'phase': phase, 'step': step, 'tokens': step * tokens_per_update,
                'logged_loss': match[2], 'logged_perplexity': match[3],
                **values, 'log': str(path.relative_to(ROOT)), 'line': line_number,
            })
            count += 1
        logs.append({'path': str(path.relative_to(ROOT)), 'snapshot_sha256': sha256(data),
                     'snapshot_bytes': len(data), 'exported_observations': count})
    if not records or max(r['step'] for r in records) != args.through_step:
        raise ValueError('Requested final evaluation step is absent from these logs')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    csv_bytes = buffer.getvalue().encode()
    output = ROOT / 'docs/tables' / f'validation_through_{args.through_step:07d}'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.csv').write_bytes(csv_bytes)
    evidence = {
        'format': 'opensml-report-evidence-v1', 'through_step': args.through_step,
        'tokens_per_update': tokens_per_update, 'observations': len(records),
        'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha256(recipe_bytes),
        'logs': logs, 'table_sha256': sha256(csv_bytes),
        'notes': [
            'Live log hashes describe the snapshot read and will change as training proceeds.',
            'Loss and perplexity retain printed precision; per-source losses retain log precision.',
            'Tokens are reconstructed from step and verified unchanged update geometry.',
            'All matching observations are retained, including repeated evaluations.',
        ],
    }
    output.with_suffix('.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(f'Exported {len(records)} observations through step {args.through_step}: {output}.csv')


if __name__ == '__main__':
    main()
