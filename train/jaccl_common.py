"""CPU-only validation and filesystem helpers for the isolated JACCL benchmark."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess

INPUT_FORMAT = 'sml-jaccl-benchmark-input-v1'
CHECKPOINT_FORMAT = 'sml-jaccl-benchmark-checkpoint-v1'
PACKAGES = ('mlx', 'mlx-metal', 'numpy', 'datasets', 'tokenizers')
ULTRA_BATCHES = (32, 40, 48)
PEER_BATCHES = (8, 12, 16, 20, 24)
BALANCE_CASE = (35, 23, 23, 23)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temp.open('w') as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def file_record(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha256(path))


def verify_files(root, files):
    root = Path(root).resolve()
    for name, expected in files.items():
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe input filename: {name}')
        if file_record(path) != expected:
            raise ValueError(f'Integrity check failed: {path}')


def verify_input(root):
    root = Path(root)
    manifest = json.loads((root / 'input.json').read_text())
    if manifest.get('format') != INPUT_FORMAT:
        raise ValueError('Not a JACCL benchmark input')
    required = {'bundle/model.safetensors', 'bundle/model.safetensors.optimizer.safetensors',
                'bundle/model.safetensors.json', 'bundle/manifest.json',
                'code/train/jaccl_worker.py', 'code/train/jaccl_common.py',
                'code/tokenizer/cosmo2/manifest.json', 'code/tokenizer/cosmo2/tokenizer.json'}
    if not required <= manifest['files'].keys():
        raise ValueError('Incomplete benchmark input')
    verify_files(root, manifest['files'])
    return dict(step=manifest['step'], model_sha256=manifest['files']['bundle/model.safetensors']['sha256'],
                input_sha256=sha256(root / 'input.json'),
                packages={name: importlib.metadata.version(name) for name in PACKAGES})


def parse_splits(value):
    splits = [int(part) for part in value.split(',')]
    if not splits or any(n not in PEER_BATCHES for n in splits) or splits != sorted(set(splits)):
        raise ValueError('Splits must be an increasing subset of 8,12,16,20,24')
    return splits


def validate_batches(batches):
    if batches == [32] or tuple(batches) == BALANCE_CASE:
        return batches
    if (len(batches) != 4 or batches[0] not in ULTRA_BATCHES
            or batches[1] not in PEER_BATCHES or len(set(batches[1:])) != 1):
        raise ValueError('Use batch 32, U/P/P/P (U=32,40,48; P=8,12,16,20,24), or the 35/23/23/23 benchmark')
    return batches


def parse_cases(value):
    cases = [validate_batches([int(part) for part in case.split('/')])
             for case in value.split(',')]
    if any(len(case) != 4 for case in cases) or len({tuple(case) for case in cases}) != len(cases):
        raise ValueError('Cases must be unique four-Mac splits, e.g. 32/24/24/24,40/24/24/24')
    return cases


def tokens_per_update(batches, accum=2, seq_len=256):
    if not batches or any(b <= 0 for b in batches) or accum < 1 or seq_len < 1:
        raise ValueError('Invalid training shape')
    return sum(batches) * accum * seq_len


def rank_scale(rank, batches):
    # nn.average_gradients divides by world; undo that before applying token weights.
    return len(batches) * batches[rank] / sum(batches)


def training_processes():
    output = subprocess.check_output(['/bin/ps', '-axo', 'pid=,command='], text=True)
    pattern = re.compile(r'(?:/|\s)(?:train|train_sft|sft_jaccl|pretrain_jaccl|benchmark_training|profile_components|sample_checkpoint|evaluate_sft)\.py(?:\s|$)|-m\s+(?:train\.(?:train|train_sft|sft_jaccl|pretrain_jaccl|jaccl_worker)\b|mlx_lm\b)')
    matches = []
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 3 or int(fields[0]) == os.getpid():
            continue
        executable = Path(fields[1]).name.lower()
        # mlx.launch embeds its children's full command line but does no GPU
        # training itself. Workers are checked separately on every host.
        script = Path(fields[2].split()[0])
        if (script.name == 'mlx.launch'
                or script in (Path.home() / 'sml-mlx/sml-mlx-v1/train/jaccl_ring_transport.py',
                              Path.home() / 'sml-mlx/train/jaccl_ring_transport.py')):
            # The wired SSH launcher waits for rank 0 but never trains a model.
            continue
        if ('python' in executable or executable.startswith('mlx_lm')) and pattern.search(' '.join(fields[1:])):
            matches.append(line.strip())
    return matches
