"""Verify checkpoint bytes and read metadata without importing MLX or weights."""
import json
import gzip
from pathlib import Path

from .common import file_sha256

FORMAT='sml-pretrain-bundle-v1'


def verify_bundle_metadata(checkpoint):
    path=Path(checkpoint).absolute()
    if path.is_symlink():raise ValueError('Checkpoint path must not be a symlink')
    if path.suffix=='.json':
        pointer=json.loads(path.read_text())
        name=pointer.get('bundle','')
        if pointer.get('format')!=FORMAT or not name.startswith('step_') or Path(name).name!=name:
            raise ValueError('Invalid checkpoint pointer')
        path=path.parent/name
    if path.is_symlink():raise ValueError('Checkpoint bundle must not be a symlink')
    manifest=json.loads((path/'manifest.json').read_text())
    required={'model.safetensors','model.safetensors.optimizer.safetensors','model.safetensors.json',
              'model.safetensors.rank0.data_state.json.gz'}
    if manifest.get('format')!=FORMAT or not required<=manifest.get('files',{}).keys():
        raise ValueError('Incomplete resume bundle')
    for name,info in manifest['files'].items():
        if Path(name).name!=name:raise ValueError('Unsafe checkpoint filename')
        f=path/name
        if f.is_symlink() or f.stat().st_size!=info['bytes'] or file_sha256(f)!=info['sha256']:
            raise ValueError('Checkpoint checksum mismatch: '+name)
    metadata=json.loads((path/'model.safetensors.json').read_text())
    if metadata['step']!=manifest['step']:raise ValueError('Checkpoint step differs from manifest')
    with gzip.open(path/'model.safetensors.rank0.data_state.json.gz','rt') as stream:
        cursor=json.load(stream)
    offsets=cursor['stream_state']['offsets']
    if (cursor['step']!=metadata['step'] or any(type(n) is not int or n<0 for n in offsets.values())
            or sum(offsets.values())!=metadata['tokens']):
        raise ValueError('Checkpoint cursor/token accounting differs')
    return metadata
