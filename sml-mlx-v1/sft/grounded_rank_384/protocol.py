"""Freeze parent identity, training selection, runtime and resumable experiment."""
import importlib.metadata
from pathlib import Path
from sml_v1.common import file_sha256, fingerprint, read_json

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
OUTPUT = ROOT / 'runs/sft_grounded_rank_384_v1'


def code_files():
    paths = list(DIR.glob('*.py')) + list((ROOT / 'sml_v1').glob('*.py'))
    for package in ('skill_recovery_768', 'skill_balance', 'transfer_control', 'reading_repair',
                    'response_repair', 'intact_smoltalk_base_pilot', 'natural_control'):
        paths += list((ROOT / 'sft' / package).glob('*.py'))
    paths += [ROOT / 'evaluation/full_benchmarks/prepare.py', ROOT / 'evaluation/full_benchmarks/spec.py']
    return sorted(set(paths))


def validate(cfg):
    if (cfg['source_step'], cfg['updates']) != (384, 256): raise ValueError('Expected 384 + 256 updates')
    if cfg['per_update'] != dict(grounded=4, instruction=3, replay=2, ranking=4):
        raise ValueError('Unexpected task mix')
    if set(cfg['loss_weights']) != set(cfg['per_update']) or abs(sum(cfg['loss_weights'].values()) - 1) > 1e-9:
        raise ValueError('Invalid family loss weights')
    if cfg['evaluation_updates'] != list(range(0, 257, 32)) or cfg['checkpoint_every'] != 32:
        raise ValueError('Expected retained 32-update development checkpoints')
    if cfg['context'] != 1024 or cfg['anchor_context'] != 256:
        raise ValueError('Unexpected training context')
    if not 0 < cfg['final_lr'] <= cfg['peak_lr'] <= 3e-6 or not 0 < cfg['warmup_updates'] < cfg['updates']:
        raise ValueError('Invalid learning-rate schedule')
    parent = ROOT / cfg['source_bundle']; meta = read_json(parent / 'model.safetensors.json')
    if (meta['step'], meta['source_step'], meta['source_sft_step'], meta['additional_updates']) != (384, 73243, 128, 384):
        raise ValueError('Wrong 384 model lineage')
    if file_sha256(parent / 'model.safetensors') != cfg['source_model_sha256']:
        raise ValueError('Preserved checkpoint 384 weights changed')


def contract(cfg):
    validate(cfg); selection = read_json(DIR / 'selection.json')
    if selection['config'] != fingerprint(cfg): raise ValueError('Prepared configuration changed')
    review = read_json(DIR / 'review.json')
    if review.get('status') != 'accepted' or review['sample_sha256'] != file_sha256(DIR / 'review_samples.json'):
        raise ValueError('Prepared training samples require a recorded content review')
    from .data import SOURCES
    paths = [ROOT / cfg['source_bundle'] / n for n in ('model.safetensors', 'model.safetensors.json', 'manifest.json', 'model.safetensors.optimizer.safetensors')]
    paths += [DIR / n for n in ('config.json', 'prepared.json', 'selection.json', 'review_samples.json', 'review.json')]
    paths += [DIR / '_sources' / (name + '.parquet') for name in SOURCES]
    paths += [Path(p) for p in selection['exclusions']]
    paths += [ROOT / 'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json']
    paths += list((ROOT / 'tokenizer/bytebpe32k_v1').rglob('*'))
    for path, sha in selection['exclusions'].items():
        if file_sha256(path) != sha: raise ValueError('Exclusion protocol changed')
    return dict(config=fingerprint(cfg), selection=fingerprint(selection), tokenizer=selection['tokenizer'],
        source=str(ROOT / cfg['source_bundle']), source_sha256=cfg['source_model_sha256'],
        protected={str(p): file_sha256(p) for p in set(paths) if p.is_file()},
        code={str(p.relative_to(ROOT)): file_sha256(p) for p in code_files()},
        runtime={n: importlib.metadata.version(n) for n in ('mlx', 'numpy', 'tokenizers', 'pyarrow', 'requests')})


def verify(frozen):
    for path, sha in frozen['protected'].items():
        if file_sha256(path) != sha: raise ValueError('Protected input changed: ' + path)
    for path, sha in frozen['code'].items():
        if file_sha256(ROOT / path) != sha: raise ValueError('Implementation changed: ' + path)


def plan(cfg):
    return dict(source=str(ROOT / cfg['source_bundle']), source_step=384, additional_updates=256, final_step=640,
        per_update=cfg['per_update'], loss_weights=cfg['loss_weights'], kl_weight=cfg['kl_weight'],
        method='Family-balanced grounded/instruction/replay SFT + character-normalized labeled choice ranking + frozen-384 TRAIN-prose KL',
        optimizer='Fresh FP32 AdamW; different objective, not continuation of old optimizer',
        peak_lr=cfg['peak_lr'], final_lr=cfg['final_lr'], retained_steps=[384 + u for u in cfg['evaluation_updates']],
        reserved_test_evaluated=False, official_benchmarks_used_for_training=False,
        automatic_promotion=False, output=str(OUTPUT))
