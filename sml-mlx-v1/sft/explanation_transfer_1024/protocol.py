"""Pin independent 1024 continuation and all rebuild/training inputs."""
import importlib.metadata
from pathlib import Path
from sml_v1.common import read_json, file_sha256, fingerprint
from sft.skill_recovery_768.protocol import code_files as replay_code, contract as replay_contract

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
OUTPUT = ROOT/'runs/sft_explanation_transfer_1024_v1'


def code_files():
    return sorted(set(replay_code() + list(DIR.glob('*.py'))))


def validate(cfg):
    if (cfg['source_step'], cfg['updates']) != (1024, 512): raise ValueError('Expected 1024 + 512')
    if cfg['per_update'] != dict(explanation=6, commonsense=2, instruction=3, reading=4): raise ValueError('Changed mix')
    if abs(sum(cfg['loss_weights'].values())-1.) > 1e-9: raise ValueError('Loss weights must sum to one')
    if cfg['context'] != 1024 or cfg['checkpoint_every'] != 64: raise ValueError('Changed geometry')
    if cfg['evaluation_updates'] != list(range(0, 513, 64)): raise ValueError('Every saved stage must be evaluated')
    parent = ROOT/cfg['source_bundle']
    meta = read_json(parent/'model.safetensors.json')
    if meta['step'] != 1024 or meta['source_step'] != 768: raise ValueError('Wrong parent lineage')
    if file_sha256(parent/'model.safetensors') != cfg['source_model_sha256']: raise ValueError('Parent weights changed')


def contract(cfg):
    validate(cfg)
    selection = read_json(DIR/'selection.json'); review = read_json(DIR/'review.json')
    if selection['config'] != fingerprint(cfg) or review['selection'] != fingerprint(selection) or review['status'] != 'sample-reviewed':
        raise ValueError('Data selection/review stale')
    old = replay_contract(read_json(ROOT/'sft/skill_recovery_768/config.json'))
    protected = dict(old['protected'])
    paths = [DIR/name for name in ('config.json','selection.json','review.json','probes.json','rejections.json')]
    paths += [ROOT/cfg['source_bundle']/name for name in ('model.safetensors','model.safetensors.json')]
    paths += [ROOT/'sft/skill_recovery_768'/name for name in ('config.json','selection.json','review.json','probes.json')]
    protected.update({str(p): file_sha256(p) for p in paths})
    return dict(config=fingerprint(cfg), selection=fingerprint(selection), source=str(ROOT/cfg['source_bundle']),
        tokenizer=selection['tokenizer'], protected=protected,
        code={str(p.relative_to(ROOT)): file_sha256(p) for p in code_files()},
        replay_contract=fingerprint(old), prose=old['prose'],
        runtime={p: importlib.metadata.version(p) for p in ('mlx','numpy','tokenizers','requests','pyarrow')})


def verify(frozen):
    for path, sha in frozen['protected'].items():
        if file_sha256(path) != sha: raise ValueError('Protected input changed: '+path)
    for path, sha in frozen['code'].items():
        if file_sha256(ROOT/path) != sha: raise ValueError('Code changed: '+path)
    if frozen['runtime'] != {p: importlib.metadata.version(p) for p in frozen['runtime']}:
        raise ValueError('Runtime changed')


def plan(cfg):
    return dict(source=str(ROOT/cfg['source_bundle']),source_step=1024,additional_updates=512,final_step=1536,
        method='Assistant-only explanation SFT + existing skill rehearsal/ranking + frozen-parent prose KL; no RL',
        new_data='3,072 independent e-SNLI TRAIN scenes; balanced supported/contradicted/unknown; human explanations',
        per_update=cfg['per_update'], loss_weights=cfg['loss_weights'], kl_weight=cfg['kl_weight'],
        peak_lr=cfg['peak_lr'],final_lr=cfg['final_lr'],optimizer='fresh FP32 AdamW; FP32 1024 weights',
        checkpoint_steps=[1024+u for u in cfg['evaluation_updates']], all_checkpoints_retained=True,
        main_mac_only=True,teacher=False,raw_dataset_disk_cache=False,automatic_promotion=False,
        selection='Review conclusion correctness AND actual explanation quality; preserve old skills. Reserved test only after frozen selection.',
        output=str(OUTPUT))
