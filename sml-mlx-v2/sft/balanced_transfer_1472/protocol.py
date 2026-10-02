"""Freeze the 1472 parent, prepared data, implementation, and evaluation scope."""
import importlib.metadata
from pathlib import Path

from sml_v2.common import file_sha256, fingerprint, read_json

ROOT=Path(__file__).resolve().parents[2]
DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_balanced_transfer_1472_v1'


def code_files():
    paths=list((ROOT/'sml_v2').glob('*.py'))
    for name in ('balanced_transfer_1472','context_reasoning_1024','explanation_transfer_1024',
                 'skill_recovery_768','skill_balance','skill_balance_768','two_turn_640',
                 'transfer_control','reading_repair','base_curriculum_v2','clean_reply_ab'):
        paths+=list((ROOT/'sft'/name).glob('*.py'))
    return sorted(set(paths))


def validate(cfg):
    if (cfg['source_step'],cfg['updates'])!=(1472,512):
        raise ValueError('Expected a 512-update continuation of preserved 1472')
    if cfg['per_update']!={'new_answer':2,'new_choice':2,'old_choice':2,
                           'instruction':2,'reading':3,'explanation':2}:
        raise ValueError('Changed training mix')
    if abs(sum(cfg['loss_weights'].values())-1)>1e-9:
        raise ValueError('Loss weights must sum to one')
    if cfg['context']!=1024 or cfg['anchor_context']!=256:
        raise ValueError('Unexpected context')
    if cfg['evaluation_updates']!=list(range(0,513,64)):
        raise ValueError('Every 64-update evaluation checkpoint must be retained')
    if not 0<cfg['final_lr']<=cfg['peak_lr'] or not 0<cfg['warmup_updates']<cfg['updates']:
        raise ValueError('Invalid learning-rate schedule')
    parent=ROOT/cfg['source_bundle'];meta=read_json(parent/'model.safetensors.json')
    if meta['step']!=1472 or meta['source_step']!=1024 or meta['additional_updates']!=448:
        raise ValueError('Wrong 1472 lineage')
    if file_sha256(parent/'model.safetensors')!=cfg['source_model_sha256']:
        raise ValueError('Preserved 1472 weights changed')


def contract(cfg):
    validate(cfg)
    prepared=DIR/'prepared.json';selection=read_json(DIR/'selection.json')
    if not prepared.exists() or selection['config']!=fingerprint(cfg):
        raise ValueError('Run --prepare before --check/--run')
    paths=[ROOT/cfg['source_bundle']/'model.safetensors',
           ROOT/cfg['source_bundle']/'model.safetensors.json',
           DIR/'config.json',DIR/'selection.json',prepared,
           ROOT/'sft/context_reasoning_1024/selection_main.json',
           ROOT/'sft/explanation_transfer_1024/selection.json',
           ROOT/'sft/context_reasoning_1024/sample_review_record.json',
           ROOT/'sft/context_reasoning_1024/probes.json',
           ROOT/'sft/explanation_transfer_1024/probes.json',
           ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json']
    paths+=list((ROOT/'tokenizer/bytebpe32k_v1').rglob('*'))
    from sft.transfer_control.launch import prose_texts
    return dict(config=fingerprint(cfg),selection=fingerprint(selection),
        prepared=file_sha256(prepared),tokenizer=selection['tokenizer'],
        source=str(ROOT/cfg['source_bundle']),
        protected={str(p):file_sha256(p) for p in set(paths) if p.is_file()},
        code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},
        prose=fingerprint(prose_texts()),
        runtime={n:importlib.metadata.version(n) for n in ('mlx','numpy','tokenizers','pyarrow','requests')})


def verify(frozen):
    for path,sha in frozen['protected'].items():
        if file_sha256(path)!=sha:
            raise ValueError('Protected input changed: '+path)
    for path,sha in frozen['code'].items():
        if file_sha256(ROOT/path)!=sha:
            raise ValueError('Implementation changed: '+path)


def plan(cfg):
    return dict(source=str(ROOT/cfg['source_bundle']),source_step=1472,
        additional_updates=cfg['updates'],final_step=1472+cfg['updates'],
        independent_new_train_situations=2048,
        new_sources='1,024 screened CosmosQA TRAIN contexts + 1,024 screened WinoGrande XL TRAIN situations',
        per_update=cfg['per_update'],loss_weights=cfg['loss_weights'],
        method='Assistant-only CE + labeled choice ranking + frozen-1472 prose KL',
        peak_lr=cfg['peak_lr'],final_lr=cfg['final_lr'],
        checkpoint_steps=[1472+u for u in cfg['evaluation_updates']],
        selection='Review saved actual content, completion, and retention answers; no automatic promotion',
        main_mac_only=True,teacher=False,automatic_promotion=False,output=str(OUTPUT))
