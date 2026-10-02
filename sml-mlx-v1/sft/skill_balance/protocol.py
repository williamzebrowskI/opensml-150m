"""Freeze parent, tokenizer, data, implementation, and evaluation boundaries."""
import importlib.metadata
from pathlib import Path
from sml_v1.common import read_json, file_sha256, fingerprint

ROOT=Path(__file__).resolve().parents[2]
DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_skill_balance_v1'


def code_files():
    paths=list((ROOT/'sml_v1').glob('*.py'))
    for name in ('skill_balance','transfer_control','reading_repair','response_repair','response_expansion'):
        paths+=list((ROOT/'sft'/name).glob('*.py'))
    paths+=list((ROOT/'evaluation/base_audit').glob('*.py'))
    return sorted(set(paths))


def validate(cfg):
    if cfg['source_step']!=896 or cfg['updates']!=1536: raise ValueError('Expected frozen 896 + 1536-update recipe')
    if cfg['per_update']!={'commonsense':4,'instruction':4,'reading':2}: raise ValueError('Family exposure changed')
    if abs(sum(cfg['loss_weights'].values())-1)>1e-9: raise ValueError('Invalid loss weights')
    if cfg['context']!=1024 or cfg['anchor_context']!=256: raise ValueError('Unexpected context')
    if not 0<cfg['ranking_temperature'] or not 0<=cfg['kl_weight']: raise ValueError('Invalid regularization')
    if cfg['evaluation_updates'][0]!=0 or cfg['evaluation_updates'][-1]!=cfg['updates']: raise ValueError('Evaluation boundaries')
    source=ROOT/cfg['source_bundle']; meta=read_json(source/'model.safetensors.json')
    if meta['step']!=896 or meta['training_format']!='plain-user-assistant-eos-v1': raise ValueError('Parent mismatch')
    if file_sha256(source/'model.safetensors')!=cfg['source_model_sha256']: raise ValueError('Parent weights changed')


def contract(cfg):
    validate(cfg)
    selection=read_json(DIR/'selection.json')
    if selection['config']!=fingerprint(cfg): raise ValueError('Data selection is stale; prepare first')
    protected={}
    directories=[ROOT/cfg['source_bundle'],ROOT/'runs/grpo_reading_v1',ROOT/'tokenizer/bytebpe32k_v1']
    for directory in directories:
        for p in directory.rglob('*'):
            if p.is_file(): protected[str(p)]=file_sha256(p)
    for p in (ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json',
              ROOT/'runs/sft_reading_repair_v1/best.json',ROOT/'sft/reading_repair/selection.json',
              ROOT/'diagnostics/base_capability_20260924/items.json'):
        protected[str(p)]=file_sha256(p)
    from .data import exclusions
    for p in exclusions()[2]+[DIR/'probes.json']:
        protected[str(p)]=file_sha256(p)
    for p in (DIR/'_tagger').rglob('*'):
        if p.is_file(): protected[str(p)]=file_sha256(p)
    return dict(config=fingerprint(cfg),selection=fingerprint(selection),tokenizer=selection['tokenizer'],source=str(ROOT/cfg['source_bundle']),
        protected=protected,code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},
        runtime={p:importlib.metadata.version(p) for p in ('mlx','numpy','tokenizers','pyarrow','datasets')})


def verify(frozen):
    for name,sha in frozen['protected'].items():
        if file_sha256(name)!=sha: raise ValueError('Protected input changed: '+name)
    for name,sha in frozen['code'].items():
        if file_sha256(ROOT/name)!=sha: raise ValueError('Implementation changed: '+name)


def plan(cfg):
    return dict(source=str(ROOT/cfg['source_bundle']),source_step=896,additional_updates=cfg['updates'],
        final_step=896+cfg['updates'],exposures=cfg['updates']*10,new_task_rows=cfg['updates']*8,
        unique_new_source_examples=cfg['updates']*6,common_gen_views_per_source=2,
        main_mac_only=True,teacher=False,reference='frozen copy of OpenSML 896; no reference-model training',
        data='Pinned public TRAIN sources in bounded RAM; no raw dataset disk cache; no target benchmark items.',
        algorithm='Balanced supervised CE + correct-choice ranking + frozen-parent prose KL; not RL or DPO',
        sources=['CommonsenseQA','SocialIQA','CommonGen with verified formatting','previously consumed reading examples'],
        per_update=cfg['per_update'],loss_weights=cfg['loss_weights'],kl_weight=cfg['kl_weight'],
        peak_lr=cfg['peak_lr'],final_lr=cfg['final_lr'],optimizer='fresh FP32 AdamW from saved 896 weights',
        selection='development candidate only; actual-answer review and frozen selection precede reserved/public testing',
        output=str(OUTPUT),automatic_promotion=False)
