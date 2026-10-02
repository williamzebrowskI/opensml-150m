"""Freeze one-pass mixed curriculum, base weights and resume dependencies."""
import importlib.metadata
from pathlib import Path
from sml_v1.common import read_json,file_sha256,fingerprint
from .data import rebuild
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_base_curriculum_v1'
def code_files():
    files=list((ROOT/'sml_v1').glob('*.py'))+list(DIR.glob('*.py'))
    for name in ('base_restart','natural_control','conversation_ab','skill_balance','transfer_control','reading_repair','response_repair','response_expansion'):
        files+=list((ROOT/'sft'/name).glob('*.py'))
    files+=list((ROOT/'evaluation/base_audit').glob('*.py'))
    return sorted(set(files))
def load_data(cfg,cancelled=lambda:False):return rebuild(cfg,cancelled)
def contract(cfg):
    count=sum(cfg['training_counts'].values())
    if count!=8192 or cfg['updates']*cfg['batch']!=count or cfg['epochs']!=1:raise ValueError('Wrong one-pass curriculum')
    parent=ROOT/cfg['source_bundle'];meta=read_json(parent/'model.safetensors.json')
    if cfg['source_step']!=73243 or meta['step']!=73243 or file_sha256(parent/'model.safetensors')!=cfg['source_model_sha256']:raise ValueError('Wrong pretrained parent')
    manifest=read_json(DIR/'selection.json')
    if manifest['config']!=fingerprint(cfg):raise ValueError('Selection stale')
    paths=[*parent.glob('*'),*(ROOT/'tokenizer/bytebpe32k_v1').rglob('*'),DIR/'config.json',DIR/'selection.json',DIR/'review_prompts.json',ROOT/'sft/base_restart/examples.json']
    paths+=list((ROOT/'runs/sft_skill_balance_v1/step_0001920_628a0a65099b').glob('*'))
    paths+=list((ROOT/'sft/skill_balance/_tagger').rglob('*'))
    from sft.skill_balance.data import exclusions
    paths+=exclusions()[2]
    return dict(config=fingerprint(cfg),selection=file_sha256(DIR/'selection.json'),protected={str(p):file_sha256(p) for p in sorted(set(paths)) if p.is_file()},code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},runtime={p:importlib.metadata.version(p) for p in ('mlx','numpy','tokenizers','requests','pyarrow','fsspec')})
def plan(cfg):
    p=dict(source=str(ROOT/cfg['source_bundle']),source_step=73243,training_examples=sum(cfg['training_counts'].values()),mix=cfg['training_counts'],epochs=1,updates=cfg['updates'],batch=cfg['batch'],microbatch=1,peak_lr=cfg['peak_lr'],final_lr=cfg['final_lr'],context=cfg['context'],objective='Equal-example assistant-only cross entropy including EOS; fresh FP32 AdamW',data='Pinned public TRAIN sources streamed; selected filtered examples in bounded RAM; no raw dataset disk cache',teacher=False,main_mac_only=True,automatic_promotion=False,output=str(OUTPUT))
    if (DIR/'selection.json').exists():p['accepted_data']=read_json(DIR/'selection.json')['stats']
    return p
