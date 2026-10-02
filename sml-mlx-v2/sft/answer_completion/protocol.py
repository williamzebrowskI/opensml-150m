"""Freeze the mixed continuation, step-512 weights and resume dependencies."""
import importlib.metadata
from pathlib import Path
from sml_v2.common import read_json,file_sha256,fingerprint
from .data import rebuild
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_answer_completion_v1'
def code_files():
    files=list((ROOT/'sml_v2').glob('*.py'))+list(DIR.glob('*.py'))
    for name in ('base_curriculum_v2','base_restart','natural_control','conversation_ab','skill_balance','transfer_control','reading_repair','response_repair','response_expansion'):
        files+=list((ROOT/'sft'/name).glob('*.py'))
    files+=list((ROOT/'evaluation/base_audit').glob('*.py'))
    return sorted(set(files))
def load_data(cfg,cancelled=lambda:False):return rebuild(cfg,cancelled)
def contract(cfg):
    count=sum(cfg['training_counts'].values())
    if count!=4096 or cfg['updates']*cfg['batch']!=count or cfg['epochs']!=1:raise ValueError('Wrong one-pass curriculum')
    parent=ROOT/cfg['source_bundle'];meta=read_json(parent/'model.safetensors.json')
    if cfg['source_step']!=512 or meta['step']!=512 or file_sha256(parent/'model.safetensors')!=cfg['source_model_sha256']:raise ValueError('Wrong SFT 512 parent')
    if sum(cfg['new_counts'].values())!=2048 or sum(cfg['rehearsal_counts'].values())!=2048 or cfg['batch']!=16:raise ValueError('Wrong new/rehearsal mix')
    manifest=read_json(DIR/'selection.json')
    if manifest['config']!=fingerprint(cfg):raise ValueError('Selection stale')
    paths=[*parent.glob('*'),*(ROOT/'tokenizer/bytebpe32k_v1').rglob('*'),DIR/'config.json',DIR/'selection.json',DIR/'review_prompts.json',DIR/'reserved_prompts.json',ROOT/'sft/base_restart/examples.json']
    paths+=list((ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea').glob('*'))
    paths+=list((ROOT/'sft/base_curriculum_v2').glob('*.json'))
    paths+=list((ROOT/'runs/sft_skill_balance_v1/step_0001920_628a0a65099b').glob('*'))
    paths+=list((ROOT/'sft/skill_balance/_tagger').rglob('*'))
    from sft.skill_balance.data import exclusions
    paths+=exclusions()[2]
    return dict(config=fingerprint(cfg),selection=file_sha256(DIR/'selection.json'),protected={str(p):file_sha256(p) for p in sorted(set(paths)) if p.is_file()},code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},runtime={p:importlib.metadata.version(p) for p in ('mlx','numpy','tokenizers','requests','pyarrow','fsspec')})
def plan(cfg):
    p=dict(source=str(ROOT/cfg['source_bundle']),source_step=512,final_step=512+cfg['updates'],new_examples=2048,rehearsal_examples=2048,training_examples=sum(cfg['training_counts'].values()),mix=cfg['training_counts'],epochs=1,updates=cfg['updates'],batch=cfg['batch'],microbatch=1,peak_lr=cfg['peak_lr'],final_lr=cfg['final_lr'],context=cfg['context'],objective='Equal-example assistant-only cross entropy including EOS; fresh FP32 AdamW',data='Pinned public TRAIN sources streamed; selected filtered examples in bounded RAM; no raw dataset disk cache',teacher=False,main_mac_only=True,automatic_promotion=False,output=str(OUTPUT))
    if (DIR/'selection.json').exists():p['accepted_data']=read_json(DIR/'selection.json')['stats']
    return p
