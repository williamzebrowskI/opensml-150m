"""Freeze 768, training/evaluation selections, shared code and immutable input files."""
import importlib.metadata
from pathlib import Path
from sml_v1.common import read_json,file_sha256,fingerprint
from sft.clean_reply_ab.protocol import code_files as shared_code
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_skill_balance_768_v1'
def code_files():
    return sorted(set(shared_code()+list(DIR.glob('*.py'))+list((ROOT/'sft/two_turn_640').glob('*.py'))))
def validate(cfg):
    if (cfg['source_step'],cfg['updates'])!=(768,512):raise ValueError('Expected 768 + 512 updates')
    if cfg['per_update']!={'commonsense':6,'instruction':3,'reading':3}:raise ValueError('Changed mix')
    if abs(sum(cfg['loss_weights'].values())-1)>1e-9:raise ValueError('Loss weights')
    if cfg['context']!=1024 or cfg['anchor_context']!=256:raise ValueError('Context')
    meta=read_json(ROOT/cfg['source_bundle']/'model.safetensors.json')
    if meta['step']!=768 or meta['source_step']!=640 or meta['additional_updates']!=128:raise ValueError('Wrong parent')
    if file_sha256(ROOT/cfg['source_bundle']/'model.safetensors')!=cfg['source_model_sha256']:raise ValueError('Parent hash changed')
def contract(cfg):
    validate(cfg);selection=read_json(DIR/'selection.json');review=read_json(DIR/'review.json')
    if selection['config']!=fingerprint(cfg) or review['selection']!=fingerprint(selection) or review['status']!='sample-reviewed':raise ValueError('Stale data/review')
    paths=[ROOT/cfg['source_bundle']/'model.safetensors',ROOT/cfg['source_bundle']/'model.safetensors.json',DIR/'config.json',DIR/'selection.json',DIR/'review.json',DIR/'probes.json',ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json']
    paths+=list((ROOT/'tokenizer/bytebpe32k_v1').rglob('*'))+list((ROOT/'sft/skill_balance/_tagger').rglob('*'))
    paths+=[ROOT/'sft/two_turn_640'/n for n in ('config.json','selection.json','review.json')]
    paths+=[ROOT/'sft/clean_reply_ab'/n for n in ('authored.json','dev.json')]
    from .data import exclusions
    paths+=exclusions()[2]
    from sft.transfer_control.launch import prose_texts
    return dict(config=fingerprint(cfg),selection=fingerprint(selection),tokenizer=selection['tokenizer'],source=str(ROOT/cfg['source_bundle']),protected={str(p):file_sha256(p) for p in set(paths) if p.is_file()},code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},prose=fingerprint(prose_texts()),runtime={n:importlib.metadata.version(n) for n in ('mlx','numpy','tokenizers','pyarrow','requests')})
def verify(frozen):
    for p,sha in frozen['protected'].items():
        if file_sha256(p)!=sha:raise ValueError('Protected file changed: '+p)
    for p,sha in frozen['code'].items():
        if file_sha256(ROOT/p)!=sha:raise ValueError('Code changed: '+p)
def plan(cfg):
    return dict(source=str(ROOT/cfg['source_bundle']),source_step=768,additional_updates=512,final_step=1280,exposures=6144,public_commonsense=3072,public_instruction=1536,replay_exposures=1536,replay_unique=1024,per_update=cfg['per_update'],replay_note='Internal reading family = two-turn and clean-reply targets already consumed by 768',sources=['CommonsenseQA TRAIN','SocialIQA TRAIN','CommonGen TRAIN, one formatted view per source','768 replay'],data='Pinned public files streamed into bounded RAM; selected rows kept in RAM, no raw dataset disk cache. No math/code tasks.',method='Family-balanced assistant CE + correct-choice ranking + frozen-parent KL on training-only social narratives/questions',loss_weights=cfg['loss_weights'],kl_weight=cfg['kl_weight'],peak_lr=cfg['peak_lr'],final_lr=cfg['final_lr'],optimizer='fresh FP32 AdamW; frozen 768 anchor',main_mac_only=True,teacher=False,automatic_promotion=False,selection='Development candidate only. Review complete answers before freezing a test/public benchmark candidate.',output=str(OUTPUT))
