"""Freeze source checkpoint, data, code and exact resumption contracts."""
import importlib.metadata
from pathlib import Path
from sml_v2.common import read_json,file_sha256,fingerprint
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_clean_reply_ab_v1'

def arm_config(cfg,arm):
    return dict(cfg,**cfg['arms'][arm],arm=arm)

def code_files():
    files=list((ROOT/'sml_v2').glob('*.py'))+list(DIR.glob('*.py'))
    for name in ('answer_completion','base_curriculum_v2','base_restart','natural_control','conversation_ab','skill_balance','transfer_control','reading_repair','response_repair','response_expansion'):
        files+=list((ROOT/'sft'/name).glob('*.py'))
    files+=list((ROOT/'evaluation/base_audit').glob('*.py'))
    return sorted(set(files))

def contract(cfg):
    if cfg['epochs']!=4 or cfg['updates']!=128 or cfg['batch']!=12 or cfg['source_step']!=512:raise ValueError('Unsupported experiment geometry')
    source=ROOT/cfg['source_bundle'];meta=read_json(source/'model.safetensors.json')
    if meta['step']!=512 or file_sha256(source/'model.safetensors')!=cfg['source_model_sha256']:raise ValueError('Wrong 512 parent')
    if read_json(DIR/'replay_review.json')['status']!='reviewed':raise ValueError('Rehearsal review incomplete')
    paths=list(source.glob('*'))+list((ROOT/'tokenizer/bytebpe32k_v1').rglob('*'))
    paths+=list((ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea').glob('*'))
    paths+=list((ROOT/'runs/sft_skill_balance_v1/step_0001920_628a0a65099b').glob('*'))
    paths+=list((ROOT/'sft/base_curriculum_v2').glob('*.json'))
    paths+=list((ROOT/'sft/skill_balance/_tagger').rglob('*'))
    from sft.skill_balance.data import exclusions
    paths+=exclusions()[2]
    from sft.transfer_control.launch import prose_texts
    return dict(config=fingerprint(cfg),data={n:file_sha256(DIR/n) for n in ('config.json','authored.json','dev.json','test.json','selection.json','replay_review.json')},
        protected={str(p):file_sha256(p) for p in sorted(set(paths)) if p.is_file()},code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},prose=fingerprint(prose_texts()),
        runtime={p:importlib.metadata.version(p) for p in ('mlx','numpy','tokenizers','requests','pyarrow','fsspec')})

def plan(cfg):
    return dict(source=str(ROOT/cfg['source_bundle']),source_step=512,arms=cfg['arms'],independent_initialization=True,
        new_content_groups=128,new_training_phrasings=256,annotated_parent_rehearsal=128,unique_examples=384,epochs=4,exposures_per_arm=1536,updates_per_arm=128,final_step_per_arm=640,
        per_update='8 original examples + 4 previously consumed skill examples',objective='Same equal-example assistant CE including EOS in both arms; fresh FP32 AdamW',
        data='Authored local tasks; public parent TRAIN selections streamed into bounded RAM, no raw corpus cache',teacher=False,
        checks='trained examples, untrained phrasings of trained tasks, new development tasks, retained skills and prose',reserved_test='32 separate prompts, never evaluated automatically',main_mac_only=True,automatic_promotion=False,output=str(OUTPUT))
