"""Pin original 640, local dataset definitions and shared training implementation."""
from pathlib import Path
import importlib.metadata
from sml_v1.common import read_json,file_sha256,fingerprint
from sft.clean_reply_ab.protocol import code_files as shared_code
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_two_turn_640_v1'
def code_files():return sorted(set(shared_code()+list(DIR.glob('*.py'))))
def contract(cfg):
 if (cfg['source_step'],cfg['batch'],cfg['updates'])!=(640,12,128):raise ValueError('Invalid run geometry')
 review=read_json(DIR/'review.json')
 if review.get('status')!='sample-reviewed' or review['selection']!=fingerprint(read_json(DIR/'selection.json')):raise ValueError('Review missing or stale')
 source=ROOT/cfg['source_bundle']
 if read_json(source/'model.safetensors.json')['step']!=640 or file_sha256(source/'model.safetensors')!=cfg['source_model_sha256']:raise ValueError('Wrong High LR 640 parent')
 paths=[source/'model.safetensors',source/'model.safetensors.json',ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json',ROOT/'sft/clean_reply_ab/authored.json',ROOT/'sft/clean_reply_ab/dev.json']
 paths+=list((ROOT/'tokenizer/bytebpe32k_v1').rglob('*'))
 from sft.transfer_control.launch import prose_texts
 return dict(config=fingerprint(cfg),data={n:file_sha256(DIR/n) for n in ('config.json','selection.json','review.json')},protected={str(p):file_sha256(p) for p in paths if p.is_file()},code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},prose=fingerprint(prose_texts()),runtime={n:importlib.metadata.version(n) for n in ('mlx','numpy','tokenizers')})
def plan(cfg):
 return dict(source=str(ROOT/cfg['source_bundle']),source_step=640,final_step=768,updates=128,method='Full-parameter assistant-only SFT including EOS, fresh FP32 AdamW; not LoRA/RL/DPO',new_constructed_scenarios=256,new_targets=768,new_target_passes=1,rehearsal_unique=256,rehearsal_passes=3,total_exposures=1536,per_update='6 new targets + 6 previously consumed rehearsal targets',peak_lr=cfg['peak_lr'],final_lr=cfg['final_lr'],context=cfg['context'],data='Original constructed local conversations plus existing authored TRAIN examples. Generated in RAM; no network or new teacher.',evaluation='Unseen settings and phrasing; paired follow-ups; own-generated first replies; existing skill/prose diagnostics. Reserved test not evaluated.',limitations='Templates with varied content, not independent human dialogue collection; not proven to improve general chat.',main_mac_only=True,automatic_promotion=False,output=str(OUTPUT))
