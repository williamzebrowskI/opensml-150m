import importlib.metadata
from pathlib import Path
from sml_v2.common import read_json,file_sha256,fingerprint
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_conversation_ab_v1/A'
def code_files():
    paths=list((ROOT/'sml_v2').glob('*.py'))
    for name in ('conversation_ab','skill_balance','transfer_control','reading_repair','response_repair','response_expansion'):
        paths+=list((ROOT/'sft'/name).glob('*.py'))
    paths+=list((ROOT/'evaluation/base_audit').glob('*.py'))
    return sorted(set(paths))
def contract(cfg):
    if cfg['branch']!='A' or cfg['updates']!=120 or cfg['epochs']!=3 or cfg['batch']!=4:raise ValueError('Unexpected frozen pilot settings')
    parent=ROOT/cfg['source_bundle'];meta=read_json(parent/'model.safetensors.json')
    if meta['step']!=1920 or file_sha256(parent/'model.safetensors')!=cfg['source_model_sha256']:raise ValueError('Wrong parent')
    sel=read_json(DIR/'selection.json')
    if sel['train_count']*cfg['epochs']!=cfg['updates']*cfg['batch']:raise ValueError('Exposure mismatch')
    paths=[*parent.glob('*'),*(ROOT/'tokenizer/bytebpe32k_v1').rglob('*')]
    paths += [DIR/x for x in ('config.json','selection.json','originals.json','review.json')]
    # Include all local inputs used by the reused source-selection/eval code.
    for name in ('skill_balance','reading_repair','response_repair','response_expansion','transfer_control'):
        paths+=list((ROOT/'sft'/name).glob('*.json'))
    paths+=list((ROOT/'sft/skill_balance/_tagger').rglob('*'))
    paths += [ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json',ROOT/'runs/sft_reading_repair_v1/step_0000896_b6e5f5f75873/model.safetensors.json',ROOT/'diagnostics/base_capability_20260924/items.json',ROOT/'evaluation/reading_transfer/protocol.json',ROOT/'sft/natural_control/probes.json',ROOT/'diagnostics/playground_sft_comparison_20260924/protocol.json']
    paths+=list((ROOT/'evaluation/full_benchmarks/_cache').glob('*.json'))
    for name in ('generation_comparison_1920_v1','continuation_audit_1920_v1'):
        paths.append(ROOT/'diagnostics'/name/'responses.jsonl')
    return dict(config=fingerprint(cfg),selection=fingerprint(sel),protected={str(p):file_sha256(p) for p in sorted(set(paths)) if p.is_file()},code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},runtime={p:importlib.metadata.version(p) for p in ('mlx','numpy','tokenizers','pyarrow','datasets')})
def plan(cfg):
    return dict(branch='A: ordinary supervised training',source=str(ROOT/cfg['source_bundle']),source_step=1920,updates=cfg['updates'],final_step=2040,unique_examples=160,epochs=3,exposures=480,batch=4,mixture={'oasst2':96,'original':24,'rehearsal':40},peak_lr=cfg['peak_lr'],final_lr=cfg['final_lr'],objective='Equal-example assistant-only CE including EOS; fresh FP32 AdamW; no unlikelihood, KL, RL or DPO',data='Pinned OASST2 streamed into bounded RAM; reviewed IDs only; no raw corpus disk cache',main_mac_only=True,teacher=False,automatic_promotion=False,output=str(OUTPUT))
