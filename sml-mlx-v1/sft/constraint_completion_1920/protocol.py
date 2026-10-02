"""Pin parent identity, source selections, tokenization, code and runtime."""
import importlib.metadata
from pathlib import Path
from sml_v1.common import read_json, file_sha256, fingerprint
from sft.skill_balance.protocol import code_files as inherited_code

ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_constraint_completion_1920_v1'
INPUTS=('config.json','selection.json','review.json','review_samples.json','probes.json')


def code_files():
    return sorted(set(inherited_code()+list(DIR.glob('*.py'))))


def validate(cfg):
    if (cfg['source_step'],cfg['updates'])!=(1920,128):raise ValueError('Expected preserved 1920 plus 128 updates')
    if cfg['per_update']!=dict(complete=8,commonsense=2,instruction=3,reading=3):raise ValueError('Changed balanced batch')
    if cfg['loss_weights']!=dict(complete=.5,commonsense=0.,instruction=.15,reading=.20,ranking=.15):raise ValueError('Changed objective')
    if cfg['evaluation_updates']!=[0,16,32,64,96,128] or cfg['checkpoint_every']!=16:raise ValueError('Changed review/checkpoint schedule')
    if cfg['context']!=1024 or cfg['anchor_context']!=256:raise ValueError('Changed context')
    if not 0<cfg['final_lr']<=cfg['peak_lr'] or not 0<cfg['warmup_updates']<cfg['updates']:raise ValueError('LR schedule')
    p=ROOT/cfg['source_bundle'];m=read_json(p/'model.safetensors.json')
    if (m['step'],m['source_step'],m['additional_updates'])!=(1920,896,1024):raise ValueError('Wrong ancestry')
    if file_sha256(p/'model.safetensors')!=cfg['source_model_sha256']:raise ValueError('Preserved 1920 changed')


def contract(cfg):
    validate(cfg);sel=read_json(DIR/'selection.json');review=read_json(DIR/'review.json')
    if sel['config']!=fingerprint(cfg) or sel['review']!=fingerprint(review) or review['status']!='sample-reviewed':raise ValueError('Data/review not frozen')
    if review['sample_sha256']!=file_sha256(DIR/'review_samples.json'):raise ValueError('Review sample changed')
    from sft.skill_balance.data import exclusions
    from sft.reading_repair.source import evaluation_exclusions
    from sft.transfer_control.launch import prose_texts
    paths=[DIR/n for n in INPUTS]+[ROOT/cfg['source_bundle']/n for n in ('model.safetensors','model.safetensors.json')]
    paths+=[ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json',ROOT/'diagnostics/base_capability_20260924/items.json']
    paths+=exclusions()[2]+evaluation_exclusions()[2]
    for name in ('skill_balance','reading_repair','response_repair','response_expansion'):
        paths+=list((ROOT/'sft'/name).glob('*.json'))
    paths+=list((ROOT/'sft/skill_balance/_tagger').rglob('*'))+list((ROOT/'tokenizer/bytebpe32k_v1').rglob('*'))
    paths+=[ROOT/read_json(ROOT/'sft/skill_balance/config.json')['source_bundle']/'model.safetensors.json']
    return dict(config=fingerprint(cfg),selection=fingerprint(sel),source=str(ROOT/cfg['source_bundle']),tokenizer=sel['tokenizer'],
        protected={str(p):file_sha256(p) for p in sorted(set(paths)) if p.is_file()},
        prose=fingerprint(prose_texts()),code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},
        runtime={p:importlib.metadata.version(p) for p in ('mlx','numpy','tokenizers','pyarrow','requests','datasets')})


def verify(frozen):
    for p,h in frozen['protected'].items():
        if file_sha256(p)!=h:raise ValueError('Protected input changed: '+p)
    for p,h in frozen['code'].items():
        if file_sha256(ROOT/p)!=h:raise ValueError('Code changed: '+p)
    if frozen['runtime']!={p:importlib.metadata.version(p) for p in frozen['runtime']}:raise ValueError('Runtime changed')


def plan(cfg):
    return dict(source=str(ROOT/cfg['source_bundle']),source_step=1920,additional_updates=cfg['updates'],final_step=2048,
        unique_new_training_tasks=1024,previously_consumed_replay_exposures=1024,passes_over_new_tasks=1,
        dataset='SmolTalk smol-constraints TRAIN; pinned, filtered, topic/answer deduplicated; no raw disk cache',
        method='Assistant-only supervised CE including EOS; QA ranking replay; frozen 1920 prose KL',
        per_update=cfg['per_update'],loss_weights=cfg['loss_weights'],peak_lr=cfg['peak_lr'],final_lr=cfg['final_lr'],
        checkpoint_steps=list(range(1920,2049,16)),evaluation_steps=[1920+x for x in cfg['evaluation_updates']],
        all_checkpoints_retained=True,optimizer='fresh FP32 AdamW',main_mac_only=True,teacher=False,
        output=str(OUTPUT),automatic_promotion=False,
        selection='Review correct content AND task completion alongside format compliance. Freeze selection before reserved/public tests.')
