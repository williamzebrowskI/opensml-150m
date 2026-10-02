"""Freeze source checkpoint, selected data, implementation and runtime."""
from pathlib import Path
import importlib.metadata
from sml_v1.common import file_sha256,fingerprint,read_json
ROOT=Path(__file__).resolve().parents[2]
DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_preference_long_640_v1'


def code_files():
    paths=list(DIR.glob('*.py'))+list((ROOT/'sml_v1').glob('*.py'))
    for package in ('corrective_640','grounded_rank_384','skill_recovery_768','transfer_control','reading_repair','response_repair','intact_smoltalk_base_pilot','natural_control','skill_balance'):
        paths+=list((ROOT/'sft'/package).glob('*.py'))
    paths += list((ROOT/'evaluation/full_benchmarks').glob('*.py'))
    return sorted(set(paths))


def validate(cfg):
    if (cfg['source_step'],cfg['updates'])!=(640,2048):raise ValueError('Expected long 640-to-2688 branch')
    if cfg['preference_train_pairs']*cfg['preference_epochs']!=cfg['updates']*cfg['per_update']['preference']:raise ValueError('Preference exposure mismatch')
    if abs(sum(cfg['loss_weights'].values())-1)>1e-9 or min(cfg['loss_weights'].values())<=0:raise ValueError('Invalid weights')
    if not 0<cfg['final_lr']<=cfg['peak_lr']<=1e-6 or not 0<cfg['warmup_updates']<cfg['updates']:raise ValueError('Invalid schedule')
    parent=ROOT/cfg['source_bundle'];meta=read_json(parent/'model.safetensors.json')
    if (meta['step'],meta['source_step'],meta['additional_updates'])!=(640,384,256):raise ValueError('Wrong parent lineage')
    if file_sha256(parent/'model.safetensors')!=cfg['source_model_sha256']:raise ValueError('Parent weights changed')
    if meta['contract']!=fingerprint(read_json(parent.parent/'contract.json')):raise ValueError('Parent contract changed')


def contract(cfg):
    validate(cfg);selection=read_json(DIR/'selection.json');review=read_json(DIR/'review.json')
    if selection['config']!=fingerprint(cfg):raise ValueError('Prepared configuration changed')
    if review['status']!='accepted' or review['sample_sha256']!=file_sha256(DIR/'review_samples.json'):raise ValueError('Current samples require recorded review')
    parent=ROOT/cfg['source_bundle']
    paths=list(parent.iterdir())+[parent.parent/'contract.json']
    paths += [DIR/n for n in ('config.json','prepared.json','selection.json','review_samples.json','review.json','review_rejections.json','README.md')]
    paths += [ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json']
    paths += list((ROOT/'tokenizer/bytebpe32k_v1').glob('*'))
    protected={str(p):file_sha256(p) for p in paths if p.is_file()}
    for p,h in selection['inputs'].items():
        if file_sha256(p)!=h:raise ValueError('Prepared input changed: '+p)
        protected[p]=h
    return dict(config=fingerprint(cfg),selection=fingerprint(selection),tokenizer=selection['tokenizer'],source=str(parent),source_sha256=cfg['source_model_sha256'],protected=protected,code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},runtime={k:importlib.metadata.version(k) for k in ('mlx','numpy','tokenizers','pyarrow')})


def verify(frozen):
    for p,h in frozen['protected'].items():
        if file_sha256(p)!=h:raise ValueError('Protected input changed: '+p)
    for p,h in frozen['code'].items():
        if file_sha256(ROOT/p)!=h:raise ValueError('Implementation changed: '+p)


def plan(cfg):
    return dict(source=str(ROOT/cfg['source_bundle']),source_step=640,updates=cfg['updates'],final_step=640+cfg['updates'],method='Reference DPO + preferred-answer CE + knowledge/instruction/conversation rehearsal + frozen-640 chat/prose KL',preference_unique_pairs=cfg['preference_train_pairs'],preference_epochs=cfg['preference_epochs'],per_update=cfg['per_update'],loss_weights=cfg['loss_weights'],peak_lr=cfg['peak_lr'],early_regression_stop=True,automatic_promotion=False,output=str(OUTPUT))
