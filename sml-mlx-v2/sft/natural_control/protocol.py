"""Frozen inputs and an explicit four-fit experiment, with no automatic winner."""
import importlib.metadata
from pathlib import Path
from sml_v2.common import file_sha256,fingerprint,read_json
from sft.transfer_control.launch import protected,prose_texts
ROOT=Path(__file__).resolve().parents[2]
DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_natural_control_v1'


def validate(cfg):
    assert cfg['models']==['opensml','reference']
    assert sum(cfg['training_counts'].values())==cfg['batch']*cfg['updates']
    assert 0<cfg['microbatch']<=cfg['batch'] and cfg['batch']%cfg['microbatch']==0
    assert 0<cfg['warmup_updates']<cfg['updates']
    assert cfg['context']<=2048 and cfg['minimum_answer_targets']>=1000000
    assert cfg['evaluation_updates']==sorted(set(cfg['evaluation_updates']))
    assert cfg['evaluation_updates'][0]==0 and cfg['evaluation_updates'][-1]==cfg['updates']
    assert all(x>0 for x in cfg['learning_rates'])


def arms(cfg):
    return [dict(name=f'{m}-lr{rate:.0e}',model=m,peak_lr=rate) for m in cfg['models'] for rate in cfg['learning_rates']]


def code_files():
    files=list(DIR.glob('*.py'))+list((ROOT/'sml_v2').glob('*.py'))
    for folder in ('sft/transfer_control','evaluation/base_audit','sft/reading_repair'):
        files+=list((ROOT/folder).glob('*.py'))
    return sorted(set(files))


def contract(cfg):
    validate(cfg)
    manifest=read_json(DIR/'selection.json')
    if manifest['config']!=fingerprint(cfg):raise ValueError('Prepare the selection for this exact configuration')
    saved=protected()
    for relative,expected in (
        ('runs/sft_reading_repair_v1/step_0000896_b6e5f5f75873/model.safetensors','021bc6da816aa2b7b657a144e28b18065159554d666ba2b2159a87d0f4d66a69'),
        ('runs/grpo_reading_v1/step_0000906_07ed58a03de1/model.safetensors','68738ce4ee81650dff48014feca3e91c9fe5e430d7bd0296043e3b942ef240f7')):
        path=ROOT/relative
        if file_sha256(path)!=expected:raise ValueError('Preserved model changed: '+relative)
        saved[str(path)]=expected
    return dict(config=fingerprint(cfg),selection=file_sha256(DIR/'selection.json'),
        probes=file_sha256(DIR/'probes.json'),protected=saved,prose=fingerprint(prose_texts()),
        code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},
        runtime={p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','numpy','transformers','tokenizers','pyarrow','fsspec')})


def plan(cfg):
    p=dict(mode='plan',source='preserved OpenSML Stage B pretrained base; separate pinned SmolLM2-135M base control',
        fits=arms(cfg),examples_per_fit=sum(cfg['training_counts'].values()),updates_per_fit=cfg['updates'],
        total_updates=len(arms(cfg))*cfg['updates'],context=cfg['context'],batch=cfg['batch'],microbatch=cfg['microbatch'],
        objective='token-weighted assistant-only cross entropy including EOS; fresh FP32 AdamW per fit',
        teacher=False,main_mac_only=True,automatic_promotion=False,
        data='Pinned public Parquet streamed once per invocation; selected conversations held in RAM; no raw disk cache',
        output=str(OUTPUT),selection='No automatic best; review blinded development answers, then freeze selection before test')
    if (DIR/'selection.json').exists():
        manifest=read_json(DIR/'selection.json')
        if manifest['config']==fingerprint(cfg):p['accepted_data']=manifest['stats']
        else:p['data_status']='Selection must be rebuilt for this configuration'
    return p
