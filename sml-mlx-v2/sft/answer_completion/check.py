"""Full data audit plus disposable step-512 updates and exact resume checks."""
import gc,math,tempfile
from collections import Counter
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from sml_v2.checkpoint_bundle import save_bundle
from . import engine
from .protocol import load_data,OUTPUT
from .launch import restore,metadata
from .evaluate import assess,teacher_stats
from sft.reading_repair.generation import verify_generation

def difference(a,b):
    assert a.keys()==b.keys()
    return max(float(mx.max(mx.abs(a[k]-b[k])).item()) for k in a)

def run_checks(cfg,frozen,prepared=None):
    d=prepared if prepared is not None else load_data(cfg)
    b=engine.load(cfg)
    from .data import quality
    assert all(quality(r) and r['id'] not in cfg['excluded_ids'] for r in d['train'])
    for split in ('train','dev','test'):
        rows=d[split]
        assert len({r['id'] for r in rows})==len(rows)
        assert Counter(r['family'] for r in rows)==(cfg['training_counts'] if split=='train' else cfg['validation_counts'])
        for r in rows:
            e=engine.encode(b,r,cfg['context']);head=b.encode('User: '+r['prompt']+'\nAssistant:')
            assert e['y'][:len(head)-1]==[-100]*(len(head)-1)
            assert e['targets']==r['answer_targets']
            assert e['targets']<=cfg['max_answer_tokens'] and len(e['x'])<=cfg['context']
            assert e['y'][-1]==b.eos and e['targets']==len(e['y'])-len(head)+1
            assert all(x not in (b.pad,2,3) for x in e['y'][len(head)-1:])
    assert len(d['manual_review'])==len(d['reserved_review'])==24
    assert not {r['prompt'] for r in d['reserved_review']} & {r['prompt'] for r in d['train']}
    assert not {r['prompt'] for r in d['manual_review']} & {r['prompt'] for r in d['reserved_review']}
    assert not {r['prompt'] for r in d['manual_review']}&{r['prompt'] for r in d['train']}
    groups={s:{r['group'] for r in d[s]} for s in ('train','dev','test')}
    assert not groups['train']&groups['dev'] and not groups['train']&groups['test'] and not groups['dev']&groups['test']
    assert len({r['id'] for r in d['train']})==4096
    for start in range(0,4096,16):
        assert Counter(r['role'] for r in d['train'][start:start+16])=={'new':8,'rehearsal':8}
    assert len({r['group'] for r in d['train']})==4096
    assert not {r['group'] for r in d['train']} & set(d['parent_reserved_groups'])
    assert {r['id'] for r in d['train'] if r['role']=='rehearsal'} <= set(d['parent_train_ids'])
    assert not {r['group'] for r in d['train'] if r['role']=='new'} & set(d['parent_all_groups'])
    assert engine.batch_at(d,cfg,100)==d['train'][1600:1616]
    assert engine.learning_rate(cfg['warmup_updates'],cfg)==cfg['peak_lr']
    assert abs(engine.learning_rate(cfg['updates'],cfg)-cfg['final_lr'])<1e-12
    print('[check] All 4,096 rows tokenized; masks, one-pass cursor and disjoint source groups passed.',flush=True)
    parity=verify_generation(b,[d['train'][0]['prompt'],d['train'][4]['prompt']],16)
    r=d['train'][8];e=engine.encode(b,r,cfg['context']);x,y=engine.arrays([e],b.pad)
    measured=teacher_stats(b,r,cfg)
    assert abs(measured['nll']-float(engine.objective(b.model,x,y).item()))<1e-5
    assert 0<=measured['correct_tokens']<=measured['targets'] and 0<=measured['eos_probability']<=1
    # Padding must not enter loss or change this example's target likelihood.
    xx=mx.concatenate([x,mx.zeros((1,7),dtype=mx.int32)],axis=1)
    yy=mx.concatenate([y,mx.full((1,7),-100,dtype=mx.int32)],axis=1)
    assert abs(measured['nll']-float(engine.objective(b.model,xx,yy).item()))<1e-5
    smoke=assess(b,dict(train_probes=d['train_probes'][:1],dev=d['dev'][:1],manual_review=d['manual_review'][:1]),dict(cfg,max_new_tokens=16))
    assert all(math.isfinite(smoke[k]['metrics']['assistant_nll']) for k in ('train','heldout','manual_review'))
    opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters())
    with tempfile.TemporaryDirectory(prefix='opensml-base-restart-check-') as temp:
        p=Path(temp)
        first=engine.update(b,opt,engine.batch_at(d,cfg,0),cfg,1)
        saved=save_bundle(p/'resume',b.model,opt,metadata(1,frozen,cfg),None,keep=1)
        second=engine.update(b,opt,engine.batch_at(d,cfg,1),cfg,2)
        mx.save_safetensors(str(p/'weights.safetensors'),dict(tree_flatten(b.model.parameters())))
        mx.save_safetensors(str(p/'state.safetensors'),dict(tree_flatten(opt.state)))
        del opt;gc.collect();mx.clear_cache();opt=engine.optimizer(cfg)
        assert restore(b,opt,saved,frozen,cfg)==1
        repeated=engine.update(b,opt,engine.batch_at(d,cfg,1),cfg,2)
        wg=difference(mx.load(str(p/'weights.safetensors')),dict(tree_flatten(b.model.parameters())))
        sg=difference(mx.load(str(p/'state.safetensors')),dict(tree_flatten(opt.state)))
        assert max(wg,sg,abs(second['loss']-repeated['loss']))<1e-6
    stats={k:dict(examples=len(v),assistant_targets=sum(engine.encode(b,r,cfg['context'])['targets'] for r in v),maximum_tokens=max(len(engine.encode(b,r,cfg['context'])['x']) for r in v)) for k,v in d.items() if k in ('train','dev','test')}
    del b,opt;gc.collect();mx.clear_cache()
    return dict(data_stats=stats,mask_and_cursor_checks=True,teacher_likelihood_and_padding_checks=True,cache_parity=parity,evaluation_smoke={k:smoke[k]['metrics'] for k in ('train','heldout','manual_review')},first_disposable_update=first,second_disposable_update=second,model_resume_gap=wg,optimizer_resume_gap=sg,production_updates=0,disposable_updates=3)
