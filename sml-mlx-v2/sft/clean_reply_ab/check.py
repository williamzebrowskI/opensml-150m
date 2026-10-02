"""Data separation, balanced exposure, objective and disposable resume checks."""
import gc,math,tempfile
from pathlib import Path
from collections import Counter
import mlx.core as mx
from mlx.utils import tree_flatten
from sml_v2.checkpoint_bundle import save_bundle
from sml_v2.common import fingerprint
from . import engine
from .launch import restore,metadata
from .evaluate import assess,teacher_stats
from sft.reading_repair.generation import verify_generation

def difference(a,b):
    assert a.keys()==b.keys()
    return max(float(mx.max(mx.abs(a[k]-b[k])).item()) for k in a)

def run_checks(cfg,frozen,d):
    b=engine.load(cfg)
    assert len(d['train'])==384 and len(d['new'])==256 and len(d['replay'])==128
    assert len(d['dev'])==len(d['test'])==32 and len(d['paraphrases'])==128
    for rows in [d['train'],d['dev'],d['test'],d['paraphrases']]:
        assert len({r['prompt'].strip().casefold() for r in rows})==len(rows)
        for r in rows:
            e=engine.encode(b,r,cfg['context']);head=b.encode('User: '+r['prompt']+'\nAssistant:')
            assert e['y'][:len(head)-1]==[-100]*(len(head)-1)
            assert e['targets']==r['answer_targets'] and e['y'][-1]==b.eos
            assert len(e['x'])<=cfg['context'] and all(x not in (b.pad,2,3) for x in e['y'][len(head)-1:])
    sets=[{r['prompt'].strip().casefold() for r in d[k]} for k in ('train','dev','test','paraphrases')]
    for i,a in enumerate(sets):
        for z in sets[i+1:]:assert not a&z
    groups={k:{r['group'] for r in d[k]} for k in ('train','dev','test')}
    assert not groups['train']&groups['dev'] and not groups['train']&groups['test'] and not groups['dev']&groups['test']
    expected=Counter(r['id'] for r in d['train']);batchhash=[]
    for epoch in range(4):
        seen=Counter()
        for u in range(epoch*32,(epoch+1)*32):
            rows=engine.batch_at(d,cfg,u);assert Counter(r['role'] for r in rows)=={'new':8,'rehearsal':4}
            seen.update(r['id'] for r in rows);batchhash.append(fingerprint([r['id'] for r in rows]))
        assert seen==expected
    assert engine.learning_rate(cfg['warmup_updates'],cfg)==cfg['peak_lr']
    assert abs(engine.learning_rate(cfg['updates'],cfg)-cfg['final_lr'])<1e-12
    parity=verify_generation(b,[d['new'][0]['prompt'],d['new'][96]['prompt']],12)
    r=d['new'][40];e=engine.encode(b,r,cfg['context']);x,y=engine.arrays([e],b.pad);measured=teacher_stats(b,r,cfg)
    assert abs(measured['nll']-float(engine.objective(b.model,x,y).item()))<1e-5
    xx=mx.concatenate([x,mx.zeros((1,7),dtype=mx.int32)],axis=1);yy=mx.concatenate([y,mx.full((1,7),-100,dtype=mx.int32)],axis=1)
    assert abs(measured['nll']-float(engine.objective(b.model,xx,yy).item()))<1e-5
    small={k:d[k][:1] for k in ('train_probes','paraphrases','dev','retention')}
    smoke=assess(b,small,dict(cfg,max_new_tokens=8))
    assert all(math.isfinite(smoke[k]['metrics']['assistant_nll']) for k in ('trained','paraphrase','new_tasks','retention'))
    opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters())
    with tempfile.TemporaryDirectory(prefix='clean-reply-check-') as temp:
        p=Path(temp);first=engine.update(b,opt,engine.batch_at(d,cfg,0),cfg,1)
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
        # Cross-arm resumes must be rejected even though the step counts match.
        try:restore(b,opt,saved,dict(frozen,config='wrong-arm'),cfg)
        except ValueError:pass
        else:raise AssertionError('Wrong arm accepted')
    del b,opt;gc.collect();mx.clear_cache()
    print('[checked]',cfg['arm'],'data, generation, optimizer updates and exact resume',flush=True)
    return dict(batch_schedule=fingerprint(batchhash),training_examples=384,exposures=1536,disposable_updates=3,production_updates=0,model_resume_gap=wg,optimizer_resume_gap=sg,first_disposable_update=first,second_disposable_update=second,cache_parity=parity,evaluation_smoke={k:smoke[k]['metrics'] for k in ('trained','paraphrase','new_tasks','retention')})
