"""Disposable real optimizer/generation/resume checks; no production training."""
import gc
import math
import tempfile
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from sml_v2.common import atomic_json,file_sha256
from sml_v2.checkpoint_bundle import save_bundle
from sft.reading_repair.generation import verify_generation
from . import engine
from .constraints import fixtures,verify as verify_rules
from .protocol import verify


def run_checks(cfg,frozen):
    from .launch import rebuild,restore,metadata
    from .evaluate import assess,review_template,validate_review
    result={'verifiers':fixtures()};data=rebuild(cfg)
    b=engine.load(cfg);anchor=engine.load(cfg);seen=set();masks=0
    for split in ('train','dev','test'):
        for family,rows in data[split].items():
            for row in rows:
                if row['id'] in seen:continue
                seen.add(row['id']);enc=engine.encode(b,row,cfg['context']);head=b.encode(engine.prefix(row['prompt']))
                assert enc['y'][:len(head)-1]==[-100]*(len(head)-1) and enc['y'][-1]==b.eos
                assert len(enc['x'])==len(enc['y'])<=cfg['context']
                assert enc['targets']==sum(v!=-100 for v in enc['y'])
                if family=='complete':assert all(verify_rules(row['answer'],row['rules']))
                masks+=1
    # Check every consumed batch has exactly the configured family sizes and no
    # new-task repetition. New-task contexts are not duplicated to fill quotas.
    ids=[]
    for u in range(cfg['updates']):
        rr=engine.batch_at(data,cfg,u);ids.extend(r['id'] for r in rr['complete'])
    assert len(ids)==len(set(ids))==1024
    class Tiny(nn.Module):
        def __init__(self):super().__init__();self.weight=mx.zeros((1,4,7))
        def __call__(self,x):return {'logits':self.weight}
    tiny=Tiny();x=mx.zeros((1,4),dtype=mx.int32);y=mx.array([[-100,1,2,3]])
    loss,g=nn.value_and_grad(tiny,lambda:engine.objective(tiny,x,y,False))();mx.eval(loss,g)
    assert abs(float(loss.item())-math.log(7))<1e-6
    assert float(mx.max(mx.abs(g['weight'][:,0])).item())==0
    assert float(mx.max(mx.abs(g['weight'][:,-1])).item())>0
    result['generation_parity']=verify_generation(b,['Hello!',data['train']['complete'][0]['prompt']],24)
    small=dict(data,dev={f:rr[:2] for f,rr in data['train'].items()})
    # Evaluator smoke uses TRAIN new tasks, not new held-out responses.
    smoke=assess(b,small,'dev',dict(cfg,max_new_tokens=24))
    template=review_template(smoke)
    try:validate_review(template,smoke)
    except ValueError:pass
    else:raise AssertionError('Unreviewed answers passed')
    assert math.isfinite(smoke['metrics']['assistant_nll'])
    first=engine.batch_at(data,cfg,0);frozen_anchor={k:mx.array(v) for k,v in tree_flatten(anchor.model.parameters())};mx.eval(frozen_anchor)
    def nll():
        out=[]
        for r in first['complete']:
            xx,yy=engine.arrays([engine.encode(b,r,cfg['context'])],b.pad)
            out.append(float(engine.objective(b.model,xx,yy,False).item()))
        return sum(out)/len(out)
    opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    before=nll();history=[]
    for step in range(1,9):
        history.append(engine.update(b,anchor,opt,first,data['anchors'][0],cfg,step))
        print('[disposable-check]',step,'complete_nll',history[-1]['complete_ce'],flush=True)
    after=nll()
    if not after<before:raise ValueError('TRAIN learnability check failed')
    delta=max(float(mx.max(mx.abs(v-frozen_anchor[k])).item()) for k,v in tree_flatten(b.model.parameters()))
    assert delta>0 and all(r['grad_norm']>0 for r in history)
    with tempfile.TemporaryDirectory(prefix='constraints-1920-check-') as temp:
        temp=Path(temp);saved=save_bundle(temp,b.model,opt,metadata(cfg,frozen,8,history),None,keep=1000000)
        atomic_json(temp/'contract.json',frozen)
        from evaluation.full_benchmarks import spec,core
        key='constraint-completion-disposable';spec.MODELS[key]=dict(bundle=saved,step=1928,sha256=file_sha256(Path(saved)/'model.safetensors'))
        try:
            for suite in ('multiple-choice','ifeval'):assert core.frozen(key,suite)['training'] is False
        finally:spec.MODELS.pop(key,None)
        rr=engine.batch_at(data,cfg,8);expected_loss=engine.update(b,anchor,opt,rr,data['anchors'][8],cfg,9)
        mx.save_safetensors(str(temp/'expected.safetensors'),dict(tree_flatten(b.model.parameters())))
        mx.save_safetensors(str(temp/'state.safetensors'),dict(tree_flatten(opt.state)))
        del opt;gc.collect();mx.clear_cache();opt=engine.optimizer(cfg)
        u,_=restore(b,opt,saved,frozen,cfg);assert u==8
        actual_loss=engine.update(b,anchor,opt,rr,data['anchors'][8],cfg,9)
        expected=mx.load(str(temp/'expected.safetensors'));actual=dict(tree_flatten(b.model.parameters()))
        gap=max(float(mx.max(mx.abs(v-actual[k])).item()) for k,v in expected.items())
        expected_state=mx.load(str(temp/'state.safetensors'));actual_state=dict(tree_flatten(opt.state))
        state_gap=max(float(mx.max(mx.abs(v.astype(mx.float32)-actual_state[k].astype(mx.float32))).item()) for k,v in expected_state.items())
        assert gap<=1e-7 and state_gap<=1e-7 and abs(expected_loss['loss']-actual_loss['loss'])<=1e-7
        assert all(bool(mx.array_equal(v,frozen_anchor[k]).item()) for k,v in tree_flatten(anchor.model.parameters()))
    del b,anchor,opt;gc.collect();mx.clear_cache();verify(frozen)
    return dict(**result,unique_masked_rows_checked=masks,prompt_gradient_zero=True,eos_gradient_nonzero=True,
        fixed_train_batch_nll_before=before,fixed_train_batch_nll_after=after,parameter_change_max=delta,
        exact_resume_model_max_error=gap,exact_resume_optimizer_max_error=state_gap,frozen_anchor_unchanged=True,
        benchmark_contracts_valid=True,public_benchmark_rows_scored=0,reserved_generations=0,
        production_updates=0,temporary_weights_removed=True)
