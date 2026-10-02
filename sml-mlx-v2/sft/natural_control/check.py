"""Disposable numerical and checkpoint checks. Never writes production weights."""
import gc,math,tempfile,time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from sml_v2.common import fingerprint
from sml_v2.checkpoint_bundle import save_bundle
from sft.natural_control import engine
from sft.natural_control.launch import restore

FIXTURES=[
 dict(id='check-short',prompt='The wool scarf is folded beside the basket. Describe its location.',answer='The folded wool scarf is beside the basket.'),
 dict(id='check-long',prompt='Please turn this into a polite request: Close the garden gate.',answer='Could you please close the garden gate after you come through? Thank you for your help.')]


def norms(a,b):
    aa=sorted(tree_flatten(a));bb=sorted(tree_flatten(b))
    if len(aa)!=len(bb):raise ValueError('Parameter count differs')
    pairs=list(zip(aa,bb))
    if any(x[0]!=y[0] or x[1].shape!=y[1].shape for x,y in pairs):raise ValueError('Parameter structure differs')
    diff=mx.array(0.);base=mx.array(0.);maximum=0.
    for (_,x),(_,y) in pairs:
        diff+=((x.astype(mx.float32)-y.astype(mx.float32))**2).sum();base+=(x.astype(mx.float32)**2).sum()
        maximum=max(maximum,float(mx.max(mx.abs(x-y)).item()))
    return dict(relative_l2=float(mx.sqrt(diff/mx.maximum(base,1e-30)).item()),max_abs=maximum)


def run_checks(cfg,frozen):
    results={}
    for model in cfg['models']:
        print('[check]',model,'masking, padding, gradients and disposable optimizer resume',flush=True)
        b=engine.load(model);opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters())
        encoded=[engine.encode(b,r,cfg) for r in FIXTURES]
        for row,enc in zip(FIXTURES,encoded):
            head=b.encode(engine.prefix(row['prompt']));target=[x for x in enc['y'] if x!=-100]
            assert all(x==-100 for x in enc['y'][:len(head)-1])
            assert enc['y'][-1]==b.eos and len(target)==enc['targets']
            assert enc['x'][len(head):]==target[:-1]
        x,y=engine.arrays(encoded,b.pad)
        batch=float(engine.objective(b.model,x,y,model=='reference').item())
        losses=[]
        for enc in encoded:
            xx,yy=engine.arrays([enc],b.pad)
            losses.append(float(engine.objective(b.model,xx,yy,model=='reference').item()))
        separate=sum(v*r['targets'] for v,r in zip(losses,encoded))/sum(r['targets'] for r in encoded)
        if abs(batch-separate)>.002:raise ValueError('Padding/token-weighting mismatch')
        # Mutating future padded inputs must not affect earlier causal predictions.
        changed=x.at[0,len(encoded[0]['x']):].add(7)
        pad_gap=abs(batch-float(engine.objective(b.model,changed,y,model=='reference').item()))
        if pad_gap>.002:raise ValueError('Right padding affected valid targets')
        v1,g1=engine.gradients(b.model,x,y,model=='reference',1)
        v2,g2=engine.gradients(b.model,x,y,model=='reference',2)
        diff=norms(g1,g2)
        if diff['relative_l2']>.01 or abs(float(v1.item())-float(v2.item()))>.002:raise ValueError('Accumulation differs from token-weighted batch gradient')
        del g1,g2;gc.collect();mx.clear_cache()
        arm=dict(name='disposable-'+model,model=model,peak_lr=cfg['learning_rates'][0])
        with tempfile.TemporaryDirectory(prefix='opensml-natural-check-') as temp:
            t=time.monotonic();first=engine.update(b,opt,FIXTURES,cfg,1,arm['peak_lr'])
            meta=dict(step=1,arm=arm,contract=fingerprint(frozen),next_example=cfg['batch'],training=[first])
            saved=save_bundle(Path(temp)/'resume',b.model,opt,meta,None,keep=1)
            second=engine.update(b,opt,FIXTURES[::-1],cfg,2,arm['peak_lr'])
            mx.save_safetensors(str(Path(temp)/'expected-model.safetensors'),dict(tree_flatten(b.model.parameters())))
            mx.save_safetensors(str(Path(temp)/'expected-optimizer.safetensors'),dict(tree_flatten(opt.state)))
            del opt;gc.collect();mx.clear_cache()
            opt=engine.optimizer(cfg)
            step,_=restore(b,opt,saved,frozen,arm,cfg);assert step==1
            repeated=engine.update(b,opt,FIXTURES[::-1],cfg,2,arm['peak_lr'])
            from mlx.utils import tree_unflatten
            expected=tree_unflatten(list(mx.load(str(Path(temp)/'expected-model.safetensors')).items()))
            weight_gap=norms(expected,b.model.parameters());del expected
            expected=tree_unflatten(list(mx.load(str(Path(temp)/'expected-optimizer.safetensors')).items()))
            state_gap=norms(expected,opt.state);del expected
            if weight_gap['max_abs']>1e-6 or state_gap['max_abs']>1e-6 or abs(second['loss']-repeated['loss'])>1e-6:
                raise ValueError('Saved optimizer/cursor did not reproduce the next update')
            elapsed=time.monotonic()-t
        # A separate base load verifies actual inference plumbing after tests; no
        # claim that answers from this untrained base are useful chat replies.
        del b,opt;gc.collect();mx.clear_cache();b=engine.load(model)
        probe=engine.generate(b,'Can you suggest a pleasant way to spend a quiet evening?',limit=12)
        results[model]=dict(mask_and_eos=True,batch_loss=batch,token_weighted_single_loss=separate,
            padding_loss_gap=pad_gap,microbatch_gradient=diff,resume_weight_gap=weight_gap,resume_optimizer_gap=state_gap,
            disposable_updates_executed=3,disposable_example_exposures=6,check_seconds=elapsed,generation_probe=probe,
            production_updates=0)
        print('[check-passed]',model,json_summary(results[model]),flush=True)
        del b;gc.collect();mx.clear_cache()
    return results


def json_summary(r):
    import json
    return json.dumps({k:r[k] for k in ('microbatch_gradient','resume_weight_gap','resume_optimizer_gap','production_updates')})
