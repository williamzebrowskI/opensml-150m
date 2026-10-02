"""Meaningful readiness: masks, DPO gradients, frozen anchor and resumable updates."""
from collections import Counter
from pathlib import Path
import gc,math,tempfile,time
from sml_v2.common import fingerprint
from .protocol import verify


def check(cfg,frozen,data):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from sml_v2.checkpoint_bundle import save_bundle
    from sft.corrective_640.launch import restore
    from sft.reading_repair.generation import verify_generation
    from sft.grounded_rank_384.data import instruction_pass
    from .data import encode_reply
    from . import engine
    from .evaluate import assess
    b=engine.load(cfg);anchor=engine.load(cfg);checked=Counter()
    anchor_start=dict(tree_flatten(anchor.model.parameters()))
    for split in ('train','dev','test'):
        for family,rows in data[split].items():
            # Rehearsal/epochs intentionally repeat; inspect every unique record.
            seen=set()
            for row in rows:
                if row['id'] in seen:continue
                seen.add(row['id'])
                if family=='preference':
                    for kind in ('answer','rejected'):
                        enc=encode_reply(b.tokenizer,row['prompt'],row[kind],cfg['context'])
                        head=b.encode('User: '+row['prompt']+'\nAssistant:')
                        if enc['y'][-1]!=b.eos or any(t!=-100 for t in enc['y'][:len(head)-1]):raise ValueError('Preference prompt/EOS mask')
                elif family=='replay':engine.validate_conversation(b,row,cfg['context'])
                elif family=='ranking':engine.choice_arrays(b,row,cfg['context'])
                else:
                    x,y,p=engine.supervised_arrays(b,row,cfg['context'])
                    if int(y[0,-1].item())!=b.eos:raise ValueError('Supervised EOS missing')
                    if family=='instruction' and not instruction_pass(row,row['answer']):raise ValueError('Instruction reference fails checker')
                checked[split+':'+family]+=1
    # DPO convention, sign and preferred-CE gradient against analytic derivatives.
    lp=mx.array([-10.,-14.]);counts=mx.array([5.,7.]);ref=mx.array([-10.,-14.])
    def scalar(z):
        d,c,_=engine.preference_terms(z,counts,ref,cfg)
        return cfg['loss_weights']['preference']*d+cfg['loss_weights']['chosen_ce']*c
    grad=mx.grad(scalar)(lp)
    expected=mx.array([-cfg['loss_weights']['preference']*cfg['dpo_beta']/2-cfg['loss_weights']['chosen_ce']/5,cfg['loss_weights']['preference']*cfg['dpo_beta']/2])
    if float(mx.max(mx.abs(grad-expected)).item())>1e-6:raise ValueError('DPO gradient sign/scale wrong')
    fixture=dict(prompt='The garden contains roses.',answer='Roses.',rejected='Rocks.')
    x,y=engine.pair_arrays(b,fixture,cfg['context']);lp,counts=engine.sequence_logps(b.model,x,y)
    reference,_=engine.sequence_logps(anchor.model,x,y);dpo,_,_=engine.preference_terms(lp,counts,reference,cfg)
    if abs(float(dpo.item())-math.log(2))>1e-6:raise ValueError('Initial DPO is not log(2)')
    errors=[]
    for kind,score in zip(('answer','rejected'),lp.tolist()):
        enc=encode_reply(b.tokenizer,fixture['prompt'],fixture[kind],cfg['context']);expected_score=0.
        for i,label in enumerate(enc['y']):
            if label==-100:continue
            z=b.logits(mx.array([enc['x'][:i+1]],dtype=mx.int32))[0,-1].astype(mx.float32)
            expected_score+=float((z[label]-mx.logsumexp(z)).item())
        errors.append(abs(expected_score-score))
    if max(errors)>.01:raise ValueError('Independent preference sum/mask score mismatch')
    opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    started=time.monotonic();first=engine.update(b,anchor,opt,engine.batch_at(data,cfg,0),data['anchors'][0],cfg,1);first['seconds']=time.monotonic()-started
    with tempfile.TemporaryDirectory(prefix='opensml-preference640-check-') as temp:
        meta=dict(step=641,source_step=640,additional_updates=1,next_examples=cfg['per_update'],contract=fingerprint(frozen),training=[first])
        bundle=save_bundle(Path(temp)/'resume',b.model,opt,meta,None,keep=1)
        restored=engine.load(cfg);ropt=engine.optimizer(cfg);u,_=restore(restored,ropt,bundle,frozen,cfg)
        if u!=1:raise ValueError('Resume cursor wrong')
        for aa,bb in ((b.model.parameters(),restored.model.parameters()),(opt.state,ropt.state)):
            a=dict(tree_flatten(aa));c=dict(tree_flatten(bb))
            if a.keys()!=c.keys() or any(not bool(mx.array_equal(a[k],c[k]).item()) for k in a):raise ValueError('Restoration not bitwise exact')
        rows=engine.batch_at(data,cfg,1);start=time.monotonic()
        second=engine.update(b,anchor,opt,rows,data['anchors'][1],cfg,2);second['seconds']=time.monotonic()-start
        engine.update(restored,anchor,ropt,rows,data['anchors'][1],cfg,2)
        diffs={}
        for label,aa,bb in (('weights',b.model.parameters(),restored.model.parameters()),('optimizer',opt.state,ropt.state)):
            a=dict(tree_flatten(aa));c=dict(tree_flatten(bb))
            for k in a:
                if bool(mx.array_equal(a[k],c[k]).item()):continue
                err=float(mx.max(mx.abs(a[k]-c[k])).item());scale=max(float(mx.max(mx.abs(a[k])).item()),float(mx.max(mx.abs(c[k])).item()))
                if label=='optimizer' and k.endswith(('.m','.v')):
                    if err>8*1.1920928955078125e-7*scale:raise ValueError('Optimizer resume rounding too large: '+k)
                elif k in ('embed.weight','embed.weight.master'):
                    mag=mx.maximum(mx.maximum(mx.abs(a[k]),mx.abs(c[k])),1.1754943508222875e-38)
                    relative=float(mx.max(mx.abs(a[k]-c[k])/mag).item())
                    if relative>8*1.1920928955078125e-7 or err>second['lr']*1e-4:raise ValueError('Embedding resume rounding too large')
                else:raise ValueError('Unexpected resumed update difference: '+k)
                diffs[label+':'+k]=err
        if any(not bool(mx.array_equal(v,dict(tree_flatten(anchor.model.parameters()))[k]).item()) for k,v in anchor_start.items()):raise ValueError('Frozen anchor changed')
        smoke=assess(restored,anchor,data,dict(cfg,max_new_tokens=32),smoke=True)
        decode=verify_generation(restored,['Return only the word violet.','Hello! Can I ask you something?'],limit=16)
        del restored,ropt
    del b,anchor,opt;gc.collect();mx.clear_cache();verify(frozen)
    return dict(unique_rows_checked=dict(checked),assistant_and_eos_masks=True,dpo_analytic_gradient_checked=True,initial_dpo=math.log(2),independent_logp_max_error=max(errors),first_disposable_update=first,second_disposable_update=second,restoration_bitwise_equal=True,next_update_fp32_equivalent=True,next_update_rounding_differences=diffs,frozen_anchor_unchanged=True,smoke_metrics=smoke['metrics'],decode_parity=decode,production_updates=0,source_640_unchanged=True)
