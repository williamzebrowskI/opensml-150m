"""Reference DPO + preferred-answer CE + full-conversation CE and forward KL."""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from sft.skill_balance.engine import load,optimizer,learning_rate
from sft.conversation_foundation_v1.engine import generate
from sft.text_followup_512_v1.engine import assess as parent_assess
from sft.transfer_control.engine import arrays
from sml_v2.common import read_json
from .data import encode_reply,turns,PARENT


def pair_arrays(b,row,cfg):
    return arrays([encode_reply(b.tokenizer,row['messages'],row[k],cfg['context']) for k in ('chosen','rejected')],b.pad)


def sequence_logps(model,x,y):
    logits=model(x)['logits'].astype(mx.float32)
    ce=nn.losses.cross_entropy(logits,mx.maximum(y,0),reduction='none');mask=y!=-100
    return -(ce*mask).sum(axis=1),mask.sum(axis=1).astype(mx.float32)


def preference_terms(logps,counts,reference,beta):
    margin=beta*((logps[0]-logps[1])-(reference[0]-reference[1]))
    return mx.logaddexp(mx.array(0.,dtype=mx.float32),-margin),-logps[0]/counts[0],margin


def pair_objective(model,x,y,ref,cfg):
    lp,n=sequence_logps(model,x,y);dpo,ce,margin=preference_terms(lp,n,ref,cfg['dpo_beta'])
    w=cfg['loss_weights']
    return w['dpo']*dpo+w['chosen_ce']*ce,mx.stack([dpo,ce,margin])


def reference_logp(anchor,x,positions):
    z=anchor.model(x)['logits'][0,positions].astype(mx.float32)
    return mx.stop_gradient(z-mx.logsumexp(z,axis=-1,keepdims=True))


def replay_objective(model,x,y,positions,ref,cfg):
    z=model(x)['logits'][0,positions].astype(mx.float32)
    lp=z-mx.logsumexp(z,axis=-1,keepdims=True)
    ce=nn.losses.cross_entropy(z,y[0,positions],reduction='mean')
    kl=(mx.exp(ref)*(ref-lp)).sum(axis=-1).mean();w=cfg['loss_weights']
    return w['replay_ce']*ce+w['chat_kl']*kl,mx.stack([ce,kl])


def update(b,anchor,opt,rows,cfg,step):
    from sml_v2.pretrain import clip_gradients
    b.model.train();anchor.model.eval();total=None;value=0.;dpos=[];ces=[];margins=[];replays=[];kls=[];turn_count=0
    def add(loss,grad,weight):
        nonlocal total,value
        grad=tree_map(lambda v:v.astype(mx.float32)*weight,grad)
        total=grad if total is None else tree_map(lambda a,c:a+c,total,grad)
        mx.eval(loss,total);value+=float(loss.item())*weight
    fn=nn.value_and_grad(b.model,lambda x,y,r:pair_objective(b.model,x,y,r,cfg))
    for row in rows['preference']:
        x,y=pair_arrays(b,row,cfg);ref,_=sequence_logps(anchor.model,x,y);ref=mx.stop_gradient(ref);mx.eval(ref)
        (loss,aux),grad=fn(x,y,ref);add(loss,grad,1/len(rows['preference']));mx.eval(aux)
        dpo,ce,margin=aux.tolist();dpos.append(dpo);ces.append(ce);margins.append(margin)
    fn=nn.value_and_grad(b.model,lambda x,y,p,r:replay_objective(b.model,x,y,p,r,cfg))
    for row in rows['replay']:
        ee=turns(b.tokenizer,row,cfg['context']);ce_sum=0.;kl_sum=0.
        for e in ee:
            x,y=arrays([e],b.pad);pos=mx.array([i for i,t in enumerate(e['y']) if t!=-100])
            ref=reference_logp(anchor,x,pos);mx.eval(ref)
            (loss,aux),grad=fn(x,y,pos,ref);add(loss,grad,1/len(rows['replay'])/len(ee));mx.eval(aux)
            ce,kl=aux.tolist();ce_sum+=ce/len(ee);kl_sum+=kl/len(ee);turn_count+=1
        replays.append(ce_sum);kls.append(kl_sum)
    total,norm=clip_gradients(total,cfg['clip_norm']);mx.eval(total,norm)
    metrics=dict(dpo_loss=sum(dpos)/len(dpos),chosen_nll=sum(ces)/len(ces),preference_margin=sum(margins)/len(margins),replay_nll=sum(replays)/len(replays),chat_kl=sum(kls)/len(kls))
    if not all(math.isfinite(v) for v in [value,norm.item(),*metrics.values()]):raise ValueError('Nonfinite update; weights not changed')
    rate=learning_rate(step,cfg);opt.learning_rate=rate;opt.update(b.model,total);mx.eval(b.model.parameters(),opt.state)
    return dict(update=step,loss=value,lr=rate,grad_norm=float(norm.item()),replay_turns=turn_count,preference_pairs=len(dpos),**metrics)


def assess(b,anchor,data,cfg):
    # Precisely the original 768 development set, ordering, generation count,
    # native prompt format and checks; do not silently replace the baseline.
    result=parent_assess(b,read_json(PARENT)['data'],cfg)
    b.model.eval();anchor.model.eval();values=[]
    for row in data['preference_dev'][:cfg.get('preference_eval_pairs',64)]:
        x,y=pair_arrays(b,row,cfg);ref,_=sequence_logps(anchor.model,x,y);lp,n=sequence_logps(b.model,x,y)
        dpo,ce,margin=preference_terms(lp,n,ref,cfg['dpo_beta']);mx.eval(dpo,ce,margin)
        values.append(dict(id=row['id'],dpo=float(dpo.item()),chosen_nll=float(ce.item()),relative_margin=float(margin.item())))
    result['preference_scores']=values;m=result['metrics']
    m['preference_dev_pairs']=len(values);m['preference_dev_loss']=sum(v['dpo'] for v in values)/len(values)
    m['preference_relative_win_rate']=sum(1 if v['relative_margin']>1e-6 else .5 if abs(v['relative_margin'])<=1e-6 else 0 for v in values)/len(values)
    m['preference_chosen_nll']=sum(v['chosen_nll'] for v in values)/len(values)
    result['note']+=' Preference win rate measures a likelihood-ratio preference against frozen 768 on upstream-labeled held-out pairs. It is not a human chat win rate. Reserved preference test remains unused. All old 768 development checks stay fixed.'
    return result
