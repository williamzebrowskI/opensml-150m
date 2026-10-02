"""Reference DPO plus preferred-answer CE, knowledge rehearsal and chat/prose KL."""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from sft.corrective_640.engine import load,optimizer,learning_rate,reference_kl,conversation_arrays,validate_conversation,supervised_arrays,supervised_loss,choice_arrays,choice_scores,ranking_loss
from sft.transfer_control.engine import arrays
from .data import encode_reply


def pair_arrays(b,row,context):
    return arrays([encode_reply(b.tokenizer,row['prompt'],row[k],context) for k in ('answer','rejected')],b.pad)


def sequence_logps(model,x,y):
    values=model(x)['logits'].astype(mx.float32)
    ce=nn.losses.cross_entropy(values,mx.maximum(y,0),reduction='none')
    mask=y!=-100
    return -(ce*mask).sum(axis=1),mask.sum(axis=1).astype(mx.float32)


def preference_terms(logps,counts,reference,cfg):
    # DPO uses summed answer+EOS log probabilities, not a length-normalized proxy.
    margin=cfg['dpo_beta']*((logps[0]-logps[1])-(reference[0]-reference[1]))
    dpo=mx.logaddexp(mx.array(0.,dtype=mx.float32),-margin)
    chosen_ce=-logps[0]/counts[0]
    return dpo,chosen_ce,margin


def pair_loss(model,x,y,reference,cfg):
    logps,counts=sequence_logps(model,x,y)
    dpo,ce,_=preference_terms(logps,counts,reference,cfg)
    return cfg['loss_weights']['preference']*dpo+cfg['loss_weights']['chosen_ce']*ce


def selected_kl(model,x,positions,reference):
    scores=model(x)['logits'][0,positions].astype(mx.float32)
    logp=scores-mx.logsumexp(scores,axis=-1,keepdims=True)
    return (mx.exp(reference)*(reference-logp)).sum(axis=-1).mean()


def batch_at(data,cfg,cursor):
    if not 0<=cursor<cfg['updates']:raise ValueError('Cursor out of range')
    result={f:data['train'][f][cursor*n:(cursor+1)*n] for f,n in cfg['per_update'].items()}
    if any(len(result[f])!=n for f,n in cfg['per_update'].items()):raise ValueError('Incomplete training batch')
    return result


def update(b,anchor,opt,rows,prose_ids,cfg,step):
    b.model.train();anchor.model.eval();total=None;combined=0.;metrics={}
    def add(loss,grad,weight):
        nonlocal total,combined
        grad=tree_map(lambda x:x.astype(mx.float32)*weight,grad)
        total=grad if total is None else tree_map(lambda x,y:x+y,total,grad)
        mx.eval(loss,total);combined+=float(loss.item())*weight
    dpos=[];ces=[];margins=[]
    fn=nn.value_and_grad(b.model,lambda x,y,ref:pair_loss(b.model,x,y,ref,cfg))
    for row in rows['preference']:
        x,y=pair_arrays(b,row,cfg['context'])
        ref,_=sequence_logps(anchor.model,x,y);ref=mx.stop_gradient(ref);mx.eval(ref)
        # Metrics are pre-update, like every other family loss.
        logps,counts=sequence_logps(b.model,x,y);dpo,ce,margin=preference_terms(logps,counts,ref,cfg)
        mx.eval(dpo,ce,margin);dpos.append(float(dpo.item()));ces.append(float(ce.item()));margins.append(float(margin.item()))
        loss,grad=fn(x,y,ref);add(loss,grad,1/len(rows['preference']))
    metrics.update(dpo_loss=sum(dpos)/len(dpos),chosen_nll=sum(ces)/len(ces),preference_margin=sum(margins)/len(margins))
    fn=nn.value_and_grad(b.model,lambda x,y,p:supervised_loss(b.model,x,y,p))
    for family in ('instruction','replay'):
        losses=[];kls=[]
        for row in rows[family]:
            parts=list(conversation_arrays(b,row,cfg['context'])) if family=='replay' else [supervised_arrays(b,row,cfg['context'])]
            example=0.;example_kl=0.
            for x,y,p in parts:
                loss,grad=fn(x,y,p);add(loss,grad,cfg['loss_weights'][family]/len(rows[family])/len(parts));example+=float(loss.item())/len(parts)
                if family=='replay':
                    logits=anchor.logits(x)[0,p].astype(mx.float32)
                    ref=mx.stop_gradient(logits-mx.logsumexp(logits,axis=-1,keepdims=True));mx.eval(ref)
                    kfn=nn.value_and_grad(b.model,lambda xx,pp,rr:selected_kl(b.model,xx,pp,rr))
                    kl,grad=kfn(x,p,ref);add(kl,grad,cfg['chat_kl_weight']/len(rows[family])/len(parts));example_kl+=float(kl.item())/len(parts)
            losses.append(example);kls.append(example_kl)
        metrics[family+'_ce']=sum(losses)/len(losses)
        if family=='replay':metrics['chat_kl']=sum(kls)/len(kls)
    fn=nn.value_and_grad(b.model,lambda x,y,c,g:ranking_loss(b.model,x,y,c,g,cfg['ranking_temperature']))
    rr=[]
    for row in rows['ranking']:
        x,y,c=choice_arrays(b,row,cfg['context']);loss,grad=fn(x,y,c,row['gold'])
        add(loss,grad,cfg['loss_weights']['ranking']/len(rows['ranking']));rr.append(float(loss.item()))
    metrics['ranking_ce']=sum(rr)/len(rr)
    x=mx.array([prose_ids[:-1]],dtype=mx.int32);logits=anchor.logits(x).astype(mx.float32)
    ref=mx.stop_gradient(logits-mx.logsumexp(logits,axis=-1,keepdims=True));mx.eval(ref)
    fn=nn.value_and_grad(b.model,lambda x,r:reference_kl(b.model,x,r))
    loss,grad=fn(x,ref);add(loss,grad,cfg['prose_kl_weight']);metrics['prose_kl']=float(loss.item())
    from sml_v1.pretrain import clip_gradients
    total,norm=clip_gradients(total,cfg['clip_norm']);mx.eval(total,norm)
    if not all(math.isfinite(v) for v in [combined,float(norm.item()),*metrics.values()]):raise ValueError('Nonfinite loss/gradient; update not applied')
    rate=learning_rate(step,cfg);opt.learning_rate=rate;opt.update(b.model,total);mx.eval(b.model.parameters(),opt.state)
    return dict(update=step,loss=combined,lr=rate,grad_norm=float(norm.item()),**metrics)
