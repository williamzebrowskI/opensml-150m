"""On-policy sequence UL; no gradients through decoding or reference answers.

Penalize token spans belonging to second/subsequent occurrences of generated
four-token n-grams. Prompt/history, EOS and reference-answer tokens are never
negative labels. This is a bounded adaptation, not a paper reproduction.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map,tree_flatten
from sft.conversation_ab.engine import load,optimizer,learning_rate,batch_at,encode,arrays,gradients
from sft.transfer_control.engine import objective
from sft.reading_repair.generation import generate
from sml_v2.pretrain import clip_gradients

def repetition_mask(tokens,ngram=4,special=(0,1,2,3)):
    seen=set();mask=[False]*len(tokens)
    for end in range(ngram,len(tokens)+1):
        span=tuple(tokens[end-ngram:end])
        if any(t in special for t in span):continue
        if span in seen:
            for j in range(end-ngram,end):mask[j]=True
        seen.add(span)
    return mask

def negative_row(b,prompt,tokens,ngram):
    head=b.encode('User: '+prompt+'\nAssistant:')
    mask=repetition_mask(tokens,ngram,(b.pad,b.eos,2,3))
    labels=[t if flag else -100 for t,flag in zip(tokens,mask)]
    return dict(x=(head+tokens)[:-1],y=[-100]*(len(head)-1)+labels,targets=sum(mask))

def ul_objective(model,x,y,epsilon):
    mask=y!=-100
    ce=nn.losses.cross_entropy(model(x)['logits'].astype(mx.float32),mx.maximum(y,0),reduction='none')
    # Clamped at p=1 for finite FP32 behavior. No EOS target enters this loss.
    penalties=-mx.log(mx.maximum(1-mx.exp(-ce),epsilon))
    counts=mask.sum(axis=1)
    return ((penalties*mask).sum(axis=1)/mx.maximum(counts,1)).mean()

def collect(b,rows,cfg):
    ul=cfg['unlikelihood'];encoded=[];records=[]
    for r in rows:
        if r['source']=='rehearsal':continue
        g=generate(b,r['prompt'],ul['max_new_tokens'])
        e=negative_row(b,r['prompt'],g['token_ids'],ul['ngram'])
        if len(e['x'])>cfg['context']:raise ValueError('UL context overflow')
        encoded.append(e)
        records.append(dict(id=r['id'],generation=g,negative_tokens=e['targets']))
    return encoded,records

def ul_gradients(model,encoded,cfg):
    # Zero-repeat sequences contribute zero; normalize over ALL collected rows.
    total=None;value=mx.array(0.,dtype=mx.float32)
    fn=nn.value_and_grad(model,lambda x,y:ul_objective(model,x,y,cfg['unlikelihood']['epsilon']))
    for e in encoded:
        if not e['targets']:continue
        x,y=arrays([e]);loss,grad=fn(x,y)
        grad=tree_map(lambda t:t.astype(mx.float32)/len(encoded),grad)
        total=grad if total is None else tree_map(lambda a,b:a+b,total,grad)
        value=value+loss/len(encoded);mx.eval(value,total)
    return value,total

def update(b,opt,rows,cfg,step):
    negatives,records=collect(b,rows,cfg)
    b.model.train();encoded=[encode(b,r,cfg['context']) for r in rows]
    x,y=arrays(encoded,b.pad);ce,grad=gradients(b.model,x,y,reference=False,microbatch=1)
    ul,ugrad=ul_gradients(b.model,negatives,cfg)
    if ugrad is not None:grad=tree_map(lambda c,u:c+cfg['unlikelihood']['weight']*u,grad,ugrad)
    value=ce+cfg['unlikelihood']['weight']*ul
    grad,norm=clip_gradients(grad,cfg['clip_norm']);mx.eval(value,ce,ul,grad,norm)
    if not all(math.isfinite(float(v.item())) for v in (value,ce,ul,norm)):raise ValueError('Nonfinite loss/gradient norm')
    if not all(bool(mx.all(mx.isfinite(t)).item()) for _,t in tree_flatten(grad)):raise ValueError('Nonfinite gradient')
    opt.learning_rate=learning_rate(step,cfg);opt.update(b.model,grad);mx.eval(b.model.parameters(),opt.state)
    return dict(update=step,loss=float(value.item()),ce=float(ce.item()),ul=float(ul.item()),lr=learning_rate(step,cfg),grad_norm=float(norm.item()),assistant_targets=sum(e['targets'] for e in encoded),negative_tokens=sum(e['targets'] for e in negatives),rollouts=records)
