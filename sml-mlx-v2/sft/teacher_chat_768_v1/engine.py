"""Conversation-balanced assistant-only SFT; every complete reply includes EOS."""
import math
from collections import Counter
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from sft.skill_balance.engine import load, optimizer, learning_rate
from sft.transfer_control.engine import arrays
from .data import turns, visible_prefix


def update(b,opt,rows,cfg,step):
    from sml_v2.pretrain import clip_gradients
    b.model.train(); acc=None; combined=0.; targets=0; count=0
    def loss_fn(x,y,positions):
        logits=b.model(x)['logits'][0,positions].astype(mx.float32)
        return nn.losses.cross_entropy(logits,y[0,positions],reduction='mean')
    fn=nn.value_and_grad(b.model,loss_fn)
    for row in rows:
        encoded=turns(b.tokenizer,row,cfg['context'])
        # Each conversation has equal weight; each of its assistant turns shares it.
        weight=1/(len(rows)*len(encoded))
        for e in encoded:
            x,y=arrays([e],b.pad)
            positions=mx.array([i for i,t in enumerate(e['y']) if t!=-100],dtype=mx.int32)
            loss,grad=fn(x,y,positions)
            grad=tree_map(lambda v:v.astype(mx.float32)*weight,grad)
            acc=grad if acc is None else tree_map(lambda a,c:a+c,acc,grad)
            mx.eval(loss,acc);combined+=float(loss.item())*weight
            targets+=e['targets'];count+=1
    acc,norm=clip_gradients(acc,cfg['clip_norm']);mx.eval(acc,norm)
    if not math.isfinite(combined) or not math.isfinite(norm.item()): raise ValueError('Nonfinite update')
    rate=learning_rate(step,cfg);opt.learning_rate=rate;opt.update(b.model,acc)
    mx.eval(b.model.parameters(),opt.state)
    return dict(update=step,loss=combined,lr=rate,grad_norm=float(norm.item()),
                conversations=len(rows),assistant_turns=count,assistant_targets=targets)


def generate(b,messages,limit,full=False):
    ids=b.encode(visible_prefix(messages));budget=min(limit,2048-len(ids))
    if budget<=0:return dict(text='',tokens=[],stop='context_limit')
    b.model.eval();out=[];stop='context_limit' if budget<limit else 'length'
    logits,cache=b.model.logits(mx.array([ids],dtype=mx.int32));scores=logits[:,-1,:];mx.eval(scores,cache)
    for _ in range(budget):
        token=int(mx.argmax(scores[0]).item())
        if token==b.eos:stop='eos';break
        out.append(token)
        if len(out)<budget:
            if full:
                logits,_=b.model.logits(mx.array([ids+out],dtype=mx.int32));scores=logits[:,-1,:]
            else:
                scores,cache=b.model.step(mx.array([[token]],dtype=mx.int32),caches=cache)
                if scores.ndim==3:scores=scores[:,-1,:]
            mx.eval(scores)
    return dict(text=b.decode(out),tokens=out,stop=stop)


def repeated(tokens):
    counts=Counter(tuple(tokens[i:i+4]) for i in range(len(tokens)-3))
    return bool(counts and sum(n-1 for n in counts.values())/max(1,len(tokens)-3)>.2)


def assess(b,data,cfg,smoke=False):
    from sml_v2.common import read_json
    from .data import ROOT,PARENT
    from sft.text_followup_512_v1.engine import assess as retention_assess
    old=read_json(ROOT/'sft/text_followup_512_v1/config.json')
    if smoke:old.update(validation_conversations=2,generation_conversations=5,constraint_eval_conversations=1,max_new_tokens=8)
    result=retention_assess(b,read_json(PARENT)['data'],old)
    fresh=[]
    for row in data['scenarios']['dev'][:1 if smoke else None]:
        history=list(row['prefix']);generated=[]
        for prompt in (None,row['followup']):
            if prompt is not None:history.append(dict(role='user',content=prompt))
            g=generate(b,history,8 if smoke else cfg['max_new_tokens'])
            generated.append(dict(history=list(history),**g))
            history.append(dict(role='assistant',content=g['text']))
            if not g['text'].strip():break
        fresh.append(dict(group=row['group'],category=row['category'],turns=generated))
        if len(fresh)%8==0:print('[fresh-answers]',len(fresh),'conversations',flush=True)
    result['fresh_answers']=fresh
    ts=[t for a in fresh for t in a['turns']]
    result['fresh_metrics']=dict(turns=len(ts),stopped=sum(t['stop']=='eos' for t in ts)/len(ts),
                                repeated=sum(repeated(t['tokens']) for t in ts)/len(ts),
                                mean_tokens=sum(len(t['tokens']) for t in ts)/len(ts))
    result['note']+=' New source-group-disjoint development conversations are generated with the model’s own history. Reserved test prompts are not evaluated.'
    return result
