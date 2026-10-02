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
    from sml_v1.pretrain import clip_gradients
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


def assess(b,data,cfg):
    b.model.eval();weighted=0.;targets=0;conv_losses=[]
    for row in data['dev'][:cfg['validation_conversations']]:
        losses=[]
        for e in turns(b.tokenizer,row,cfg['context']):
            x,y=arrays([e],b.pad);positions=mx.array([i for i,t in enumerate(e['y']) if t!=-100])
            loss=nn.losses.cross_entropy(b.model(x)['logits'][0,positions].astype(mx.float32),y[0,positions],reduction='mean')
            mx.eval(loss);v=float(loss.item());losses.append(v);weighted+=v*e['targets'];targets+=e['targets']
        conv_losses.append(sum(losses)/len(losses))
    answers=[]
    # Fixed source-balanced development sample, separate from public MT-Bench.
    per_source=cfg['generation_conversations']//len(cfg['training_counts'])
    selected=[row for s in cfg['training_counts'] for row in [r for r in data['dev'] if r['source']==s][:per_source]]
    for row in selected:
        history=[];turn_answers=[]
        for m in row['messages']:
            if m['role']=='system':history.append(m);continue
            if m['role']!='user':continue
            history.append(m);g=generate(b,history,cfg['max_new_tokens'])
            turn_answers.append(dict(prompt=m['content'],**g));history.append(dict(role='assistant',content=g['text']))
            if len(turn_answers)==2 or not g['text'].strip():break
        answers.append(dict(id=row['id'],source=row['source'],reference_messages=row['messages'],turns=turn_answers))
    generated=[g for a in answers for g in a['turns']]
    metrics=dict(validation_token_nll=weighted/targets,validation_conversation_nll=sum(conv_losses)/len(conv_losses),
                 validation_conversations=len(conv_losses),validation_targets=targets,
                 generated_turns=len(generated),stopped=sum(g['stop']=='eos' for g in generated)/len(generated),
                 repeated=sum(repeated(g['tokens']) for g in generated)/len(generated),
                 empty=sum(not g['text'].strip() for g in generated)/len(generated),
                 mean_generated_tokens=sum(len(g['tokens']) for g in generated)/len(generated),
                 stop_counts=dict(Counter(g['stop'] for g in generated)))
    return dict(metrics=metrics,answers=answers,public_benchmarks=False,
                note='Likelihood/stopping/repetition do not score relevance or correctness; review saved answers. Reserved test not evaluated.')
