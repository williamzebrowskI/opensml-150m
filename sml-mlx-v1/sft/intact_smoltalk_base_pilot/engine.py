"""Full-history assistant/EOS token CE, variable-length microbatches, fresh FP32 AdamW."""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from sft.skill_balance.engine import load, optimizer, learning_rate as scheduled_learning_rate
from sft.transfer_control.engine import arrays
from .data import turns, visible_prefix


def learning_rate(step,cfg):
    # Match the first 128 updates of the previous schedule; only the parent and
    # maximum production budget change. This is not a new shortened cosine.
    if not 1 <= step <= cfg['updates']:raise ValueError('Invalid pilot update')
    return scheduled_learning_rate(step,dict(cfg,updates=cfg['lr_schedule_updates']))


def objective(model,x,y):
    # Gather supervised positions before vocabulary-sized CE; do not score the
    # duplicated historical replies, role labels, system/user text, or padding.
    import numpy as np
    positions = np.flatnonzero(np.asarray(y[0]) != -100)
    logits = model(x)['logits'][0,mx.array(positions)].astype(mx.float32)
    return nn.losses.cross_entropy(logits,y[0,mx.array(positions)],reduction='mean')


def update(b,opt,rows,cfg,step):
    from sml_v1.pretrain import clip_gradients
    b.model.train(); encoded = [e for r in rows for e in turns(b.tokenizer,r,cfg['context'])]
    target_count = sum(r['targets'] for r in encoded)
    acc = None; combined = 0.
    # Index positions are passed as ordinary arrays to the differentiated graph;
    # no conversion of a traced array to numpy inside autodiff.
    def loss_fn(x,y,positions):
        scores = b.model(x)['logits'][0,positions].astype(mx.float32)
        return nn.losses.cross_entropy(scores,y[0,positions],reduction='mean')
    fn = nn.value_and_grad(b.model,loss_fn)
    for row in encoded:
        x,y = arrays([row],b.pad)
        positions = mx.array([i for i,t in enumerate(row['y']) if t!=-100],dtype=mx.int32)
        loss,grad = fn(x,y,positions); w = row['targets']/target_count
        grad = tree_map(lambda t:t.astype(mx.float32)*w,grad)
        acc = grad if acc is None else tree_map(lambda a,c:a+c,acc,grad)
        mx.eval(loss,acc); combined += float(loss.item())*w
    acc,norm = clip_gradients(acc,cfg['clip_norm']); mx.eval(acc,norm)
    if not math.isfinite(combined) or not math.isfinite(float(norm.item())):
        raise ValueError('Nonfinite loss or gradient')
    rate = learning_rate(step,cfg); opt.learning_rate = rate
    opt.update(b.model,acc); mx.eval(b.model.parameters(),opt.state)
    return dict(update=step,loss=combined,lr=rate,grad_norm=float(norm.item()),
                conversations=len(rows),assistant_turns=len(encoded),assistant_targets=target_count)


def generate(b,messages,limit,cancelled=lambda:False,full=False):
    ids = b.encode(visible_prefix(messages))
    if len(ids)+limit>2048: raise ValueError('Generation budget would truncate history')
    b.model.eval(); output = []; stop='length'
    scores,cache=b.model.logits(mx.array([ids],dtype=mx.int32));scores=scores[:,-1,:];mx.eval(scores,cache)
    for _ in range(limit):
        if cancelled(): raise InterruptedError('Stopped during generation')
        token=int(mx.argmax(scores[0]).item())
        if token==b.eos:stop='end';break
        if token in (0,2,3):stop='invalid_structure';break
        output.append(token)
        if len(output)<limit:
            if full:
                scores=b.model.logits(mx.array([ids+output],dtype=mx.int32))[0][:,-1,:]
            else:
                scores,cache=b.model.step(mx.array([[token]],dtype=mx.int32),caches=cache)
            mx.eval(scores)
    return dict(text=b.decode(output),tokens=len(output),token_ids=output,stop=stop)
