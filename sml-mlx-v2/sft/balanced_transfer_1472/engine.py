"""Assistant-only CE, labeled choice ranking, and frozen-parent prose anchor."""
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from sft.context_reasoning_1024.engine import (
    load, optimizer, learning_rate, choice_arrays, ranking_loss, reference_kl,
)
from sft.transfer_control.engine import encode, arrays, objective


def batch_at(data, cfg, cursor):
    if not 0 <= cursor < cfg['updates']:
        raise ValueError('Cursor out of range')
    result = {}
    for family, size in cfg['per_update'].items():
        result[family] = data['train'][family][cursor*size:(cursor+1)*size]
        if len(result[family]) != size:
            raise ValueError('Incomplete '+family)
    return result


def update(backend, anchor, opt, rows, prose_ids, cfg, step):
    backend.model.train(); anchor.model.eval()
    total = None; metrics = {}; combined = 0.

    def add(value, grad, scale):
        nonlocal total, combined
        grad = tree_map(lambda t: t.astype(mx.float32)*scale, grad)
        total = grad if total is None else tree_map(lambda a,b: a+b, total, grad)
        mx.eval(value, total); combined += float(value.item())*scale

    ce_fn = nn.value_and_grad(backend.model,
        lambda x,y: objective(backend.model,x,y,False))
    for family in ('new_answer','instruction','reading','explanation'):
        losses=[]; targets=0
        for row in rows[family]:
            enc=encode(backend,row,cfg['context']); targets+=enc['targets']
            x,y=arrays([enc],backend.pad); loss,grad=ce_fn(x,y)
            add(loss,grad,cfg['loss_weights'][family]/len(rows[family]))
            losses.append(float(loss.item()))
        metrics[family+'_ce']=sum(losses)/len(losses)
        metrics[family+'_targets']=targets

    # The old MC rows are ranked only, avoiding a one-word generic assistant.
    rank_rows = rows['new_answer']+rows['new_choice']+rows['old_choice']
    rank_fn = nn.value_and_grad(backend.model,
        lambda x,y,g: ranking_loss(backend.model,x,y,g,cfg['ranking_temperature']))
    rank_losses=[]
    for row in rank_rows:
        x,y=choice_arrays(backend,row)
        loss,grad=rank_fn(x,y,row['gold'])
        add(loss,grad,cfg['loss_weights']['ranking']/len(rank_rows))
        rank_losses.append(float(loss.item()))
    metrics['ranking_ce']=sum(rank_losses)/len(rank_losses)
    if len(prose_ids)<2:
        raise ValueError('Empty prose anchor')
    x=mx.array([prose_ids[:-1]],dtype=mx.int32)
    logits=anchor.model(x)['logits'].astype(mx.float32)
    ref=mx.stop_gradient(logits-mx.logsumexp(logits,axis=-1,keepdims=True));mx.eval(ref)
    kl_fn=nn.value_and_grad(backend.model,
        lambda xx,rr: reference_kl(backend.model,xx,rr))
    loss,grad=kl_fn(x,ref);add(loss,grad,cfg['kl_weight'])
    metrics['anchor_kl']=float(loss.item())
    from sml_v2.pretrain import clip_gradients
    total,norm=clip_gradients(total,cfg['clip_norm']);mx.eval(total,norm)
    if not all(math.isfinite(v) for v in [combined,float(norm.item()),*metrics.values()]):
        raise ValueError('Nonfinite objective/gradient')
    rate=learning_rate(step,cfg);opt.learning_rate=rate;opt.update(backend.model,total)
    mx.eval(backend.model.parameters(),opt.state)
    return dict(update=step,loss=combined,lr=rate,grad_norm=float(norm.item()),**metrics)
