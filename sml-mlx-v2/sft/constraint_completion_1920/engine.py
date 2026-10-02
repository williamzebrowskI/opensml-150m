"""Complete-answer CE plus 1920 rehearsal; fresh FP32 optimizer, frozen prose KL."""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from sft.skill_balance.engine import (load, optimizer, learning_rate, batch_at,
    choice_arrays, choice_scores, ranking_loss, reference_kl)
from sft.transfer_control.engine import encode, arrays, objective, prefix


def update(b, anchor, opt, rows, prose_ids, cfg, step):
    b.model.train();anchor.model.eval();total=None;combined=0.;metrics={}
    def add(value,grad,weight):
        nonlocal total,combined
        grad=tree_map(lambda v:v.astype(mx.float32)*weight,grad)
        total=grad if total is None else tree_map(lambda x,y:x+y,total,grad)
        mx.eval(value,total);combined+=float(value.item())*weight
    fn=nn.value_and_grad(b.model,lambda x,y:objective(b.model,x,y,False))
    for family,rr in rows.items():
        if cfg['loss_weights'][family]==0:continue
        values=[]
        for row in rr:
            x,y=arrays([encode(b,row,cfg['context'])],b.pad);loss,grad=fn(x,y)
            add(loss,grad,cfg['loss_weights'][family]/len(rr));values.append(float(loss.item()))
        metrics[family+'_ce']=sum(values)/len(values)
    values=[]
    for row in rows['commonsense']:
        x,y=choice_arrays(b,row)
        fn=nn.value_and_grad(b.model,lambda xx,yy:ranking_loss(b.model,xx,yy,row['gold'],cfg['ranking_temperature']))
        loss,grad=fn(x,y);add(loss,grad,cfg['loss_weights']['ranking']/len(rows['commonsense']));values.append(float(loss.item()))
    metrics['ranking_ce']=sum(values)/len(values)
    if len(prose_ids)<2:raise ValueError('Empty prose anchor')
    x=mx.array([prose_ids[:-1]],dtype=mx.int32)
    logits=anchor.model(x)['logits'].astype(mx.float32)
    ref=mx.stop_gradient(logits-mx.logsumexp(logits,axis=-1,keepdims=True));mx.eval(ref)
    fn=nn.value_and_grad(b.model,lambda xx,rr:reference_kl(b.model,xx,rr))
    loss,grad=fn(x,ref);add(loss,grad,cfg['kl_weight']);metrics['anchor_kl']=float(loss.item())
    from sml_v2.pretrain import clip_gradients
    total,norm=clip_gradients(total,cfg['clip_norm']);mx.eval(total,norm)
    if not all(math.isfinite(v) for v in [combined,float(norm.item()),*metrics.values()]):raise ValueError('Nonfinite objective')
    lr=learning_rate(step,cfg);opt.learning_rate=lr;opt.update(b.model,total);mx.eval(b.model.parameters(),opt.state)
    return dict(update=step,loss=combined,grad_norm=float(norm.item()),lr=lr,**metrics)
