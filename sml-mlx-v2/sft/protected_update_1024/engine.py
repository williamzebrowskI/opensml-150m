"""Experimental Adam-step projection onto two replay-loss halfspaces.

Inspired by GEM/A-GEM, not an exact implementation of either paper. Projection
is applied AFTER Adam preconditioning; moments follow the raw objective, while
FP32 masters follow accepted parameters. Constraints are first-order and sampled.
"""
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map,tree_flatten,tree_unflatten
from sft.skill_recovery_768.engine import (load,optimizer,learning_rate,choice_arrays,choice_scores,ranking_loss,reference_kl,batch_at,encode,arrays,objective,prefix)

def dot(a,b):
    aa=dict(tree_flatten(a));bb=dict(tree_flatten(b))
    if aa.keys()!=bb.keys():raise ValueError('Gradient structure mismatch')
    return sum((aa[k].astype(mx.float32)*bb[k].astype(mx.float32)).sum() for k in aa)

def coefficients(gram, products):
    """Projection min ||d'-d||^2 s.t. g_i dot d' <= 0 (two constraints).

Return nonnegative multipliers for d' = d - sum_i lambda_i g_i.
Enumerate the empty, one-active and two-active KKT solutions in float64.
"""
    G=np.asarray(gram,dtype=np.float64); b=np.asarray(products,dtype=np.float64)
    if G.shape!=(2,2) or b.shape!=(2,) or not(np.isfinite(G).all() and np.isfinite(b).all()):raise ValueError('Invalid projection geometry')
    scales=np.sqrt(np.maximum(np.diag(G),0)); active=scales>1e-15
    if not active.any() or np.all(b[active]<=0):return np.zeros(2)
    scale=np.where(active,scales,1.)
    A=G/scale[:,None]/scale[None,:]; z=b/scale
    tol=1e-9*max(float(np.max(np.abs(z))),1e-20)
    candidates=[]
    for i in range(2):
        if not active[i]:continue
        lam=np.zeros(2);lam[i]=max(z[i],0)/A[i,i]
        if np.all((z-A@lam)[active]<=tol):candidates.append(lam)
    lam=np.linalg.lstsq(A,z,rcond=1e-12)[0]
    if np.all(lam>=-tol) and np.all((z-A@np.maximum(lam,0))[active]<=tol):candidates.append(np.maximum(lam,0))
    if not candidates:raise ValueError('No feasible replay projection; refusing an unconstrained update')
    best=min(candidates,key=lambda v:float(v@A@v))
    return best/scale

def project(delta, guards):
    G=[[float(dot(a,b).item()) for b in guards] for a in guards]
    products=[float(dot(g,delta).item()) for g in guards]
    lam=coefficients(G,products)
    projected=tree_map(lambda d,a,b:d-float(lam[0])*a-float(lam[1])*b,delta,*guards)
    mx.eval(projected)
    return projected,dict(would_conflict=int(any(v>0 for v in products)),projected=int(any(lam>0)),
      format_direction_before=products[0],reply_direction_before=products[1],
      format_direction_after=float(dot(guards[0],projected).item()),reply_direction_after=float(dot(guards[1],projected).item()))

def sync_masters(opt,parameters):
    state=dict(tree_flatten(opt.state))
    for name,value in tree_flatten(parameters):
        key=name+'.master'
        if key not in state:raise ValueError('Missing FP32 master '+key)
        state[key]=value.astype(mx.float32)
    opt.state=tree_unflatten(list(state.items()))

def update(backend,anchor,opt,rows,prose_ids,cfg,step):
    backend.model.train();anchor.model.eval(); total=None;metrics={};combined=0.;guards=[]
    def add(value,grad,scale):
        nonlocal total,combined
        scaled=tree_map(lambda t:t.astype(mx.float32)*scale,grad)
        total=scaled if total is None else tree_map(lambda a,b:a+b,total,scaled)
        mx.eval(value,total);combined+=float(value.item())*scale
    fn=nn.value_and_grad(backend.model,lambda x,y:objective(backend.model,x,y,False))
    # Keep two independent guards: content/format exercises and complete replies.
    for family in ('instruction','reading'):
        rr=rows[family];grad_sum=None;values=[];targets=0
        for row in rr:
            enc=encode(backend,row,cfg['context']);targets+=enc['targets']
            x,y=arrays([enc],backend.pad);v,g=fn(x,y)
            scaled=tree_map(lambda t:t.astype(mx.float32)/len(rr),g)
            grad_sum=scaled if grad_sum is None else tree_map(lambda a,b:a+b,grad_sum,scaled)
            mx.eval(v,grad_sum);values.append(float(v.item()))
        v=mx.array(sum(values)/len(values));add(v,grad_sum,cfg['loss_weights'][family]);guards.append(grad_sum)
        metrics[family+'_ce']=float(v.item());metrics[family+'_targets']=targets
    values=[]
    for row in rows['commonsense']:
        x,y=choice_arrays(backend,row)
        fn=nn.value_and_grad(backend.model,lambda a,b:ranking_loss(backend.model,a,b,row['gold'],cfg['ranking_temperature']))
        v,g=fn(x,y);add(v,g,cfg['loss_weights']['ranking']/len(rows['commonsense']));values.append(float(v.item()))
    metrics['ranking_ce']=sum(values)/len(values)
    x=mx.array([prose_ids[:-1]],dtype=mx.int32)
    logits=anchor.model(x)['logits'].astype(mx.float32)
    ref=mx.stop_gradient(logits-mx.logsumexp(logits,axis=-1,keepdims=True));mx.eval(ref)
    fn=nn.value_and_grad(backend.model,lambda x,r:reference_kl(backend.model,x,r))
    v,g=fn(x,ref);add(v,g,cfg['kl_weight']);metrics['anchor_kl']=float(v.item())
    from sml_v2.pretrain import clip_gradients
    total,norm=clip_gradients(total,cfg['clip_norm']);mx.eval(total,norm)
    if not all(math.isfinite(v) for v in [combined,float(norm.item()),*metrics.values()]):raise ValueError('Nonfinite gradients')
    # FP32 arrays are immutable; retain pre-update tree to compute actual Adam step.
    before=tree_map(lambda t:t,backend.model.parameters());mx.eval(before)
    rate=learning_rate(step,cfg);opt.learning_rate=rate;opt.update(backend.model,total);mx.eval(backend.model.parameters(),opt.state)
    delta=tree_map(lambda a,b:a-b,backend.model.parameters(),before);mx.eval(delta)
    adjusted,pmetrics=project(delta,guards)
    if cfg['arm']=='protected' and pmetrics['projected']:
        updated=tree_map(lambda p,d:p+d,before,adjusted);mx.eval(updated)
        # Account for FP32 rounding of the final parameters, not only ideal delta.
        actual=tree_map(lambda p,q:p-q,updated,before)
        residual=[float(dot(g,actual).item()) for g in guards]
        norms=[math.sqrt(max(float(dot(g,g).item()),0.)) for g in guards]
        dn=math.sqrt(max(float(dot(delta,delta).item()),0.))
        if any(v>cfg['projection_tolerance']*max(n*dn,1e-12) for v,n in zip(residual,norms)):
            # Fail closed rather than silently applying a materially conflicting step.
            updated=before;residual=[0.,0.];pmetrics['roundoff_skipped']=1
        else:pmetrics['roundoff_skipped']=0
        backend.model.update(updated);sync_masters(opt,backend.model.parameters());mx.eval(backend.model.parameters(),opt.state)
        pmetrics['format_direction_after'],pmetrics['reply_direction_after']=residual
    else:
        pmetrics['roundoff_skipped']=0;pmetrics['projected']=0
        pmetrics['format_direction_after']=pmetrics['format_direction_before'];pmetrics['reply_direction_after']=pmetrics['reply_direction_before']
    return dict(update=step,loss=combined,lr=rate,grad_norm=float(norm.item()),**metrics,**pmetrics)
