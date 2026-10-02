"""Branch A: equal-example assistant CE, including EOS; no UL, RL or KL."""
import math,random
import mlx.core as mx
from mlx.utils import tree_map
from sft.skill_balance.engine import load,optimizer,learning_rate
from sft.transfer_control.engine import encode,arrays,gradients,objective
from sml_v2.pretrain import clip_gradients

def batch_at(data,cfg,update):
    count=len(data['train']);per_epoch=count//cfg['batch']
    if count%cfg['batch'] or not 0<=update<cfg['updates']:raise ValueError('Invalid data cursor')
    epoch,offset=divmod(update,per_epoch)
    order=list(range(count));random.Random(cfg['seed']+epoch).shuffle(order)
    return [data['train'][i] for i in order[offset*cfg['batch']:(offset+1)*cfg['batch']]]

def update(b,opt,rows,cfg,step):
    b.model.train();encoded=[encode(b,r,cfg['context']) for r in rows]
    x,y=arrays(encoded,b.pad)
    value,grad=gradients(b.model,x,y,reference=False,microbatch=1)
    grad,norm=clip_gradients(grad,cfg['clip_norm']);mx.eval(value,grad,norm)
    if not math.isfinite(float(value.item())) or not math.isfinite(float(norm.item())):raise ValueError('Nonfinite training values')
    if not all(bool(mx.all(mx.isfinite(t)).item()) for _,t in __import__('mlx.utils',fromlist=['tree_flatten']).tree_flatten(grad)):raise ValueError('Nonfinite gradient')
    lr=learning_rate(step,cfg);opt.learning_rate=lr;opt.update(b.model,grad);mx.eval(b.model.parameters(),opt.state)
    return dict(update=step,loss=float(value.item()),lr=lr,grad_norm=float(norm.item()),assistant_targets=sum(e['targets'] for e in encoded))
