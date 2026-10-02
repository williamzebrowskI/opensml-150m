"""Same visible format and token-weighted assistant objective on both base models."""
import gc,math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from evaluation.base_audit.scorer import Backend
from sft.transfer_control.engine import encode as original_encode,arrays,prefix
from sml_v1.precision import MasterAdamW


def load(arm):
    b=Backend('current' if arm=='opensml' else 'reference')
    if arm=='opensml':
        # Native GQA attention; reference FFN, external CE. No architecture change.
        for block in b.model.blocks:block.attn.attention_impl='fast';block.ffn.ffn_impl='reference'
    b.model.update(tree_map(lambda x:x.astype(mx.float32),b.model.parameters()))
    b.model.train();mx.eval(b.model.parameters());mx.set_cache_limit(512*1024**2)
    return b


def encode(b,row,cfg):return original_encode(b,row,cfg['context'])
def logits(model,x,reference):
    out=model(x);return out if reference else out['logits']


def objective(model,x,y,reference=False):
    mask=y!=-100
    ce=nn.losses.cross_entropy(logits(model,x,reference).astype(mx.float32),mx.maximum(y,0),reduction='none')
    return (ce*mask).sum()/mx.maximum(mask.sum(),1)


def gradients(model,x,y,reference,microbatch):
    total=int((y!=-100).sum().item())
    if total<1:raise ValueError('No assistant targets')
    fn=nn.value_and_grad(model,lambda xx,yy:objective(model,xx,yy,reference))
    acc=None;value=mx.array(0.,dtype=mx.float32)
    for i in range(0,len(x),microbatch):
        yy=y[i:i+microbatch];w=(yy!=-100).sum().astype(mx.float32)/total
        loss,grad=fn(x[i:i+microbatch],yy)
        grad=tree_map(lambda t:t.astype(mx.float32)*w,grad)
        acc=grad if acc is None else tree_map(lambda a,b:a+b,acc,grad)
        value=value+loss*w;mx.eval(value,acc)
    return value,acc


def optimizer(cfg):return MasterAdamW(learning_rate=0.,betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'])
def lr(step,cfg,peak):
    if not 1<=step<=cfg['updates']:raise ValueError('Invalid update index')
    if step<=cfg['warmup_updates']:return peak*step/cfg['warmup_updates']
    t=(step-cfg['warmup_updates'])/(cfg['updates']-cfg['warmup_updates'])
    end=peak*cfg['final_lr_fraction'];return end+(peak-end)*(1+math.cos(math.pi*t))/2


def update(b,opt,rows,cfg,step,peak):
    from sml_v1.pretrain import clip_gradients
    b.model.train();encoded=[encode(b,r,cfg) for r in rows];x,y=arrays(encoded,b.pad)
    value,grad=gradients(b.model,x,y,b.name=='reference',cfg['microbatch'])
    grad,norm=clip_gradients(grad,cfg['clip_norm']);mx.eval(value,grad,norm)
    if not math.isfinite(float(value.item())) or not math.isfinite(float(norm.item())):raise ValueError('Nonfinite loss/gradient')
    rate=lr(step,cfg,peak);opt.learning_rate=rate;opt.update(b.model,grad)
    mx.eval(b.model.parameters(),opt.state)
    return dict(step=step,loss=float(value.item()),grad_norm=float(norm.item()),lr=rate,assistant_targets=sum(r['targets'] for r in encoded))


def generate(b,prompt,limit=192):
    ids=b.encode(prefix(prompt));out=[];reason='length';b.model.eval()
    special=set(b.tokenizer.all_special_ids) if b.name=='reference' else {0,2,3}
    for _ in range(limit):
        if len(ids)>=2048:reason='context';break
        next_id=int(mx.argmax(b.logits(mx.array([ids],dtype=mx.int32))[0,-1]).item())
        if next_id==b.eos:reason='eos';break
        if next_id in special:reason='invalid_structure';break
        out.append(next_id);ids.append(next_id)
    return dict(text=b.decode(out).strip(),tokens=len(out),stop=reason)


def nll(b,rows,cfg):
    b.model.eval();loss=0.;count=0
    for row in rows:
        encoded=encode(b,row,cfg);x,y=arrays([encoded],b.pad);n=encoded['targets']
        loss+=float(objective(b.model,x,y,b.name=='reference').item())*n;count+=n
    return dict(nll=loss/count,targets=count,examples=len(rows))


def free(b,opt=None):
    del b,opt;gc.collect();mx.clear_cache()
