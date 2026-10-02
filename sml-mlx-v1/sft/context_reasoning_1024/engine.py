"""Group-balanced CE + supervised choice ranking + training-prose KL replay.

The anchor is checkpoint 1024, never a stronger teacher. Its parameters are not
part of the differentiated model. No rewards, PPO, DPO, or sampled targets.
"""
import math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from sml_v1.common import read_json
from sml_v1.precision import MasterAdamW
from sft.transfer_control.engine import encode, arrays, objective, prefix

ROOT = Path(__file__).resolve().parents[2]


def load(cfg, weights=None):
    from types import SimpleNamespace
    from sml_v1.model import TransformerConfig, TransformerLM
    from sml_v1.tokenization import Tokenizer
    architecture = read_json(ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json')['recipe']['model']
    architecture = dict(architecture, attention_impl='vanilla', ffn_impl='reference', ce_impl='reference', loss_dtype='float32')
    model = TransformerLM(TransformerConfig(**architecture))
    model.load_weights(str(weights or ROOT/cfg['source_bundle']/'model.safetensors'), strict=True)
    model.update(tree_map(lambda t: t.astype(mx.float32), model.parameters()))
    model.eval(); mx.eval(model.parameters()); mx.set_cache_limit(256*1024**2)
    tok = Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    return SimpleNamespace(model=model, tokenizer=tok, encode=tok.encode, decode=tok.decode,
                           eos=tok.eos, pad=tok.pad, name='opensml',
                           logits=lambda x: model(x)['logits'])


def optimizer(cfg):
    return MasterAdamW(learning_rate=0., betas=tuple(cfg['betas']), weight_decay=cfg['weight_decay'])


def learning_rate(step, cfg):
    if not 1 <= step <= cfg['updates']: raise ValueError('Invalid optimizer update')
    if step <= cfg['warmup_updates']: return cfg['peak_lr']*step/cfg['warmup_updates']
    phase = (step-cfg['warmup_updates'])/(cfg['updates']-cfg['warmup_updates'])
    return cfg['final_lr'] + (cfg['peak_lr']-cfg['final_lr'])*(1+math.cos(math.pi*phase))/2


def choice_arrays(backend, row):
    head = backend.encode(row['ranking_prompt']); encoded = []
    for choice in row['choices']:
        ids = backend.encode(row['ranking_prompt'] + ' ' + choice)
        if ids[:len(head)] != head or not len(head) < len(ids) <= 1024: raise ValueError('Choice boundary/length')
        encoded.append(dict(x=ids[:-1], y=[-100]*(len(head)-1)+ids[len(head):], targets=len(ids)-len(head)))
    return arrays(encoded, backend.pad)


def choice_scores(model, x, y):
    values = model(x)['logits'].astype(mx.float32)
    ce = nn.losses.cross_entropy(values, mx.maximum(y, 0), reduction='none')
    mask = y != -100
    return -(ce*mask).sum(axis=1)/mx.maximum(mask.sum(axis=1), 1)


def ranking_loss(model, x, y, gold, temperature):
    scores = choice_scores(model, x, y)/temperature
    return mx.logsumexp(scores)-scores[gold]


def reference_kl(model, x, reference_logp):
    logits = model(x)['logits'].astype(mx.float32)
    policy_logp = logits-mx.logsumexp(logits, axis=-1, keepdims=True)
    reference_logp = mx.stop_gradient(reference_logp)
    return (mx.exp(reference_logp)*(reference_logp-policy_logp)).sum(axis=-1).mean()


def batch_at(data, cfg, cursor):
    if not 0 <= cursor < cfg['updates']: raise ValueError('Cursor out of range')
    result = {}
    for family, size in cfg['per_update'].items():
        result[family] = data['train'][family][cursor*size:(cursor+1)*size]
        if len(result[family]) != size: raise ValueError('Incomplete family batch')
    return result


def update(backend, anchor, opt, rows, prose_ids, cfg, step):
    backend.model.train(); anchor.model.eval()
    total = None; metrics = {}; combined = 0.

    def add(value, grad, scale):
        nonlocal total, combined
        grad = tree_map(lambda t: t.astype(mx.float32)*scale, grad)
        total = grad if total is None else tree_map(lambda a,b:a+b, total, grad)
        mx.eval(value, total); combined += float(value.item())*scale

    # Normalize per example, then per family. Long natural answers cannot drown
    # out short reading targets through their token count.
    fn = nn.value_and_grad(backend.model, lambda x,y: objective(backend.model, x, y, False))
    for family, rr in rows.items():
        if cfg['loss_weights'][family] == 0:
            continue  # QA choices teach ranking, never terse assistant completions.
        values = []; targets = 0
        for row in rr:
            enc = encode(backend, row, cfg['context']); targets += enc['targets']
            x,y = arrays([enc], backend.pad); loss,grad = fn(x,y)
            add(loss, grad, cfg['loss_weights'][family]/len(rr)); values.append(float(loss.item()))
        metrics[family+'_ce'] = sum(values)/len(values)
        metrics[family+'_targets'] = targets
    # Every labeled QA row contributes a choice-ranking gradient. No EOS or
    # assistant-prefix CE is imposed on these often one-word answer options.
    ranking_values=[]
    for row in rows['commonsense']:
        x,y = choice_arrays(backend, row)
        fn = nn.value_and_grad(backend.model, lambda a,b: ranking_loss(backend.model,a,b,row['gold'],cfg['ranking_temperature']))
        loss,grad = fn(x,y)
        add(loss,grad,cfg['loss_weights']['ranking']/len(rows['commonsense']))
        ranking_values.append(float(loss.item()))
    metrics['ranking_ce']=sum(ranking_values)/len(ranking_values)
    if len(prose_ids)<2: raise ValueError('Empty prose anchor')
    x = mx.array([prose_ids[:-1]], dtype=mx.int32)
    logits = anchor.model(x)['logits'].astype(mx.float32)
    ref = mx.stop_gradient(logits-mx.logsumexp(logits,axis=-1,keepdims=True)); mx.eval(ref)
    fn = nn.value_and_grad(backend.model,lambda xx,rr:reference_kl(backend.model,xx,rr))
    loss,grad = fn(x,ref); add(loss,grad,cfg['kl_weight']); metrics['anchor_kl'] = float(loss.item())
    from sml_v1.pretrain import clip_gradients
    total,norm = clip_gradients(total,cfg['clip_norm']); mx.eval(total,norm)
    if not all(math.isfinite(v) for v in [combined,float(norm.item()),*metrics.values()]): raise ValueError('Nonfinite objective/gradient')
    rate=learning_rate(step,cfg); opt.learning_rate=rate; opt.update(backend.model,total)
    mx.eval(backend.model.parameters(),opt.state)
    return dict(update=step,loss=combined,lr=rate,grad_norm=float(norm.item()),**metrics)
