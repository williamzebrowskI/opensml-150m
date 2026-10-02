"""Shared assistant-only objective for both backends, with identical text format."""
import math
from collections import Counter, defaultdict
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from evaluation.base_audit.scorer import Backend
from .data import normalized, OPEN


def load(name):
    backend = Backend('current' if name == 'opensml' else 'reference')
    backend.model.update(tree_map(lambda x: x.astype(mx.bfloat16), backend.model.parameters()))
    mx.eval(backend.model.parameters())
    backend.model.train()
    return backend


def prefix(prompt):
    # Same visible format on both bases, no newly added vocabulary or untrained
    # chat role IDs. This differs from the deleted run and must be reported.
    return f'User: {prompt}\nAssistant:'


def encode(backend, row, context=2048):
    p = prefix(row['prompt'])
    head = backend.encode(p)
    ids = backend.encode(p + ' ' + row['answer'])
    if not head or ids[:len(head)] != head: raise ValueError('Tokenization changed at answer boundary')
    ids += [backend.eos]
    if not row['answer'].strip() or len(ids) - 1 > context: raise ValueError('Empty answer or context overflow')
    y = [-100] * (len(head)-1) + ids[len(head):]
    assert len(y) == len(ids)-1 and sum(t != -100 for t in y) == len(ids)-len(head)
    return dict(x=ids[:-1], y=y, targets=len(ids)-len(head))


def arrays(rows, pad=0):
    width = max(len(r['x']) for r in rows)
    x = np.full((len(rows), width), pad, dtype=np.int32)
    y = np.full_like(x, -100)
    for i, r in enumerate(rows):
        x[i, :len(r['x'])] = r['x']; y[i, :len(r['y'])] = r['y']
    return mx.array(x), mx.array(y)


def logits(model, x, reference):
    out = model(x)
    return out if reference else out['logits']


def objective(model, x, y, reference=False):
    mask = y != -100
    values = nn.losses.cross_entropy(logits(model, x, reference).astype(mx.float32), mx.maximum(y, 0), reduction='none')
    counts = mask.sum(axis=1)
    values = (values * mask).sum(axis=1) / mx.maximum(counts, 1)
    return values.sum() / mx.maximum((counts > 0).sum(), 1)


def gradients(model, x, y, reference=False, microbatch=2):
    count = int(mx.any(y != -100, axis=1).sum().item())
    if count < 1 or microbatch < 1: raise ValueError('No supervised examples')
    fn = nn.value_and_grad(model, lambda a, b: objective(model, a, b, reference))
    total, value = None, mx.array(0., dtype=mx.float32)
    for i in range(0, len(x), microbatch):
        a, b = x[i:i+microbatch], y[i:i+microbatch]
        weight = mx.any(b != -100, axis=1).sum().astype(mx.float32) / count
        loss, grad = fn(a, b)
        grad = tree_map(lambda t: t.astype(mx.float32) * weight, grad)
        total = grad if total is None else tree_map(lambda a, b: a + b, total, grad)
        value = value + loss * weight
        mx.eval(value, total)
    return value, total


def learning_rate(step, cfg):
    warm = cfg['warmup_updates']
    if step <= warm: return cfg['peak_lr'] * step / warm
    t = min(1., (step-warm) / (cfg['updates']-warm))
    return cfg['final_lr'] + (cfg['peak_lr']-cfg['final_lr']) * (1+math.cos(math.pi*t))/2


def generate(backend, prompt, limit=48):
    ids = backend.encode(prefix(prompt)); output = []; stop = 'length'
    # Deliberately uncached for a common, easily checked path on both models.
    backend.model.eval()
    try:
        for _ in range(limit):
            if len(ids) >= 2048: raise ValueError('Generation context overflow')
            token = int(mx.argmax(backend.logits(mx.array([ids], dtype=mx.int32))[0, -1]).item())
            if token == backend.eos: stop = 'end'; break
            if (backend.name != 'reference' and token in (0, 2, 3)) or (
                backend.name == 'reference' and token in backend.tokenizer.all_special_ids):
                stop = 'invalid_structure'; break
            ids.append(token); output.append(token)
    finally: backend.model.train()
    return dict(text=backend.decode(output).strip(), stop=stop, tokens=len(output))


def grade(row, generation):
    text = normalized(generation['text'])
    exact = generation['stop'] == 'end' and text in [normalized(s) for s in row['references']]
    # Lenient factual-content diagnostic, not semantic correctness: still read
    # raw output for unsupported extra claims. Never selects checkpoints.
    mentions = any(normalized(ref) in text for ref in row['references'])
    foil = normalized(row['foil'])
    foil_present = f' {foil} ' in f' {text} '
    return dict(exact=exact, fact_mention=mentions and not foil_present,
                prompt_echo=text == normalized(row['prompt']) or text == '',
                foil_present=foil_present)


def evaluate(backend, rows, limit=48, include_open=True):
    records, families, pairs = [], defaultdict(list), defaultdict(list)
    encoded = [encode(backend, r) for r in rows]
    losses = []
    backend.model.eval()
    try:
        for i in range(0, len(encoded), 2):
            batch = encoded[i:i+2]; x, y = arrays(batch, backend.pad)
            losses.append((float(objective(backend.model, x, y, backend.name == 'reference').item()), len(batch)))
    finally: backend.model.train()
    for row in rows:
        gen = generate(backend, row['prompt'], limit)
        rec = dict(**row, **gen, **grade(row, gen)); records.append(rec)
        families[row['family']].append(rec); pairs[row['pair']].append(rec)
    if any(len(rr) != 2 for rr in pairs.values()): raise ValueError('Incomplete paired evaluation')
    stats = {f: dict(count=len(rr), exact=sum(r['exact'] for r in rr)/len(rr),
                     fact_mention=sum(r['fact_mention'] for r in rr)/len(rr),
                     predictions=dict(Counter(normalized(r['text']) for r in rr))) for f, rr in families.items()}
    return dict(assistant_nll=sum(v*n for v,n in losses)/sum(n for _,n in losses),
                exact=sum(r['exact'] for r in records)/len(records),
                both_correct=sum(all(r['exact'] for r in rr) for rr in pairs.values())/len(pairs),
                fact_mention=sum(r['fact_mention'] for r in records)/len(records),
                stopped=sum(r['stop']=='end' for r in records)/len(records),
                same_answer_pairs=sum(normalized(rr[0]['text'])==normalized(rr[1]['text']) for rr in pairs.values())/len(pairs),
                by_family=stats, records=records,
                open=[dict(prompt=p, **generate(backend, p, limit)) for p in OPEN] if include_open else [])


def prose_loss(backend, texts):
    """A small, fixed same-text drift diagnostic; compare within model only."""
    total = count = 0
    backend.model.eval()
    try:
        for text in texts:
            ids = backend.encode(text)
            if not 2 <= len(ids) <= 2048: raise ValueError('Prose overflow')
            x = mx.array([ids[:-1]], dtype=mx.int32); y = mx.array([ids[1:]], dtype=mx.int32)
            v = nn.losses.cross_entropy(backend.logits(x).astype(mx.float32), y, reduction='sum')
            total += float(v.item()); count += len(ids)-1
    finally: backend.model.train()
    return total/count
