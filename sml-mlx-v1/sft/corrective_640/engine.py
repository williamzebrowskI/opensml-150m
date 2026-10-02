"""Example- and family-balanced CE, character-normalized ranking, parent KL."""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from sft.skill_recovery_768.engine import load, optimizer, learning_rate, reference_kl
from sft.transfer_control.engine import encode, arrays
from sft.intact_smoltalk_base_pilot.data import turns, visible_prefix


def validate_conversation(b, row, context):
    encoded = turns(b.tokenizer, row, context)
    if len(encoded) != sum(m['role'] == 'assistant' for m in row['messages']):
        raise ValueError('Conversation lost an assistant turn')
    for part in encoded:
        if part['y'][-1] != b.eos: raise ValueError('Conversation target missing EOS')
        index = part['message_index']
        prefix_ids = b.encode(visible_prefix(row['messages'][:index]))
        if any(t != -100 for t in part['y'][:len(prefix_ids) - 1]):
            raise ValueError('Earlier history or user tokens received supervision')
    return encoded


def conversation_arrays(b, row, context):
    for encoded in validate_conversation(b, row, context):
        x, y = arrays([encoded], b.pad)
        positions = mx.array([i for i, t in enumerate(encoded['y']) if t != -100], dtype=mx.int32)
        yield x, y, positions


def batch_at(data, cfg, cursor):
    if not 0 <= cursor < cfg['updates']: raise ValueError('Training cursor out of range')
    result = {family: data['train'][family][cursor * n:(cursor + 1) * n]
              for family, n in cfg['per_update'].items()}
    if any(len(result[f]) != n for f, n in cfg['per_update'].items()):
        raise ValueError('Incomplete family batch')
    return result


def supervised_arrays(b, row, context):
    encoded = encode(b, row, context)
    x, y = arrays([encoded], b.pad)
    positions = mx.array([i for i, token in enumerate(encoded['y']) if token != -100], dtype=mx.int32)
    return x, y, positions


def supervised_loss(model, x, y, positions):
    # Score the answer and EOS only. Each example contributes a mean loss.
    scores = model(x)['logits'][0, positions].astype(mx.float32)
    return nn.losses.cross_entropy(scores, y[0, positions], reduction='mean')


def choice_arrays(b, row, context=1024):
    head = b.encode(row['ranking_prompt']); encoded = []
    for choice in row['choices']:
        ids = b.encode(row['ranking_prompt'] + ' ' + choice)
        if ids[:len(head)] != head or not len(head) < len(ids) <= context:
            raise ValueError('Choice boundary/context violation')
        encoded.append(dict(x=ids[:-1], y=[-100] * (len(head) - 1) + ids[len(head):]))
    x, y = arrays(encoded, b.pad)
    chars = mx.array([len(c) for c in row['choices']], dtype=mx.float32)
    if any(not c for c in row['choices']): raise ValueError('Empty candidate')
    return x, y, chars


def choice_scores(model, x, y, chars):
    # Match the public benchmark's acc_norm: answer log-likelihood divided by
    # Unicode character count, not token count. No chat prefix or answer EOS.
    values = model(x)['logits'].astype(mx.float32)
    ce = nn.losses.cross_entropy(values, mx.maximum(y, 0), reduction='none')
    raw = -(ce * (y != -100)).sum(axis=1)
    return raw, raw / chars


def ranking_loss(model, x, y, chars, gold, temperature):
    _, normalized = choice_scores(model, x, y, chars)
    scores = normalized / temperature
    return mx.logsumexp(scores) - scores[gold]


def update(b, anchor, opt, rows, prose_ids, cfg, step):
    b.model.train(); anchor.model.eval(); accumulated = None; combined = 0.; metrics = {}

    def add(value, gradient, weight):
        nonlocal accumulated, combined
        gradient = tree_map(lambda t: t.astype(mx.float32) * weight, gradient)
        accumulated = gradient if accumulated is None else tree_map(lambda a, c: a + c, accumulated, gradient)
        mx.eval(value, accumulated); combined += float(value.item()) * weight

    fn = nn.value_and_grad(b.model, lambda x, y, p: supervised_loss(b.model, x, y, p))
    for family in ('grounded', 'instruction', 'replay'):
        losses = []
        for row in rows[family]:
            parts = list(conversation_arrays(b, row, cfg['context'])) if family == 'replay' else [supervised_arrays(b, row, cfg['context'])]
            example_loss = 0.
            for x, y, positions in parts:
                value, gradient = fn(x, y, positions)
                add(value, gradient, cfg['loss_weights'][family] / len(rows[family]) / len(parts))
                example_loss += float(value.item()) / len(parts)
            losses.append(example_loss)
        metrics[family + '_ce'] = sum(losses) / len(losses)
    fn = nn.value_and_grad(b.model, lambda x, y, c, g: ranking_loss(b.model, x, y, c, g, cfg['ranking_temperature']))
    losses = []
    for row in rows['ranking']:
        x, y, chars = choice_arrays(b, row, cfg['context'])
        value, gradient = fn(x, y, chars, row['gold'])
        add(value, gradient, cfg['loss_weights']['ranking'] / len(rows['ranking']))
        losses.append(float(value.item()))
    metrics['ranking_ce'] = sum(losses) / len(losses)
    if len(prose_ids) < 2: raise ValueError('Empty training prose anchor')
    x = mx.array([prose_ids[:-1]], dtype=mx.int32)
    logits = anchor.logits(x).astype(mx.float32)
    reference = mx.stop_gradient(logits - mx.logsumexp(logits, axis=-1, keepdims=True)); mx.eval(reference)
    fn = nn.value_and_grad(b.model, lambda xx, rr: reference_kl(b.model, xx, rr))
    value, gradient = fn(x, reference); add(value, gradient, cfg['kl_weight'])
    metrics['anchor_kl'] = float(value.item())
    from sml_v1.pretrain import clip_gradients
    accumulated, norm = clip_gradients(accumulated, cfg['clip_norm']); mx.eval(accumulated, norm)
    if not all(math.isfinite(v) for v in [combined, float(norm.item()), *metrics.values()]):
        raise ValueError('Nonfinite objective/gradient; update not applied')
    rate = learning_rate(step, cfg); opt.learning_rate = rate
    opt.update(b.model, accumulated); mx.eval(b.model.parameters(), opt.state)
    return dict(update=step, loss=combined, lr=rate, grad_norm=float(norm.item()), **metrics)
