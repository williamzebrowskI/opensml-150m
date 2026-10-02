"""Freeze screened new TRAIN situations and previously consumed skill replay."""
import random
from collections import Counter
from pathlib import Path

from sml_v2.common import fingerprint, read_json
from sft.context_reasoning_1024.data import build as context_build
from sft.explanation_transfer_1024.data import build as explanation_build
from sft.explanation_transfer_1024.data import LABELS
from sft.skill_balance.data import encode_stats

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent


def take(rows, count, seed):
    unique = list({r['id']: r for r in rows}.values())
    random.Random(seed).shuffle(unique)
    chosen=[];groups=set()
    for row in unique:
        if row['group'] in groups:continue
        chosen.append(row);groups.add(row['group'])
        if len(chosen)==count:break
    if len(chosen) < count:
        raise ValueError(f'Only {len(chosen)} independent groups; need {count}')
    return chosen


def cycle(rows, count, seed):
    unique = list({r['id']: r for r in rows}.values())
    if not unique:
        raise ValueError('Empty rehearsal pool')
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        rng.shuffle(unique)
        result.extend(unique[:count-len(result)])
    return result


def build(cfg, cancelled=lambda: False):
    from sml_v2.tokenization import Tokenizer
    context_cfg = read_json(ROOT/'sft/context_reasoning_1024/config.json')
    context_cfg['arm'] = 'main'
    context, context_selection = context_build(context_cfg, cancelled)
    if context_selection != read_json(ROOT/'sft/context_reasoning_1024/selection_main.json'):
        raise ValueError('Screened context selection changed')
    explanation_cfg = read_json(ROOT/'sft/explanation_transfer_1024/config.json')
    explanation, explanation_selection = explanation_build(explanation_cfg, cancelled)
    if explanation_selection != read_json(ROOT/'sft/explanation_transfer_1024/selection.json'):
        raise ValueError('Explanation selection changed')

    n = cfg['updates']; per = cfg['per_update']; seed = cfg['seed']
    new = context['new_split']; old = explanation['replay']
    cosmos = [r for r in new['train'] if r['source'] == 'cosmosqa']
    wino = [r for r in new['train'] if r['source'] == 'winogrande']
    if len(cosmos) < 2*n or len(wino) < 2*n:
        raise ValueError('Insufficient independent new TRAIN rows')
    data = {'train': {}, 'dev': {}, 'anchors': context['anchors']}
    data['train']['new_answer'] = take(cosmos, per['new_answer']*n, seed+1)
    data['train']['new_choice'] = take(wino, per['new_choice']*n, seed+2)
    old_qa = old['train']['commonsense']
    qa_sources = sorted({r['source'] for r in old_qa})
    if len(qa_sources) != 2 or per['old_choice'] != 2:
        raise ValueError('Expected one old choice per source and update')
    pools = [cycle([r for r in old_qa if r['source'] == s], n, seed+10+i)
             for i, s in enumerate(qa_sources)]
    data['train']['old_choice'] = [r for pair in zip(*pools) for r in pair]
    data['train']['instruction'] = cycle(old['train']['instruction'], per['instruction']*n, seed+20)
    data['train']['reading'] = cycle(old['train']['reading'], per['reading']*n, seed+21)
    # The first 448 original updates consumed 2,688 explanation scenes. Replay
    # only those scenes so this branch does not silently change new-data scope.
    consumed = explanation['train']['explanation'][:448*6]
    if len(consumed) != 2688:
        raise ValueError('Unexpected source-1472 explanation cursor')
    labels = [LABELS[i % 3] for i in range(per['explanation']*n)]
    epools = {label: iter(cycle([r for r in consumed if r['label'] == label], labels.count(label), seed+30+i))
              for i, label in enumerate(LABELS)}
    data['train']['explanation'] = [next(epools[label]) for label in labels]

    def source_sample(source, count):
        return [r for r in new['dev'] if r['source'] == source][:count]
    replay_dev = old['dev']
    dev_mc_sources = sorted({r['source'] for r in replay_dev['commonsense']})
    data['dev'] = {
        'knowledge': source_sample('cosmosqa', 64)+source_sample('winogrande', 64),
        'commonsense': [r for s in dev_mc_sources
                        for r in [x for x in replay_dev['commonsense'] if x['source'] == s][:32]],
        'instruction': replay_dev['instruction'][:48],
        'reading': replay_dev['reading'][:64],
        'explanation': [r for label in LABELS
                        for r in [x for x in explanation['dev']['explanation'] if x['label'] == label][:16]]
    }
    if len(data['dev']['explanation']) != 48:
        raise ValueError('Missing label-balanced explanation development rows')
    seen = {r['group'] for r in data['train']['new_answer']+data['train']['new_choice']}
    if seen & {r['group'] for r in data['dev']['knowledge']}:
        raise ValueError('New-source group leakage')
    for family, count in per.items():
        if len(data['train'][family]) != n*count:
            raise ValueError('Incomplete '+family)
    tok = Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    stats = {}
    for split in ('train', 'dev'):
        stats[split] = {}
        for family, rows in data[split].items():
            sizes = [encode_stats(tok, row, cfg['context']) for row in rows]
            if max(a for a, _ in sizes) > cfg['context']:
                raise ValueError('Truncated '+family)
            stats[split][family] = dict(exposures=len(rows), unique=len({r['id'] for r in rows}),
                groups=len({r['group'] for r in rows}), hash=fingerprint(rows),
                maximum_tokens=max(a for a, _ in sizes), answer_tokens=sum(b for _, b in sizes),
                sources=dict(Counter(r.get('source', 'authored') for r in rows)))
    selection = dict(config=fingerprint(cfg), tokenizer=tok.fingerprint,
        context_selection=fingerprint(context_selection), explanation_selection=fingerprint(explanation_selection),
        anchor_hash=fingerprint(data['anchors']), stats=stats,
        limitation='New contextual rows are from TRAIN subpartitions and were screened and sample-reviewed, '
        'not exhaustively semantically verified. Prior 1472 explanation scenes and earlier skills are replay only. '
        'Development groups are disjoint from new training groups; public benchmark overlap screening is heuristic.')
    return data, selection
