"""Pinned human explanation TRAIN data, grouped before selection; no raw cache."""
import csv
import hashlib
import heapq
import io
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from sml_v2.common import fingerprint, read_json
from sft.skill_balance.data import BLOCK, ROLE, exclusions, finite_sentence, grams, norm
from sft.skill_recovery_768.data import build as replay_build
from sft.skill_balance.data import encode_stats

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
SOURCE = dict(
    url='https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/7b585a3f077fdea899780eb0473940522ae44a2e/dataset/esnli_train_1.csv',
    revision='7b585a3f077fdea899780eb0473940522ae44a2e',
    bytes=90169741,
    sha256='7311c7bc16ad9f6a9adcd116a62ef991e1803dc0d71e253ce66b975c2aba8ee5',
    license='e-SNLI repository MIT; underlying SNLI CC-BY-SA-4.0',
    scope='First original training CSV only; no official development/test files')
LABELS = ('entailment', 'contradiction', 'neutral')
LEADS = {'entailment': 'Supported.', 'contradiction': 'Contradicted.', 'neutral': 'Not enough information.'}
META = re.compile(r'\b(pictures?|photos?|photographs?|images?|captions?|premises?|hypotheses|sentences?|saturn|unicorns?)\b', re.I)


def raw_source(cancelled):
    import requests
    raw = io.BytesIO(); digest = hashlib.sha256(); size = 0
    with requests.get(SOURCE['url'], stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        for block in response.iter_content(65536):
            if cancelled(): raise InterruptedError('Stopped during source download')
            size += len(block)
            if size > SOURCE['bytes']: raise ValueError('Source exceeds pinned size')
            raw.write(block); digest.update(block)
    if size != SOURCE['bytes'] or digest.hexdigest() != SOURCE['sha256']:
        raise ValueError('e-SNLI source hash/length mismatch')
    raw.seek(0)
    print('[source-verified] e-SNLI TRAIN CSV; raw bytes remain in RAM', flush=True)
    return raw


def csv_rows(raw):
    raw.seek(0); wrapper = io.TextIOWrapper(raw, encoding='utf8', newline='')
    try: yield from csv.DictReader(wrapper)
    finally: wrapper.detach()


def scene_id(row):
    scene = row['pairID'].split('#', 1)[0]
    return re.sub(r'r\d+[enc]$', '', scene) if scene.startswith('vg_') else scene


def grouped_scenes(raw, cancelled):
    # Captions of one image AND identical premises attached to different images
    # form a connected component. Never split a component across partitions.
    parents = {}; premises = {}
    def find(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]; x = parents[x]
        return x
    for i, row in enumerate(csv_rows(raw)):
        if i % 4096 == 0 and cancelled(): raise InterruptedError('Stopped during grouping')
        scene = scene_id(row); key = norm(row['Sentence1'])
        old = premises.setdefault(key, scene)
        a, b = find(scene), find(old)
        if a != b: parents[max(a, b)] = min(a, b)
    return {s: find(s) for s in parents}


def split_of(group):
    n = int(fingerprint(['explanation-transfer-v1', group])[:8], 16) % 10
    return 'train' if n < 8 else 'dev' if n == 8 else 'test'


def clean(text):
    return ' '.join(text.replace('\u2019', "'").split())


def target_row(row, group):
    premise, claim, reason = (clean(row[k]) for k in ('Sentence1', 'Sentence2', 'Explanation_1'))
    label = row['gold_label']
    reason = reason[:1].upper() + reason[1:]
    if reason[-1:] not in '.!?': reason += '.'
    instructions = (
        'Does the claim follow from the description? Give the verdict, then a brief explanation.',
        'Compare the claim with the description. State your conclusion and explain why.',
        'Use the description to assess the claim. Explain the evidence for your conclusion.',
        'What can we conclude about this claim from the description? Give a short reason.')
    variant = int(fingerprint(row['pairID'])[:8], 16) % len(instructions)
    prompt = (instructions[variant] + '\nTreat both as describing the same people or objects at the same moment. '
              'Start with Supported., Contradicted., or Not enough information. '
              'Do not assume details that are not given.\nDescription: ' + premise + '\nClaim: ' + claim)
    return dict(id='esnli:' + row['pairID'], group=group, source='esnli-train-1', split=split_of(group),
                label=label, premise=premise, claim=claim, reason=reason, prompt=prompt,
                answer=LEADS[label] + ' ' + reason)


def select_new(cfg, cancelled=lambda: False):
    from sml_v2.tokenization import Tokenizer
    raw = raw_source(cancelled); groups = grouped_scenes(raw, cancelled)
    denied, denied_grams, _ = exclusions()
    probes = read_json(DIR/'probes.json')
    for rows in probes.values():
        for r in rows:
            denied.add(norm(r['prompt'])); denied_grams.update(grams(r['prompt']))
    reject = read_json(DIR/'rejections.json'); counts = Counter(); heaps = defaultdict(list)
    quota = {'train': cfg['updates'] * cfg['per_update']['explanation'] // 3,
             'dev': cfg['dev_per_label'], 'test': cfg['test_per_label']}
    tok = Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    try:
        for i, row in enumerate(csv_rows(raw)):
            if i % 4096 == 0 and cancelled(): raise InterruptedError('Stopped during selection')
            if row['gold_label'] not in LABELS: continue
            if 'esnli:' + row['pairID'] in reject: counts['review_rejected'] += 1; continue
            p, c, e = (clean(row[k]) for k in ('Sentence1', 'Sentence2', 'Explanation_1'))
            if not (6 <= len(norm(p).split()) <= 40 and 4 <= len(norm(c).split()) <= 30
                    and 10 <= len(norm(e).split()) <= 40): continue
            joined = ' '.join((p, c, e))
            if BLOCK.search(joined) or ROLE.search(joined) or META.search(joined): continue
            # Reject obvious label/explanation conflicts and speculative
            # rationales; this cannot replace full human semantic review.
            if row['gold_label'] == 'entailment' and re.search(
                    r"\b(does not|do not|doesn't|don't|not necessarily|cannot|can't|assum\w*|probably|likely|usually|might|may|could)\b", e, re.I): continue
            if row['gold_label'] == 'contradiction' and re.search(
                    r'\b(usually|unlikely|probably|most likely|extent of|not necessarily|does not imply|does not mean)\b', e, re.I): continue
            if re.search(r'\b(samd|rephasing|pepople|OOOHS|AAAHS|playing beach|PLAY ING)\b', joined, re.I): continue
            if any(t in e for t in ('/', '. .', '  ')): continue
            if any(t in joined for t in ('http', '@', '<', '>')): continue
            if any(norm(t) in denied or grams(t) & denied_grams for t in (p, c, e)): continue
            words = norm(e).split(); triples = [tuple(words[j:j+3]) for j in range(len(words)-2)]
            if len(set(triples)) != len(triples) or norm(e) in (norm(p), norm(c)): continue
            group = groups[scene_id(row)]; split = split_of(group); label = row['gold_label']
            key = (split, label); counts[str(key)] += 1
            priority = int(fingerprint(['esnli-order-v1', row['pairID']]), 16)
            heap = heaps[key]; limit = quota[split] * 8 + 128
            if len(heap) == limit and priority >= -heap[0][0]: continue
            item = (-priority, row['pairID'], row, group)
            if len(heap) < limit: heapq.heappush(heap, item)
            else: heapq.heapreplace(heap, item)
    finally: raw.close()
    selected = {s: [] for s in quota}; used_groups = set(); used_reasons = set()
    # One independent scene per row, not a hundred variants of one situation.
    for split in ('train', 'dev', 'test'):
        for label in LABELS:
            accepted = []
            for _, _, row, group in sorted(heaps[(split, label)], reverse=True):
                reason = norm(row['Explanation_1'])
                if group in used_groups or reason in used_reasons: continue
                if not finite_sentence(row['Explanation_1']): continue
                r = target_row(row, group)
                n, _ = encode_stats(tok, r, cfg['context'])
                if n > cfg['context']: raise ValueError('No training truncation allowed')
                accepted.append(r); used_groups.add(group); used_reasons.add(reason)
                if len(accepted) == quota[split]: break
            if len(accepted) != quota[split]: raise ValueError('Insufficient independent rows: ' + repr((split, label)))
            selected[split].extend(accepted)
    # Every six-row training micro-curriculum contains two examples per label.
    rng = random.Random(cfg['seed']); balanced = []
    pools = {label: [r for r in selected['train'] if r['label'] == label] for label in LABELS}
    for rows in pools.values(): rng.shuffle(rows)
    for index in range(quota['train']):
        block = [pools[label][index] for label in LABELS]; rng.shuffle(block); balanced += block
    selected['train'] = balanced
    print('[explanation-data]', json_summary(selected), flush=True)
    return selected, dict(source=SOURCE,eligible=dict(counts),groups=len(groups))


def json_summary(selected):
    return {s: dict(rows=len(rr), labels=dict(Counter(r['label'] for r in rr))) for s, rr in selected.items()}


def replay_rows(rows, n, rng):
    pool = list({r['id']: r for r in rows}.values()); result = []
    while len(result) < n:
        rng.shuffle(pool); result.extend(pool[:n-len(result)])
    return result


def build(cfg, cancelled=lambda: False):
    from sml_v2.tokenization import Tokenizer
    replay_cfg = read_json(ROOT/'sft/skill_recovery_768/config.json')
    replay, old_selection = replay_build(replay_cfg, cancelled)
    if old_selection != read_json(ROOT/'sft/skill_recovery_768/selection.json'):
        raise ValueError('Parent rehearsal selection changed')
    new, new_stats = select_new(cfg, cancelled)
    data = dict(replay=replay, anchors=replay['anchors'], train={}, dev={'explanation': new['dev']},
                test={'explanation': new['test']})
    data['train']['explanation'] = new['train']; rng = random.Random(cfg['seed'])
    # Consume one item from each commonsense source in every update.
    pool = replay['train']['commonsense']
    sources = [replay_rows([r for r in pool if r['source'] == s], cfg['updates'], rng)
               for s in ('commonsenseqa', 'socialiqa')]
    data['train']['commonsense'] = [r for pair in zip(*sources) for r in pair]
    data['train']['instruction'] = replay_rows(replay['train']['instruction'], cfg['updates']*3, rng)
    # Preserve the parent's new/rehearsal proportions; never consume dev/test.
    old = replay['train']['reading']; repeated = []
    while len(repeated) < cfg['updates']*4:
        cycle = list(old); rng.shuffle(cycle); repeated.extend(cycle[:cfg['updates']*4-len(repeated)])
    data['train']['reading'] = repeated
    tok = Tokenizer(ROOT/'tokenizer/bytebpe32k_v1'); stats = {}
    for split in ('train', 'dev', 'test'):
        stats[split] = {}
        for family, rows in data[split].items():
            sizes = [encode_stats(tok, r, cfg['context']) for r in rows]
            stats[split][family] = dict(exposures=len(rows), unique=len({r['id'] for r in rows}),
                groups=len({r['group'] for r in rows}), hash=fingerprint(rows),
                maximum_tokens=max(n for n, _ in sizes), answer_tokens=sum(t for _, t in sizes))
    for family, n in cfg['per_update'].items():
        if len(data['train'][family]) != cfg['updates']*n: raise ValueError('Incomplete '+family)
    sets = [{r['group'] for r in new[s]} for s in ('train', 'dev', 'test')]
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]: raise ValueError('Scene leakage')
    selection = dict(config=fingerprint(cfg), tokenizer=tok.fingerprint, replay=fingerprint(old_selection),
        new_source=new_stats, stats=stats, anchors=fingerprint(data['anchors']),
        limitations='Human annotations are noisy and filtered heuristically, not all semantically verified. '
        'Scene grouping plus duplicate-premise grouping reduces leakage; wording/knowledge may recur. '
        'Public benchmarks used only for exclusion. Base pretraining overlap is unknown. '
        'Rehearsal/dev diagnostics were consumed or inspected previously. No claim of general-chat improvement.')
    return data, selection
