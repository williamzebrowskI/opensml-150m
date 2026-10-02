"""Pinned TRAIN data, independent group splits, and executable task targets.

Public evaluation and earlier diagnostic text is used solely for exclusion.
Programmatic scenes are fictional; they teach relationships and output tasks,
not world facts. Human annotations are used without invented explanations.
"""
from collections import Counter
import json
from pathlib import Path
import random
import re

from sml_v2.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v2.tokenization import Tokenizer
from sft.skill_balance.data import norm, grams
from sft.intact_smoltalk_base_pilot.data import visible_prefix

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
SOURCES = {
    'arc_easy': dict(repo='allenai/ai2_arc', revision='210d026faf9955653af8916fad021475a3f00453', file='ARC-Easy/train-00000-of-00001.parquet', bytes=330598, sha256='b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5', license='CC-BY-SA-4.0'),
    'arc_challenge': dict(repo='allenai/ai2_arc', revision='210d026faf9955653af8916fad021475a3f00453', file='ARC-Challenge/train-00000-of-00001.parquet', bytes=189909, sha256='e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb', license='CC-BY-SA-4.0'),
    'piqa': dict(repo='baber/piqa', revision='142f6d7367fd9877f0fb3b5734ea6a545f54cdd1', file='piqa_train.parquet', bytes=2638413, sha256='3114602b31fa724309a6c493cc86368c4eab7ab6e79e285056fd5ce3b60ddf02', license='unknown', license_source='https://huggingface.co/datasets/ybisk/piqa'),
    'hellaswag': dict(repo='Rowan/hellaswag', revision='218ec52e09a7e7462a5400043bb9a69a41d06b76', file='data/train-00000-of-00001.parquet', bytes=24365524, sha256='cacb12587faa63d7f723a72d61d12bfa94b140446f5a6a0a2e1c6906ab88bf02', license='MIT'),
    'commonsenseqa': dict(repo='tau/commonsense_qa', revision='94630fe30dad47192a8546eb75f094926d47e155', file='data/train-00000-of-00001.parquet', bytes=1247103, sha256='b0449767ed986bfc2ca52b1244a46ef12f732756727f3cb0a4ab69ac8b3d282b', license='MIT'),
    'squad': dict(repo='rajpurkar/squad_v2', revision='3ffb306f725f7d2ce8394bc1873b24868140c412', file='squad_v2/train-00000-of-00001.parquet', bytes=16369982, sha256='f6da32ffb482ff463ad056477740d1bb284b96a45db3a08bee6a225ca6abf291', license='CC-BY-SA-4.0'),
}
NAMES = 'Ada Ben Cora Dario Esme Farah Giles Hana Ivan Jaya Kira Leon Mina Nico Orla Pavel Quinn Rafi Sora Tessa Uma Vera Wynn Yara Zane'.split()
ITEMS = ['lantern', 'scarf', 'notebook', 'basket', 'compass', 'jacket', 'camera', 'mug', 'sketchbook', 'helmet', 'thermos', 'apron', 'brush', 'pencil', 'tray', 'key']
PLACES = ['drawer', 'locker', 'shelf', 'cupboard', 'counter', 'bench', 'closet', 'box', 'attic', 'studio', 'office', 'pantry']
ROOMS = ['studio', 'office', 'hall', 'classroom', 'courtyard', 'library', 'auditorium', 'meeting room']
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
WORDS = 'amber birch cedar daisy elm fern grape hazel iris jade kiwi lemon maple nutmeg olive peach quince reed sage thyme violet willow yarrow'.split()


def partition(group):
    bucket = int(fingerprint(['grounded-rank-384-split', group])[:8], 16) % 10
    return 'train' if bucket < 8 else 'dev' if bucket == 8 else 'test'


def download(name):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    spec = SOURCES[name]; path = DIR / '_sources' / (name + '.parquet')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        url = f'https://huggingface.co/datasets/{spec["repo"]}/resolve/{spec["revision"]}/{spec["file"]}'
        temporary = path.with_suffix('.download')
        try:
            with session.get(url, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                with temporary.open('wb') as out:
                    size = 0
                    for block in response.iter_content(65536):
                        size += len(block)
                        if size > spec['bytes']: raise ValueError('Pinned source too large')
                        out.write(block)
            if temporary.stat().st_size != spec['bytes'] or file_sha256(temporary) != spec['sha256']:
                raise ValueError('Pinned source checksum mismatch: ' + name)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True); session.close()
    if path.stat().st_size != spec['bytes'] or file_sha256(path) != spec['sha256']:
        raise ValueError('Cached source checksum mismatch: ' + name)
    print('[source-verified]', name, 'TRAIN', flush=True)
    return path


def raw_rows(name):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(download(name)).iter_batches(batch_size=128):
        yield from batch.to_pylist()


def exclusions():
    texts = []; paths = []
    for name in ('arc_easy', 'arc_challenge', 'piqa', 'hellaswag', 'ifeval'):
        path = ROOT / 'evaluation/full_benchmarks/_cache' / (name + '.json'); paths.append(path)
        for row in read_json(path):
            texts += [row['prompt'], row['prompt'].removeprefix('Question: ').removesuffix('\nAnswer:')] + row.get('choices', [])
    path = ROOT / 'diagnostics/intact_base_384_review_v1/fresh_probes.json'; paths.append(path)
    probes = read_json(path)
    # The saved diagnostic protocol is excluded in full, including follow-ups.
    def extract(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('prompt', 'followup', 'current') and isinstance(item, str): texts.append(item)
                else: extract(item)
        elif isinstance(value, list):
            for item in value: extract(item)
    extract(probes)
    path = ROOT / 'sft/intact_smoltalk_base_pilot/prepared.json'; paths.append(path)
    pilot = read_json(path)['data']
    for split in ('dev', 'test'):
        texts += [m['content'] for row in pilot[split] for m in row['messages'] if m['role'] == 'user']
    paths.append(ROOT / 'evaluation/full_benchmarks/_cache/raw/hellaswag.parquet')
    return ({norm(t) for t in texts if len(norm(t).split()) >= 4},
            set().union(*(grams(t) for t in texts)), paths, pilot)


def overlaps(texts, denied, dgrams):
    return any((len(norm(t).split()) >= 4 and norm(t) in denied) or grams(t) & dgrams for t in texts)


def choose(rows, count, seed, used=None):
    rows = sorted(rows, key=lambda r: fingerprint([seed, r.get('selection_id', r['id'])]))
    chosen = []; seen = set() if used is None else used
    for row in rows:
        if row['group'] in seen: continue
        chosen.append(row); seen.add(row['group'])
        if len(chosen) == count: return chosen
    raise ValueError(f'Insufficient independent groups: {len(chosen)}/{count}')


def row_valid(tok, row, context):
    p = 'User: ' + row['prompt'] + '\nAssistant:'
    head = tok.encode(p); ids = tok.encode(p + ' ' + row['answer'])
    return (bool(head) and ids[:len(head)] == head and len(ids) <= context
            and bool(row['answer'].strip()) and len(tok.encode(row['answer'])) <= 256)


def ranking_row(name, raw, index):
    from evaluation.full_benchmarks.prepare import preprocess
    if name.startswith('arc_') or name == 'commonsenseqa':
        choices = list(raw['choices']['text']); gold = list(raw['choices']['label']).index(raw['answerKey'])
        question = raw['question']; prompt = 'Question: ' + question + '\nAnswer:'
    elif name == 'piqa':
        choices = [raw['sol1'], raw['sol2']]; gold = int(raw['label']); question = raw['goal']
        prompt = 'Question: ' + question + '\nAnswer:'
    else:
        choices = [preprocess(x) for x in raw['endings']]; gold = int(raw['label'])
        question = preprocess(raw['ctx_a'] + ' ' + raw['ctx_b'].capitalize())
        prompt = preprocess(raw['activity_label'] + ': ' + raw['ctx_a'] + ' ' + raw['ctx_b'].capitalize())
    group = (fingerprint(['hellaswag-origin', raw['source_id']]) if name == 'hellaswag'
             else fingerprint(['public-context', norm(question)]))
    selection_id = f'{name}:train:{raw.get("id", raw.get("ind", index))}'
    # HellaSwag's ind is reused between video and WikiHow sources. Include its
    # original source identity, retaining the old selection key/order.
    unique_id = selection_id + ':' + raw['source_id'] if name == 'hellaswag' else selection_id
    return dict(id=unique_id, selection_id=selection_id, group=group,
                origin_group=raw.get('source_id') if name == 'hellaswag' else None,
                source=name, source_split='train', split=partition(group), prompt=question,
                ranking_prompt=prompt, choices=choices, gold=gold, answer=choices[gold])


def relation_rows(split, count, seed):
    rng = random.Random(seed); rows = []; seen = set()
    while len(rows) < count:
        names = rng.sample(NAMES, 3); item = rng.choice(ITEMS); places = rng.sample(PLACES, 2)
        days = rng.sample(DAYS, 2); times = rng.sample(['9 am', '10 am', '11 am', '1 pm', '3 pm', '5 pm'], 2)
        scene = dict(names=names, item=item, places=places, days=days, times=times)
        group = fingerprint(['relation-scene', scene])
        if group in seen or partition(group) != split: continue
        seen.add(group); a, b, c = names; p, q = places; d, e = days; t, u = times
        noun = ('an ' if item[0] in 'aeiou' else 'a ') + item
        kind = len(rows) % 8
        if kind == 0:
            context = f'{a} owns {noun}. {a} lent it to {b} for the weekend.'
            question, answer = f'Who owns the {item}?', a
        elif kind == 1:
            context = f'{a} lent {noun} to {b}. {c} borrowed a different item.'
            question, answer = f'Who borrowed the {item}?', b
        elif kind == 2:
            prep = 'on' if p in ('shelf', 'counter', 'bench') else 'in'
            context = f'The {item} was {prep} the {p}. {a} moved it to the {q}. Nobody moved it afterward.'
            question, answer = f'Where is the {item} now?', q
        elif kind == 3:
            context = f'{a} did not bring the {item}. {b} brought it. {c} helped arrange the chairs.'
            question, answer = f'Who brought the {item}?', b
        elif kind == 4:
            context = f'The workshop was scheduled for {t} on {d}. Its starting time changed to {u}; the day stayed the same.'
            question, answer = 'What is the updated starting time?', u
        elif kind == 5:
            context = f'The workshop was planned for {d}. It was moved to {e}. {a} will host it.'
            question, answer = 'Which day is the workshop now scheduled for?', e
        elif kind == 6:
            prep = 'on' if p in ('shelf', 'counter', 'bench') else 'in'
            other = 'on' if q in ('shelf', 'counter', 'bench') else 'in'
            context = f'{a} keeps the {item} {prep} the {p}. {b} keeps a coat {other} the {q}.'
            question, answer = f'Where does {a} keep the {item}?', p
        else:
            prep = 'on' if p in ('shelf', 'counter', 'bench') else 'in'
            context = f'{a} put {noun} {prep} the {p}. The note does not describe its color.'
            question, answer = f'What color is the {item}?', 'Not stated'
        prompt = f'Use only the passage. Give a brief answer; if the information is missing, reply "Not stated".\n\nPassage: {context}\n\nQuestion: {question}'
        rows.append(dict(id='relation:' + group, group=group, source='authored-relations',
                         split=split, task=str(kind), prompt=prompt, answer=answer, references=[answer]))
    return rows


def instruction_rows(split, count, seed):
    rng = random.Random(seed); rows = []; seen = set()
    while len(rows) < count:
        a, b = rng.sample(NAMES, 2); d, e = rng.sample(DAYS, 2); p = rng.choice(ROOMS)
        words = rng.sample(WORDS, 4); item = rng.choice(ITEMS); kind = len(rows) % 12
        facts = dict(a=a, b=b, d=d, e=e, p=p, words=words, item=item, kind=kind)
        group = fingerprint(['instruction-scene', facts])
        if group in seen or partition(group) != split: continue
        seen.add(group)
        if kind == 0:
            prompt = f'{a} will meet {b} on {d}. Return only a JSON object with exactly two keys: "person" for the person {a} will meet, and "day" for the meeting day.'
            answer = json.dumps(dict(person=b, day=d)); rule = 'json'
        elif kind == 1:
            prompt = 'Sort these words alphabetically. Return only the words separated by commas, without spaces: ' + ', '.join(words)
            answer = ','.join(sorted(words)); rule = 'csv'
        elif kind == 2:
            prompt = f'Give exactly two bullet points, starting each with "- ": say that {a} has the {item}, then say that {b} is in the {p}. Add nothing else.'
            answer = f'- {a} has the {item}.\n- {b} is in the {p}.'; rule = 'bullets'
        elif kind == 3:
            prompt = f'Return exactly two sentences. In the first, say that the workshop is on {d}. In the second, say that {a} will host it. Add no other information.'
            answer = f'The workshop is on {d}. {a} will host it.'; rule = 'sentences'
        elif kind == 4:
            prompt = f'Rewrite this invitation with the updated day {e}, preserving the other details. Return only the revised invitation.\nInvitation: Please join {a} in the {p} on {d}.'
            answer = f'Please join {a} in the {p} on {e}.'; rule = 'rewrite'
        elif kind == 5:
            prompt = f'Summarize this note in one sentence, mentioning the organizer, event day, and location. Add no other facts.\nNote: {a} is organizing a workshop on {d}. The location is the {p}. {b} has offered to help set up.'
            answer = f'{a} is organizing a workshop on {d} in the {p}.'; rule = 'summary'
        elif kind == 6:
            prompt = f'Return only these words converted to uppercase, in the same order and separated by single spaces: {" ".join(words)}'
            answer = ' '.join(words).upper(); rule = 'upper'
        elif kind == 7:
            prompt = f'Return only a numbered list of three steps: first collect the {item}, then take it to the {p}, then give it to {a}. Use the numbers 1, 2, and 3 followed by a period.'
            answer = f'1. Collect the {item}.\n2. Take it to the {p}.\n3. Give it to {a}.'; rule = 'numbered'
        elif kind == 8:
            prompt = f'Write an update saying that {a} lent the {item} to {b}. Begin with "Update: " and end with "Confirmed." Add no other information.'
            answer = f'Update: {a} lent the {item} to {b}. Confirmed.'; rule = 'prefix-suffix'
        elif kind == 9:
            prompt = f'Write a subject line of exactly four words using these facts: {d} workshop, {p} location, {a} organizer. Use the words in that order, omit the organizer, and use no punctuation.'
            # Use a one-word room in this exact four-word task.
            if ' ' in p: continue
            answer = f'{d} workshop {p} location'; rule = 'four-words'
        elif kind == 10:
            prompt = f'Extract the day from this note. Return only a JSON object with the key "day" and a lowercase value.\nNote: {a} reserved the {p} for {d}.'
            answer = json.dumps(dict(day=d.lower())); rule = 'json-lower'
        else:
            prompt = f'Convert this comma-separated list to one word per line. Preserve the order, use lowercase, and add no bullets or other text: {", ".join(w.upper() for w in words)}'
            answer = '\n'.join(words); rule = 'lines'
        rows.append(dict(id='instruction:' + group, group=group, source='authored-executable-instructions',
                         split=split, task=rule, prompt=prompt, answer=answer, references=[answer]))
    return rows


def instruction_pass(row, text):
    kind = row['task']; target = row['answer']; text = text.strip()
    if kind.startswith('json'):
        try:
            def unique(pairs):
                result = {}
                for k, v in pairs:
                    if k in result: raise ValueError('Duplicate key')
                    result[k] = v
                return result
            return json.loads(text, object_pairs_hook=unique) == json.loads(target)
        except (ValueError, TypeError): return False
    if kind in ('csv', 'upper', 'lines', 'four-words'): return text == target
    # Exact wording is required for transformations; for authored summaries and
    # sentence tasks this is a conservative completion proxy, not human grading.
    return ' '.join(text.split()) == ' '.join(target.split())


def review_samples(data):
    samples = {}
    for family, rows in data['train'].items():
        strata = {}
        for row in rows:
            key = (row['source'], row.get('task', ''), row.get('unknown', False))
            strata.setdefault(key, []).append(row)
        samples[family] = [r for key in sorted(strata) for r in
            sorted(strata[key], key=lambda r: fingerprint(['review-384', r.get('selection_id', r['id'])]))[:2]]
    return samples


def build(cfg):
    tok = Tokenizer(ROOT / 'tokenizer/bytebpe32k_v1')
    denied, dgrams, excluded_paths, pilot = exclusions()
    data = {s: {f: [] for f in cfg['per_update']} for s in ('train', 'dev', 'test')}
    rejections = Counter(); counts = {}; rank_pools = {}
    import pyarrow.parquet as pq
    held_origins = {r['source_id'] for r in pq.read_table(ROOT / 'evaluation/full_benchmarks/_cache/raw/hellaswag.parquet', columns=['source_id']).to_pylist()}
    for name in ('arc_easy', 'arc_challenge', 'piqa', 'hellaswag', 'commonsenseqa'):
        pools = {s: [] for s in data}
        for index, raw in enumerate(raw_rows(name)):
            row = ranking_row(name, raw, index)
            if name == 'hellaswag' and row['origin_group'] in held_origins:
                rejections[name + ':evaluation-video-or-article-overlap'] += 1; continue
            if row['id'] in cfg.get('excluded_training_ids', []) or row['selection_id'] in cfg.get('excluded_training_ids', []):
                rejections[name + ':sample-review'] += 1; continue
            if overlaps([row['ranking_prompt']] + row['choices'], denied, dgrams):
                rejections[name + ':evaluation-overlap'] += 1; continue
            if not 0 <= row['gold'] < len(row['choices']) or len(set(row['choices'])) != len(row['choices']):
                rejections[name + ':bad-options'] += 1; continue
            head = tok.encode(row['ranking_prompt']); valid = bool(head)
            for choice in row['choices']:
                ids = tok.encode(row['ranking_prompt'] + ' ' + choice)
                valid &= ids[:len(head)] == head and len(head) < len(ids) <= cfg['context']
            if not valid: rejections[name + ':encoding'] += 1; continue
            pools[row['split']].append(row)
        counts[name] = {s: len(rows) for s, rows in pools.items()}; rank_pools[name] = pools
    for split in data:
        total = cfg['updates'] if split == 'train' else 32
        used = set(); selected = {}
        for i, name in enumerate(rank_pools):
            n = total // 2 if name.startswith('arc_') else total
            selected[name] = choose(rank_pools[name][split], n, cfg['seed'] + i, used)
        science = [row for pair in zip(selected['arc_easy'], selected['arc_challenge']) for row in pair]
        data[split]['ranking'] = [r for group in zip(science, selected['piqa'], selected['hellaswag'], selected['commonsenseqa']) for r in group]
    # Reuse the strict human span and topic filter, partitioning by article title.
    from sft.reading_repair.source import candidate
    natural = {s: {False: [], True: []} for s in data}
    for raw in raw_rows('squad'):
        row, reason = candidate(raw, tok, denied, dgrams)
        if row is None: rejections['squad:' + str(reason)] += 1; continue
        if not row_valid(tok, row, cfg['context']): rejections['squad:context'] += 1; continue
        # Same article partition as earlier reading runs; never move their held
        # articles into training. One source paragraph per selected example.
        row.update(source='squad', article_group=row['group'],
                   group=fingerprint(['squad-paragraph', norm(row['context'])]))
        natural[row['split']][row['unknown']].append(row)
    for split in data:
        n = cfg['updates'] * 2 if split == 'train' else 32
        used = set(); humans = choose(natural[split][False], n * 3 // 4, cfg['seed'] + 30, used)
        humans += choose(natural[split][True], n // 4, cfg['seed'] + 31, used)
        random.Random(cfg['seed'] + 32).shuffle(humans)
        relations = relation_rows(split, n, cfg['seed'] + 40)
        data[split]['grounded'] = [r for pair in zip(humans, relations) for r in pair]
        data[split]['instruction'] = instruction_rows(split, cfg['updates'] * 3 if split == 'train' else 48, cfg['seed'] + 50)
        for row in data[split]['grounded'] + data[split]['instruction']:
            if not row_valid(tok, row, cfg['context']): raise ValueError('Task encoding overflow')
            if overlaps([row['prompt']], denied, dgrams) and split == 'train':
                raise ValueError('Authored training overlaps an excluded evaluation prompt')
    # Replay existing selected TRAIN conversations, preserving source instruction
    # and actual history. Every source conversation contributes at most one turn.
    for split in data:
        pools = {source: [] for source in ('everyday-conversations', 'smollm-rewrite-30k', 'smol-summarize-20k')}
        for conversation in pilot[split]:
            if conversation['source'] not in pools: continue
            indices = [i for i, m in enumerate(conversation['messages']) if m['role'] == 'assistant']
            index = indices[-1] if conversation['source'] == 'everyday-conversations' else indices[0]
            messages = conversation['messages'][:index]; answer = conversation['messages'][index]['content']
            # Strip only the outer formatter's fixed wrapper, then normal encode
            # reconstructs the byte-identical native prefix with full history.
            prefix = visible_prefix(messages)
            row = dict(id=conversation['id'] + ':replay:' + str(index), group=conversation['group'],
                       source=conversation['source'], split=split, prompt=prefix[6:-11],
                       answer=answer, messages=messages, references=[answer])
            if row['id'] in cfg.get('excluded_training_ids', []): continue
            excluded = split == 'train' and overlaps([m['content'] for m in messages if m['role'] == 'user'], denied, dgrams)
            if not excluded and row_valid(tok, row, cfg['context']) and len(tok.encode(answer)) <= 192:
                pools[row['source']].append(row)
        total = cfg['updates'] * 2 if split == 'train' else 16
        used = set(); ordinary = choose(pools['everyday-conversations'], total // 2, cfg['seed'] + 60, used)
        edits = choose(pools['smollm-rewrite-30k'], total // 4, cfg['seed'] + 61, used)
        edits += choose(pools['smol-summarize-20k'], total // 4, cfg['seed'] + 62, used)
        random.Random(cfg['seed'] + 63).shuffle(edits)
        data[split]['replay'] = [r for pair in zip(ordinary, edits) for r in pair]
    # Anchors come exclusively from TRAIN continuations and TRAIN source passages.
    anchor_texts = [r['ranking_prompt'] + ' ' + r['answer'] for r in data['train']['ranking'] if r['source'] == 'hellaswag']
    data['anchors'] = [tok.encode(text)[:cfg['anchor_context']] for text in anchor_texts]
    stats = {}
    groups = {}; paragraphs = {}; articles = {}
    for split in ('train', 'dev', 'test'):
        groups[split] = {r['group'] for rows in data[split].values() for r in rows}
        paragraphs[split] = {norm(r['context']) for r in data[split]['grounded'] if r['source'] == 'squad'}
        articles[split] = {r['article_group'] for r in data[split]['grounded'] if r['source'] == 'squad'}
        stats[split] = {family: dict(examples=len(rows), groups=len({r['group'] for r in rows}),
            hash=fingerprint(rows), sources=dict(Counter(r['source'] for r in rows))) for family, rows in data[split].items()}
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        if groups[a] & groups[b] or paragraphs[a] & paragraphs[b] or articles[a] & articles[b]:
            raise ValueError('Selected train/development/test group leakage')
    for family, per in cfg['per_update'].items():
        if len(data['train'][family]) != cfg['updates'] * per: raise ValueError('Incomplete ' + family)
    data['prose_diagnostic'] = __import__('sft.transfer_control.launch', fromlist=['prose_texts']).prose_texts()
    # A sample is frozen for manual review before production updates.
    samples = review_samples(data)
    selection = dict(config=fingerprint(cfg), tokenizer=tok.fingerprint, sources=SOURCES,
        public_training_pool_counts=counts, rejections=dict(rejections), stats=stats,
        anchor_hash=fingerprint(data['anchors']), sample_hash=fingerprint(samples),
        exclusions={str(p): file_sha256(p) for p in excluded_paths},
        limitations='Human labels and sampled semantic review, not exhaustive factual verification. '
        'Programmatic dev/test share training task templates with disjoint scene groups. '
        'Public TRAIN data may have appeared in pretraining or older branches. '
        'Exact and 13-word overlap exclusion is heuristic; no proof of zero semantic contamination. '
        'Reserved test is frozen and not used in per-checkpoint development evaluation.')
    return data, selection, samples


def prepare(cfg):
    data, selection, samples = build(cfg)
    atomic_json(DIR / 'prepared.json', data); atomic_json(DIR / 'selection.json', selection)
    atomic_json(DIR / 'review_samples.json', samples)
    return selection
