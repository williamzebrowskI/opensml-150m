"""Fresh correction tasks plus explicit retention replay; official eval is excluded."""
from collections import Counter
import json
from pathlib import Path
import random
import re

from sml_v1.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v1.tokenization import Tokenizer
from sft.grounded_rank_384 import data as base
from sft.intact_smoltalk_base_pilot.data import visible_prefix, turns
from sft.natural_control.data import BLOCK, MEDICAL, LIVE, BAD_RESPONSE

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
SOURCES = base.SOURCES
NAMES = base.NAMES
WORDS = base.WORDS
DAYS = base.DAYS
ROOMS = base.ROOMS
ITEMS = base.ITEMS


def partition(group):
    bucket = int(fingerprint(['corrective-640-scenes', group])[:8], 16) % 10
    return 'train' if bucket < 8 else 'dev' if bucket == 8 else 'test'


def instruction_rows(split, count, seed):
    """Eight combinations; hold wording variants 2/3 out of TRAIN entirely."""
    rng = random.Random(seed); rows = []; used = set()
    while len(rows) < count:
        a, b = rng.sample(NAMES, 2); day, old = rng.sample(DAYS, 2)
        room = rng.choice(ROOMS); item = rng.choice(ITEMS); words = rng.sample(WORDS, 4)
        kind = len(rows) % 8
        facts = dict(a=a, b=b, day=day, old=old, room=room, item=item, words=words, kind=kind)
        group = fingerprint(['corrective-combined-instructions', facts])
        if group in used or partition(group) != split: continue
        variant = rng.randrange(2) if split == 'train' else 2 if split == 'dev' else 3
        if kind == 0:
            note = f'{a} originally booked the {room} for {old}. The booking changed to {day}.'
            task = 'json-updated-lower'
            request = 'Extract the current day and room. Use exactly the JSON keys "day" and "room", lowercase both values, and add no other text.'
            answer = json.dumps(dict(day=day.lower(), room=room.lower()))
        elif kind == 1:
            note = ', '.join(words)
            task = 'json-sorted-upper'
            request = 'Sort these words alphabetically, convert them to uppercase, and return only a JSON object with the single key "words" containing the ordered array.'
            answer = json.dumps(dict(words=[w.upper() for w in sorted(words)]))
        elif kind == 2:
            note = f'{a} has the {item}. {b} is in the {room}.'
            task = 'bullets-upper'
            request = 'Return exactly two bullet points beginning with "- ". State who has the item first and where the other person is second. Use uppercase throughout and add nothing else.'
            answer = f'- {a} has the {item}.\n- {b} is in the {room}.'.upper()
        elif kind == 3:
            note = ' / '.join(w.upper() for w in words)
            task = 'numbered-reverse-lower'
            request = 'List these four words in reverse order. Use lowercase, one word per line, and number the lines 1. through 4. Add no other text.'
            answer = '\n'.join(f'{i}. {w}' for i, w in enumerate(reversed(words), 1))
        elif kind == 4:
            note = ', '.join(w.upper() for w in words)
            task = 'csv-descending-lower'
            request = 'Sort the words in reverse alphabetical order and convert them to lowercase. Separate them with commas without spaces. Return the list only.'
            answer = ','.join(sorted(words, reverse=True))
        elif kind == 5:
            note = f'Please join {a} in the {room} on {old}. The updated day is {day}.'
            task = 'rewrite-updated-labelled'
            request = 'Rewrite the invitation with the updated day, preserving the host and location. Begin with "Invitation: " and return just one sentence.'
            answer = f'Invitation: Please join {a} in the {room} on {day}.'
        elif kind == 6:
            note = f'{a} owns the {item}; {b} borrowed it. The borrowing happened on {day}.'
            task = 'json-owner-day'
            request = 'Return only JSON with exactly the keys "owner" and "day". Identify the owner rather than the borrower, and make the day uppercase.'
            answer = json.dumps(dict(owner=a, day=day.upper()))
        else:
            if ' ' in room: continue
            note = f'Organizer: {a}. Event: workshop. Day: {day}. Location: {room}.'
            task = 'six-words-lower'
            request = 'Write exactly six lowercase words in this order: organizer, "hosts", "workshop", "on", day, location. Use spaces only, with no punctuation or explanation.'
            answer = f'{a.lower()} hosts workshop on {day.lower()} {room.lower()}'
        if variant == 0: prompt = request + '\n\nInput: ' + note
        elif variant == 1: prompt = 'Input: ' + note + '\n\n' + request
        elif variant == 2: prompt = 'Apply all of these requirements to the information below.\n' + request + '\nInformation: ' + note
        else: prompt = 'Information to process:\n' + note + '\nRequired response:\n' + request
        rows.append(dict(id='combined:' + group, group=group, source='authored-combined-instructions',
                         split=split, task=task, variant=variant, prompt=prompt, answer=answer, references=[answer]))
        used.add(group)
    return rows


def instruction_pass(row, text):
    text = text.strip()
    if row['task'].startswith('json-'):
        def unique(pairs):
            result = {}
            for k, v in pairs:
                if k in result: raise ValueError('Duplicate key')
                result[k] = v
            return result
        try: return json.loads(text, object_pairs_hook=unique) == json.loads(row['answer'])
        except (ValueError, TypeError): return False
    return text == row['answer']


def authored_grounded(split, count, seed):
    """Half new relationship problems, half explanations of fictional test evidence."""
    rng = random.Random(seed); rows = []; seen = set()
    while len(rows) < count:
        a, b, c = rng.sample(NAMES, 3); item = rng.choice(ITEMS)
        p, q = rng.sample(ROOMS, 2); d, e = rng.sample(DAYS, 2)
        kind = len(rows) % 8
        group = fingerprint(['corrective-evidence', a, b, c, item, p, q, d, e, kind])
        if group in seen or partition(group) != split: continue
        if kind == 0:
            context = f'{a} owns the {item}. {b} borrowed it and carried it to the {p}. Later, {b} gave it back to {a}.'
            question, answer = f'Who owns the {item}?', a
        elif kind == 1:
            context = f'{a} placed the {item} in the {p}. {b} moved it to the {q}. {c} moved it back to the {p}.'
            question, answer = f'Where is the {item} now?', p
        elif kind == 2:
            context = f'{a} planned to meet {b} on {d} in the {p}. They changed the day to {e} and kept the location.'
            question, answer = 'What is the current meeting day?', e
        elif kind == 3:
            context = f'{a} owns the {item}. {b} borrowed it on {d}. The note describes its use but gives no color.'
            question, answer = f'What color is the {item}?', 'Not stated'
        elif kind == 4:
            context = f'In a fictional circuit test, sample {a} let the bulb light. Sample {b} left the bulb dark. The battery and bulb were unchanged.'
            question = 'Which sample allowed current through the circuit? Briefly explain using the observations.'
            answer = f'Sample {a}. The bulb lit with that sample while the battery and bulb were unchanged.'
        elif kind == 5:
            context = f'In a fictional plant experiment, plant {a} received water and plant {b} did not. Their light, soil, and containers were the same.'
            question = 'What variable was changed between the plants? Briefly explain using the setup.'
            answer = 'Water. One plant received water and the other did not, while light, soil, and containers stayed the same.'
        elif kind == 6:
            context = f'In a fictional light test, object {a} cast a dark shadow. Object {b} let the light pass through. Both were tested with the same lamp.'
            question = 'Which object blocked the light? Briefly explain using the test results.'
            answer = f'Object {a}. It cast a dark shadow, whereas the other object let the light pass through.'
        else:
            context = f'In a fictional seed test, seeds in tray {a} sprouted after watering. Seeds in tray {b} did not sprout. The observer recorded no temperatures.'
            question, answer = 'What temperature did the trays have?', 'Not stated'
        source = 'authored-relations' if kind < 4 else 'authored-science-evidence'
        prompt = f'Use only the passage. Answer briefly; if the information is missing, reply "Not stated".\n\nPassage: {context}\n\nQuestion: {question}'
        rows.append(dict(id='evidence:' + group, group=group, source=source, split=split,
                         task=str(kind), prompt=prompt, answer=answer, references=[answer], unknown=answer == 'Not stated'))
        seen.add(group)
    return rows


def authored_replay(split, count, seed):
    """Complete short dialogue with a checked summary or rewrite on the last turn."""
    rng = random.Random(seed); rows = []; seen = set()
    while len(rows) < count:
        a, b = rng.sample(NAMES, 2); day, old = rng.sample(DAYS, 2)
        room = rng.choice(ROOMS); item = rng.choice(ITEMS); kind = len(rows) % 4
        time = rng.choice(['9 am', '10 am', '11 am', '1 pm', '3 pm', '5 pm'])
        group = fingerprint(['corrective-complete-dialogue', a, b, day, old, room, item, kind, time])
        if group in seen or partition(group) != split: continue
        if kind == 0:
            first = 'I need help drafting a short invitation.'
            reply = 'Please share the host, day, time, location, and any request you want included.'
            current = f'{a} is hosting a workshop on {day} at {time} in the {room}. Ask {b} to bring the {item}. Write the invitation in two sentences and add no other facts.'
            answer = f'Please join {a} for a workshop on {day} at {time} in the {room}. {b}, please bring the {item}.'
            task = 'invitation'
        elif kind == 1:
            first = 'Can you summarize a meeting update without losing the schedule?'
            reply = 'Yes. Share the update, and I will keep the current day, time, and location in the summary.'
            current = f'Note: {a} planned to meet {b} on {old}. The meeting is now on {day} at {time} in the {room}. Summarize the current plan in one sentence, naming both people.'
            answer = f'{a} will meet {b} on {day} at {time} in the {room}.'
            task = 'schedule-summary'
        elif kind == 2:
            first = 'Help me make a request more polite while keeping all the details.'
            reply = 'Please send the request and the details that must stay in the revised version.'
            current = f'Rewrite this as a polite two-sentence message to {b}, preserving the deadline and the question: Send the {item} to {a} in the {room} by {day} at {time}. Can you confirm when it has been sent?'
            answer = f'{b}, please send the {item} to {a} in the {room} by {day} at {time}. Could you confirm when it has been sent?'
            task = 'polite-request'
        else:
            first = 'I have a note with an old plan and a correction. Can you help write the final version?'
            reply = 'Yes. Share the original plan and the correction so I can preserve the unchanged details.'
            current = f'Original: {a} will bring the {item} to {b} in the {room} on {old} at {time}. Correction: the day is {day}. Write only the corrected sentence.'
            answer = f'{a} will bring the {item} to {b} in the {room} on {day} at {time}.'
            task = 'corrected-plan'
        messages = [dict(role='user', content=first), dict(role='assistant', content=reply),
                    dict(role='user', content=current), dict(role='assistant', content=answer)]
        rows.append(dict(id='dialogue:' + group, group=group, source='authored-complete-dialogue',
                         split=split, task=task, messages=messages, prompt=visible_prefix(messages[:-1])[6:-11],
                         answer=answer, references=[answer]))
        seen.add(group)
    return rows


def exclusions():
    denied, dgrams, paths, pilot = base.exclusions()
    old_path = base.DIR / 'prepared.json'; paths.append(old_path)
    old = read_json(old_path); old_groups = set(); old_ids = set(); old_contexts = set()
    for split in ('train', 'dev', 'test'):
        for family, rows in old[split].items():
            for row in rows:
                old_groups.add(row['group']); old_ids.add(row['id'])
                if 'context' in row: old_contexts.add(base.norm(row['context']))
                texts = [row['prompt']] + ([row['context']] if 'context' in row else [])
                if 'ranking_prompt' in row: texts += [row['ranking_prompt']] + row['choices']
                for text in texts:
                    if len(base.norm(text).split()) >= 4: denied.add(base.norm(text))
                # Source passages/questions/options count as overlap. Repeated
                # "answer using the passage" instructions are boilerplate, not
                # a previously trained paragraph or benchmark question.
                if family == 'ranking':
                    for text in texts: dgrams |= base.grams(text)
                elif family == 'grounded':
                    content = [row['context'], row['question']] if 'context' in row else [row['prompt'].split('\n\nPassage: ', 1)[-1]]
                    for text in content: dgrams |= base.grams(text)
    return denied, dgrams, paths, pilot, old_groups, old_ids, old_contexts


def build(cfg):
    tok = Tokenizer(ROOT / 'tokenizer/bytebpe32k_v1')
    denied, dgrams, paths, pilot, old_groups, old_ids, old_contexts = exclusions()
    data = {s: {f: [] for f in cfg['per_update']} for s in ('train', 'dev', 'test')}
    rejects = Counter(); pools_by_source = {}; census = {}
    import pyarrow.parquet as pq
    origins = {r['source_id'] for r in pq.read_table(ROOT / 'evaluation/full_benchmarks/_cache/raw/hellaswag.parquet', columns=['source_id']).to_pylist()}
    for name in ('arc_challenge', 'piqa', 'hellaswag'):
        pools = {s: [] for s in data}
        for index, raw in enumerate(base.raw_rows(name)):
            row = base.ranking_row(name, raw, index)
            if row['group'] in old_groups or row['id'] in old_ids:
                rejects[name + ':previous-round'] += 1; continue
            if name == 'hellaswag' and row['origin_group'] in origins:
                rejects[name + ':evaluation-origin'] += 1; continue
            if base.overlaps([row['ranking_prompt']] + row['choices'], denied, dgrams):
                rejects[name + ':evaluation-or-old-overlap'] += 1; continue
            head = tok.encode(row['ranking_prompt'])
            if not head or len(set(row['choices'])) != len(row['choices']): continue
            valid = True
            for choice in row['choices']:
                ids = tok.encode(row['ranking_prompt'] + ' ' + choice)
                valid &= ids[:len(head)] == head and len(head) < len(ids) <= cfg['context']
            if not valid: rejects[name + ':encoding'] += 1; continue
            pools[row['split']].append(row)
        pools_by_source[name] = pools; census[name] = {s: len(v) for s, v in pools.items()}
    for split in data:
        n = cfg['updates'] if split == 'train' else 32
        used = set()
        science = base.choose(pools_by_source['arc_challenge'][split], n * 2, cfg['seed'], used)
        piqa = base.choose(pools_by_source['piqa'][split], n, cfg['seed'] + 1, used)
        hella = base.choose(pools_by_source['hellaswag'][split], n, cfg['seed'] + 2, used)
        data[split]['ranking'] = [r for batch in zip(science[::2], science[1::2], piqa, hella) for r in batch]
    from sft.reading_repair.source import candidate
    natural = {s: {False: [], True: []} for s in data}
    for raw in base.raw_rows('squad'):
        if 'squad-' + raw['id'] in cfg.get('excluded_training_ids', []):
            rejects['squad:sample-review'] += 1; continue
        if base.norm(raw['context']) in old_contexts:
            rejects['squad:previous-paragraph'] += 1; continue
        row, reason = candidate(raw, tok, denied, dgrams)
        if row is None: rejects['squad:' + str(reason)] += 1; continue
        if not base.row_valid(tok, row, cfg['context']): continue
        row.update(source='squad', article_group=row['group'], group=fingerprint(['squad-paragraph', base.norm(row['context'])]))
        natural[row['split']][row['unknown']].append(row)
    for split in data:
        n = cfg['updates'] if split == 'train' else 32
        unknown = 72 if split == 'train' else 8
        used = set()
        humans = base.choose(natural[split][False], n * 3 - unknown, cfg['seed'] + 10, used)
        humans += base.choose(natural[split][True], unknown, cfg['seed'] + 11, used)
        random.Random(cfg['seed'] + 12).shuffle(humans)
        authored = authored_grounded(split, n, cfg['seed'] + 20)
        data[split]['grounded'] = [r for i in range(n) for r in humans[i * 3:(i + 1) * 3] + [authored[i]]]
        data[split]['instruction'] = instruction_rows(split, n * 5 if split == 'train' else 64, cfg['seed'] + 30)
        for row in data[split]['grounded'] + data[split]['instruction']:
            if not base.row_valid(tok, row, cfg['context']): raise ValueError('Task encoding overflow')
            if base.norm(row['prompt']) in denied: raise ValueError('Old/evaluation prompt reused')
            if split == 'train' and base.overlaps([row['prompt']], denied, dgrams):
                raise ValueError('Correction task overlaps evaluation text')
    # Original whole source conversations are explicit retention replay. They
    # are newly selected for this round, but may have trained the ancestor 384.
    for split in data:
        pool = []
        for conversation in pilot[split]:
            if conversation['source'] != 'everyday-conversations' or conversation['group'] in old_groups: continue
            messages = conversation['messages']
            text = '\n'.join(m['content'] for m in messages)
            if BLOCK.search(text) or MEDICAL.search(text) or LIVE.search(text) or BAD_RESPONSE.search(text): continue
            users = [m['content'] for m in messages if m['role'] == 'user']
            if split == 'train' and base.overlaps(users, denied, dgrams): continue
            try: encoded = turns(tok, conversation, cfg['context'])
            except (ValueError, OverflowError): continue
            if any(len(tok.encode(m['content'])) > 192 for m in messages if m['role'] == 'assistant'): continue
            answer = messages[-1]['content']
            row = dict(id=conversation['id'] + ':full-replay', group=conversation['group'], source=conversation['source'],
                       split=split, messages=messages, prompt=visible_prefix(messages[:-1])[6:-11], answer=answer,
                       references=[answer], assistant_turns=len(encoded), retention_replay=True)
            pool.append(row)
        n = cfg['updates'] if split == 'train' else 16
        selected = base.choose(pool, n, cfg['seed'] + 40)
        authored = authored_replay(split, n, cfg['seed'] + 41)
        if split == 'train' and any(base.overlaps([m['content'] for m in r['messages'] if m['role'] == 'user'], denied, dgrams) for r in authored):
            raise ValueError('Authored dialogue overlaps evaluation text')
        data[split]['replay'] = [r for pair in zip(selected, authored) for r in pair]
    data['anchors'] = [tok.encode(r['ranking_prompt'] + ' ' + r['answer'])[:cfg['anchor_context']]
                       for r in data['train']['ranking'] if r['source'] == 'hellaswag']
    data['prose_diagnostic'] = __import__('sft.transfer_control.launch', fromlist=['prose_texts']).prose_texts()
    stats = {}; groups = {}; articles = {}
    for split in ('train', 'dev', 'test'):
        groups[split] = {r['group'] for rr in data[split].values() for r in rr}
        articles[split] = {r['article_group'] for r in data[split]['grounded'] if r['source'] == 'squad'}
        stats[split] = {f: dict(examples=len(rr), groups=len({r['group'] for r in rr}), hash=fingerprint(rr),
            sources=dict(Counter(r['source'] for r in rr))) for f, rr in data[split].items()}
        stats[split]['grounded']['answerable_human'] = sum(r['source'] == 'squad' and not r['unknown'] for r in data[split]['grounded'])
        stats[split]['grounded']['unanswerable_human'] = sum(r['source'] == 'squad' and r['unknown'] for r in data[split]['grounded'])
        stats[split]['replay']['supervised_assistant_turns'] = sum(len(turns(tok, r, cfg['context'])) for r in data[split]['replay'])
        for r in data[split]['replay']: turns(tok, r, cfg['context'])
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        if groups[a] & groups[b] or articles[a] & articles[b]: raise ValueError('Group/article leakage')
    if any(groups[s] & old_groups for s in groups): raise ValueError('Previous-round group reused')
    for family, n in cfg['per_update'].items():
        if len(data['train'][family]) != cfg['updates'] * n: raise ValueError('Incomplete family: ' + family)
    samples = {}
    for family, rr in data['train'].items():
        strata = {}
        for r in rr: strata.setdefault((r['source'], r.get('task', ''), r.get('unknown', False)), []).append(r)
        samples[family] = [r for key in sorted(strata) for r in sorted(strata[key], key=lambda r: fingerprint(['review-640', r['id']]))[:6 if key[0] == 'squad' else 2]]
    selection = dict(config=fingerprint(cfg), tokenizer=tok.fingerprint, sources=SOURCES,
        public_training_pool_counts=census, rejections=dict(rejects), stats=stats,
        anchor_hash=fingerprint(data['anchors']), sample_hash=fingerprint(samples),
        exclusions={str(p): file_sha256(p) for p in set(paths)},
        limitations='TRAIN labels are human annotated, not exhaustively audited. Fictional science targets explain only supplied observations. '
        'Original whole conversations are retention replay and may have appeared in ancestor training; they are disjoint from the last round. '
        'New authored scenes use separate groups and held-out presentation variants, but share task families. '
        'Exact/13-word overlap filters are heuristic, not proof of zero pretraining or semantic contamination. Reserved test is not used for selection.')
    return data, selection, samples


def prepare(cfg):
    data, selection, samples = build(cfg)
    atomic_json(DIR / 'prepared.json', data); atomic_json(DIR / 'selection.json', selection)
    atomic_json(DIR / 'review_samples.json', samples)
    return selection
