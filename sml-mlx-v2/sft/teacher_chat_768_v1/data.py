"""Frozen source groups; teacher targets conditioned on the student's own history.

No public benchmark item is used to construct or supervise a target. JSON is
only a transport/receipt format. Every student target is a plain-text response.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path
from sml_v2.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v2.tokenization import Tokenizer
from sft.conversation_foundation_v1.data import exclusions, exclusion_files, grams, norm, group_id
from sft.intact_smoltalk_base_pilot.data import turns as all_turns, visible_prefix
from sft.text_followup_512_v1.data import eligible, BANNED

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
FOUNDATION = ROOT/'sft/conversation_foundation_v1/prepared.json'
PARENT = ROOT/'sft/text_followup_512_v1/prepared.json'
CATEGORIES = ['revision', 'history', 'completion', 'grounded', 'text_instructions', 'topic_switch']


def turns(tok, row, context):
    encoded = all_turns(tok, row, context)
    return encoded[-1:] if row.get('last_only') else encoded


def decode_object(text):
    t = text.strip()
    if t.startswith('```'):
        t = re.sub(r'^```(?:json)?\s*', '', t)
        t = re.sub(r'\s*```$', '', t)
    # Some local teachers emit literal line breaks inside quoted strings.
    # Accept those characters without repairing missing keys or fabricated text.
    value = json.loads(t, strict=False)
    if not isinstance(value, dict):
        raise ValueError('Teacher must return an object')
    return value


def annotate(tok, row, cfg):
    quality_messages = [dict(m,content='Earlier assistant reply.')
                        if row.get('last_only') and m['role']=='assistant' and i<len(row['messages'])-1
                        else m for i,m in enumerate(row['messages'])]
    if not eligible(quality_messages) or BANNED.search(' '.join(m['content'] for m in row['messages'])):
        raise ValueError('Nontext, malformed, or repetitive conversation')
    enc = turns(tok, row, cfg['context'])
    if max(e['targets'] for e in enc) > cfg['max_assistant_tokens']:
        raise ValueError('Complete answer is too long; never truncate targets')
    row.update(id=fingerprint(row['messages']), assistant_turns=len(enc),
               max_context=max(len(e['x']) for e in enc),
               assistant_targets=sum(e['targets'] for e in enc))
    return row


def denied_sets():
    denied, ng = exclusions()
    # Retained-model holdouts and previously used development questions also stay out.
    for path in (FOUNDATION, PARENT):
        data = read_json(path)['data']
        for split in ('dev', 'test'):
            for row in data[split]:
                for m in row['messages']:
                    if m['role'] == 'user':
                        denied.add(norm(m['content'])); ng |= grams(m['content'])
    from sft.conversation_foundation_v2.behavior import suite
    for split in ('dev', 'test'):
        for row in suite(split):
            for q in row['prompts']:
                denied.add(norm(q)); ng |= grams(q)
    return denied, ng


def overlaps(messages, denied, ng):
    return any(norm(m['content']) in denied or bool(grams(m['content']) & ng) for m in messages)


def make_plan(cfg, path):
    inputs = {str(p):file_sha256(p) for p in [FOUNDATION, PARENT] + exclusion_files()}
    identity = dict(config=fingerprint(cfg), inputs=inputs,
                    data_code=file_sha256(__file__))
    if path.exists():
        saved = read_json(path)
        if saved['identity'] != identity:
            raise ValueError('Frozen data plan inputs changed; use a new experiment directory')
        return saved
    foundation = read_json(FOUNDATION)['data']; parent = read_json(PARENT)['data']
    tok = Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    denied, ng = denied_sets()
    consumed = foundation['train'][:8192] + parent['train']
    used_groups = {group_id(r['messages']) for r in consumed}
    used_prompts = {norm(m['content']) for r in consumed for m in r['messages'] if m['role']=='user'}
    used_grams = set().union(*(grams(m['content']) for r in consumed for m in r['messages'] if m['role']=='user'))
    candidates = []
    seen = set(); rejects = Counter()
    pool = list(foundation['train'][8192:]); random.Random(cfg['seed']).shuffle(pool)
    for row in pool:
        messages = row['messages']; group = group_id(messages)
        if group in used_groups or group in seen or not eligible(messages): continue
        # Keep the seed's system instruction and first task. Original assistant
        # answer is retained only for source audit; never supplied as a teacher target.
        index = next(i for i,m in enumerate(messages) if m['role']=='assistant')
        prefix = messages[:index]
        if any(norm(m['content']) in used_prompts or grams(m['content']) & used_grams for m in prefix): continue
        if overlaps(prefix, denied, ng): continue
        if len(tok.encode(visible_prefix(prefix))) > 650: continue
        seen.add(group)
        candidates.append(dict(group=group, prefix=prefix, origin={k:row[k] for k in ('repo','file','row_index','source','id')}))
    needed = sum(cfg['candidate_groups'].values())
    if len(candidates) < needed:
        raise ValueError(f'Only {len(candidates)} fresh eligible groups; need {needed}')
    # Assign whole source groups before generating either student or teacher text.
    groups = {}; cursor = 0; held_ng = set()
    for split in ('test','dev','train'):
        groups[split] = []; current = set()
        while len(groups[split]) < cfg['candidate_groups'][split]:
            if cursor >= len(candidates): raise ValueError('Insufficient groups after cross-split filtering')
            row = candidates[cursor]; cursor += 1
            qg = set().union(*(grams(m['content']) for m in row['prefix']))
            if qg & held_ng: continue
            row['category'] = CATEGORIES[len(groups[split]) % len(CATEGORIES)]
            groups[split].append(row); current |= qg
        held_ng |= current
    replay = []; replay_ids=set()
    # Preserve the original chat/text-followup mix, not just short repair replies.
    for source, count in [('conversation',1024),('constraints',512),('reading',160),('science',160),('writing',192)]:
        pool = [r for r in parent['train'] if r['source']==source]
        random.Random(cfg['seed']+len(source)).shuffle(pool)
        for original in pool:
            row = dict(original, source='replay', original_source=source)
            try: row=annotate(tok,row,cfg)
            except (ValueError, OverflowError): continue
            if row['id'] in replay_ids: continue
            replay.append(row); replay_ids.add(row['id'])
            if sum(r['original_source']==source for r in replay)==count:break
        if sum(r['original_source']==source for r in replay)!=count:raise ValueError('Insufficient replay: '+source)
    assert len(replay)==cfg['replay_records']
    result=dict(identity=identity, groups=groups, replay=replay,
                policy='Fresh source groups relative to retained 512 and 768; normalized exact and word-ngram exclusion, not proof of semantic decontamination.')
    atomic_json(path,result)
    return result


def batch_at(data, cfg, cursor):
    per_epoch = len(data['train'])//cfg['batch_conversations']
    epoch, offset = divmod(cursor, per_epoch)
    if not 0 <= cursor < cfg['updates']: raise ValueError('Invalid update cursor')
    batch = []
    for i, source in enumerate(('teacher','replay')):
        pool = [r for r in data['train'] if r['source']==source]
        random.Random(cfg['seed']+epoch*31+i).shuffle(pool)
        batch.extend(pool[offset*8:(offset+1)*8])
    if len(batch)!=16:raise ValueError('Incomplete balanced batch')
    return batch


def validate_review(obj):
    keys=('scope','first_complete','final_complete','history_correct','supported','instruction_following')
    if any(type(obj.get(k)) is not bool for k in keys):raise ValueError('Invalid review schema')
    return all(obj[k] for k in keys)
