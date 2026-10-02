"""Whole source conversations, unchanged messages, native playground prefixes.

Each assistant reply is a target with its entire preceding history. We do not
drop greetings, shorten replies, add system messages, or flatten to first pairs.
The native formatter puts existing system text in Instruction: and history in
Previous messages:. This is an OpenSML adaptation, not PetitGPT's role-token format.
"""
from collections import Counter, defaultdict
import heapq
import random
import re
from pathlib import Path
from sml_v1.common import fingerprint, read_json
from sml_v1.tokenization import Tokenizer
from sft.natural_control.data import REPO, REV, FILES, stream, norm, grams, exclusions

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
PARENT_EVAL = ROOT/'runs/sft_constraint_completion_1920_v1/evaluations/update_00128.json'


def visible_prefix(messages):
    """Exactly the existing plain-SFT playground format; messages end in user."""
    instruction = messages[0]['content'].strip() if messages[0]['role'] == 'system' else ''
    body = messages[1:] if messages[0]['role'] == 'system' else messages
    if not body or body[-1]['role'] != 'user':
        raise ValueError('Generation prefix must end in user')
    for i, m in enumerate(body):
        if m['role'] != ('user' if i % 2 == 0 else 'assistant'):
            raise ValueError('Nonalternating conversation')
    if not instruction and len(body) == 1:
        return 'User: '+body[-1]['content']+'\nAssistant:'
    parts = []
    if instruction:
        parts.append('Instruction: '+instruction)
    if len(body) > 1:
        parts.append('Previous messages:\n'+'\n'.join(m['role'].title()+': '+m['content'] for m in body[:-1]))
    parts.append('Current message: '+body[-1]['content'])
    return 'User: '+'\n\n'.join(parts)+'\nAssistant:'


def turns(tok, row, context):
    encoded = []
    for i, m in enumerate(row['messages']):
        if m['role'] != 'assistant':
            continue
        p = visible_prefix(row['messages'][:i])
        head = tok.encode(p)
        full = tok.encode(p+' '+m['content'])
        if not head or full[:len(head)] != head:
            raise ValueError('Assistant token boundary changed')
        full += [tok.eos]
        if len(full)-1 > context:
            raise OverflowError('Whole conversation exceeds native context')
        encoded.append(dict(x=full[:-1], y=[-100]*(len(head)-1)+full[len(head):],
                            targets=len(full)-len(head), message_index=i))
    if not encoded:
        raise ValueError('No assistant turns')
    return encoded


def validate_messages(messages):
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    if any(set(m) != {'role', 'content'} or not isinstance(m['content'], str)
           or not m['content'].strip() for m in messages):
        return False
    body = messages[1:] if messages[0]['role'] == 'system' else messages
    return (len(body) % 2 == 0 and all(m['role'] == ('user' if i % 2 == 0 else 'assistant')
                                    for i, m in enumerate(body)))


def retention_rows():
    old = read_json(PARENT_EVAL)
    # Previously inspected development items; diagnostics only, never gradients.
    return dict(reading=[{k:r[k] for k in ('id','prompt','references')}
                         for r in old['retention']['reading_answers']],
                constraints=[{k:r[k] for k in ('id','prompt','reference','rules')}
                             for r in old['answers'][:16]],
                qa=[{k:r[k] for k in ('id','prompt','reference')}
                    for r in old['retention']['qa_answers'][:16]])


def build(cfg, cancelled=lambda: False):
    tok = Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    denied, denied_grams = exclusions()
    # Existing release audit is evaluation-only.
    audit = ROOT/'evaluation/release_chat_comparison/protocol.json'
    if audit.exists():
        for r in read_json(audit)['scenarios']:
            for k in ('prompt','followup'):
                if k in r:
                    denied.add(norm(r[k])); denied_grams |= grams(r[k])
    for rows in retention_rows().values():
        for r in rows:
            denied.add(norm(r['prompt'])); denied_grams |= grams(r['prompt'])
    heaps = defaultdict(list); census = Counter(); rejects = Counter()
    rejected_ids=read_json(DIR/'review_rejections.json')['reject']
    counts = {'train':cfg['training_counts'], 'dev':cfg['validation_counts'], 'test':cfg['reserved_counts']}
    for file, index, raw in stream(cancelled):
        source = raw['source']; census[source] += 1
        if source not in cfg['training_counts']:
            rejects['excluded_source'] += 1; continue
        messages = raw['messages']
        if not validate_messages(messages):
            rejects['malformed_roles_or_empty_content'] += 1; continue
        # Whole rows are preserved, including all greetings and source systems.
        # Match the playground request's total character bound.
        if sum(len(m['content']) for m in messages) > 32000 or len(messages) > 100:
            rejects['native_request_limit'] += 1; continue
        users = [m['content'] for m in messages if m['role'] == 'user']
        group_text = next((p for p in users if len(norm(p).split()) >= 5), '\n'.join(users))
        group = fingerprint(norm(group_text))
        bucket = int(group[:8],16) % 100
        split = 'dev' if bucket < 10 else 'test' if bucket < 20 else 'train'
        identity = fingerprint(messages)
        if identity in rejected_ids:
            rejects['sample_review_rejection']+=1;continue
        if source=='smol-summarize-20k' and messages[0]['role']=='system':
            instruction=messages[0]['content'].lower()
            answer=messages[-1]['content']
            sentences=len(re.split(r'(?<=[.!?])\s+(?=[A-Z])',answer.strip()))
            limit=1 if 'one very short sentence' in instruction else 3
            if sentences>limit or ('without using second or third person pronouns' in instruction
                                  and re.search(r'\b(you|your|he|his|him|she|her|they|their|them|it|its)\b',answer,re.I)):
                rejects['explicit_summary_instruction_violation']+=1;continue
        row = dict(id=identity, group=group, source=source, file=file, row_index=index, messages=messages)
        rank = int(fingerprint([cfg['seed'],identity]),16)
        key = (split,source); cap = counts[split][source]*5
        entry = (-rank, identity, file+':'+str(index), row)
        if len(heaps[key]) < cap:
            heapq.heappush(heaps[key],entry)
        elif entry[:3] > heaps[key][0][:3]:
            heapq.heapreplace(heaps[key],entry)
    selected = {s:[] for s in counts}; seen_ids = set(); seen_groups = set(); seen_user_grams = set()
    # Reserve whole groups first. Shared short greetings/system boilerplate do
    # not count as leakage, but substantive 13-word user overlap does.
    for split in ('test','dev','train'):
        split_grams = set(); split_groups = set()
        for source,n in counts[split].items():
            added = 0
            for _,_,_,row in sorted(heaps[(split,source)],key=lambda x:x[:3],reverse=True):
                if cancelled(): raise InterruptedError('Stopped during selection')
                users = [m['content'] for m in row['messages'] if m['role']=='user']
                gg = set().union(*(grams(t) for t in users))
                # This source consists of shared formatting boilerplate. Require
                # exact prompt/group separation, but do not treat a shared rule
                # such as "exactly five bullet points" as a shared passage.
                if source=='smol-contraints': gg=set()
                if row['id'] in seen_ids or row['group'] in seen_groups or gg & seen_user_grams:
                    rejects['split_or_duplicate_overlap'] += 1; continue
                if any(norm(t) in denied for t in users) or gg & denied_grams:
                    rejects['evaluation_overlap'] += 1; continue
                try:
                    enc = turns(tok,row,cfg['context'])
                except OverflowError:
                    rejects['whole_conversation_context_overflow'] += 1; continue
                row = dict(row, assistant_turns=len(enc), assistant_targets=sum(r['targets'] for r in enc),
                           max_context=max(len(r['x']) for r in enc))
                selected[split].append(row); seen_ids.add(row['id'])
                split_groups.add(row['group']); split_grams |= gg; added += 1
                if added == n: break
            if added != n:
                raise ValueError(f'Insufficient intact {split}/{source}: {added}/{n}. Census: {dict(census)}')
        seen_groups |= split_groups; seen_user_grams |= split_grams
    for split in selected:
        random.Random(cfg['seed']+len(split)).shuffle(selected[split])
    stats = {s:dict(conversations=len(rr), assistant_turns=sum(r['assistant_turns'] for r in rr),
                   assistant_targets=sum(r['assistant_targets'] for r in rr),
                   multi_turn_conversations=sum(r['assistant_turns']>1 for r in rr),
                   distinct_groups=len({r['group'] for r in rr}),
                   source_counts=dict(Counter(r['source'] for r in rr)), hash=fingerprint(rr))
             for s,rr in selected.items()}
    receipt = dict(config=fingerprint(cfg), repo=REPO, revision=REV, files=[list(x) for x in FILES],
                   stats=stats, source_census=dict(census), rejections=dict(rejects),
                   selection={s:[{k:r[k] for k in ('id','source','file','row_index')} for r in rr]
                              for s,rr in selected.items()},
                   integrity='Pinned revision, ranged Parquet file size, selected-message hashes; upstream full-file hashes are metadata, not locally rehashed ranged reads.',
                   truncated_conversations=0, injected_system_messages=0,
                   reviewed_rejections=fingerprint(read_json(DIR/'review_rejections.json')),
                   limitations='Synthetic source answers, not fact-verified. Whole-group/phrase separation is not semantic or pretraining decontamination. Five publisher-generated subsets; incorporated OpenHermes, coding and persona sources excluded. Prior historical SmolTalk exposure may overlap TRAIN selections. Original messages are retained; native serialization differs from PetitGPT.')
    return selected, receipt


def batch_at(data, cfg, update):
    per_epoch = len(data['train']) // cfg['batch_conversations']
    epoch, offset = divmod(update,per_epoch)
    if not 0 <= epoch < cfg['epochs']: raise ValueError('Cursor out of range')
    order = list(range(len(data['train'])))
    random.Random(cfg['seed']+100+epoch).shuffle(order)
    start = offset*cfg['batch_conversations']
    return [data['train'][i] for i in order[start:start+cfg['batch_conversations']]]


def review_rows(data, cfg, split='dev'):
    result = []
    known=set(read_json(DIR/'review_rejections.json').get('sample_reviewed_ids',[])) if split=='train' else set()
    for source in cfg['training_counts']:
        rows = [r for r in data[split] if r['source']==source]
        multi = [r for r in rows if r['assistant_turns']>1]
        single = [r for r in rows if r['assistant_turns']==1]
        chosen = [r for r in rows if r['id'] in known][:6]
        for row in multi[:3]+single[:3]:
            if len(chosen)>=6:break
            if row['id'] not in {r['id'] for r in chosen}:chosen.append(row)
        for row in rows:
            if len(chosen)>=6: break
            if row['id'] not in {r['id'] for r in chosen}: chosen.append(row)
        result += chosen
    return result
