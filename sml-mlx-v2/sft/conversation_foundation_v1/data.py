"""Pinned sources, whole conversations, deterministic disjoint selection."""
import hashlib
import heapq
import io
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from sml_v2.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v2.tokenization import Tokenizer
from sft.intact_smoltalk_base_pilot.data import turns, visible_prefix, validate_messages
from sft.natural_control.data import REPO as SMOL, REV as SMOL_REV, FILES as SMOL_FILES, norm, grams
from sft.complete_transfer_1024.data import REPO as ULTRA, REV as ULTRA_REV, FILES as ULTRA_FILES

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
SOURCES = [(SMOL, SMOL_REV, SMOL_FILES), (ULTRA, ULTRA_REV, ULTRA_FILES)]


def exclusion_files():
    files = [ROOT/'evaluation/mt_bench_local/upstream/question.jsonl']
    files += [ROOT/f'evaluation/full_benchmarks/_cache/{name}.json' for name in
              ('arc_easy', 'arc_challenge', 'piqa', 'hellaswag', 'ifeval')]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f'Benchmark exclusion input missing: {f}')
    return files


def exclusions():
    prompts = []
    for f in exclusion_files():
        if f.suffix == '.jsonl':
            prompts += [p for line in f.read_text().splitlines() for p in __import__('json').loads(line)['turns']]
        else:
            prompts += [r['prompt'] for r in read_json(f)]
    return {norm(p) for p in prompts}, set().union(*(grams(p) for p in prompts))


def stream():
    # Verify full raw bytes before reading; discard each in-memory Parquet file.
    import requests
    import pyarrow.parquet as pq
    for repo, revision, files in SOURCES:
        for filename, size, sha in files:
            buf = io.BytesIO(); digest = hashlib.sha256(); total = 0
            url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(1024*1024):
                    total += len(chunk)
                    if total > size:
                        raise ValueError('Unexpected upstream size')
                    digest.update(chunk); buf.write(chunk)
            if total != size or digest.hexdigest() != sha:
                raise ValueError('Upstream integrity mismatch: '+filename)
            print('[source-verified]', repo, filename, flush=True)
            buf.seek(0); index = 0
            for batch in pq.ParquetFile(buf).iter_batches(batch_size=128):
                for raw in batch.to_pylist():
                    yield repo, filename, index, raw
                    index += 1
            buf.close()


def basic_quality(messages):
    if not validate_messages(messages): return 'roles_or_empty'
    if len(messages) > 32 or any(len(m['content']) > 12000 for m in messages): return 'request_limit'
    if sum(len(m['content']) for m in messages) > 16000: return 'oversized_candidate'
    for m in messages:
        text = m['content']
        if re.search(r'<\|(?:im_start|im_end|endoftext)\|>', text): return 'foreign_role_tokens'
        if m['role'] != 'assistant': continue
        words = norm(text).split()
        repeated = Counter(tuple(words[i:i+8]) for i in range(len(words)-7))
        if repeated and max(repeated.values()) >= 3: return 'repetitive_target'
        if text.count('```') % 2: return 'unclosed_code_fence'
        if re.search(r'!\[[^\]]*\]\(https?://',text): return 'external_image_target'
    return None


def group_id(messages):
    users = [norm(m['content']) for m in messages if m['role'] == 'user']
    return fingerprint(next((p for p in users if len(p.split()) >= 5), '\n'.join(users)))


def build(cfg):
    tok = Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    heaps = defaultdict(list); rejects = Counter(); census = Counter()
    caps = {s:dict(cfg['training_counts']) if s == 'train' else
            {k:cfg['holdout_per_source'] for k in cfg['training_counts']} for s in ('train','dev','test')}
    cache=DIR/'candidate_pool.json'
    cache_key=fingerprint(dict(seed=cfg['seed'],sources=SOURCES,version=1))
    cached=None
    if cache.exists() and (DIR/'candidate_pool_receipt.json').exists():
        rec=read_json(DIR/'candidate_pool_receipt.json')
        if rec['key']==cache_key and rec['sha256']==file_sha256(cache):cached=read_json(cache)
    for repo, filename, index, raw in ([] if cached else stream()):
        source = 'ultrachat' if repo == ULTRA else raw['source']
        census[source] += 1
        if source not in cfg['training_counts']: continue
        messages = raw['messages']
        if (not validate_messages(messages) or len(messages)>32 or
                any(len(m['content'])>12000 for m in messages) or
                sum(len(m['content']) for m in messages)>16000):
            rejects['structure_or_size']+=1;continue
        identity = fingerprint(messages); group = group_id(messages)
        bucket = int(group[:8],16) % 100
        split = 'dev' if bucket < 5 else 'test' if bucket < 10 else 'train'
        row = dict(id=identity, group=group, source=source, repo=repo,
                   file=filename, row_index=index, messages=messages)
        rank = int(fingerprint([cfg['seed'], identity]),16)
        entry = (-rank, identity, filename+':'+str(index), row)
        heap = heaps[(split,source)]; cap = caps[split][source]*4
        if len(heap) < cap: heapq.heappush(heap,entry)
        elif entry[:3] > heap[0][:3]: heapq.heapreplace(heap,entry)
    if cached:
        for split in caps:
            for source in cfg['training_counts']:
                heaps[(split,source)]=[tuple(entry) for entry in cached['heaps'][split][source]]
        census.update(cached['census']);rejects.update(cached['rejections'])
    else:
        atomic_json(cache,dict(heaps={split:{source:heaps[(split,source)] for source in cfg['training_counts']} for split in caps},
                              census=dict(census),rejections=dict(rejects)))
        atomic_json(DIR/'candidate_pool_receipt.json',dict(key=cache_key,sha256=file_sha256(cache)))
    denied, denied_grams = exclusions()
    selected = {s:[] for s in caps}; ids = set(); groups = set(); held_grams = set(); prompts_seen = set()
    reviewed = ROOT/'sft/intact_smoltalk_base_pilot/review_rejections.json'
    manual_reject = set(read_json(reviewed)['reject']) | set(read_json(DIR/'review_rejections.json'))
    for split in ('test','dev','train'):
        split_grams = set()
        for source, count in caps[split].items():
            added = 0
            for *_, row in sorted(heaps[(split,source)], key=lambda x:x[:3], reverse=True):
                if row['id'] in ids or row['group'] in groups or row['id'] in manual_reject:
                    rejects['duplicate_or_reviewed_rejection'] += 1; continue
                reason=basic_quality(row['messages'])
                if reason:rejects[reason]+=1;continue
                if source=='everyday-conversations' and re.search(
                        r'\b(weather|forecast|snow|rain|thirsty|hydrated|hydration|calories|diet|medication|symptoms|diagnosis)\b',
                        ' '.join(m['content'] for m in row['messages']),re.I):
                    rejects['everyday_live_or_health_claims']+=1;continue
                if source=='smol-summarize-20k' and row['messages'][0]['role']=='system':
                    instruction=row['messages'][0]['content'].lower();answer=row['messages'][-1]['content']
                    sentences=len(re.split(r'(?<=[.!?])\s+(?=[A-Z])',answer.strip()))
                    limit=1 if 'one very short sentence' in instruction else 3
                    if sentences>limit or ('without using second or third person pronouns' in instruction and
                                          re.search(r'\b(you|your|he|his|him|she|her|they|their|them|it|its)\b',answer,re.I)):
                        rejects['summary_instruction_violation']+=1;continue
                users = [m['content'] for m in row['messages'] if m['role'] == 'user']
                substantive = {norm(p) for p in users if len(norm(p).split()) >= 5}
                if substantive & prompts_seen: rejects['duplicate_user_prompt'] += 1; continue
                allgrams = set().union(*(grams(m['content']) for m in row['messages']))
                usergrams = set().union(*(grams(p) for p in users))
                if any(norm(m['content']) in denied for m in row['messages']) or allgrams & denied_grams:
                    rejects['benchmark_overlap'] += 1; continue
                if usergrams & held_grams: rejects['holdout_phrase_overlap'] += 1; continue
                try: enc = turns(tok,row,cfg['context'])
                except OverflowError: rejects['whole_history_too_long'] += 1; continue
                if max(e['targets'] for e in enc) > cfg['max_assistant_tokens']:
                    rejects['long_answer'] += 1; continue
                row.update(assistant_turns=len(enc), assistant_targets=sum(e['targets'] for e in enc),
                           max_context=max(len(e['x']) for e in enc), answer_lengths=[e['targets'] for e in enc])
                selected[split].append(row); ids.add(row['id']); groups.add(row['group'])
                prompts_seen |= substantive; split_grams |= usergrams; added += 1
                if added == count: break
            print('[selected]',split,source,added,'/',count,flush=True)
            if added != count: raise ValueError(f'Insufficient eligible data: {split}/{source}: {added}/{count}')
        held_grams |= split_grams
        random.Random(cfg['seed']+len(split)).shuffle(selected[split])
    stats = {s:dict(conversations=len(rr), assistant_turns=sum(r['assistant_turns'] for r in rr),
                   assistant_targets=sum(r['assistant_targets'] for r in rr),
                   multi_turn=sum(r['assistant_turns']>1 for r in rr),
                   source_counts=dict(Counter(r['source'] for r in rr)), hash=fingerprint(rr)) for s,rr in selected.items()}
    receipt = dict(config=fingerprint(cfg), tokenizer=tok.fingerprint, stats=stats, sources=SOURCES,
                   raw_files_fully_verified=True, exclusions={str(p):file_sha256(p) for p in exclusion_files()},
                   census=dict(census),rejections=dict(rejects),truncated_answers=0,
                   limitations='Synthetic answers are not comprehensively fact-checked. Exact/13-word overlap checks do not establish semantic or pretraining decontamination.')
    atomic_json(DIR/'prepared.json',dict(data=selected,receipt=receipt))
    sample = [r for source in cfg['training_counts'] for r in
              [x for x in selected['train'] if x['source']==source][:3]]
    atomic_json(DIR/'review_samples.json',sample)
    atomic_json(DIR/'review.json',dict(status='unreviewed',sample_sha256=file_sha256(DIR/'review_samples.json')))
    print('[prepared]',stats,flush=True)
    return selected


def batch_at(data,cfg,cursor):
    if not 0 <= cursor < cfg['updates']: raise ValueError('Invalid cursor')
    start = cursor*cfg['batch_conversations']
    result = data['train'][start:start+cfg['batch_conversations']]
    if len(result) != cfg['batch_conversations']: raise ValueError('Incomplete batch')
    return result
