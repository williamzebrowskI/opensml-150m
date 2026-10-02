"""Immutable public TRAIN sources; bounded RAM, no raw dataset disk cache.

Public evaluation text is used only for overlap rejection, never as targets.
Source partitions are grouped before any target formatting is applied.
"""
import hashlib
import heapq
import io
import json
import random
import re
import sys
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path
from sml_v2.common import fingerprint, read_json

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
SOURCES = {
    'commongen': dict(repo='allenai/common_gen', revision='792db818e8199aff8f77e183caf7d03de451b426',
        file='data/train-00000-of-00001.parquet', bytes=3232448,
        sha256='5641eb79211ff3022efd7f0d4df83ceb0e894e7c553340fd418d4075d319ea0c', license='MIT'),
    'socialiqa': dict(repo='allenai/social_i_qa', revision='537a2ec8ec565adc0b70b70752893e59e024df26',
        file='default/train/0000.parquet', bytes=3475355,
        sha256='7c4fabfbb03842c565303e30b84bf52189981a4c10610d2eb84ebb410a62e630', license='CC-BY-4.0',
        note='Pinned HF Parquet conversion of public train; original source revision 8835ceb9141d7896d9d968634a9b21ae440e3ec5'),
    'commonsenseqa': dict(repo='tau/commonsense_qa', revision='94630fe30dad47192a8546eb75f094926d47e155',
        file='data/train-00000-of-00001.parquet', bytes=1247103,
        sha256='b0449767ed986bfc2ca52b1244a46ef12f732756727f3cb0a4ab69ac8b3d282b', license='MIT')}
BLOCK = re.compile(r'```|https?://|<\||\b(python|javascript|coding|programming|algorithm|algebra|calculus|equation|arithmetic|mathematics|calculate|multiplication|division|dosage|diagnosis|investment|prescription|suicide|rape)\b|\d\s*[+*/=]\s*\d', re.I)
QUANT = re.compile(r'\b(how many|how much|what percentage|compute|calculate|count the)\b', re.I)
ROLE = re.compile(r'(^|\n)\s*(User|Assistant|System|Human):|<\|', re.I)
REVIEW_REJECTIONS = {
    'commongen:0d59495e73ed01c2b9ce994654e3aa8eb077675f8f82bbe206bade0a4481076b': 'ungrammatical_to_sale',
    'commongen:5f88f085ffbbd2c6d07aca00c5dbfba417e9b81819135382d2cd605c400325b7': 'incorrect_hanged_for_ties',
    'commongen:dd236a6234389ce1c4e274c2706ed887ec67011511600bb4d790b4c2d75c2256': 'unhelpful_niche_slang',
}


@lru_cache(maxsize=1)
def language_tools():
    sys.path.insert(0,str(ROOT/'evaluation/full_benchmarks/_runtime'))
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.tag import PerceptronTagger
    from sml_v2.common import file_sha256
    directory=DIR/'_tagger'; manifest=read_json(directory/'manifest.json')
    for name,sha in manifest['files'].items():
        if file_sha256(directory/name)!=sha: raise ValueError('POS tagger changed')
    nltk.data.path.insert(0,str(directory))
    return PorterStemmer(),PerceptronTagger()


def stems(text):
    stemmer,_=language_tools()
    return {stemmer.stem(w) for w in norm(text).split()}


def finite_sentence(text):
    # Heuristic syntax screen, not a factual/grammatical correctness proof.
    _,tagger=language_tools()
    text=text[:1].upper()+text[1:].lower()
    tagged=tagger.tag(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^\w\s]",text))
    for i,(_,tag) in enumerate(tagged):
        if tag in ('VBP','VBZ','MD') and any(t.startswith('NN') or t=='PRP' for _,t in tagged[:i]):
            return True
    return False


def norm(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def grams(text):
    words = norm(text).split()
    return {hashlib.blake2b(' '.join(words[i:i+13]).encode(), digest_size=8).digest()
            for i in range(len(words)-12)}


def partition(group):
    bucket = int(fingerprint(['skill-balance-v1', group])[:8], 16) % 10
    return 'train' if bucket < 8 else 'dev' if bucket == 8 else 'test'


def stream(name, cancelled=lambda: False):
    import requests
    import pyarrow.parquet as pq
    source = SOURCES[name]
    url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}"
    # Each raw file is at most 3.5 MB, bounded explicitly, freed after decoding.
    raw = io.BytesIO(); digest = hashlib.sha256(); size = 0
    with requests.get(url, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        for block in response.iter_content(65536):
            if cancelled():
                raise InterruptedError('Stopped during data preparation')
            size += len(block)
            if size > source['bytes']:
                raise ValueError('Source exceeded pinned size')
            digest.update(block); raw.write(block)
    if size != source['bytes'] or digest.hexdigest() != source['sha256']:
        raise ValueError('Source hash mismatch: ' + name)
    raw.seek(0)
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=128):
        if cancelled():
            raise InterruptedError('Stopped during data preparation')
        yield from batch.to_pylist()
    raw.close()
    print('[source-verified]', name, flush=True)


def exclusions():
    texts = []; paths = []
    for name in ('arc_easy', 'arc_challenge', 'piqa', 'hellaswag', 'ifeval'):
        path = ROOT/'evaluation/full_benchmarks/_cache'/f'{name}.json'
        paths.append(path)
        for row in read_json(path):
            texts += [row['prompt']] + row.get('choices', [])
    for path in (ROOT/'evaluation/reading_transfer/protocol.json', ROOT/'sft/natural_control/probes.json', DIR/'probes.json'):
        paths.append(path); value = read_json(path)
        groups = [value['rows']] if 'rows' in value else value.values()
        texts.extend(row['prompt'] for group in groups for row in group)
    return {norm(t) for t in texts if len(norm(t).split()) >= 4}, set().union(*(grams(t) for t in texts)), paths


def formats(answer, kind):
    """Exact reversible formatting, not fabricated factual content."""
    if kind == 'plain': return '', answer
    if kind == 'lower': return 'Use lowercase letters throughout.', answer.lower()
    if kind == 'upper': return 'Use uppercase letters throughout.', answer.upper()
    if kind == 'quote': return 'Enclose the entire answer in double quotation marks.', json.dumps(answer, ensure_ascii=False)
    if kind == 'json': return 'Return only a JSON object with the single key "answer".', json.dumps({'answer': answer}, ensure_ascii=False)
    if kind == 'bullet': return 'Return exactly one bullet, starting with "- ".', '- ' + answer
    if kind == 'lower_bullet': return 'Use lowercase letters and return exactly one bullet starting with "- ".', '- ' + answer.lower()
    if kind == 'lower_json': return 'Return only a JSON object with the single key "answer"; use lowercase letters.', json.dumps({'answer': answer.lower()}, ensure_ascii=False)
    raise ValueError(kind)


def unpack(text, kind):
    s = text.strip()
    if kind in ('json', 'lower_json'):
        try:
            obj = json.loads(s)
            if not isinstance(obj, dict) or set(obj) != {'answer'} or not isinstance(obj['answer'], str): return None
            if kind == 'lower_json' and s != s.lower(): return None
            return obj['answer']
        except (ValueError, TypeError): return None
    if kind == 'quote':
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, str) else None
        except ValueError: return None
    if kind in ('bullet', 'lower_bullet'):
        if not s.startswith('- ') or '\n' in s: return None
        if kind == 'lower_bullet' and s != s.lower(): return None
        return s[2:]
    if kind == 'lower' and s != s.lower(): return None
    if kind == 'upper' and s != s.upper(): return None
    return s


def eligible(raw, source, denied, denygrams):
    row_id = source + ':' + fingerprint(raw)
    if row_id in REVIEW_REJECTIONS: return None, 'review_' + REVIEW_REJECTIONS[row_id]
    if source == 'commongen':
        concepts = sorted({norm(w) for w in raw['concepts']})
        if not raw['target'].strip()[:1].isupper(): return None,'lowercase_caption'
        answer = re.sub(r'\s+([.!?,;:])',r'\1',raw['target'].strip())
        if not 8 <= len(answer.split()) <= 40 or not 3 <= len(concepts) <= 5: return None, 'length'
        if '\n' in answer or re.search(r'[.!?]\s+\S', answer): return None, 'multiple_sentences'
        if not answer.endswith(('.', '!', '?')): answer += '.'
        answer=answer[0].upper()+answer[1:]
        if re.search(r'\b(that|which|who|whose)\b',answer,re.I): return None,'relative_clause_ambiguity'
        if not finite_sentence(answer): return None,'caption_fragment'
        if not stems(' '.join(concepts)) <= stems(answer): return None, 'missing_concept'
        prompt = 'Write one sensible sentence about an everyday situation using these concepts: ' + ', '.join(concepts) + '. You may change word forms to make the sentence grammatical.'
        group = 'concepts:' + fingerprint(concepts)
        row = dict(prompt=prompt, answer=answer, concepts=concepts, group=group)
    else:
        question = raw['question'].strip()
        context = raw.get('context', '').strip()
        if source == 'socialiqa':
            choices = [raw['answerA'].strip(), raw['answerB'].strip(), raw['answerC'].strip()]
            gold = int(raw['label']) - 1
            group = 'situation:' + norm(context)
        else:
            choices = [c.strip() for c in raw['choices']['text']]
            gold = raw['choices']['label'].index(raw['answerKey'])
            group = 'question:' + norm(question)
        if not 0 <= gold < len(choices) or len({norm(c) for c in choices}) != len(choices): return None, 'labels'
        if len({tuple(sorted(stems(c))) for c in choices}) != len(choices): return None,'duplicate_choice_forms'
        if any(not 1 <= len(c.split()) <= 24 or c.lower() in ('none', 'none of the above', 'all of the above') for c in choices): return None, 'answer_length'
        if QUANT.search(question): return None, 'math_question'
        base = (context + '\n' if context else '') + question
        if not 4 <= len(base.split()) <= 110: return None, 'prompt_length'
        prompt = 'Choose the most plausible answer. Reply with the answer text only.\n\n' + base
        prompt += '\n\nOptions:\n' + '\n'.join('- ' + c for c in choices)
        row = dict(prompt=prompt, answer=choices[gold], choices=choices, gold=gold, group=group,
                   ranking_prompt='Question: ' + base + '\nAnswer:')
    text = row['prompt'] + '\n' + row['answer']
    if BLOCK.search(text) or ROLE.search(text): return None, 'scope'
    if re.search(r'\b(\w+)\s+\1\b',text,re.I): return None,'repeated_adjacent_word'
    checked = [row['prompt'], row['answer']] + row.get('choices', [])
    if any(norm(t) in denied or grams(t) & denygrams for t in checked): return None, 'evaluation_overlap'
    row.update(id=row_id, source=source, split=partition(row['group']), format='plain')
    return row, None


def encode_stats(tok, row, context):
    prefix = 'User: ' + row['prompt'] + '\nAssistant:'
    head = tok.encode(prefix); full = tok.encode(prefix + ' ' + row['answer'])
    if not head or full[:len(head)] != head: raise ValueError('Tokenization boundary changed')
    if len(full) > context: raise ValueError('Example exceeds context; truncation is forbidden')
    return len(full), len(full) - len(head) + 1


def build(cfg, cancelled=lambda: False):
    from sml_v2.tokenization import Tokenizer
    from sft.reading_repair.source import build as old_build
    tok = Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    denied, denygrams, paths = exclusions(); rejected = Counter(); pools = defaultdict(list)
    source_counts = {'commongen': cfg['updates']*2, 'socialiqa': cfg['updates']*2, 'commonsenseqa': cfg['updates']*2}
    for source in SOURCES:
        groups = set(); answer_texts = set(); answer_grams = set()
        for raw in stream(source, cancelled):
            row, reason = eligible(raw, source, denied, denygrams)
            if row is None: rejected[source + ':' + reason] += 1; continue
            # Whole concept set/situation belongs to one partition; one target per group.
            if row['group'] in groups: rejected[source + ':duplicate_group'] += 1; continue
            if source == 'commongen':
                if norm(row['answer']) in answer_texts or grams(row['answer']) & answer_grams:
                    rejected[source + ':duplicate_answer'] += 1; continue
                answer_texts.add(norm(row['answer'])); answer_grams |= grams(row['answer'])
            groups.add(row['group']); split = row['split']
            quota = source_counts[source] if split == 'train' else cfg[split + '_per_source']
            rank = fingerprint([cfg['seed'], row['id']]); entry = (-int(rank, 16), row['id'], row)
            heap = pools[(source, split)]
            if len(heap) < quota: heapq.heappush(heap, entry)
            elif entry > heap[0]: heapq.heapreplace(heap, entry)
        for split in ('train', 'dev', 'test'):
            need = source_counts[source] if split == 'train' else cfg[split + '_per_source']
            if len(pools[(source, split)]) != need:
                raise ValueError(f'Insufficient {source}/{split}: {len(pools[(source, split)])}/{need}; do not silently shrink the recipe')
    data = {s: dict(commonsense=[], instruction=[]) for s in ('train', 'dev', 'test')}
    kinds = ('plain', 'lower', 'upper', 'quote', 'json', 'bullet', 'lower_bullet', 'lower_json')
    for (source, split), entries in sorted(pools.items()):
        for n, (_, _, original) in enumerate(sorted(entries, reverse=True)):
            views = (0,3) if source=='commongen' and split=='train' else (0,)
            for offset in views:
                row = dict(original,source_id=original['id'])
                if source == 'commongen':
                    kind = kinds[(n+offset) % len(kinds)]
                    rule, answer = formats(row['answer'], kind)
                    row.update(id=row['id']+':'+kind,base_answer=row['answer'], format=kind, answer=answer,
                               prompt=row['prompt'] + ('\n' + rule if rule else ''))
                    if unpack(answer, kind) is None: raise ValueError('Invalid generated target')
                family = 'instruction' if source == 'commongen' else 'commonsense'
                encode_stats(tok, row, cfg['context']); data[split][family].append(row)
    # Rebuild the exact earlier source selection, and rehearse only what 896 consumed.
    old, selection = old_build(tok)
    if selection != read_json(ROOT/'sft/reading_repair/selection.json'): raise ValueError('Reading source changed')
    meta = read_json(ROOT/cfg['source_bundle']/'model.safetensors.json')
    seen = old['train'][:meta['next_example']]
    if meta['step'] != 896 or len(seen) != 3584: raise ValueError('Unexpected parent exposure')
    for split in ('dev', 'test'):
        # Existing diagnostic: explicitly not a fresh reserved reading test.
        natural = sorted(old['natural'][split], key=lambda r: fingerprint(r['id']))[:64]
        authored = sorted(old['authored'][split], key=lambda r: fingerprint(r['id']))[:64]
        data[split]['reading'] = natural + authored
    rng = random.Random(cfg['seed'])
    data['train']['reading'] = []
    while len(data['train']['reading']) < cfg['updates']*2:
        cycle = list(seen); rng.shuffle(cycle); data['train']['reading'].extend(cycle)
    data['train']['reading'] = data['train']['reading'][:cfg['updates']*2]
    for family in ('commonsense', 'instruction'): rng.shuffle(data['train'][family])
    # Prose regularization contexts come only from previously consumed TRAIN passages.
    texts = sorted({r['context'] for r in seen if r.get('context')})
    data['anchors'] = [tok.encode(t)[:cfg['anchor_context']] for t in texts]
    if not data['anchors']: raise ValueError('No training-only prose anchors')
    rng.shuffle(data['anchors'])
    stats = {}
    for split in ('train', 'dev', 'test'):
        stats[split] = {}
        for family, rows in data[split].items():
            sizes = [encode_stats(tok, r, cfg['context']) for r in rows]
            stats[split][family] = dict(exposures=len(rows), unique=len({r['id'] for r in rows}),
                unique_sources=len({r.get('source_id',r['id']) for r in rows}),
                answer_tokens=sum(t for _, t in sizes), maximum_tokens=max(n for n, _ in sizes),
                sources=dict(Counter(r.get('source', r.get('origin', 'original')) for r in rows)),
                formats=dict(Counter(r.get('format', 'plain') for r in rows)), hash=fingerprint(rows))
    new_splits = [{r['group'] for f in ('commonsense', 'instruction') for r in data[s][f]} for s in ('train', 'dev', 'test')]
    if any(new_splits[i] & new_splits[j] for i in range(3) for j in range(i)): raise ValueError('Group leakage')
    manifest = dict(config=fingerprint(cfg), sources=SOURCES, tokenizer=tok.fingerprint, stats=stats,
        syntax_filter=read_json(DIR/'_tagger/manifest.json'),
        rejections=dict(rejected), reviewed_rejections=REVIEW_REJECTIONS,
        anchor_count=len(data['anchors']), anchor_hash=fingerprint(data['anchors']),
        reading_selection_hash=fingerprint(selection),
        ids={s:{f:[r['id'] for r in rows] for f,rows in data[s].items()} for s in ('train','dev','test')},
        limitations='Heuristic scope and 13-word overlap filters; not semantic decontamination or full factual review. New tests use unseen groups but familiar task types. Reading diagnostics have been inspected previously. Format/concept coverage is a proxy, not semantic correctness.')
    return data, manifest
