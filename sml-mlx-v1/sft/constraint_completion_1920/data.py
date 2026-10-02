"""Pinned TRAIN-only Smol-Constraints in RAM; one new context per exposure.

Screening is conservative, not full semantic certification. Source data is
synthetic. Only row IDs, hashes, counts and small review samples are persisted.
"""
import hashlib
import io
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from sml_v1.common import read_json, fingerprint
from sft.skill_balance.data import norm, grams, encode_stats
from .constraints import parse, verify, repetition, words

ROOT=Path(__file__).resolve().parents[2]
DIR=Path(__file__).resolve().parent
SOURCE=dict(repo='HuggingFaceTB/smoltalk',revision='5feaf2fd3ffca7c237fc38d1861bc30365d48ffa',
    file='data/smol-constraints/train-00000-of-00001.parquet',bytes=16862908,
    sha256='3369eaff911d3511ee21561a3b8607e1acfd773a6ade80f4759490ffdcce7ba4',license='Apache-2.0',
    origin='Publisher synthetic examples; no local teacher generation')
BLOCK=re.compile(r'```|https?://|<\||\b(?:python|javascript|programming|coding|algorithm|calculate|equation|diagnosis|dosage|medication|medical|therapy|investment|retirement|stock market|legal advice|suicide|rape|disease|cancer|treatment|diet|weight loss|depression|anxiety|current price|latest|as of 20\d\d)\b|\d\s*[+*/=]\s*\d|\$',re.I)
TRUNCATED=re.compile(r'\b(?:and|or|the|a|an|of|to|with|for|by|in|is|are|such as)\s*[.!?]?$',re.I)
FILLER=re.compile(r'\b(?:first point|second point|third point|point one|point two|point three|lorem ipsum|insert here|your text|sample text|as an ai)\b',re.I)
STOPWORDS=set('a an the this that these those to for of in on with by from and or is are be can could would what why how which do does please me about using use benefits advantages main key best top some ways important importance explain describe provide discuss give steps process following question query'.split())


def stream(cancelled=lambda:False):
    import requests
    import pyarrow.parquet as pq
    raw=io.BytesIO();digest=hashlib.sha256();size=0
    url=f"https://huggingface.co/datasets/{SOURCE['repo']}/resolve/{SOURCE['revision']}/{SOURCE['file']}"
    with requests.get(url,stream=True,timeout=(30,120)) as r:
        r.raise_for_status()
        for block in r.iter_content(65536):
            if cancelled():raise InterruptedError('Stopped during source download')
            size+=len(block)
            if size>SOURCE['bytes']:raise ValueError('Unpinned source size')
            digest.update(block);raw.write(block)
    if size!=SOURCE['bytes'] or digest.hexdigest()!=SOURCE['sha256']:raise ValueError('Unpinned source bytes')
    raw.seek(0)
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=128):
        if cancelled():raise InterruptedError('Stopped during source decoding')
        yield from batch.to_pylist()
    raw.close()


def screen(raw):
    m=raw.get('messages',[])
    if len(m)!=2 or [r['role'] for r in m]!=['user','assistant']:return None,'turns'
    p=m[0]['content'].strip();a=m[1]['content'].strip()
    if not 30<=len(a.split())<=150:return None,'answer_length'
    if not 12<=len(p.split())<=190:return None,'prompt_length'
    if BLOCK.search(p+'\n'+a):return None,'scope'
    if re.search(r'(?im)^\s*(?:User|Assistant|System|Human):',p+'\n'+a):return None,'role_marker'
    if '[' in a or ']' in a:return None,'unfilled_placeholder'
    if re.search(r'\b(?:step.by.step|recipe|instructions for making)\b',p,re.I):return None,'procedural_completion_uncertain'
    if FILLER.search(a) or repetition(a) or TRUNCATED.search(a):return None,'filler_or_truncation'
    if not re.search(r'[.!?]["\x27*]*$',a):return None,'unfinished_end'
    parsed=parse(p)
    if not parsed:return None,'unsupported_or_no_task'
    task,rules=parsed
    if not all(verify(a,rules)):return None,'target_rule_failure'
    if norm(a) in norm(p) or grams(a)&grams(p):return None,'embedded_answer'
    content=set(norm(task).split())-STOPWORDS
    if len(content)<3:return None,'weak_topic'
    # Reject answers unrelated even at the lexical topic level; semantic review is separate.
    if len(content & set(norm(a).split()))<min(2,len(content)):return None,'weak_relevance'
    row=dict(id='smol-constraints:'+fingerprint(raw),prompt=p,answer=a,task=task,rules=rules,
        source='smol-constraints',content_words=sorted(content),format='source_constraints')
    return row,None


def new_rows(cfg,cancelled=lambda:False):
    from sml_v1.tokenization import Tokenizer
    from sft.skill_balance.data import exclusions
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');denied,denygrams,_=exclusions()
    if (DIR/'probes.json').exists():
        for rr in read_json(DIR/'probes.json').values():
            for r in rr:denied.add(norm(r['prompt']));denygrams|=grams(r['prompt'])
    pools=[];rejected=Counter()
    review=read_json(DIR/'review.json') if (DIR/'review.json').exists() else {'reject':{}}
    for raw in stream(cancelled):
        row,reason=screen(raw)
        if row is None:rejected[reason]+=1;continue
        if row['id'] in review['reject']:rejected['review_rejection']+=1;continue
        if any(norm(t) in denied or grams(t)&denygrams for t in (row['task'],row['answer'])):
            rejected['public_or_previous_eval_overlap']+=1;continue
        try:encode_stats(tok,row,cfg['context'])
        except ValueError:rejected['context_overflow']+=1;continue
        pools.append(row)
    # Topic-word similarity catches requests differing only in formatting. Keep
    # ONE representative per topic cluster, so no variants leak between splits.
    kept=[];inverted=defaultdict(set);answergrams=set();answers=set()
    for row in sorted(pools,key=lambda r:fingerprint([cfg['seed'],r['id']])):
        ws=set(row['content_words']);candidates=set().union(*(inverted[w] for w in ws))
        if any(len(ws&set(kept[j]['content_words']))/len(ws|set(kept[j]['content_words']))>=.60 for j in candidates):
            rejected['near_topic_duplicate']+=1;continue
        ag=grams(row['answer'])
        if norm(row['answer']) in answers or ag&answergrams:
            rejected['near_answer_duplicate']+=1;continue
        for w in ws:inverted[w].add(len(kept))
        row['group']=fingerprint(row['content_words']);kept.append(row)
        answergrams|=ag;answers.add(norm(row['answer']))
    groups={s:[] for s in ('train','dev','test')}
    for row in kept:
        bucket=int(fingerprint(['constraints-1920-v1',row['group']])[:8],16)%20
        split='dev' if bucket<2 else 'test' if bucket<4 else 'train'
        groups[split].append(dict(row,split=split))
    return groups,dict(eligible_before_dedup=len(pools),available={s:len(rr) for s,rr in groups.items()},rejected=dict(rejected))


def build(cfg,cancelled=lambda:False):
    from sml_v1.tokenization import Tokenizer
    from sft.skill_balance.data import build as replay_build
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    groups,audit=new_rows(cfg,cancelled)
    for split,need in [('train',cfg['updates']*cfg['per_update']['complete']),('dev',cfg['dev_examples']),('test',cfg['test_examples'])]:
        if len(groups[split])<need:raise ValueError(f'Insufficient independent {split} contexts: {len(groups[split])}/{need}; revise the recipe explicitly')
        groups[split]=groups[split][:need]
    old_cfg=read_json(ROOT/'sft/skill_balance/config.json');old,receipt=replay_build(old_cfg,cancelled)
    if receipt!=read_json(ROOT/'sft/skill_balance/selection.json'):raise ValueError('Original 1920 data selection changed')
    meta=read_json(ROOT/cfg['source_bundle']/'model.safetensors.json')
    data={s:{'complete':groups[s]} for s in groups};rng=random.Random(cfg['seed'])
    for family in ('commonsense','instruction','reading'):
        seen=old['train'][family][:meta['next_examples'][family]]
        need=cfg['updates']*cfg['per_update'][family]
        if need>len(seen):raise ValueError('Replay quota exceeds consumed rows')
        rng.shuffle(seen);data['train'][family]=seen[:need]
        for split in ('dev','test'):
            # Balanced source prefixes for MC; these diagnostics have been seen before.
            rr=old[split][family]
            if family=='commonsense':
                data[split][family]=[r for source in ('commonsenseqa','socialiqa') for r in [x for x in rr if x['source']==source][:32]]
            else:data[split][family]=rr[:48 if family=='instruction' else 64]
    data['anchors']=old['anchors']
    # Prove no retained source group or exact prompt appears in two new splits.
    seen=[{r['group'] for r in groups[s]} for s in ('train','dev','test')]
    if any(seen[i]&seen[j] for i in range(3) for j in range(i)):raise ValueError('New-task group leakage')
    stats={s:{f:dict(count=len(rr),unique_sources=len({r.get('source_id',r['id']) for r in rr}),
        hash=fingerprint(rr),answer_tokens=sum(encode_stats(tok,r,cfg['context'])[1] for r in rr),
        rules=dict(Counter(k['kind'] for r in rr for k in r.get('rules',[])))) for f,rr in data[s].items()} for s in groups}
    manifest=dict(config=fingerprint(cfg),source=SOURCE,tokenizer=tok.fingerprint,audit=audit,stats=stats,
        ids={s:{f:[r['id'] for r in rr] for f,rr in data[s].items()} for s in groups},
        replay_selection=fingerprint(receipt),anchor_hash=fingerprint(data['anchors']),review=fingerprint(read_json(DIR/'review.json')),
        limitation='Targets pass implemented surface checks; semantic quality is sampled, not exhaustively verified. Near-topic/13-word dedup is heuristic. Existing retention/public suites have been inspected before.')
    return data,manifest
