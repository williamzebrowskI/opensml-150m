"""Pinned streams, bounded selections; source groups stay in one split."""
import hashlib, heapq, json, random, re
from collections import Counter, defaultdict
from pathlib import Path
from sml_v2.common import fingerprint, read_json, atomic_json
from sml_v2.tokenization import Tokenizer
from sft.natural_control import data as old

ROOT=Path(__file__).resolve().parents[2]; DIR=Path(__file__).resolve().parent
SQUAD=dict(repo='rajpurkar/squad_v2',revision='3ffb306f725f7d2ce8394bc1873b24868140c412',file='squad_v2/train-00000-of-00001.parquet',license='CC-BY-SA-4.0')
CREATIVE=re.compile(r'\b(write|create|tell|compose|invent)\b.*\b(story|stories|poem|poetry|tale|fiction|dialogue)\b',re.I|re.S)
EXPLAIN=re.compile(r'^(?:why|how does|how do|what is|what are|explain|describe|what.s the difference|what is the difference)\b',re.I)

EDIT_FILLER=re.compile(r"\b(?:I(?:'ve| have)? (?:made|changed|removed|replaced|added)|here(?:'s| is) (?:a|the) (?:revised|rephrased|edited)|these changes|suggested (?:edits|revision))\b",re.I)
FALSE_SELF=re.compile(r"\b(?:I (?:learn|am learning|adapt|am trained) (?:from|through) (?:our|your|each)|constantly being trained|my (?:personal )?experience|I (?:remember when|have always loved))\b",re.I)
EDIT_REQUEST=re.compile(r"\b(?:edit(?:ing|s)?|revis(?:e|ion)|rephras\w*|rewrit\w*|gramma\w*|punctua\w*|polish\w*|improve this (?:sentence|paragraph))\b",re.I)

SCOPE_EXTRA=re.compile(r"\b(?:triangles?|equilateral|trapez\w*|hypotenuse|degrees? angles?|vectors?|mathematic\w*|HTTP|HTTPS|Content-Type|Content-Disposition|multi.?thread\w*|software|application development|web server|XML|CSV)\b",re.I)
META_EDIT=re.compile(r"\b(?:more concise|revised (?:sentence|paragraph|text)|improv\w* (?:the |this |your )?(?:writing|sentence|paragraph)|writing in this sentence|by removing|I achieved concision)\b",re.I)
PERSONA_EXTRA=re.compile(r"\b(?:I need you to be|I want to speak to (?:a|an)|your name is|your name will be|your character|what is your name|my name is (?:Bob|Raven))\b",re.I)

def quality(row):
    text=row['answer'];words=old.norm(text).split()
    if SCOPE_EXTRA.search(row['prompt']+' '+text) or PERSONA_EXTRA.search(row['prompt']):return False
    if row['family'] not in ('rewrite','summary') and META_EDIT.search(row['prompt']+' '+text):return False
    required=re.search(r'\b(?:exactly )?(\d+)[ -]words?\b',row['prompt'],re.I)
    if required and len(text.split())!=int(required.group(1)):return False
    grams=Counter(tuple(words[i:i+5]) for i in range(len(words)-4))
    if grams and max(grams.values())>=2:return False
    if EDIT_FILLER.search(text) or FALSE_SELF.search(text):return False
    # Dedicated rewrite/summary sources provide the result without meta-edit explanations.
    if row['family'] in ('conversation','explanation','followup') and EDIT_REQUEST.search(row['prompt']):return False
    if row['family']=='followup' and len(row['prompt'].split())>200:return False
    if row['family']=='rewrite':
        if not set(re.findall(r'\b\d+\b',row.get('content',row['prompt'])))<=set(re.findall(r'\b\d+\b',text)):return False
    return True

def partition(group):
    n=int(fingerprint(['base-curriculum',group])[:8],16)%100
    return 'dev' if n<5 else 'test' if n<10 else 'train'

def valid(tok,row,cfg):
    p,a=row['prompt'],row['answer']
    if not quality(row):return False
    if old.BLOCK.search(p+' '+a) or old.MATH.search(p+' '+a) or old.QUANT.search(p):return False
    head=tok.encode('User: '+p+'\nAssistant:'); full=tok.encode('User: '+p+'\nAssistant: '+a)
    if full[:len(head)]!=head:raise ValueError('Prompt/answer boundary changed')
    row['answer_targets']=len(full)-len(head)+1
    minimum=2 if row['family'] in ('reading','commonsense','grounded') else cfg['min_answer_tokens']
    return len(full)<=cfg['context'] and minimum<=row['answer_targets']<=cfg['max_answer_tokens']

def smol_rows(tok,cfg,cancelled):
    # Reuse established topic, persona, language, completion and factual screens.
    # The source's actual earlier turns are retained for follow-ups.
    for file,index,raw in old.stream(cancelled):
        if index%25000==0:print('[scan]',file,index,flush=True)
        if raw['source'] not in old.SOURCES:continue
        msgs=raw['messages']; system=[]
        if msgs and msgs[0]['role']=='system':system=msgs[:1]; msgs=msgs[1:]
        if len(msgs)<2 or msgs[0]['role']!='user':continue
        first=next((m['content'] for m in msgs if m['role']=='user' and len(m['content'].split())>=4),msgs[0]['content'])
        group=fingerprint(old.norm(first))
        for j in (0,2):
            if len(msgs)<j+2 or [r['role'] for r in msgs[:j+2]]!=['user','assistant']*((j+2)//2):continue
            selected=dict(raw,messages=system+msgs[j:j+2])
            row,_=old.candidate(selected,cfg,tok,tok.encode)
            if row is None:continue
            if j:
                if raw['source'] not in ('smol-magpie-ultra-short','everyday-conversations'):continue
                history='\n'.join(('User: ' if k%2==0 else 'Assistant: ')+r['content'].strip() for k,r in enumerate(msgs[:j]))
                if old.BLOCK.search(history) or old.MATH.search(history) or old.PERSONA.search(history) or old.BAD_RESPONSE.search(history):continue
                row['prompt']='Previous messages:\n'+history+'\n\nCurrent message: '+row['prompt']
                row['family']='everyday' if raw['source']=='everyday-conversations' else 'followup'
            elif row['family']=='conversation':
                if CREATIVE.search(row['prompt']):row['family']='creative'
                elif EXPLAIN.search(row['prompt']):row['family']='explanation'
            row.update(group=group,split=partition(group),source=old.REPO,source_file=file,source_row=index,turn=j//2)
            row['id']=fingerprint([raw['source'],row['prompt'],row['answer']])
            if valid(tok,row,cfg):yield row

def skill_rows(tok,cfg,cancelled):
    from sft.skill_balance import data as skill
    denied,dgrams,_=skill.exclusions()
    for source in ('socialiqa','commonsenseqa'):
        for raw in skill.stream(source,cancelled):
            row,_=skill.eligible(raw,source,denied,dgrams)
            if row is None:continue
            row['split']=partition(row['group'])
            row['content']=row['ranking_prompt']
            # Assign each source item to one task only; no duplicate formatting views.
            formatted=int(fingerprint(row['group'])[:8],16)%4==0
            row['family']='instruction' if formatted else 'commonsense'
            if formatted:
                row['prompt']=row['prompt'].replace('Reply with the answer text only.','').strip()
                kinds=['plain','lower','upper','quote','bullet','lower_bullet','json','lower_json']
                kind=kinds[int(fingerprint(['answer-format',row['group']])[:8],16)%len(kinds)]
                suffix,answer=skill.formats(row['answer'],kind)
                assert skill.unpack(answer,kind) is not None
                row['prompt']+='\n\n'+suffix;row['answer']=answer;row['format']=kind
            if valid(tok,row,cfg):yield row

def reading_rows(tok,cfg,cancelled):
    import fsspec,pyarrow.parquet as pq
    url=f"https://huggingface.co/datasets/{SQUAD['repo']}/resolve/{SQUAD['revision']}/{SQUAD['file']}"
    with fsspec.open(url,'rb',block_size=4*1024**2,cache_type='readahead') as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=128):
            if cancelled():raise InterruptedError('Stopped while preparing reading')
            for raw in batch.to_pylist():
                context=raw['context']; answers=raw['answers']['text']
                if answers:
                    answer=answers[0]; family='reading'
                    if answer not in context:raise ValueError('Reading target absent from context')
                else:answer='The passage does not provide that information.';family='uncertainty'
                row=dict(id='squad:'+raw['id'],group=fingerprint(old.norm(context)),family=family,source=SQUAD['repo'],content=context,answer=answer,prompt='Answer using only the passage. If the answer is not given, say so.\n\nPassage: '+context+'\n\nQuestion: '+raw['question'])
                row['split']=partition(row['group'])
                if old.MEDICAL.search(row['prompt']) or old.LIVE.search(row['prompt']):continue
                if valid(tok,row,cfg):yield row

_COLLECTION=None

def build(cfg,cancelled=lambda:False,review=False):
    global _COLLECTION
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1'); denied,dgrams=old.exclusions()
    from sft.skill_balance.data import exclusions
    from sft.transfer_control.launch import prose_texts
    extra,extra_grams,_=exclusions();denied|=extra;dgrams|=extra_grams
    for text in prose_texts():denied.add(old.norm(text));dgrams|=old.grams(text)
    for row in read_json(DIR/'review_prompts.json'):
        denied.add(old.norm(row['prompt']));dgrams|=old.grams(row['prompt'])
    # Already inspected tiny diagnostic remains an audit only; none enters gradients.
    for rows in read_json(ROOT/'sft/base_restart/examples.json').values():
        for row in rows:denied.add(old.norm(row['prompt']));dgrams|=old.grams(row['prompt'])
    heaps=defaultdict(list); eligible=Counter(); integrity={}
    data_key=fingerprint({k:v for k,v in cfg.items() if k not in ('training_counts','validation_counts','excluded_ids','updates','warmup_updates','mix_blocks','checkpoint_every','evaluation_updates')})
    def accept(row):
        if old.norm(row['prompt']) in denied or old.grams(row['prompt'])&dgrams:return
        key=row['split'],row['family'];eligible[key]+=1
        n=cfg['training_counts'][row['family']] if row['split']=='train' else cfg['validation_counts'][row['family']]
        entry=(-int(fingerprint([cfg['seed'],row['id']]),16),row['id'],sum(eligible.values()),row)
        h=heaps[key]
        if len(h)<8192:heapq.heappush(h,entry)
        elif entry[:2]>h[0][:2]:heapq.heapreplace(h,entry)
    if _COLLECTION is not None and _COLLECTION[0]==data_key:
        _,heaps,eligible,integrity=_COLLECTION
    else:
        for row in smol_rows(tok,cfg,cancelled):accept(row)
        for row in reading_rows(tok,cfg,cancelled):accept(row)
        for row in skill_rows(tok,cfg,cancelled):accept(row)
        _COLLECTION=(data_key,heaps,eligible,integrity)
    print('[eligible]',dict(eligible),flush=True)
    pools={s:[] for s in ('train','dev','test')}; seen=set(); seen_groups=set(); reserved_grams=set()
    for split in ('test','dev','train'):
        split_grams=set()
        for family,ntrain in cfg['training_counts'].items():
            n=ntrain if split=='train' else cfg['validation_counts'][family];added=0
            for _,_,_,row in sorted(heaps[(split,family)],key=lambda e:e[:2],reverse=True):
                if row['id'] in cfg.get('excluded_ids',[]):continue
                key=old.norm(row['prompt']);g=old.grams(row.get('content',row['prompt']))
                # One example per source conversation/passage, even within training.
                if key in seen or row['group'] in seen_groups or g&reserved_grams:continue
                pools[split].append(row);seen.add(key);seen_groups.add(row['group']);split_grams|=g;added+=1
                if added==n:break
            if added!=n:raise ValueError(f'Not enough {split}/{family}: {added}/{n}; eligible={dict(eligible)}')
        reserved_grams|=split_grams
    for split in pools:random.Random(cfg['seed']+len(split)).shuffle(pools[split])
    # Every eight optimizer updates (128 examples) sees the full intended mix.
    # Each selected source example occurs exactly once in this run.
    by_family={f:[r for r in pools['train'] if r['family']==f] for f in cfg['training_counts']}
    ordered=[];rng=random.Random(cfg['seed'])
    for block in range(cfg['mix_blocks']):
        rows=[]
        for family,count in cfg['training_counts'].items():
            if count%cfg['mix_blocks']:raise ValueError('Mix must divide into the balanced blocks')
            n=count//cfg['mix_blocks'];rows+=by_family[family][block*n:(block+1)*n]
        rng.shuffle(rows);ordered+=rows
    pools['train']=ordered
    stats={s:dict(count=len(rows),families=dict(Counter(r['family'] for r in rows)),answer_targets=sum(r['answer_targets'] for r in rows),hash=fingerprint(rows)) for s,rows in pools.items()}
    from sft.skill_balance.data import SOURCES
    manifest=dict(config=fingerprint(cfg),smol_source=dict(repo=old.REPO,revision=old.REV,files=old.FILES,license='Apache-2.0'),reading_source=SQUAD,skill_sources={k:v for k,v in SOURCES.items() if k!='commongen'},stats=stats,eligible={str(k):v for k,v in eligible.items()},limitations='Automated topical/duplicate screens plus sampled review, not exhaustive factual verification. Smol data is synthetic; reading/commonsense use human annotations. Stream revision and selected content hashes are pinned; range reads are not whole-file SHA checks.')
    if review:
        for family in cfg['training_counts']:
            rows=[r for r in pools['train'] if r['family']==family]
            for row in rows[::max(1,len(rows)//12)][:12]:print('[sample]',json.dumps(row,ensure_ascii=False),flush=True)
    pools['train_probes']=[r for family in cfg['training_counts'] for r in by_family[family][:cfg['train_probe_per_family']]]
    pools['manual_review']=read_json(DIR/'review_prompts.json')
    return pools,manifest

def rebuild(cfg,cancelled=lambda:False):
    global _COLLECTION
    pools,manifest=build(cfg,cancelled)
    _COLLECTION=None
    if fingerprint(manifest)!=fingerprint(read_json(DIR/'selection.json')):raise ValueError('Selection differs from checked stream')
    return pools

if __name__=='__main__':
    cfg=read_json(DIR/'config.json');pools,manifest=build(cfg,review=True)
    atomic_json(DIR/'selection.json',manifest)
    print('[prepared]',json.dumps(manifest['stats']),flush=True)
