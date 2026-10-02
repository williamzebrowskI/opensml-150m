"""Pinned SQuAD v2 train source, streamed; bounded selection in RAM only."""
from collections import Counter,defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import random
import re
import statistics
import requests
from sft.transfer_control.data import digest,normalized
from sft.response_repair.data import training_records as repair_training
from .cards import records as authored
ROOT=Path(__file__).resolve().parents[2]
DIR=Path(__file__).resolve().parent
REPO='rajpurkar/squad_v2'
REVISION='3ffb306f725f7d2ce8394bc1873b24868140c412'
URL=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/squad_v2/train-00000-of-00001.parquet'
SOURCE_SHA='f6da32ffb482ff463ad056477740d1bb284b96a45db3a08bee6a225ca6abf291'
SEED=20260926
# Inspectable lexical exclusion, not a semantic guarantee. No dedicated math/code.
EXCLUDE=re.compile(r'\b(python|javascript|typescript|programming|coding|source code|sql|java|algorithm|api|spark|databricks|database|calculus|algebra|equation|formula|arithmetic|calculate|calculation|math|mathematics|multiply|multiplication|divided|percentage|percent|derivative|integral|quantum computing|computer science|cryptography|investment|diagnosis|medication|dosage|lawsuit)\b|```|https?://|\b\d+\s*[+*/=]\s*\d+',re.I)
QUANTITATIVE=re.compile(r'\b(how many|how much|what percentage|what percent|compute|sum of|product of)\b',re.I)
REVIEW_EXCLUSIONS={
 '572ec82acb0c0d14000f1559':'Question says lowest altitude, reference describes an upper limit.',
 '57299bd7af94a219006aa565':'Reference Taxa is an incomplete answer to the named complex question.',
 '5a82491e31013a001a33539f':'Malformed question about banner names.',
 '5a628335f8d794001af1c086':'Ambiguous no-answer label; national crime decrease is in context.',
 '5ad2f9f5604f3c001a3fda7e':'Bishop versus administrator role is needlessly ambiguous for this subset.',
 '5a516918ce860b001aa3fd98':'Malformed question about laws issued to whom.'
}
UNKNOWN=['Not stated','The passage does not say.','Unknown','The answer is not stated in the passage.']


def stream_source():
    # Verify the entire immutable source as it streams, discard each raw chunk.
    h=hashlib.sha256();size=0
    with requests.get(URL,stream=True,timeout=(20,90)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(65536):h.update(chunk);size+=len(chunk)
    if h.hexdigest()!=SOURCE_SHA or size!=16369982:raise ValueError('Pinned SQuAD source changed')
    from datasets import load_dataset
    dataset=load_dataset(REPO,revision=REVISION,split='train',streaming=True)
    count=0
    for raw in dataset:
        yield raw;count+=1
    if count!=130319:raise ValueError('Unexpected SQuAD train row count')


def split_for(title):
    bucket=int(digest(['article',normalized(title)])[:8],16)%10
    return 'train' if bucket<8 else 'dev' if bucket==8 else 'test'


def candidate(raw,tokenizer,excluded,grams):
    if REVIEW_EXCLUSIONS.get(raw['id']):return None,'manual_training_review'
    context=raw['context'].strip();question=raw['question'].strip();title=raw['title']
    if not 15<=len(context.split())<=240 or not 4<=len(question.split())<=40:return None,'length'
    if EXCLUDE.search(' '.join((title,question,context))) or QUANTITATIVE.search(question):return None,'math_code_or_scope'
    answers=raw['answers'];unknown=not answers['text'];refs=[]
    if not unknown:
        for answer,start in zip(answers['text'],answers['answer_start']):
            if context[start:start+len(answer)]!=answer:return None,'invalid_span'
            if not 1<=len(answer.split())<=16 or len(tokenizer.encode(answer))>48:return None,'long_answer'
            if answer.strip():refs.append(answer.strip())
        if not refs:return None,'empty_answer'
    else:refs=list(UNKNOWN)
    answer=refs[0]
    if normalized(question) in excluded or any(ngrams(t)&grams for t in (question,context,answer)):return None,'evaluation_overlap'
    directive='Answer briefly using only the passage. If the answer is absent, say "Not stated".'
    prompt=f'{directive}\n\nPassage: {context}\n\nQuestion: {question}'
    head=tokenizer.encode('User: '+prompt+'\nAssistant:');ids=tokenizer.encode('User: '+prompt+'\nAssistant: '+answer)
    if ids[:len(head)]!=head:raise ValueError('Answer boundary changed')
    if len(ids)>1536:return None,'token_limit'
    return dict(id='squad-'+raw['id'],origin='squad_v2',family='unanswerable' if unknown else 'answerable',
                prompt=prompt,question=question,context=context,answer=answer,references=list(dict.fromkeys(refs)),
                unknown=unknown,foils=[],group='article-'+digest(normalized(title)),title=title,split=split_for(title)),None


def ngrams(text,n=13):
    words=normalized(text).split()
    return {hashlib.blake2b(' '.join(words[i:i+n]).encode(),digest_size=8).digest()
            for i in range(len(words)-n+1)}


def evaluation_exclusions():
    prompts=[];paths=[]
    cache=ROOT/'evaluation/full_benchmarks/_cache'
    for name in ('arc_easy','arc_challenge','piqa','hellaswag','ifeval'):
        path=cache/(name+'.json');paths.append(path)
        prompts.extend(r['prompt'] for r in json.loads(path.read_text()))
    from sft.transfer_control.data import records as legacy
    from sft.response_repair.data import new_records
    from sft.response_expansion.data import new_records as expansion
    for split in ('dev','test'):
        prompts.extend(r['prompt'] for fn in (legacy,new_records,expansion) for r in fn(split))
    prompts.extend(r['prompt'] for r in legacy('dev','familiar'))
    path=ROOT/'diagnostics/playground_sft_comparison_20260924/protocol.json';paths.append(path)
    prompts.extend(r['prompt'] for r in json.loads(path.read_text())['items'])
    path=DIR/'excluded_previous_prompts.json';paths.append(path)
    prompts.extend(json.loads(path.read_text())['prompts'])
    return {normalized(p) for p in prompts},set().union(*(ngrams(p) for p in prompts)),paths


def rehearsal():
    # Step 448 consumed only the FIRST 512 repair examples. Do not rehearse the
    # unconsumed remainder from step 512. Explicit copying/replacement is omitted.
    rows=[r for r in repair_training()[:512] if r['family'] not in ('copy_or_reply','replace','extract')]
    rows=sorted(rows,key=lambda r:digest(['rehearsal',r['id']]))[:256]
    if len(rows)!=256:raise ValueError('Insufficient previously trained rehearsal')
    return [dict(r,id='retained-'+r['id'],origin='rehearsal',group='rehearsal-'+r['id']) for r in rows]


def select_source(tokenizer,source=None):
    excluded,grams,_=evaluation_exclusions();heaps=defaultdict(list);eligible=Counter();rejected=Counter()
    for raw in stream_source() if source is None else source:
        row,reason=candidate(raw,tokenizer,excluded,grams)
        if row is None:rejected[reason]+=1;continue
        key=(row['split'],row['unknown']);eligible[key]+=1
        rank=int(digest([SEED,row['id']]),16);entry=(-rank,row['id'],row)
        # Six pools, at most 4096 candidates per pool; no full corpus accumulation.
        if len(heaps[key])<4096:heapq.heappush(heaps[key],entry)
        elif entry>heaps[key][0]:heapq.heapreplace(heaps[key],entry)
    print('[data] eligible human rows:',{str(k):v for k,v in eligible.items()},flush=True)
    natural={}
    for split in ('train','dev','test'):
        selected=[];paragraphs=Counter();titles=Counter();questions=set()
        for unknown,n in ((False,864 if split=='train' else 96),(True,288 if split=='train' else 32)):
            added=0
            for _,_,row in sorted(heaps[(split,unknown)],reverse=True):
                para=digest(normalized(row['context']));title=row['group'];question=normalized(row['question'])
                if paragraphs[para]>=2 or titles[title]>=24 or question in questions:continue
                selected.append(row);paragraphs[para]+=1;titles[title]+=1;questions.add(question);added+=1
                if added==n:break
            if added!=n:raise ValueError(f'Insufficient {split} unknown={unknown}: {added}/{n}')
        natural[split]=selected
    return natural,dict(rejected),{str(k):v for k,v in eligible.items()}


def build(tokenizer,source=None):
    natural,rejected,counts=select_source(tokenizer,source)
    rr=rehearsal();new=authored('train');nat=list(natural['train']);rng=random.Random(SEED)
    for rows in (rr,new,nat):rng.shuffle(rows)
    train=[]
    # Every 16 updates: 84 original + 36 human + 8 rehearsal. Randomized locally.
    for i in range(32):
        block=new[i*84:(i+1)*84]+nat[i*36:(i+1)*36]+rr[i*8:(i+1)*8]
        rng.shuffle(block);train.extend(block)
    result=dict(train=train,natural=natural,authored={s:authored(s) for s in ('dev','test')},challenge=json.loads((DIR/'challenge.json').read_text()))
    if len(train)!=4096 or len({r['id'] for r in train})!=4096:raise ValueError('Training count/uniqueness mismatch')
    split_groups=[{r['group'] for r in natural[s]} for s in ('train','dev','test')]
    if any(split_groups[i]&split_groups[j] for i in range(3) for j in range(i)):raise ValueError('Article leakage')
    # Also fail closed on exact/prefix paragraph leakage across selected articles.
    split_contexts=[{tuple(normalized(r['context']).split()[:60]) for r in natural[s]} for s in ('train','dev','test')]
    if any(split_contexts[i]&split_contexts[j] for i in range(3) for j in range(i)):raise ValueError('Paragraph leakage')
    excluded,_,_=evaluation_exclusions()
    held=[r for s in ('dev','test') for r in natural[s]+result['authored'][s]]+result['challenge']
    held_prompts={normalized(r['prompt']) for r in held}
    if any(normalized(r['prompt']) in excluded|held_prompts for r in train):raise ValueError('Evaluation prompt entered training')
    stats={}
    for split,rows in [('train',train)]+[(s,natural[s]+result['authored'][s]) for s in ('dev','test')]+[('challenge',result['challenge'])]:
        sizes=[];targets=[]
        for row in rows:
            p='User: '+row['prompt']+'\nAssistant:';head=tokenizer.encode(p);ids=tokenizer.encode(p+' '+row['answer'])
            if ids[:len(head)]!=head or len(ids)>2048:raise ValueError('Encoding or context violation')
            sizes.append(len(ids));targets.append(len(ids)-len(head)+1)
        stats[split]=dict(examples=len(rows),groups=len({r['group'] for r in rows}),hash=digest(rows),maximum_tokens=max(sizes),
                         assistant_targets=sum(targets),median_answer_tokens=statistics.median(targets),
                         families=dict(Counter(r['family'] for r in rows)),origins=dict(Counter(r['origin'] for r in rows)))
    manifest=dict(source=dict(repo=REPO,url=URL,revision=REVISION,sha256=SOURCE_SHA,license='CC-BY-SA-4.0',split='train'),
                  tokenizer=tokenizer.fingerprint,seed=SEED,rejections=rejected,eligible_source=counts,splits=stats,
                  selection={s:[r['id'] for r in rows] for s,rows in natural.items()},training_order=[r['id'] for r in train],
                  limitations='Lexical scope filter, not semantic guarantee. Authored data is synthetic; independent wording challenge is small. Article and paragraph separation does not prove absence of all paraphrases or pretraining overlap.')
    return result,manifest
