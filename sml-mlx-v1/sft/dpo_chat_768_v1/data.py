"""Frozen public preference pairs and reviewed model errors; old chat holdouts unchanged."""
from pathlib import Path
import random,re
from collections import Counter
from difflib import SequenceMatcher
from sml_v1.common import read_json,atomic_json,file_sha256,fingerprint
from sml_v1.tokenization import Tokenizer
from sft.text_followup_512_v1.data import eligible
from sft.conversation_foundation_original.data import turns,visible_prefix,exclusions,exclusion_files,norm,grams
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
PARENT=ROOT/'sft/text_followup_512_v1/prepared.json'
SOURCE=DIR/'_sources/helpsteer2_preference.jsonl.gz'
OUT_OF_SCOPE=re.compile(r'\b(software development|domain.driven|application development|source code|pseudocode|computer program|programming language|API|arithmetic|compute|multiply|divide|solve for|square root|probability)\b|(?:descending|ascending|highest|lowest).{0,35}(?:price|cost)|(?:price|cost).{0,35}(?:descending|ascending)|\d\s*[+*/=]\s*\d',re.I)

def text_scope(messages):
    return not any(OUT_OF_SCOPE.search(m['content']) for m in messages)

def encode_reply(tok,messages,answer,context):
    prefix=visible_prefix(messages);head=tok.encode(prefix);full=tok.encode(prefix+' '+answer)+[tok.eos]
    if not head or full[:len(head)]!=head or not answer.strip():raise ValueError('Invalid preference answer boundary')
    if len(full)-1>context:raise OverflowError('Complete reply exceeds context; no truncation')
    return dict(x=full[:-1],y=[-100]*(len(head)-1)+full[len(head):],targets=len(full)-len(head))

def pair_id(messages):return fingerprint([norm(m['content']) for m in messages if m['role']=='user'])

def batch_at(data,cfg,cursor):
    assert 0<=cursor<cfg['updates']
    return {f:[data[f][i] for i in data['schedule'][f][cursor*n:(cursor+1)*n]] for f,n in cfg['per_update'].items()}

def pool(cfg):
    import gzip,json
    source=read_json(DIR/'_sources/helpsteer2_source.json')
    for item in source['files']:assert file_sha256(DIR/'_sources'/item['local'])==item['sha256']
    ratings={(r['prompt'],r['response']):r for r in map(json.loads,gzip.open(DIR/'_sources/helpsteer2_train.jsonl.gz','rt'))}
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');denied,dgrams=exclusions();parent=read_json(PARENT)['data']
    held=set().union(*(grams(m['content']) for s in ('dev','test') for r in parent[s] for m in r['messages']))
    oldprompts={norm(m['content']) for s in ('train','dev','test') for r in parent[s] for m in r['messages'] if m['role']=='user'}
    reserved=read_json(DIR/'authored.json')
    blocked=set(read_json(DIR/'rejections.json')) if (DIR/'rejections.json').exists() else set()
    result=[];seen=set();counts=Counter()
    raw=list(map(json.loads,gzip.open(SOURCE,'rt')));random.Random(cfg['seed']).shuffle(raw)
    for r in raw:
        counts['raw']+=1
        strength=r['preference_strength']
        if r['split']!='train' or abs(strength)<1:continue
        p=r['prompt'];a=r['response_2' if strength>0 else 'response_1'];b=r['response_1' if strength>0 else 'response_2']
        rating=ratings.get((p,a),{})
        if min(rating.get(k,0) for k in ('helpfulness','correctness','coherence'))<3:continue
        messages=[dict(role='user',content=p)];source_id=fingerprint(p)
        if source_id in blocked:continue
        if not text_scope(messages+[dict(role='assistant',content=t) for t in (a,b)]):continue
        if any(not eligible(messages+[dict(role='assistant',content=t)]) for t in (a,b)):continue
        if norm(a)==norm(b) or min(len(a.split()),len(b.split()))<6:continue
        if max(len(tok.encode(a)),len(tok.encode(b)))>cfg['max_preference_tokens'] or len(tok.encode(p))>900:continue
        if len(a)>2.5*len(b) or len(b)>2.5*len(a):continue
        if sum(ord(ch)<128 for ch in p)/len(p)<.97:continue
        if re.search(r'translat|\b(?:German|Spanish|French|Chinese|Hindi|Portuguese|Russian|Italian|Korean|Japanese|Arabic|Tamil)\b|Teacher:|Definition:|Detailed Instructions|Premise:|Hypothesis:|In this task|You will be given|\bQ:|Options:|Confidence:',p+'\n'+a,re.I):continue
        if SequenceMatcher(None,a.lower().split(),b.lower().split()).ratio()>.8:continue
        # Avoid obvious missing referents and ungrounded analysis of unnamed works.
        if len(p.split())<45 and re.search(r"\b(the author|the character|the article|the passage|the story|their message|his actions)\b",p,re.I):continue
        if re.search(r"\b(as an ai|as a language model|i cannot|i can't|i am unable|i'm unable)\b",a,re.I):continue
        if any(norm(p)==norm(z['messages'][-1]['content']) for z in reserved):continue
        ag=set().union(*(grams(t) for t in (p,a,b)))
        if norm(p) in denied or norm(p) in oldprompts or ag&(dgrams|held):continue
        group=pair_id(messages)
        if group in seen:continue
        try:
            for answer in (a,b):encode_reply(tok,messages,answer,cfg['context'])
        except (ValueError,OverflowError):continue
        row=dict(id=group,source='helpsteer2',source_id=source_id,messages=messages,chosen=a,rejected=b,preference_strength=strength,chosen_ratings=rating,human_justification=r['preference_statement']+' '+r['preference_elaboration'])
        result.append(row);seen.add(group)
    print('[public-pool]',len(result),flush=True)
    atomic_json(DIR/'candidate_pool.json',dict(rows=result,source=source,config=fingerprint(cfg)))
    return result

def build(cfg):
    if (ROOT/'runs/dpo_chat_768_v1/contract.json').exists():raise ValueError('Started run is frozen')
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');public=pool(cfg)
    candidates=read_json(DIR/'model_candidates.json');decision=read_json(DIR/'model_review.json')
    assert decision['candidates_sha256']==file_sha256(DIR/'model_candidates.json')
    chosen_ids=set(decision['accepted_ids']);custom=[r for r in candidates['pairs'] if r['id'] in chosen_ids]
    assert len(custom)==len(chosen_ids)
    h=cfg['preference_holdout_pairs'];n=cfg['preference_train_pairs']-len(custom)
    assert len(public)>=n+2*h
    dev=public[:h];held=set().union(*(grams(m['content']) for r in dev for m in r['messages']))
    rest=[r for r in public[h:] if not any(grams(m['content'])&held for m in r['messages'])]
    test=rest[:h];held|=set().union(*(grams(m['content']) for r in test for m in r['messages']))
    rest=[r for r in rest[h:] if not any(grams(m['content'])&held for m in r['messages'])]
    assert len(rest)>=n,(len(rest),n)
    pref=rest[:n]+custom;random.Random(cfg['seed']+1).shuffle(pref)
    assert not any(grams(m['content'])&held for r in pref for m in r['messages'])
    parent=read_json(PARENT)['data'];replay=[];seen=set()
    foundation=read_json(ROOT/'sft/conversation_foundation_original/prepared.json')['data']['train'][:8192]
    chat=[dict(r,source='conversation') for r in foundation if r['assistant_turns']>=2]
    chat+=[r for r in parent['train'] if r['source']=='conversation']
    texts=[r for r in parent['train'] if r['source'] in ('writing','reading','science')]
    random.Random(cfg['seed']+2).shuffle(chat);random.Random(cfg['seed']+3).shuffle(texts)
    for name,rows,count in [('conversation',chat,1536),('text',texts,512)]:
        used=0
        for r in rows:
            if r['id'] in seen or not eligible(r['messages']) or not text_scope(r['messages']):continue
            if any(grams(m['content'])&held for m in r['messages']):continue
            try:ee=turns(tok,r,cfg['context'])
            except OverflowError:continue
            if max(e['targets'] for e in ee)>cfg['max_assistant_tokens']:continue
            replay.append(r);seen.add(r['id']);used+=1
            if used==count:break
        assert used==count,(name,used)
    schedule={'preference':[],'replay':[]}
    for epoch in range(2):
        rng=random.Random(cfg['seed']+100+epoch);indices=list(range(len(pref)));rng.shuffle(indices);schedule['preference']+=indices
        chats=list(range(1536));texts=list(range(1536,2048));rng.shuffle(chats);rng.shuffle(texts)
        schedule['replay'] += [j for i in range(512) for j in chats[i*3:i*3+3]+texts[i:i+1]]
    data=dict(preference=pref,preference_dev=dev,preference_test=test,replay=replay,schedule=schedule)
    inputs=[PARENT,ROOT/'sft/conversation_foundation_original/prepared.json',SOURCE,DIR/'_sources/helpsteer2_train.jsonl.gz',DIR/'_sources/helpsteer2_source.json',DIR/'authored.json',DIR/'model_candidates.json',DIR/'model_review.json',DIR/'rejections.json',DIR/'reviewed_ids.json']+exclusion_files()
    receipt=dict(config=fingerprint(cfg),data_sha256=fingerprint(data),inputs={str(p):file_sha256(p) for p in inputs},public_train=n,reviewed_model_pairs=len(custom),unique_preferences=len(pref),unique_replay=len(replay),epochs=2,public_preference_labels='Human preference annotations plus quality ratings, filtered and sample-reviewed; not independently verified for every row. UltraFeedback excluded after review.',retention='Previously seen TRAIN conversations replay intentionally. Parent dev/test held fixed. Public benchmarks excluded by exact/13-word heuristics, not proof of zero contamination.')
    atomic_json(DIR/'prepared.json',dict(data=data,receipt=receipt))
    reviewed=set(read_json(DIR/'reviewed_ids.json'))
    samples=[r for r in pref if r['source']=='helpsteer2' and r['id'] in reviewed]+[r for r in pref if r['source']=='model-768']+replay[:3]+replay[1536:1539]
    atomic_json(DIR/'review_samples.json',samples)
    print('[prepared]',{k:v for k,v in receipt.items() if k!='inputs'},flush=True)
    return receipt
