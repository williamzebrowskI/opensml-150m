"""Pinned human demonstrations + constructed complete dialogues + consumed replay."""
from collections import Counter
import hashlib,io,json,random,re
from pathlib import Path
from sml_v2.common import fingerprint,read_json
from sml_v2.tokenization import Tokenizer
from sft.conversation_ab.data import BLOCK,ROLE,norm
from .dialogues import build as dialogues

ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
OFF_SCOPE=re.compile(r'\b(poem|poetry|haiku|sonnet|limerick|rhyme|alphabetical|exercise|workout|sunscreen|medicin\w*|medical|allerg\w*|symptoms?|dosage|weight loss|weight gain|diagnos\w*|treatment)\b',re.I)


def source_rows(cancelled=lambda:False):
    import requests,pyarrow.parquet as pq
    pin=read_json(DIR/'source.json');buf=io.BytesIO();digest=hashlib.sha256();size=0
    url=f"https://huggingface.co/datasets/{pin['repo']}/resolve/{pin['revision']}/{pin['file']}"
    with requests.get(url,stream=True,timeout=(20,90)) as r:
        r.raise_for_status()
        for block in r.iter_content(65536):
            if cancelled():raise InterruptedError('Preparation stopped')
            size+=len(block)
            if size>pin['bytes']:raise ValueError('Source size bound exceeded')
            digest.update(block);buf.write(block)
    if size!=pin['bytes'] or digest.hexdigest()!=pin['sha256']:raise ValueError('Source integrity mismatch')
    buf.seek(0)
    for batch in pq.ParquetFile(buf).iter_batches(batch_size=128):yield from batch.to_pylist()
    buf.close()


def screen(raw,tok,cfg):
    m=raw['messages'];cat=raw['category']
    if cat not in cfg['human_quotas'] or len(m)!=2 or [x['role'] for x in m]!=['user','assistant']:return None
    p,a=[x['content'].strip() for x in m]
    if any(BLOCK.search(t) or ROLE.search(t) for t in [p,a]):return None
    if OFF_SCOPE.search(p+' '+a):return None
    if cat=='Brainstorm' and re.search(r'\b(recommend|best|benefits?|neighborhoods?|hotels?|restaurants?|parks?|cheapest)\b',p,re.I):return None
    if cat=='Generation' and not re.search(r'\b(write|create|draft|compose|generate|make|tell|prepare)\b',p,re.I):return None
    if not 8<=len(a.split())<=140 or len(tok.encode(' '+a))+1>cfg['max_new_tokens']:return None
    # Never truncate an answer or remove a persona/system instruction to make it fit.
    head=tok.encode('User: '+p+'\nAssistant:');full=tok.encode('User: '+p+'\nAssistant: '+a)
    if full[:len(head)]!=head or len(full)>cfg['context']:return None
    if norm(p)==norm(a) or re.search(r'\b(\w+)\s+\1\b',a,re.I):return None
    words=norm(a).split();grams=Counter(tuple(words[i:i+5]) for i in range(len(words)-4))
    if grams and max(grams.values())>=3:return None
    return dict(id='no-robots:'+raw['prompt_id'],group=raw['prompt_id'],source='no-robots',family=cat,prompt=p,answer=a)


def build(cfg,cancelled=lambda:False):
    from sft.skill_balance.data import build as old_build,exclusions,grams
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');denied,dg,_=exclusions()
    # Audited release conversations remain evaluation-only.
    audit=read_json(ROOT/'evaluation/release_chat_comparison/protocol.json')
    for row in audit['scenarios']:
        for key in ['prompt','followup']:
            if key in row:denied.add(norm(row[key]));dg|=grams(row[key])
    rejected=read_json(DIR/'review.json').get('reject',{}) if (DIR/'review.json').exists() else {}
    pool=[];seen=set()
    for raw in source_rows(cancelled):
        if cancelled():raise InterruptedError('Preparation stopped')
        row=screen(raw,tok,cfg)
        if row is None or row['id'] in rejected:continue
        if norm(row['prompt']) in denied or grams(row['prompt'])&dg or grams(row['answer'])&dg:continue
        key=norm(row['prompt'])
        if key in seen:continue
        seen.add(key);pool.append(row)
    # Group by prompt ID and reserve entire tasks before TRAIN selection.
    selected={s:[] for s in ['train','dev','test']}
    for family,n in cfg['human_quotas'].items():
        rr=sorted([r for r in pool if r['family']==family],key=lambda r:fingerprint([cfg['seed'],r['group']]))
        if len(rr)<n+16:raise ValueError(f'Not enough {family}: {len(rr)}, need {n+16}')
        selected['dev']+=rr[:8];selected['test']+=rr[8:16];selected['train']+=rr[16:16+n]
    original=dialogues(cfg)
    old,receipt=old_build(read_json(ROOT/'sft/skill_balance/config.json'),cancelled)
    if receipt!=read_json(ROOT/'sft/skill_balance/selection.json'):raise ValueError('Replay selection changed')
    meta=read_json(ROOT/'runs/sft_skill_balance_v1/step_0001920_628a0a65099b/model.safetensors.json')
    data={s:{'complete':selected[s]+original[s]} for s in selected};data['anchors']=old['anchors']
    data['dialogues']={s:original[s] for s in ['dev','test']};data['human']={s:selected[s] for s in ['dev','test']}
    for family in ['commonsense','instruction','reading']:
        consumed=old['train'][family][:meta['next_examples'][family]]
        rr=list(consumed);random.Random(cfg['seed']+len(family)).shuffle(rr)
        need=cfg['updates']*cfg['per_update'][family]
        if len(rr)<need:raise ValueError('Replay not previously consumed')
        data['train'][family]=rr[:need]
        for s in ['dev','test']:
            rows=old[s][family]
            data[s][family]=rows[:24 if family=='instruction' else 32]
            if family=='commonsense':data[s][family]=[r for src in ['commonsenseqa','socialiqa'] for r in [x for x in rows if x['source']==src][:16]]
    # 64 blocks of 24 human targets and 8 dialogue targets: all examples once.
    human=list(selected['train']);local=list(original['train'])
    random.Random(cfg['seed']).shuffle(human);random.Random(cfg['seed']+1).shuffle(local)
    schedule=[]
    for block in range(64):
        chunk=human[block*24:(block+1)*24]+local[block*8:(block+1)*8]
        random.Random(cfg['seed']+block+10).shuffle(chunk);schedule+=chunk
    data['train']['complete']=schedule
    assert len(schedule)==2048 and len({r['id'] for r in schedule})==2048
    for s in ['train','dev','test']:
        for row in data[s]['complete']:
            head=tok.encode('User: '+row['prompt']+'\nAssistant:');full=tok.encode('User: '+row['prompt']+'\nAssistant: '+row['answer'])
            if full[:len(head)]!=head or len(full)>cfg['context']:raise ValueError('Boundary/context mismatch')
            if norm(row.get('current',row['prompt'])) in denied:raise ValueError('Audited prompt entered training')
    sets=[{norm(r['prompt']) for r in data[s]['complete']} for s in ['train','dev','test']]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Split leakage')
    receipt=dict(config=fingerprint(cfg),source=read_json(DIR/'source.json'),replay=fingerprint(receipt),
                 counts={s:{f:len(rr) for f,rr in data[s].items()} for s in ['train','dev','test']},
                 train_categories=dict(Counter(r['family'] for r in schedule)),data_hash=fingerprint(data),
                 human_ids={s:[r['id'] for r in selected[s]] for s in selected},
                 limitations='1536 human-written targets; 512 targets from 256 constructed dialogues with shared templates. Exact/long-phrase rejection is not semantic/pretraining decontamination. Official No Robots TEST remains unused. CC-BY-NC-4.0 research branch.')
    return data,receipt


def batch_at(data,cfg,u):
    if not 0<=u<cfg['updates']:raise ValueError('Cursor out of bounds')
    return {f:data['train'][f][u*n:(u+1)*n] for f,n in cfg['per_update'].items()}
