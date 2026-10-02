"""Reuse checksummed public-source cache; select by behavior, never source name alone."""
from pathlib import Path
import random,re
from collections import Counter
from sml_v1.common import read_json,atomic_json,fingerprint,file_sha256
from sml_v1.tokenization import Tokenizer
from sft.conversation_foundation_original.data import turns,visible_prefix,group_id,basic_quality,exclusions,exclusion_files,norm,grams,SOURCES
from sft.text_followup_512_v1.data import eligible as legacy_scope
from .behavior import suite,verify
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
CACHE=ROOT/'sft/conversation_foundation_original/candidate_pool.json'
PUBLIC=('conversation','completion','writing','grounded')
# Conservative task filter. Ordinary textual instructions may specify a word or bullet count.
MORE_BANNED=re.compile(r'\b(compute|numerical|numeric|mathematical|geometry|trigonometry|derivative|integral|probability|statistical|coding|debug|pseudocode|binary search|spreadsheet|programming|SQL|HTML|XML|JSON|YAML|API|software development|domain.driven|application development|source code|computer program|solve for|square root)\b|\d\s*[-+*/=]\s*\d|\\(?:frac|sum|int|sqrt)\b|\$\$|<[^>]+>',re.I)

def eligible(messages):
    if not legacy_scope(messages):return False
    if MORE_BANNED.search(' '.join(m['content'] for m in messages)):return False
    instruction=messages[0]['content'].lower() if messages[0]['role']=='system' else ''
    for m in messages:
        if m['role']!='assistant':continue
        t=m['content'];words=norm(t).split()
        if 'one very short sentence' in instruction and len(re.findall(r'[.!?](?:\s|$)',t.strip()))!=1:return False
        if 'without using second or third person pronouns' in instruction and re.search(r'\b(you|your|he|his|him|she|her|they|their|them|it|its)\b',t,re.I):return False
        if len(words)>12 and len(set(words))/len(words)<.28:return False
        if re.search(r"as an ai|as a language model|I (?:do not|don't) have a (?:physical )?body|I cannot browse|I can't browse",t,re.I):return False
        if re.search(r'\[(?:insert|your|name|date|time)[^\]]*\]',t,re.I):return False
        if len(words)<35 and re.search(r"(?:I'd|I would|I'll|I will|I can) (?:be happy to |gladly )?(?:write|draft|create|compose).*?(?:please|provide|let me know)",t,re.I):return False
    return True

def reading_rows(cfg,tok,denied,dgrams,held):
    import pyarrow.parquet as pq
    from sft.reading_repair.source import candidate
    path=ROOT/'sft/grounded_rank_384/_sources/squad.parquet'
    assert file_sha256(path)=='f6da32ffb482ff463ad056477740d1bb284b96a45db3a08bee6a225ca6abf291'
    raw=pq.read_table(path).to_pylist();random.Random(cfg['seed']+901).shuffle(raw)
    out={s:[] for s in ('train','dev','test')};counts=Counter();paras=set();articles=Counter();blocked_grams=held|dgrams
    for item in raw:
        row,reason=candidate(item,tok,denied,dgrams)
        if row is None:continue
        if re.fullmatch(r'[\d\W]+',row['answer']) or re.search(r'\b(value|rate|frequency|ratio|theorem|how many|how much|total|number of)\b',row['question'],re.I):continue
        split=row['split'];unknown=row['unknown'];limit=(384 if unknown else 1152) if split=='train' else (8 if unknown else 24)
        if counts[split,unknown]>=limit or norm(row['context']) in paras or articles[split,row['group']]>=32:continue
        mm=[dict(role='user',content=row['prompt']),dict(role='assistant',content=row['answer'])]
        if not eligible(mm) or any(grams(m['content'])&blocked_grams for m in mm):continue
        if not unknown:assert row['answer'] in row['context']
        r=dict(id=fingerprint(mm),group=group_id(mm),source='grounded',messages=mm,origin=dict(dataset='rajpurkar/squad_v2',id=row['id'],revision='3ffb306f725f7d2ce8394bc1873b24868140c412'),article_group=row['group'],unknown=unknown,references=row['references'])
        ee=turns(tok,r,cfg['context']);r.update(assistant_turns=len(ee),assistant_targets=sum(e['targets'] for e in ee),max_context=max(len(e['x']) for e in ee))
        out[split].append(r);counts[split,unknown]+=1;paras.add(norm(row['context']));articles[split,row['group']]+=1
        if len(out['train'])==1536 and len(out['dev'])==len(out['test'])==32:break
    assert {s:len(v) for s,v in out.items()}==dict(train=1536,dev=32,test=32),counts
    return out,path

def classify(r):
    src=r['source'];mm=r['messages'];users=[m['content'] for m in mm if m['role']=='user']
    if src=='self-oss-instruct':return None
    if src in ('smollm-rewrite-30k','smol-summarize-20k'):return 'writing'
    if any(len(p.split())>45 and re.search(r'according to|passage:|reference text:|provided text|following (?:text|passage)|answer based on',p,re.I) for p in users):return 'grounded'
    multi=sum(m['role']=='assistant' for m in mm)>=2
    if re.search(r'\b(write|draft|compose|suggest|give me|create|list|explain|describe|recommend|summarize|rewrite)\b',users[0],re.I):return 'completion'
    if multi:return 'conversation'
    return None

def practice(count):
    names=['Mira','Jonah','Nadia','Eli','Tomas','Rosa','Dylan','Ada','Leonie','Emil','Nico','Faye']
    places=['orchard','workshop','harbor','courtyard','museum','station','bakery','studio','terrace','market','theater','pavilion']
    items=['sketchbook','basket','scarf','brush','camera','hat','vase','ribbon','glove','map','pencil','bottle']
    colors=['green','blue','orange','white','brown','yellow','black','pink','red','purple','cream','gold']
    rng=random.Random(929442);out=[];seen=set();group_seen=set()
    def pair(p,a):return [dict(role='user',content=p),dict(role='assistant',content=a)]
    while len(out)<count:
        a,b=rng.sample(names,2);p,q=rng.sample(places,2);item,other=rng.sample(items,2);color=rng.choice(colors);family=len(out)%12
        scenario=(a,b,p,q,item,other,color,family)
        if scenario in seen:continue
        seen.add(scenario)
        if family==0:
            mm=pair(f'Please note that my {color} {item} is kept at the {p}.',f'I will keep that in mind: your {item} is at the {p}.')+pair(f'I moved it to the {q}. Where should I look for it now?',f'Look for your {item} at the {q}.');checks={'include':[[q],[item]],'exclude':[p]}
        elif family==1:
            mm=pair(f'{a} prefers the {color} {item}, while {b} prefers the {other}.',f'Understood. {a} prefers the {color} {item}; {b} prefers the {other}.')+pair(f'Which item should I choose for {a}?',f'Choose the {color} {item} for {a}.');checks={'include':[[a],[color],[item]],'exclude':[other]}
        elif family==2:
            mm=pair(f'Help me write a kind note to {a}, thanking them for lending me a {color} {item}.',f'Thank you, {a}, for lending me your {color} {item}. I really appreciate your kindness.')+pair('Make it shorter and keep the specific reason for thanking them.',f'Thank you for lending me your {color} {item}, {a}!');checks={'include':[[a],[item],['thank']],'max_words':20}
        elif family==3:
            mm=pair(f'Write an invitation to join {a} for a walk near the {p} this Sunday, bringing a {color} {item}.',f'Join {a} for a relaxing walk near the {p} this Sunday. Bring a {color} {item}.')+pair(f'Change Sunday to Tuesday and replace {a} with {b}. Keep the location.',f'Join {b} for a relaxing walk near the {p} this Tuesday. Bring a {color} {item}.');checks={'include':[[b],[p],['Tuesday']],'exclude':[a,'Sunday']}
        elif family==4:
            mm=pair(f'Imagine this note: {a} left a {color} {item} at the {p}. What did {a} leave there?',f'{a} left a {color} {item}.')+pair('Does the note say when it was left there?','No. The note does not say when it was left there.');checks={'include':[['not say','does not specify']],'no_question':True}
        elif family==5:
            mm=pair(f'Fictional facts: {a} owns the {item}. {b} owns the {other}. Name the owner of the {other}, using only the name.',b);checks={'exact':b}
        elif family==6:
            mm=pair(f'Give only a two-item bullet list containing a {color} {item} and a {other}.',f'- {color} {item}\n- {other}');checks={'include':[[color],[item],[other]],'bullets':2}
        elif family==7:
            mm=pair(f'Write a short description of a {color} {item} in the {p}.',f'A {color} {item} rests quietly in the {p}.')+pair('Keep the meaning but make every letter lowercase.',f'a {color} {item} rests quietly in the {p}.');checks={'include':[[color],[item],[p]],'lowercase':True}
        elif family==8:
            mm=pair(f'Write a friendly request to borrow a {color} {item} from {a}. Promise to return it on Monday.',f'{a}, could I borrow your {color} {item}? I will return it on Monday.')+pair('Make it a statement instead of a question, keeping the promise.',f'I would like to borrow your {color} {item}, {a}. I will return it on Monday.');checks={'include':[[item],[a],['Monday']],'no_question':True}
        elif family==9:
            mm=pair(f'Write one sentence about {a} visiting the {p} and a second sentence about finding a {item}.',f'{a} visited the {p}. A {item} lay beside the entrance.')+pair('Put those two sentences in separate paragraphs.',f'{a} visited the {p}.\n\nA {item} lay beside the entrance.');checks={'include':[[a],[p],[item]],'paragraphs':2}
        elif family==10:
            mm=pair(f'We were discussing a visit to the {p} with {b} to collect a {item}.',f'What would you like to know about visiting the {p}?')+pair(f'Change of topic: write a cheerful greeting to {a}.',f'Hello, {a}! I hope you are having a lovely day.');checks={'include':[[a],['hello']],'exclude':[p],'no_question':True}
        else:
            mm=pair(f'Write a brief message telling {a} that their {item} is ready to collect at the {p}. Finish with the words: Thank you.',f'{a}, your {item} is ready to collect at the {p}. Thank you.');checks={'include':[[a],[item],[p]],'ending':'Thank you.'}
        for m in mm:m['content']=m['content'].replace('a orange','an orange').replace('A orange','An orange')
        assert verify(mm[-1]['content'],checks) and eligible(mm)
        g=group_id(mm)
        if g in group_seen:continue
        group_seen.add(g)
        out.append(dict(id=fingerprint(mm),group=group_id(mm),source='practice',messages=mm,origin='authored-fictional-practice-v1',family=family,checks=checks,scenario=fingerprint(scenario)))
    return out

def batch_at(data,cfg,cursor):
    if not 0<=cursor<cfg['updates']:raise ValueError('Invalid cursor')
    n=cfg['batch_conversations'];return [data['train'][i] for i in data['schedule'][cursor*n:(cursor+1)*n]]

def build(cfg):
    if (ROOT/'runs/sft_conversation_foundation_v1/contract.json').exists():raise ValueError('Run is frozen')
    receipt=ROOT/'sft/conversation_foundation_original/candidate_pool_receipt.json';rec=read_json(receipt)
    assert file_sha256(CACHE)==rec['sha256']
    raw=read_json(CACHE);tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');denied,dgrams=exclusions()
    behavior={s:suite(s) for s in ('dev','test')}
    behaviorgrams=set().union(*(grams(p) for ss in behavior.values() for r in ss for p in r['prompts']))
    oldfiles=[ROOT/'sft/conversation_foundation_original/prepared.json',ROOT/'sft/text_followup_512_v1/prepared.json']
    oldheld=[r for f in oldfiles for s in ('dev','test') for r in read_json(f)['data'][s]]
    held=set().union(*(grams(m['content']) for r in oldheld for m in r['messages'] if m['role']=='user'))|behaviorgrams
    manual=set(read_json(DIR/'review_rejections.json'))
    manual |= set(read_json(ROOT/'sft/conversation_foundation_original/review_rejections.json'))
    ids=set();groups=set();substantive=set();reject=Counter();selected={s:[] for s in ('test','dev','train')}
    # New root-group split. Old reserved examples excluded everywhere; old training
    # examples cannot enter this run's dev/test, so references cannot have seen them.
    oldtrain={group_id(r['messages']) for f in oldfiles for r in read_json(f)['data']['train']}
    allrows=[entry[-1] for src,entries in raw['heaps']['train'].items() if src!='self-oss-instruct' for entry in entries]
    selection_cache=DIR/'public_selection_cache.json';previous=read_json(selection_cache)
    for split in ('test','dev','train'):
        caps={k:(cfg['training_counts'][k] if split=='train' else cfg['holdout_per_source']) for k in PUBLIC if k!='grounded'}
        counts=Counter();rows=[r for r in allrows if ('test' if int(fingerprint(['v1',r['group']])[:8],16)%100<10 else 'dev' if int(fingerprint(['v1',r['group']])[:8],16)%100<20 else 'train')==split and (split=='train' or r['group'] not in oldtrain)]
        random.Random(cfg['seed']+len(split)).shuffle(rows);rows=previous[split]+rows;splitgrams=set()
        for row_number,original in enumerate(rows):
            if row_number and row_number%10000==0:print('[filtering]',split,row_number,dict(counts),flush=True)
            kind=classify(original)
            if kind not in caps or counts[kind]>=caps[kind]:continue
            r=dict(original,source=kind,origin=dict(dataset=original['repo'],file=original['file'],row_index=original['row_index'],original_source=original['source']))
            if r['id'] in ids or r['group'] in groups or r['id'] in manual:continue
            mm=r['messages']
            if not eligible(mm):reject['scope_or_quality']+=1;continue
            if sum(m['role']=='assistant' for m in mm)>4:reject['long_conversation']+=1;continue
            users=[m['content'] for m in mm if m['role']=='user'];usergrams=set().union(*(grams(p) for p in users))
            sub={norm(p) for p in users if len(norm(p).split())>=5}
            if sub&substantive or usergrams&held:reject['holdout_or_duplicate']+=1;continue
            allgrams=set().union(*(grams(m['content']) for m in mm))
            if allgrams&dgrams or any(norm(m['content']) in denied for m in mm):reject['benchmark_overlap']+=1;continue
            try:ee=turns(tok,r,cfg['context'])
            except OverflowError:reject['context']+=1;continue
            if max(e['targets'] for e in ee)>cfg['max_assistant_tokens']:reject['answer_length']+=1;continue
            r.update(assistant_turns=len(ee),assistant_targets=sum(e['targets'] for e in ee),max_context=max(len(e['x']) for e in ee))
            selected[split].append(r);ids.add(r['id']);groups.add(r['group']);substantive|=sub;splitgrams|=usergrams;counts[kind]+=1
            if all(counts[k]==caps[k] for k in caps):break
        print('[selected]',split,dict(counts),flush=True)
        assert all(counts[k]==caps[k] for k in caps),(split,counts,caps)
        held|=splitgrams
        random.Random(cfg['seed']+10+len(split)).shuffle(selected[split])
    reading,reading_path=reading_rows(cfg,tok,denied,dgrams,held)
    for split in ('test','dev','train'):
        selected[split]+=reading[split]
        random.Random(cfg['seed']+22+len(split)).shuffle(selected[split])
    practice_denied=dgrams|behaviorgrams
    for r in practice(cfg['training_counts']['practice']):
        assert r['id'] not in ids and r['group'] not in groups
        assert not any(grams(m['content'])&practice_denied or norm(m['content']) in denied for m in r['messages'])
        ee=turns(tok,r,cfg['context']);r.update(assistant_turns=len(ee),assistant_targets=sum(e['targets'] for e in ee),max_context=max(len(e['x']) for e in ee))
        selected['train'].append(r);ids.add(r['id']);groups.add(r['group'])
    schedule=[]
    for epoch in range(cfg['epochs']):
        rng=random.Random(cfg['seed']+100+epoch);indices={k:[i for i,r in enumerate(selected['train']) if r['source']==k] for k in cfg['batch_mix']}
        for a in indices.values():rng.shuffle(a)
        for b in range(cfg['updates']//cfg['epochs']):
            chunk=[i for k,n in cfg['batch_mix'].items() for i in indices[k][b*n:(b+1)*n]];rng.shuffle(chunk);schedule+=chunk
    selected.update(schedule=schedule,behavior_dev=behavior['dev'],behavior_test=behavior['test'])
    inputs=[CACHE,receipt,selection_cache,reading_path,*oldfiles,DIR/'reviewed_ids.json',DIR/'review_rejections.json',ROOT/'sft/conversation_foundation_original/review_rejections.json',*exclusion_files()]
    stats={s:dict(conversations=len(selected[s]),source_counts=dict(Counter(r['source'] for r in selected[s])),multi_turn=sum(r['assistant_turns']>1 for r in selected[s]),hash=fingerprint(selected[s])) for s in ('train','dev','test')}
    audit=dict(config=fingerprint(cfg),data_sha256=fingerprint(selected),stats=stats,inputs={str(p):file_sha256(p) for p in inputs},sources=SOURCES,rejections=dict(reject),scope='All roles filtered; self-oss-instruct excluded; no code/math/JSON tasks intended. Heuristic scope filters are imperfect.',limitations='Public targets are synthetic and sample-reviewed, not fully fact-checked. Behavior checks are proxies plus saved transcripts. Public benchmark exact/13-word exclusion is not proof of no contamination. Development/test scenario templates share a design but are separate from training practice templates. Public-source cached pool was selected by the earlier foundation recipe; this is not a fresh raw-data census.')
    atomic_json(DIR/'prepared.json',dict(data=selected,receipt=audit))
    reviewed=set(read_json(DIR/'reviewed_ids.json'))
    sample=[]
    for k in PUBLIC:
        pool=[a for a in selected['train'] if a['source']==k]
        kept=[r for r in pool if r['id'] in reviewed][:3]
        sample+=kept+[r for r in pool if r['id'] not in reviewed][:3-len(kept)]
    sample += [next(r for r in selected['train'] if r['source']=='practice' and r['family']==i) for i in range(12)]
    atomic_json(DIR/'review_samples.json',sample)
    print('[prepared]',stats,flush=True)
