"""Local, frozen plain-text curriculum; no benchmark answers used for training."""
import random
import re
from collections import Counter
from pathlib import Path
from sml_v2.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v2.tokenization import Tokenizer
from sft.conversation_foundation_v1.data import exclusions, exclusion_files, basic_quality, turns, visible_prefix, norm, grams
ROOT=Path(__file__).resolve().parents[2]
DIR=Path(__file__).resolve().parent
FOUNDATION=ROOT/'sft/conversation_foundation_v1/prepared.json'
GROUNDED=ROOT/'sft/grounded_rank_384/prepared.json'
SCIENCE=ROOT/'sft/preference_long_640/prepared.json'
BANNED=re.compile(r'```|[{}]|\b(json|xml|yaml|sql|python|javascript|programming|algorithm|equation|algebra|calculus|arithmetic|calculate|multiplication|division|math|mathematics|percentage|percentages|quadratic|statistics|dataset|spreadsheet|html|css|function|regex|coding|code|java|c\+\+|formula)\b|https?://|[=<>]',re.I)
DEFER=re.compile(r"\b(I (?:will|can|would|could)|I'll|let me) (?:write|draft|create|compose)\b",re.I)

def eligible(messages):
    if basic_quality(messages):return False
    if any('|' in m['content'] or re.search(r'\d\s*[+*/=]\s*\d|\b(?:growth rate|sales by quarter|solve for|square root)\b',m['content'],re.I) for m in messages):return False
    return not BANNED.search(' '.join(m['content'] for m in messages))

def row(source,messages,origin,**extra):
    return dict(source=source,messages=messages,origin=origin,id=fingerprint(messages),**extra)

def pair(prompt,answer):return [dict(role='user',content=prompt),dict(role='assistant',content=answer)]

def constraints(split,count):
    # Fictional scenarios, not factual claims. Disjoint people/places per split;
    # shared task templates make these diagnostics proxies, not public benchmarks.
    names={'train':['Maya','Leo','Nora','Sam','Iris','Owen','Zara','Felix'],
           'dev':['Tessa','Hugo','Cleo','Ravi'], 'test':['Anya','Bryn','Esme','Joel']}[split]
    places={'train':['library','garden','studio','courtyard','museum','cafe','park','hall'],
            'dev':['gallery','terrace','lounge','pavilion'], 'test':['theater','greenhouse','workshop','patio']}[split]
    activities=['reading club','painting session','birdwatching walk','storytelling circle','photography meetup','pottery class','gardening group','choir rehearsal']
    days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    rng=random.Random(991+['train','dev','test'].index(split));seen=set();out=[]
    while len(out)<count:
        person=rng.choice(names);place=rng.choice(places);activity=rng.choice(activities)
        day,newday=rng.sample(days,2);item=rng.choice(['notebook','scarf','water bottle','sketchbook','jacket','pencil'])
        identity=(person,place,activity,day,newday,item)
        if identity in seen:continue
        seen.add(identity);kind=len(out)%8
        genre=(len(out)//8)%6
        if genre==0:
            prompt=f'Write a brief invitation using these details. Use the updated day. {person} is organizing a {activity} at the {place} on {day}. The event has moved to {newday}. Guests should bring a {item}.'
            a=f'Join {person} for a {activity} at the {place} on {newday}.';b=f'Please bring a {item}.'
            required=[person,activity,place,newday,item]
        elif genre==1:
            prompt=f'Write a short thank-you message to {person} for bringing a {item} to our {activity} at the {place} on {newday}. Say that the help made the event easier to organize.'
            a=f'Thank you {person} for bringing a {item} to our {activity} at the {place} on {newday}.';b='Your help made the event easier to organize.'
            required=[person,activity,place,newday,item,'thank']
        elif genre==2:
            prompt=f'Draft a brief message asking {person} if I can borrow a {item} for a {activity} at the {place} on {newday}. Promise to return it afterward.'
            a=f'{person} could I borrow your {item} for a {activity} at the {place} on {newday}?';b='I will return it afterward.'
            required=[person,activity,place,newday,item,'return']
        elif genre==3:
            prompt=f'Write a short apology to {person} for missing the {activity} at the {place} on {newday}. Offer to return their {item} soon.'
            a=f'I am sorry {person} that I missed the {activity} at the {place} on {newday}.';b=f'I can return your {item} soon.'
            required=[person,activity,place,newday,item,'sorry']
        elif genre==4:
            prompt=f'Write a warm birthday message for {person}. Mention that I look forward to our {activity} at the {place} on {newday}. Keep it short and write the message itself.'
            a=f'Happy birthday {person}!';b=f'I look forward to our {activity} at the {place} on {newday}.'
            required=[person,activity,place,newday,'birthday']
        else:
            prompt=f'Write a short lost-property notice. {person} left a {item} at the {place} during a {activity} on {newday}. Ask the finder to return it to the front desk.'
            a=f'{person} left a {item} at the {place} during a {activity} on {newday}.';b='Please return it to the front desk if you find it.'
            required=[person,activity,place,newday,item,'front desk']
        first=a+' '+b
        tasks=[('bullets','Rewrite the message as exactly two bullet points. Preserve all the details. No introduction.',f'- {a}\n- {b}'),
          ('numbered','Rewrite the message as a numbered list with exactly two items and nothing else. Preserve all the details.',f'1. {a}\n2. {b}'),
          ('paragraphs','Rewrite it as exactly two short paragraphs. Preserve all the details.',f'{a}\n\n{b}'),
          ('lowercase','Rewrite the entire message in lowercase. Keep all the details.',first.lower()),
          ('uppercase','Rewrite the entire message in uppercase. Keep all the details.',first.upper()),
          ('ending','Keep all the details. Finish with the exact words: See you there!',first+' See you there!'),
          ('sentences','Rewrite it as exactly two sentences. Keep all the details.',first),
          ('quotation','Return the message inside double quotation marks. Include no text outside the quotation marks.','"'+first+'"')]
        family,follow,answer=tasks[kind]
        checks=dict(family=family,required=required,forbidden_day=day)
        out.append(row('constraints',pair(prompt,first)+pair(follow,answer),'authored-fictional-text-v1',checks=checks,scenario_group=fingerprint(identity)))
    return out

def verify_text(text,checks):
    t=text.strip();family=checks['family'];lines=t.splitlines()
    content=all(x.lower() in t.lower() for x in checks['required']) and checks['forbidden_day'].lower() not in t.lower()
    if family=='bullets':ok=len(lines)==2 and all(re.match(r'^[-*] ',x) for x in lines)
    elif family=='numbered':ok=len(lines)==2 and lines[0].startswith('1. ') and lines[1].startswith('2. ')
    elif family=='paragraphs':ok=len(re.split(r'\n\s*\n',t))==2
    elif family=='lowercase':ok=t==t.lower() and any(x.isalpha() for x in t)
    elif family=='uppercase':ok=t==t.upper() and any(x.isalpha() for x in t)
    elif family=='ending':ok=t.endswith('See you there!')
    elif family=='sentences':ok=len(re.findall(r'[.!?](?:\s|$)',t))==2
    elif family=='quotation':ok=t.startswith('"') and t.endswith('"') and t.count('"')==2
    else:raise ValueError(family)
    return dict(format=bool(ok),content_proxy=content,joint_proxy=bool(ok and content))

def build(cfg):
    parent_contract=read_json(ROOT/'runs/sft_conversation_foundation_v1/contract.json')
    assert file_sha256(FOUNDATION)==parent_contract['protected'][str(FOUNDATION)],'Parent training data changed'
    old=read_json(FOUNDATION);ground=read_json(GROUNDED);science=read_json(SCIENCE);tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    denied,denied_grams=exclusions();held_grams=set();used=set();reject=Counter()
    counts={s:(cfg['training_counts'] if s=='train' else dict(conversation=64,constraints=32,reading=16,science=16,writing=32)) for s in ('train','dev','test')}
    manual=set(read_json(DIR/'review_rejections.json'))
    result={};rng=random.Random(cfg['seed']);sample=[]
    for split in ('test','dev','train'):
        pools={k:[] for k in cfg['training_counts']}
        # Replay comes only from the exact first 8192 rows consumed by parent 512.
        original=old['data'][split][:8192] if split=='train' else old['data'][split]
        for r in original:
            if r['source'] in ('smol-magpie-ultra-short','everyday-conversations','ultrachat') and r['assistant_turns']>=2:
                pools['conversation'].append(row('conversation',r['messages'],dict(dataset=r['repo'],file=r['file'],row_index=r['row_index'],parent_id=r['id'])))
        # New writing rows may come from the parent's unconsumed training pool.
        for r in old['data'][split]:
            if r['source'] in ('smollm-rewrite-30k','smol-summarize-20k'):
                if any(DEFER.search(m['content']) for m in r['messages'] if m['role']=='assistant'):continue
                pools['writing'].append(row('writing',r['messages'],dict(dataset=r['repo'],file=r['file'],row_index=r['row_index'],parent_id=r['id'])))
        for r in ground[split]['grounded']:
            if r.get('unknown') or not r.get('context') or r['answer'] not in r['context']:continue
            prompt='Use the passage to answer the question briefly.\n\nPassage: '+r['context']+'\n\nQuestion: '+r['question']
            pools['reading'].append(row('reading',pair(prompt,r['answer']),dict(source=r['origin'],id=r['id']),references=r['references'],article_group=r['article_group']))
        for r in science[split]['ranking']:
            if re.search(r'\b(all|none|both) (of |the |a\b|b\b)|^[A-D]$',r['answer'],re.I):continue
            if r['source'] not in ('arc_easy','arc_challenge') or r.get('source_split')!='train':continue
            if re.search(r'\b(how many|how much|percent|ratio|total|sum|difference|average|solve|number of)\b|\d',r['prompt'],re.I):continue
            if norm(r['prompt']) in denied:continue
            prompt=r['prompt']+'\n\nOptions:\n'+'\n'.join('- '+choice for choice in r['choices'])+'\n\nAnswer with the correct option text only.'
            pools['science'].append(row('science',pair(prompt,r['answer']),dict(source=r['source'],split='train',id=r['id']),references=[r['answer']]))
        pools['constraints']=constraints(split,counts[split]['constraints']*2)
        chosen={};current_grams=set()
        for source,pool in pools.items():
            rng.shuffle(pool);selected=[]
            for r in pool:
                if r['id'] in manual or r['id'] in used or not eligible(r['messages']):reject['duplicate_or_nontext']+=1;continue
                allgrams=set().union(*(grams(m['content']) for m in r['messages']))
                ug=set().union(*(grams(m['content']) for m in r['messages'] if m['role']=='user'))
                # Synthetic task wording is intentionally shared; actual scenarios are split-disjoint.
                if any(norm(m['content']) in denied for m in r['messages']) or allgrams&denied_grams:reject['public_overlap']+=1;continue
                if source!='constraints' and ug&held_grams:reject['holdout_overlap']+=1;continue
                try:encoded=turns(tok,r,cfg['context'])
                except OverflowError:reject['context']+=1;continue
                if max(e['targets'] for e in encoded)>cfg['max_assistant_tokens']:reject['long_answer']+=1;continue
                r.update(assistant_turns=len(encoded),max_context=max(len(e['x']) for e in encoded),assistant_targets=sum(e['targets'] for e in encoded))
                if source=='constraints' and not verify_text(r['messages'][-1]['content'],r['checks'])['joint_proxy']:raise ValueError('Invalid authored target')
                selected.append(r);used.add(r['id']);current_grams|=ug
                if len(selected)==counts[split][source]:break
            if len(selected)!=counts[split][source]:raise ValueError(f'Insufficient {split}/{source}: {len(selected)}/{counts[split][source]}')
            chosen[source]=selected
            print('[selected]',split,source,len(selected),flush=True)
            if split=='train':sample.extend(selected[:4])
        held_grams|=current_grams
        if split=='train':
            # Every update: 8 chats, 4 constraints, and 4 reading/science/writing.
            rest=chosen['reading']+chosen['science']+chosen['writing'];rng.shuffle(rest)
            result[split]=[r for i in range(cfg['updates']) for r in (chosen['conversation'][i*8:i*8+8]+chosen['constraints'][i*4:i*4+4]+rest[i*4:i*4+4])]
        else:
            result[split]=sum(chosen.values(),[]);rng.shuffle(result[split])
    stats={s:dict(conversations=len(rr),sources=dict(Counter(r['source'] for r in rr)),assistant_turns=sum(r['assistant_turns'] for r in rr),hash=fingerprint(rr)) for s,rr in result.items()}
    receipt=dict(config=fingerprint(cfg),stats=stats,inputs={str(p):file_sha256(p) for p in [FOUNDATION,GROUNDED,SCIENCE]+exclusion_files()},rejections=dict(reject),
      limitations='Replay and writing are filtered synthetic source data, with sample review only. Constraint diagnostics share authored templates across disjoint scenarios and measure literal content proxies. No claim of complete semantic decontamination or factual verification.')
    atomic_json(DIR/'prepared.json',dict(data=result,receipt=receipt));atomic_json(DIR/'review_samples.json',sample)
    print('[prepared]',stats,flush=True)
    return result


def batch_at(data,cfg,cursor):
    if not 0<=cursor<cfg['updates']:raise ValueError('Invalid cursor')
    batch=data['train'][cursor*16:cursor*16+16]
    if len(batch)!=16:raise ValueError('Incomplete batch')
    return batch
