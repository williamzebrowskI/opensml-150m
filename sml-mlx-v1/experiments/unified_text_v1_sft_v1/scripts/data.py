"""Frozen native-format unified mixture; public TRAIN pools plus fictional follow-ups."""
from collections import Counter
from pathlib import Path
import random
import re
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json,read_json,file_sha256,fingerprint
from sml_v1.tokenization import Tokenizer
from sft.conversation_foundation_original.data import exclusions,exclusion_files,norm,grams,group_id,turns,visible_prefix
from sft.text_followup_512_v1.data import eligible,pair,row,DEFER,verify_text,constraints
EXP=Path(__file__).resolve().parents[1]
FOUNDATION=ROOT/'sft/conversation_foundation_original/prepared.json'
FOLLOWUP=ROOT/'sft/text_followup_512_v1/prepared.json'
GROUNDED=ROOT/'sft/grounded_rank_384/prepared.json'
SCIENCE=ROOT/'sft/preference_long_640/prepared.json'
INPUTS=[FOUNDATION,FOLLOWUP,GROUNDED,SCIENCE]


def recovery(split,count):
    # Simple given-fact tasks: no external knowledge, no deliberately bad assistant targets.
    people={'train':['Ada','Ben','Cara','Dion','Eli','Fern','Gus','Hana'],
            'dev':['Inez','Jules','Kira','Lars'], 'test':['Mira','Nico','Orla','Pia']}[split]
    places={'train':['atrium','orchard','boathouse','bookshop','meadow','clubhouse','cabin','market'],
            'dev':['conservatory','veranda','annex','foyer'], 'test':['observatory','kiosk','gazebo','pier']}[split]
    rng=random.Random(930601+['train','dev','test'].index(split));out=[];seen=set()
    while len(out)<count:
        person=rng.choice(people);a,b=rng.sample(places,2);day,newday=rng.sample(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],2)
        item=rng.choice(['sketchbook','scarf','lantern','notebook','jacket','basket']);kind=len(out)%8
        drink,other=rng.sample(['tea','coffee','lemonade','water','cocoa','juice'],2)
        identity=(person,a,b,day,newday,item,kind,drink,other)
        if identity in seen:continue
        seen.add(identity)
        family=['history','correction','preference','unknown','grounded','completion','shorten','topic_switch'][kind]
        if kind==0:
            p=f'{person} left a {item} at the {a} on {day}. Remember these details.';first=f'{person} left a {item} at the {a} on {day}.'
            q='Where was the item left? Give only the place.';answer=a;required=[a];forbidden=[]
        elif kind==1:
            p=f'Write a short invitation to {person} for a reading group at the {a} on {day}.';first=f'{person}, please join our reading group at the {a} on {day}.'
            q=f'Correction: it is at the {b} on {newday}. Rewrite the invitation using the corrected details.';answer=f'{person}, please join our reading group at the {b} on {newday}.';required=[person,b,newday];forbidden=[a,day]
        elif kind==2:
            p=f'At the {a} on {day}, {person} prefers {drink} to {other}. Write a sentence about this preference.';first=f'{person} prefers {drink} to {other} at the {a} on {day}.'
            q=f'What drink should we offer {person}? Answer with the drink only.';answer=drink;required=[drink];forbidden=[other]
        elif kind==3:
            p=f'The note says {person} left a {item} at the {a}. Summarize it briefly.';first=f'{person} left a {item} at the {a}.'
            q='What time was the item left? Use only the note and do not guess.';answer='The note does not say what time the item was left.';required=['does not say'];forbidden=[]
        elif kind==4:
            p=f'The {a} closed because a pipe burst. {person} moved the reading group to the {b}. Summarize these facts.';first=f'A burst pipe closed the {a}, so {person} moved the reading group to the {b}.'
            q=f'Why did the {a} close? Give the reason in one short sentence.';answer='It closed because a pipe burst.';required=['pipe','burst'];forbidden=[]
        elif kind==5:
            p=f'I need to ask {person} to bring a {item} to the {a} on {day}. What should the message include?';first=f'Include {person}, the {item}, the {a}, and {day}.'
            q='Now write the actual message. No introduction.';answer=f'{person}, please bring a {item} to the {a} on {day}.';required=[person,item,a,day];forbidden=[]
        elif kind==6:
            p=f'Write a short notice saying {person} left a {item} at the {a} on {day}, and ask the finder to return it to the front desk.';first=f'{person} left a {item} at the {a} on {day}. Please return it to the front desk if you find it.'
            q='Shorten the notice to one sentence while keeping the person, item, place, and day.';answer=f'{person} left a {item} at the {a} on {day}.';required=[person,item,a,day];forbidden=[]
        else:
            p=f'Write a brief reminder for {person} to bring a {item} to the {a} on {day}.';first=f'{person}, remember to bring a {item} to the {a} on {day}.'
            q='New topic: write a warm birthday greeting for my friend. Do not mention the previous reminder.';answer='Happy birthday! I hope you have a wonderful day.';required=['birthday'];forbidden=[person,item,a,day]
        out.append(row('recovery',pair(p,first)+pair(q,answer),'authored-given-facts-v1',
                       scenario_group=fingerprint((split,identity)),checks=dict(family=family,required=required,forbidden=forbidden,exact=answer if kind in (0,2) else None)))
    return out


def recovery_score(text,checks):
    t=text.strip();low=t.lower()
    content=all(x.lower() in low for x in checks['required']) and not any(x.lower() in low for x in checks['forbidden'])
    style=bool(t) and not re.search(r'^\s*(?:[-*] |\d+[.)] )',t,re.M)
    if checks.get('exact'):style=style and low==checks['exact'].lower()
    if checks['family'] in ('shorten','grounded'):style=style and len(re.findall(r'[.!?](?:\s|$)',t))==1 and len(t.split())<=25
    return dict(content_proxy=content,style=bool(style),joint_proxy=bool(content and style))


def build(cfg):
    foundation=read_json(FOUNDATION)['data'];follow=read_json(FOLLOWUP)['data'];ground=read_json(GROUNDED);science=read_json(SCIENCE)
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');rng=random.Random(cfg['seed'])
    denied,denied_grams=exclusions();held_groups=set();held_grams=set();held_articles=set();held_science=set()
    # Preserve all legacy held-out public conversations, not just the subset evaluated.
    for split in ('dev','test'):
        for r in foundation[split]+follow[split]:
            if r['source']=='constraints':continue
            held_groups.add(group_id(r['messages']))
            for m in r['messages']:
                if m['role']=='user':held_grams|=grams(m['content'])
        held_articles|={r['article_group'] for r in ground[split]['grounded'] if r.get('article_group')}
        held_science|={r['id'] for r in science[split]['ranking']}
    result={};used=set();reject=Counter()
    for split in ('test','dev','train'):
        counts=cfg['training_counts'] if split=='train' else dict(conversation=32,format_followup=32,recovery=32,reading=16,science=32,writing=32)
        pools={k:[] for k in counts}
        for r in foundation[split]:
            if r['source'] in ('smol-magpie-ultra-short','everyday-conversations','ultrachat'):
                pools['conversation'].append(row('conversation',r['messages'],dict(dataset=r['repo'],file=r['file'],row_index=r['row_index'],source=r['source'],parent_id=r['id'])))
            elif r['source'] in ('smollm-rewrite-30k','smol-summarize-20k') and not any(DEFER.search(m['content']) for m in r['messages'] if m['role']=='assistant'):
                pools['writing'].append(row('writing',r['messages'],dict(dataset=r['repo'],file=r['file'],row_index=r['row_index'],source=r['source'],parent_id=r['id'])))
        pools['format_followup']=[dict(r,source='format_followup') for r in constraints(split,counts['format_followup']*2)]
        pools['recovery']=recovery(split,counts['recovery']*2)
        for r in ground[split]['grounded']:
            if r.get('unknown') or not r.get('context') or r['answer'] not in r['context']:continue
            prompt='Use the passage to answer the question briefly.\n\nPassage: '+r['context']+'\n\nQuestion: '+r['question']
            pools['reading'].append(row('reading',pair(prompt,r['answer']),dict(source=r['origin'],id=r['id']),references=r['references'],article_group=r['article_group']))
        for r in science[split]['ranking']:
            if r['source'] not in ('arc_easy','arc_challenge') or r.get('source_split')!='train':continue
            if re.search(r'\b(all|none|both) (of |the |a\b|b\b)|^[A-D]$',r['answer'],re.I):continue
            if re.search(r'\b(how many|how much|percent|ratio|total|sum|difference|average|solve|number of)\b|\d',r['prompt'],re.I):continue
            if r['answer']!=r['choices'][r['gold']]:raise ValueError('Science label mismatch')
            prompt=r['prompt']+'\n\nOptions:\n'+'\n'.join('- '+c for c in r['choices'])+'\n\nAnswer with the correct option text only.'
            pools['science'].append(row('science',pair(prompt,r['answer']),dict(source=r['source'],split='train',id=r['id']),references=[r['answer']]))
        selected={}
        for category,pool in pools.items():
            rng.shuffle(pool);chosen=[];groups=set()
            # Broad conversation source coverage; avoid exhausting one short-chat source.
            quotas=dict(zip(('smol-magpie-ultra-short','ultrachat','everyday-conversations'),(4160,3136,384))) if split=='train' and category=='conversation' else None
            if category in ('format_followup','recovery'):
                families=sorted({r['checks']['family'] for r in pool})
                assert len(families)==8 and counts[category]%8==0
                quotas={f:counts[category]//8 for f in families}
            source_counts=Counter()
            for r in pool:
                if not eligible(r['messages']):reject['nontext_or_quality']+=1;continue
                synthetic=category in ('format_followup','recovery');group=r.get('scenario_group') if synthetic else group_id(r['messages'])
                if r['id'] in used or group in groups:reject['duplicate']+=1;continue
                ug=set().union(*(grams(m['content']) for m in r['messages'] if m['role']=='user'))
                ag=set().union(*(grams(m['content']) for m in r['messages']))
                if any(norm(m['content']) in denied for m in r['messages']) or ag&denied_grams:reject['public_benchmark_overlap']+=1;continue
                if split=='train' and not synthetic and (group in held_groups or ug&held_grams or r.get('article_group') in held_articles or (category=='science' and r['origin']['id'] in held_science)):
                    reject['legacy_holdout_overlap']+=1;continue
                key=r['checks']['family'] if synthetic else r['origin'].get('source')
                if quotas and source_counts[key]>=quotas[key]:continue
                try:encoded=turns(tok,r,cfg['context'])
                except OverflowError:reject['context_overflow']+=1;continue
                if max(e['targets'] for e in encoded)>cfg['max_assistant_tokens']:reject['long_answer']+=1;continue
                if category=='format_followup' and not verify_text(r['messages'][-1]['content'],r['checks'])['joint_proxy']:raise ValueError('Invalid format target')
                if category=='recovery' and not recovery_score(r['messages'][-1]['content'],r['checks'])['joint_proxy']:raise ValueError('Invalid recovery target')
                r.update(assistant_turns=len(encoded),assistant_targets=sum(e['targets'] for e in encoded),max_context=max(len(e['x']) for e in encoded))
                chosen.append(r);used.add(r['id']);groups.add(group)
                if quotas:source_counts[key]+=1
                if len(chosen)==counts[category]:break
            if len(chosen)!=counts[category]:raise ValueError(f'Insufficient {split}/{category}: {len(chosen)}/{counts[category]}, sources={dict(source_counts)}, rejects={dict(reject)}')
            selected[category]=chosen
            print('[selected]',split,category,len(chosen),flush=True)
        if split=='train':
            # Interleave supplemental pools throughout rather than finishing one category first.
            rest=[]
            for c in ('reading','science','writing'):
                n=len(selected[c]);rest.extend(((i+.5)/n,c,r) for i,r in enumerate(selected[c]))
            rest=[r for _,c,r in sorted(rest,key=lambda x:(x[0],x[1]))]
            result[split]=[]
            for i in range(cfg['updates']):
                batch=selected['conversation'][i*10:i*10+10]+selected['format_followup'][i:i+1]+selected['recovery'][i:i+1]+rest[i*4:i*4+4]
                if len(batch)!=16:raise ValueError('Incomplete unified batch')
                rng.shuffle(batch);result[split]+=batch
        else:result[split]=sum(selected.values(),[])
    stats={s:dict(conversations=len(rr),sources=dict(Counter(r['source'] for r in rr)),assistant_turns=sum(r['assistant_turns'] for r in rr),hash=fingerprint(rr)) for s,rr in result.items()}
    old512={r['id'] for r in foundation['train'][:8192]};old768={r['id'] for r in follow['train']}
    previously_used=sum(r['id'] in old512|old768 for r in result['train'])
    receipt=dict(config=fingerprint(cfg),stats=stats,inputs={str(p):file_sha256(p) for p in INPUTS+exclusion_files()},rejections=dict(reject),previously_used_in_512_or_768=previously_used,
                 source_receipts={str(p):read_json(p).get('receipt',{}) for p in (FOUNDATION,FOLLOWUP)},
                 limitations='Public source answers are filtered, not exhaustively fact checked. Authored given-fact and format checks use literal content proxies. Shared task templates are allowed across split-disjoint scenarios. No guarantee of semantic decontamination or benchmark gains.')
    atomic_json(EXP/'data/prepared.json',dict(data=result,receipt=receipt));return result


def batch_at(data,cfg,cursor):
    if not 0<=cursor<cfg['updates']:raise ValueError('Invalid cursor')
    batch=data['train'][cursor*16:cursor*16+16]
    if len(batch)!=16:raise ValueError('Incomplete batch')
    return batch
