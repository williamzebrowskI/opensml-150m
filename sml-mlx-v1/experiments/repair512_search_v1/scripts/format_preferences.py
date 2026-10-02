"""TRAIN-only formatting counterexamples with preserved answer words.

The checks prove a named formatting violation, not factual correctness.
Count constraints that require deleting content are deliberately omitted.
"""
import random
import re
from collections import Counter
from balanced_instructions import rules

def corrupt(text,check):
    family=check['family']
    if family=='lowercase':return text.upper()
    if family=='uppercase':return text.lower()
    if family=='title':return text.replace('<<','').replace('>>','')
    if family=='quotation':return text.strip()[1:-1]
    if family=='ending':return ' '.join(reversed(text.splitlines())) if '\n' in text.strip() else text+'\n.'
    if family=='bullets':return re.sub(r'(?m)^\s*([*-])\s+', '', text)
    if family=='paragraphs':return re.sub(r'\s*\*\*\*\s*|\n\s*\n',' ',text)
    if family=='no_comma':return text.replace(' ', ', ',1)
    return text

def build(train,updates,seed,tok,context):
    from sft.dpo_chat_768_v1.data import encode_reply
    rng=random.Random(seed+71);buckets={};controls=[];seen=set()
    for row in train:
        if row['group'] in seen:continue
        seen.add(row['group']);chosen=row['messages'][-1]['content'];messages=row['messages'][:-1]
        if row['source']=='instructions':
            assert all(rules.verify(chosen,row['checks']))
            for check in row['checks']:
                rejected=corrupt(chosen,check)
                if rejected==chosen or all(rules.verify(rejected,row['checks'])):continue
                # Every word is retained, with only case and punctuation altered.
                words=lambda t:re.findall(r'\w+',t.lower())
                if Counter(words(chosen))!=Counter(words(rejected)):continue
                pair=dict(group=row['group'],family=check['family'],messages=messages,chosen=chosen,rejected=rejected)
                try:pair['encoded']=[encode_reply(tok,messages,t,context) for t in (chosen,rejected)]
                except (ValueError,OverflowError):continue
                buckets.setdefault(check['family'],[]).append(pair)
        elif row['source'] in ('chat','human'):
            prompt=' '.join(m['content'] for m in messages if m['role']=='user')
            if re.search(r'\b(?:uppercase|capital|lowercase|bullet|list|format|title|quote)\b',prompt,re.I):continue
            if len(tok.encode(chosen))>192 or chosen==chosen.upper() or not re.search(r'[a-z]',chosen):continue
            pair=dict(group=row['group'],family='plain_control',messages=messages,chosen=chosen,rejected=chosen.upper())
            try:pair['encoded']=[encode_reply(tok,messages,t,context) for t in (chosen,chosen.upper())]
            except (ValueError,OverflowError):continue
            controls.append(pair)
    families=sorted(buckets);assert families and controls
    for f in families:rng.shuffle(buckets[f])
    rng.shuffle(controls);offset=Counter();schedule=[]
    for u in range(updates):
        f=families[u%len(families)];pool=buckets[f];p=pool[offset[f]%len(pool)];offset[f]+=1
        schedule.append([p,controls[u%len(controls)]])
    return dict(updates=schedule,family_pool={f:len(buckets[f]) for f in families},plain_control_pool=len(controls),
                train_only=True,answer_words_preserved=True,checks='Chosen passes every named constraint; rejected violates at least one.',
                objective='Reference-relative character-normalized pairwise logistic; beta=5; not standard sequence-sum DPO.')
