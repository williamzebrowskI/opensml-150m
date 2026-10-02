"""1,024 deterministic examples: 50% factual/format practice, 50% varied responses."""
from collections import Counter
import random
from pathlib import Path
import json
from sft.transfer_control.data import records as old_records, normalized, digest
from sft.response_expansion.data import new_records as expansion_records
from .cards import GREETINGS,POLITE,FACTS,READING,CLARIFY,EVENTS,SOCIAL

STRUCTURED=('location','ownership','missing','negation','correction')
VARIED=('greeting','copy_or_reply','polite','fact_response','reading','clarification','invitation','social')
NAMES={
 'train':'Alec Blake Cora Dena Elio Fara Huw Inez Jori Kira Luan Mina Noah Orin Rosa Saul Tori Vera Wren Zuri'.split(),
 'dev':'Adela Bevan Corin Della Eamon Fia Greta Hanif Ivana Jovan Kenza Loren Merle Nima Ottilie Piers Rafi Sela Tegan Wynn'.split(),
 'test':'Aidan Briony Carys Devika Esben Farah Goran Hester Ismail Juna Kato Liv Marta Neve Oren Paloma Rina Stellan Thea Ulric'.split()}
OBJECTS={
 'train':['mug','notebook','hat','scarf','basket','letter','drum','tin','jar','brush','bag','coat','flute','key','camera','cup'],
 'dev':['apron','satchel','binoculars','ribbon','globe','candle','shovel','comb'],
 'test':['suitcase','medal','jug','whistle','telescope','sketchbook','pillow','trumpet']}
PLACES={
 'train':['in the shed','on the table','under the chair','beside the gate','in the kitchen','on the desk','in the attic','by the door'],
 'dev':['in the pantry','on the landing','beside the pond','under the stairs','on the windowsill','in the cellar','near the fence','by the fountain'],
 'test':['in the workshop','on the balcony','beside the fireplace','under the awning','in the alcove','on the dresser','near the doorway','by the stream']}
DAYS=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']


def make(family,group,prompt,answer,*,kind='sentence',content_answers=None,format_answers=None,keywords=None,echo_text=None):
    return dict(id=family+'-'+digest([group,prompt])[:16],family=family,group=group,
                prompt=prompt,answer=answer,references=[answer],origin='new',
                purpose='factual' if family in STRUCTURED else 'response',
                rubric=dict(kind=kind,content_answers=content_answers,format_answers=format_answers,
                            keywords=keywords,echo_text=echo_text))


def wording(split,index,short):
    choices={
      'train':(['Give only the answer, with no extra words.','Return just the requested name or fact.'],
               ['Use one complete sentence.','Answer in a short sentence.']),
      'dev':(['Keep the response to the answer alone.'],['Give a full-sentence answer.']),
      'test':(['Write only the requested answer.'],['Please respond with a complete sentence.'])}[split][0 if short else 1]
    return choices[index%len(choices)]


def structured(split):
    result=[];rng=random.Random(290924+list(NAMES).index(split))
    for family in STRUCTURED:
        n=(16 if family in ('location','ownership','missing') else 8) if split=='train' else 4
        for i in range(n):
            a,b=rng.sample(NAMES[split],2);obj,other=rng.sample(OBJECTS[split],2)
            loc,foil_loc=rng.sample(PLACES[split],2);day,old_day=rng.sample(DAYS,2)
            group=f'{split}:{family}:{i}'
            for variant in (0,1):
                place,wrong=(loc,foil_loc) if variant==0 else (foil_loc,loc)
                owner,other_owner=(a,b) if variant==0 else (b,a)
                if family=='location':
                    facts=[f'The {obj} is {place}.',f'The {other} is {wrong}.']
                    question=f'Where is the {obj}?';short=place;full=f'The {obj} is {place}.'
                    contents=[short,place.split('the ',1)[-1],full]
                    formats=[loc,foil_loc,loc.split('the ',1)[-1],foil_loc.split('the ',1)[-1]]
                elif family=='ownership':
                    facts=[f'{owner} owns the {obj}.',f'{other_owner} borrowed the {obj}.']
                    if split!='train' or i%2:facts=[f'{other_owner} borrowed the {obj} from its owner, {owner}.']
                    question=f'Who owns the {obj}?';short=owner;full=f'The {obj} belongs to {owner}.'
                    contents=[short,full,f'{owner} owns the {obj}.'];formats=[a,b]
                elif family=='missing':
                    facts=[f'{b} found the {obj}.',f'The {obj} belongs to {a}.' if variant==0 else 'Its owner is not identified.']
                    question=f'Who owns the {obj}? If the passage does not say, acknowledge that.'
                    short=a if variant==0 else 'Not stated'
                    full=f'The {obj} belongs to {a}.' if variant==0 else f'The passage does not say who owns the {obj}.'
                    contents=[short,full] if variant==0 else [short,full,'The owner is not stated.','The owner is not identified.','The passage does not identify the owner.']
                    formats=[a,b,'Not stated']
                elif family=='negation':
                    available,unavailable=(obj,other) if variant==0 else (other,obj)
                    facts=[f'The shop has a {available}, but it does not have a {unavailable}.']
                    if i%2:facts=[f'The shop does not have a {unavailable}; it has a {available}.']
                    question='Which of these items is available?';short=available;full=f'The {available} is available.'
                    contents=[short,full,f'The shop has a {available}.'];formats=[obj,other]
                else:
                    new,old=(day,old_day) if variant==0 else (old_day,day)
                    facts=[f'The first notice scheduled {a}\'s visit for {old}.',f'The corrected notice schedules it for {new}.']
                    question='When is the visit according to the correction?';short=new;full=f'The visit is now scheduled for {new}.'
                    contents=[short,full,f'The visit is on {new}.'];formats=[day,old_day]
                if i%2:facts=list(reversed(facts))
                passage=' '.join(facts)
                for brief in (True,False):
                    instruction=question+' '+wording(split,i,brief)
                    prompt=passage+'\n\n'+instruction if i%2==0 else instruction+'\n\n'+passage
                    row=make(family,group,prompt,short if brief else full,kind='brief' if brief else 'sentence',
                             content_answers=contents,format_answers=formats)
                    row.update(variant=variant,requested_form='brief' if brief else 'sentence')
                    result.append(row)
    return result


def lexical_keywords(answer):
    # Diagnostic only. Grounded accuracy is scored separately by exact factual
    # alternatives above; lexical overlap is not proof of semantic correctness.
    stop=set('a an the is are was were be been to of and or for from at in on it its they them their that this with so as we our you your me my i by had has have can could would will how what why which who'.split())
    return sorted(set(normalized(answer).split())-stop)


def varied(split):
    result=[]
    # Training includes observed lowercase/typo failures. They are not counted as
    # fresh successes later. New dev/test prompts are kept out of all gradients.
    for i,message in enumerate(GREETINGS[split]):
        for variant in (message,message.upper()):
            result.append(make('greeting',f'greeting:{message}',variant,'Hello! How can I help you?',
                               kind='greeting',echo_text=variant))
        for copy in (False,True):
            prompt=(f'Repeat the following greeting exactly, preserving its spelling: {message}' if copy else
                    f'Respond to this greeting with a friendly reply: {message}')
            if split=='dev':prompt=(f'Copy this greeting verbatim: {message}' if copy else f'Give a welcoming response to this greeting: {message}')
            if split=='test':prompt=(f'Return exactly this greeting text: {message}' if copy else f'Write a friendly answer to this greeting: {message}')
            result.append(make('copy_or_reply',f'copy:{message}',prompt,message if copy else 'Hi! What can I help you with?',
                               kind='copy' if copy else 'greeting',echo_text=None if copy else message))
    for i,action in enumerate(POLITE[split]):
        prompts=[f'Rewrite this command politely: {action}.',f'Make this into a friendly request: {action}.']
        if split=='dev':prompts=[f'How can I ask someone to {action} politely?',f'Use polite wording to ask: {action}.']
        if split=='test':prompts=[f'Give a courteous way to say: {action}.',f'Ask for this kindly: {action}.']
        for prompt in prompts:
            result.append(make('polite',f'polite:{action}',prompt,f'Could you please {action}?',kind='polite',keywords=lexical_keywords(action)))
    for i,(question,short,full) in enumerate(FACTS[split]):
        for brief in (True,False):
            result.append(make('fact_response',f'fact:{question}',question+' '+wording(split,i,brief),
                               short if brief else full,kind='brief' if brief else 'sentence',
                               content_answers=[short,full],format_answers=[short]))
    for i,(passage,question,answer) in enumerate(READING[split]):
        forms=[f'{passage}\n\n{question}',f'{question}\n\nPassage: {passage}',
               f'Use the information below to answer. {question}\n{passage}',f'Account: {passage}\n{question}']
        if split=='dev':forms=[f'Based on this account, {question[0].lower()+question[1:]}\n{passage}',f'{passage}\nRead carefully and answer: {question}',f'Passage: {passage}\nQuestion: {question}',f'{question}\nUse only this account: {passage}']
        if split=='test':forms=[f'From the following passage, answer the question.\n{passage}\n{question}',f'{question}\nHere are the facts: {passage}',f'Given this information: {passage}\n{question}',f'Read: {passage}\nYour task: {question}']
        for prompt in forms:result.append(make('reading',f'reading:{passage}',prompt,answer,keywords=lexical_keywords(answer)))
    for i,(situation,answer) in enumerate(CLARIFY[split]):
        asks=['Ask one question to understand my needs.','Respond with one useful clarifying question.',
              'Before suggesting an option, ask me one relevant question.','Ask something that will help me choose.']
        if split=='dev':asks=['Find out what matters to me by asking a question.','Reply with a question before you recommend anything.','Ask for one detail that would help you advise me.','Give one question to narrow my choice.']
        if split=='test':asks=['What is one useful question you would ask me first?','Use a question to learn what I need before advising me.','Please ask for a relevant detail about my preference.','Ask one helpful question rather than recommending an option now.']
        for ask in asks:result.append(make('clarification',f'clarification:{situation}',situation+' '+ask,answer,kind='question',keywords=lexical_keywords(answer)))
    for i,event in enumerate(EVENTS[split]):
        location=PLACES[split][i%len(PLACES[split])]
        # Event venues have natural complete prepositional phrases.
        venue=['at the community center','in the garden','at the local hall','in the park'][i%4]
        if split=='dev':venue=['at the arts center','by the riverside','in the school hall','at the village green'][i%4]
        if split=='test':venue=['at the recreation center','in the orchard','at the scout hall','in the courtyard'][i%4]
        day=DAYS[i%7];details=f'{event} {venue} on {day}'
        forms=[f'Write a brief invitation to {details}.',f'Invite a friend to {details} in one sentence.',
               f'Create a friendly invitation for {details}.',f'Please ask someone to join us for {details}.']
        if split=='dev':forms=[f'Turn this into an invitation: {details}.',f'Use a friendly sentence to invite someone: {details}.',f'Give a short welcoming invite to {details}.',f'Write one sentence inviting a neighbor to {details}.']
        if split=='test':forms=[f'Write the invitation itself for {details}.',f'What could I say to invite someone to {details}?',f'Compose a short invite for {details}.',f'Address a friendly invitation to the reader for {details}.']
        answer=f'Join us for {details}!'
        for prompt in forms:result.append(make('invitation',f'invitation:{event}',prompt,answer,kind='invitation',keywords=lexical_keywords(details)))
    for i,(task,answer) in enumerate(SOCIAL[split]):
        forms=[task+' Use one friendly sentence.',task+' Keep it short and warm.',task+' Write the message itself.',task+' Use a brief, kind message.']
        if split=='dev':forms=[task+' Give the actual message in a short sentence.',task+' Write a warm reply.',task+' Keep your message concise.',task+' A friendly line is enough.']
        if split=='test':forms=[task+' Please provide the message, not advice.',task+' Respond with a short message I could send.',task+' Write a kind line for me to use.',task+' Make it brief and welcoming.']
        for prompt in forms:result.append(make('social',f'social:{task}',prompt,answer,kind='social',keywords=lexical_keywords(answer)))
    return result


def new_records(split):
    if split not in NAMES:raise ValueError(split)
    return structured(split)+varied(split)


def training_records():
    counts=Counter();replay=[]
    for r in old_records('train'):
        if counts[r['family']]<32:
            replay.append(dict(r,id='rehearsal-'+r['id'],origin='rehearsal',purpose='factual'))
            counts[r['family']]+=1
    factual=structured('train')+replay;response=varied('train')
    rng=random.Random(290924512);rng.shuffle(factual);rng.shuffle(response)
    result=[]
    for i in range(0,512,4):
        batch=factual[i:i+4]+response[i:i+4];rng.shuffle(batch);result.extend(batch)
    return result


def audit():
    rows=training_records();splits={s:new_records(s) for s in NAMES}
    if len(rows)!=1024 or Counter(r['purpose'] for r in rows)!=dict(factual=512,response=512):
        raise ValueError('Expected exact 50/50 mixture')
    for i in range(0,1024,8):
        if Counter(r['purpose'] for r in rows[i:i+8])!=dict(factual=4,response=4):raise ValueError('Unbalanced update')
    if len({r['id'] for r in rows})!=1024:raise ValueError('Duplicate training IDs')
    groups=set();prompts=set()
    for split,rr in splits.items():
        if len({r['prompt'] for r in rr})!=len(rr):raise ValueError(f'Duplicate prompts: {split}')
        gg={r['group'] for r in rr};pp={normalized(r['prompt']) for r in rr}
        if groups&gg or prompts&pp:raise ValueError('Train/dev/test leakage')
        groups|=gg;prompts|=pp
    old_eval=[r for s in ('dev','test') for r in old_records(s)]+old_records('dev','familiar')
    old_eval += [r for s in ('dev','test') for r in expansion_records(s)]
    path=Path(__file__).resolve().parents[2]/'diagnostics/playground_sft_comparison_20260924/protocol.json'
    if path.exists():old_eval+=json.loads(path.read_text())['items']
    if {normalized(r['prompt']) for r in rows}&{normalized(r['prompt']) for r in old_eval}:
        raise ValueError('A previous evaluation prompt entered training')
    previous_train={r['id']:r for r in old_records('train')}
    for r in rows:
        if r['origin']=='rehearsal':
            old=previous_train[r['id'].removeprefix('rehearsal-')]
            assert (r['prompt'],r['answer'])==(old['prompt'],old['answer'])
    return dict(training_examples=len(rows),rehearsal_examples=256,new_factual_examples=256,new_response_examples=512,
                per_update=dict(factual=4,response=4),train_hash=digest(rows),
                splits={s:dict(examples=len(rr),groups=len({r['group'] for r in rr}),hash=digest(rr)) for s,rr in splits.items()},
                new_training_counts=dict(Counter(r['family'] for r in splits['train'])),
                provenance='Original local examples plus prior training-only rehearsal; no external teacher or downloads.',
                limitations='Controlled tasks and repeated phrasings; observed hello/hii failures are training examples, not fresh evaluation.')
