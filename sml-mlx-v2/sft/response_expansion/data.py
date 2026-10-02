"""Frozen original examples: 768 new responses plus 256 training-only rehearsals."""
from collections import Counter
import random
from sft.transfer_control.data import records as old_records, normalized, digest, OPEN as OLD_OPEN
from .cards import EXPLANATIONS, CAUSES, POLITE, CLARIFY, GREETINGS

COUNTS = dict(location=64, owner=64, correction=64, missing=64, polite=64, edit=64,
              explanation=64, grounded_explanation=64, summary=64, description=64,
              invitation=64, greeting=32, clarification=32)
NAMES = {
 'train': 'Aaron Bella Calvin Delia Edgar Flora Gavin Hazel Isaac Josie Kelvin Lila Marcel Nella Oscar Pia'.split(),
 'dev': 'Amira Bowen Celia Dorian Ewan Freya Galen Hilda Idris Jess Kian Lucia Maeve Nolan Olive Perry'.split(),
 'test': 'Alina Basil Casper Daphne Emmett Frida Griffin Heidi Ilona Jasper Kit Linus Mabel Nestor Odette Porter'.split()}
OBJECTS = ['lantern','scarf','parcel','umbrella','bottle','rucksack','pencil','towel','tray','vase','purse','suitcase']
PLACES = ['on the bench','under the awning','beside the pond','in the cupboard','near the fountain',
          'on the shelf','inside the tent','by the window','behind the counter','under the desk']
DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
VERBS = ['keeps the garden tidy','enjoys listening to music','visits the market','likes to draw flowers',
         'walks beside the canal','collects smooth stones','feeds the chickens','reads by the window']


def pack(family, group, prompt, answer, required=(), forbidden=(), exact_only=False,
         question=False, min_words=4, references=None):
    return dict(id=f'{family}-{digest([group,prompt])[:16]}', family=family, group=group,
                prompt=prompt, answer=answer, references=references or [answer],
                rubric=dict(required=list(required),forbidden=list(forbidden),exact_only=exact_only,
                            question=question,min_words=min_words,max_words=45))


def style(split,i):return i%4 if split=='train' else 4 if split=='dev' else 5


def card_rows(family,split):
    # Entire source topics/intents stay in one split, including their paraphrases.
    repeats=4 if split=='train' else 2 if split=='dev' else 3
    rows=[]
    if family=='greeting':
        for i,prompt in enumerate(GREETINGS[split]):
            answer=['Hello! How can I help you?','Hi! What would you like to discuss?',
                    'Hello! What can I help you with?'][i%3]
            rows.append(pack(family,f'greeting:{normalized(prompt)}',prompt,answer,
                             [['hello','hi','hey','good morning','good afternoon','good evening'],
                              ['help','discuss','assist']],min_words=4))
        return rows
    cards={'explanation':EXPLANATIONS,'grounded_explanation':CAUSES,
           'polite':POLITE,'clarification':CLARIFY}[family][split]
    for j,card in enumerate(cards):
        for k in range(repeats):
            # Fresh dev/test wording varies as well as topic; no test responses
            # are consulted to choose training templates or checkpoint ranking.
            if family=='explanation':
                question,answer,required=card
                endings={
                    'train':[' Give a short explanation.',' Answer in one clear sentence.',
                             ' Please explain briefly.',' Use a complete sentence.'],
                    'dev':[' Explain the reason in a brief sentence.',' Give a concise explanation, not just a keyword.'],
                    'test':[' Tell me the reason in plain language.',' A short explanatory sentence is enough.',
                            ' Respond with a brief, informative explanation.']}[split]
                rows.append(pack(family,f'explanation:{question}',question+endings[k],answer,required,min_words=6))
            elif family=='grounded_explanation':
                event,cause=card;source=f'{event} because {cause}.'
                questions={
                    'train':[f'Why did this happen? {source}',f'{source}\nExplain the reason briefly.',
                             f'Give a complete answer about the cause: {source}',f'{source}\nWhat caused this?'],
                    'dev':[f'Based on this note, explain the cause in a sentence: {source}',f'{source}\nWhat was the reason?'],
                    'test':[f'{source}\nBriefly explain why this occurred.',f'Use the information to give the reason: {source}',
                            f'{source}\nWrite a short answer explaining the cause.']}[split]
                answer=f'{event} because {cause}.'
                rows.append(pack(family,f'cause:{source}',questions[k],answer,
                                 [[cause],['because','since','due to','as a result']],min_words=6))
            elif family=='polite':
                action=card;sentence=action[0].upper()+action[1:]+'.'
                instructions={
                    'train':['Make this request polite','Rewrite this as a polite request',
                             'Ask someone to do this politely','Turn this command into a courteous request'],
                    'dev':['Soften this command without changing its meaning','Write a courteous version of this instruction'],
                    'test':['How could I ask this more politely','Give a respectful wording for this request',
                            'Keep the requested action and make the tone polite']}[split]
                answer=[f'Could you please {action}?',f'Please {action}.',f'Would you please {action}?',f'Can you please {action}?'][k%4]
                rows.append(pack(family,f'polite:{action}',instructions[k]+': '+sentence,answer,
                                 [[action],['please','could you','would you','can you']],min_words=4))
            else:
                situation,answer,required=card
                asks={
                    'train':['Ask one useful question before making a suggestion.','Ask a short question to clarify my needs.',
                             'Respond with one relevant question.','What would you ask to understand what I need?'],
                    'dev':['Please ask one question that would help you advise me.','Find out what matters to me with a question.'],
                    'test':['Reply with a question to help narrow the choice.','Ask something useful before recommending an option.',
                            'Give one clarifying question rather than an immediate recommendation.']}[split]
                rows.append(pack(family,f'clarify:{situation}',situation+' '+asks[k],answer,required,question=True))
    return rows


def structured_rows(family,split):
    n=COUNTS[family] if split=='train' else 8 if split=='dev' else 12
    rng=random.Random(240924200+list(COUNTS).index(family)*100+['train','dev','test'].index(split))
    rows=[];prompts=set()
    while len(rows)<n:
        i=len(rows);s=style(split,i);a,b,c=rng.sample(NAMES[split],3)
        obj,other=rng.sample(OBJECTS,2);where,elsewhere=rng.sample(PLACES,2);day,old_day=rng.sample(DAYS,2)
        choose_other=bool((i//2)%2);target=other if choose_other else obj
        person=b if choose_other else a;place=elsewhere if choose_other else where
        foil_person=a if choose_other else b;foil_place=where if choose_other else elsewhere
        def pair(first,second):return f'{second} {first}' if i%2 else f'{first} {second}'
        exact_only=False;question=False;min_words=4;references=None
        if family=='location':
            source=pair(f'The {obj} is {where}.',f'The {other} is {elsewhere}.')
            instructions=[f'Where can I find the {target}? Answer in a complete sentence.',f'Tell me where the {target} is in a sentence.',
                          f'Give a short sentence locating the {target}.',f'Describe the location of the {target} briefly.',
                          f'Use a whole sentence to say where the {target} can be found.',f'Where should I look for the {target}? Give a brief sentence.']
            answer=f'The {target} is {place}.';required=[[target],[place]];forbidden=[foil_place]
        elif family=='owner':
            source=pair(f'{a} owns the {obj}.',f'The {other} belongs to {b}.')
            instructions=[f'Who owns the {target}? Answer in a complete sentence.',f'Say who the {target} belongs to in a sentence.',
                          f'Write a short answer identifying the owner of the {target}.',f'Tell me whose {target} this is, using a sentence.',
                          f'Identify who owns the {target} with a full-sentence answer.',f'Explain in one sentence who the {target} belongs to.']
            answer=f'The {target} belongs to {person}.'
            required=[[f'{target} belongs to {person}',f'{person} owns the {target}',f'{target} is owned by {person}']];forbidden=[foil_person]
        elif family=='correction':
            source=pair(f'The earlier note sets {a}\'s appointment for {old_day}.',f'The updated note sets {a}\'s appointment for {day}.')
            instructions=['Give the current appointment day in a complete sentence.','When is the appointment now? Use a sentence.',
                          'State the updated appointment day in one sentence.','Write a brief answer giving the revised day.',
                          'Following the update, when is the appointment? Use a full sentence.',
                          'Tell me the present schedule in a short sentence, taking account of the update.']
            answer=f'The appointment is now on {day}.';required=[['appointment'],[day]];forbidden=[old_day]
        elif family=='missing':
            source=(pair(f'{a} owns the {obj}.',f'{b} carried the {obj} to the market.') if i%2==0 else
                    f'{b} carried the {obj} to the market. The text does not identify its owner.')
            instructions=[f'Who owns the {obj}? Use a sentence and say if the owner is not identified.',
                          f'Give a full-sentence answer about who owns the {obj}, without guessing.',
                          f'Can you identify the owner of the {obj}? Answer in a sentence using only the text.',
                          f'Explain who owns the {obj}, or say that the text does not tell us.',
                          f'Using only this account, say in a sentence who owns the {obj}. Acknowledge missing information.',
                          f'Write a sentence giving the owner of the {obj} if known from the passage; otherwise say the owner is not stated.']
            if i%2==0:
                answer=f'The {obj} belongs to {a}.';required=[[f'{obj} belongs to {a}',f'{a} owns the {obj}']];forbidden=[b]
            else:
                answer=f'The text does not identify who owns the {obj}.'
                required=[[obj],['does not identify','not stated','not mentioned','does not say','not identified']];forbidden=[a,b]
        elif family=='edit':
            sentence=f'{a} {rng.choice(VERBS)}.';source=sentence
            instructions=[f'Use {b} as the name in place of {a}; leave the other words as they are.',
                          f'Change the subject name from {a} to {b}, keeping the sentence otherwise identical.',
                          f'Update the sentence so it names {b} rather than {a}. Make no other changes.',
                          f'Swap the name {a} for {b} and reproduce the rest.',
                          f'Return the sentence with {b} replacing {a}, without any other edits.',
                          f'The sentence should refer to {b}, not {a}. Return that edited sentence only.']
            answer=sentence.replace(a,b,1);required=[];forbidden=[];exact_only=True
        elif family=='summary':
            source=pair(f'{a} moved the {obj}. It was {where} and is now {elsewhere}.',f'{b} stayed behind.')
            instructions=[f'Summarize the movement of the {obj} in one short sentence.',f'Write one sentence saying who moved the {obj} and where it ended up.',
                          f'Briefly report the transfer of the {obj}, leaving out unrelated details.',f'Give a short summary of what {a} did with the {obj}.',
                          f'Condense the account into a sentence about the {obj}\'s move and its destination.',
                          f'Provide a concise sentence capturing who moved the {obj} and its final location.']
            destination=elsewhere
            answer=f'{a} moved the {obj}; it is now {destination}.'
            required=[[f'{a} moved the {obj}',f'{obj} was moved by {a}'],[destination]];forbidden=[b,where]
        elif family=='description':
            venue=f'{a}\'s Corner';kind=rng.choice(['cafe','bakery','kiosk','stall'])
            location=rng.choice(['near the lake','beside the market','in the village','by the station'])
            food,wrong=rng.sample(['bread','tea','soup','cakes','sandwiches','fruit'],2)
            source=f'Facts: name: {venue}; type: {kind}; location: {location}; serves: {food}.'
            instructions=['Write a short description using every fact and nothing else.','Turn these facts into one complete sentence.',
                          'Describe this place briefly using only the supplied information.','Combine the given facts into a concise description.',
                          'Produce a factual sentence that includes the name, type, location and food.',
                          'Give a brief prose description, preserving all the listed facts.']
            answer=f'{venue} is a {kind} {location} that serves {food}.'
            required=[[venue],[kind],[location],[food]];forbidden=[wrong,'not','does not']
        elif family=='invitation':
            events={'train':['a picnic','a reading group','a neighborhood walk','an art afternoon','a garden visit','a music evening','a craft session','a board game gathering'],
                    'dev':['a beach cleanup','a poetry reading','a birdwatching walk','a book exchange'],
                    'test':['a pottery demonstration','a storytelling evening','a photography walk','a flower show']}[split]
            event=events[i%len(events)];location=rng.choice(['in the park','at the community hall','in the library','at the village green'])
            source=f'Event: {event}. Place: {location}. Day: {day}.'
            instructions=['Write a short, friendly invitation using these details.','Invite someone to this event in a brief sentence.',
                          'Make a friendly invitation that includes the event, place and day.','Write a welcoming invitation from the details below.',
                          'Turn this event notice into a short invitation addressed to the reader.',
                          'Ask the reader to join this event; include its location and day.']
            answer=f'Join us for {event} {location} on {day}!'
            required=[['join','invited','come'],[event.split(' ',1)[1]],[location],[day]];forbidden=[old_day]
        else:raise ValueError(family)
        instruction=instructions[s]
        prompt=f'{instruction}\n\n{source}' if i%2 else f'{source}\n\n{instruction}'
        if prompt in prompts:continue
        prompts.add(prompt)
        # Cross-split lexical/topic grouping is additionally audited below.
        group=digest([family,source])
        rows.append(pack(family,group,prompt,answer,required,forbidden,exact_only,question,min_words,references))
    return rows


def new_records(split):
    if split not in NAMES:raise ValueError(split)
    rows=[]
    for family in COUNTS:
        if family in ('explanation','grounded_explanation','polite','clarification','greeting'):
            rows.extend(card_rows(family,split))
        else:rows.extend(structured_rows(family,split))
    return rows


def training_records():
    counts=Counter();rehearsal=[]
    for row in old_records('train'):
        if counts[row['family']]<32:
            counts[row['family']]+=1
            rehearsal.append(dict(row,origin='rehearsal',id='rehearsal-'+row['id']))
    rows=[dict(r,origin='new') for r in new_records('train')]+rehearsal
    random.Random(240924384).shuffle(rows)
    return rows


def audit():
    splits={s:new_records(s) for s in NAMES};train=training_records()
    assert Counter(r['family'] for r in splits['train'])==COUNTS
    assert len(train)==1024 and sum(r['origin']=='rehearsal' for r in train)==256
    groups=set();prompts=set()
    for split,rows in splits.items():
        local_groups={r['group'] for r in rows};local_prompts={normalized(r['prompt']) for r in rows}
        if len(local_prompts)!=len(rows):raise ValueError(f'Duplicate prompts in {split}')
        if groups&local_groups or prompts&local_prompts:raise ValueError('Training/evaluation leakage')
        groups|=local_groups;prompts|=local_prompts
    known={normalized(r['prompt']) for s in ('dev','test') for r in old_records(s)}
    known|={normalized(r['prompt']) for r in old_records('dev','familiar')}|{normalized(p) for p in OLD_OPEN}
    if known & {normalized(r['prompt']) for r in train}:raise ValueError('Old evaluation copied into training')
    old_training={r['id']:r for r in old_records('train')}
    for r in train:
        if r['origin']=='rehearsal':
            old=old_training[r['id'].removeprefix('rehearsal-')]
            assert (r['prompt'],r['answer'])==(old['prompt'],old['answer'])
    return dict(train_examples=len(train),new_examples=768,rehearsal_examples=256,
                new_counts=COUNTS,development_examples=len(splits['dev']),test_examples=len(splits['test']),
                training_hash=digest(train),split_hashes={s:digest(r) for s,r in splits.items()},
                provenance='Original authored examples and controlled variations, no external teacher or dataset.',
                limitations='Related task templates; automated rubric matches are not comprehensive semantic judgments.')
