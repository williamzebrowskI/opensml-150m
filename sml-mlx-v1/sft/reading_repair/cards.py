"""Original short reading scenarios, with explicit role and question contrasts.

Generated examples are synthetic. They do not replace the separate human-written
source or independently worded development challenge. No teacher is used.
"""
import random
from sft.transfer_control.data import digest
FAMILIES=('contents','ownership','availability','updates','reference','missing','false_premise')
NAMES={'train':'Amira Boris Clara Diego Esther Finn Grace Hugo Iris Jonas Keira Leo Mabel Niko Olive Pavel Quinn Rhea Silas Uma Vera Wyatt Xena Yuri Zara'.split(),
       'dev':'Ansel Bianca Cedric Delia Emmett Freya Galen Helena Ingrid Jethro'.split(),
       'test':'Alba Basil Celine Desmond Elara Florian Giselle Henrik Ilona Jericho'.split()}
OBJECTS={'train':['lantern','kettle','helmet','teapot','thermos','sweater','puzzle','pencil','towel','vase','coat','plate','bowl','mug','camera','radio'],
         'dev':['compass','raincoat','seashell','hammock','cushion','lunchbox','snowglobe','watering can'],
         'test':['paintbrush','telescope','rucksack','headlamp','tablecloth','windchime','paperweight','fishing rod']}
PLACES={'train':['tool cabinet','laundry room','garden shed','hall cupboard','kitchen shelf','wooden chest','porch','bedroom drawer'],
        'dev':['reading nook','glass cabinet','attic shelf','covered balcony'],
        'test':['linen cupboard','storage alcove','studio shelf','spare room']}
COLORS={'train':['blue','green','yellow','red','white','purple','orange','brown'],
        'dev':['silver','pink','violet','black'], 'test':['gold','turquoise','beige','navy']}
INSTRUCTIONS={'train':['Answer the question briefly using the passage.','Give a short answer supported by this account.','Use only the information here to answer.','Keep the answer brief and factual.'],
              'dev':['Read the account and answer the question below.','Which answer follows from the information supplied?'],
              'test':['Base a concise answer on this description.','Respond using the facts in the text.']}

def render(family,style,a,b,x,y,p,q,color,flip,ask):
    # 0..7 train surface forms, 8..9 dev, 10..11 test. Same relational semantics,
    # different sentence/question wording. Final manual review remains necessary.
    u,v=(a,b) if not flip else (b,a)
    first,last=(p,q) if not flip else (q,p)
    left,right=(x,y) if not flip else (y,x)
    if family=='contents':
        facts=[f'The parcel addressed to {a} contains a {left}. The parcel for {b} contains a {right}.',
               f'{a} is receiving a {left}; a {right} is being sent to {b}.',
               f'On the delivery sheet, a {left} is listed for {a}, and a {right} for {b}.',
               f'Two parcels are ready. Inside {a}\'s parcel is a {left}. Inside {b}\'s parcel is a {right}.',
               f'A {left} has been packed for {a}. A separate parcel containing a {right} is for {b}.',
               f'The package for {b} holds a {right}, whereas the package for {a} holds a {left}.',
               f'{a}\'s order is a {left}. {b}\'s order is a {right}. Both orders have been packed.',
               f'Packing record: {b} gets a {right}; {a} gets a {left}.',
               f'There are two deliveries: one with a {left} for {a}, the other with a {right} for {b}.',
               f'A courier has a parcel for {a} holding a {left}. Another parcel, holding a {right}, goes to {b}.',
               f'The recipient of a {left} is {a}. A second delivery, a {right}, is addressed to {b}.',
               f'According to the packing slip, {a} ordered a {left}, while {b} ordered a {right}.']
        if ask:question=f'Who is receiving the {x}?';gold=a if left==x else b;full=f'{gold} is receiving the {x}.';foils=[b if gold==a else a]
        else:question=f'What is inside the parcel for {b}?';gold=right;full=f'The parcel for {b} contains a {right}.';foils=[left]
    elif family=='ownership':
        facts=[f'{u} owns a {x}. {v} borrowed it for the afternoon.',f'{v} borrowed a {x} from its owner, {u}.',
               f'{u} lent a personal {x} to {v}; it remains {u}\'s property.',f'The {x} belongs to {u}, although {v} is using it today.',
               f'{v} is using {u}\'s {x} on loan.',f'The owner of the {x} is {u}. Its temporary borrower is {v}.',
               f'{u} agreed to lend the {x} to {v}. The loan does not transfer ownership.',f'{v} has temporary use of a {x} owned by {u}.',
               f'{u} let {v} use the {x} and asked for it back later.',f'{v} took the {x} home on loan from {u}, who owns it.',
               f'{u}\'s {x} is currently on loan to {v}.',f'{v} is the borrower of the {x}; the property still belongs to {u}.']
        gold=v if ask else u;question=f'Who is {"borrowing" if ask else "the owner of"} the {x}?';full=f'{gold} {"is borrowing" if ask else "owns"} the {x}.';foils=[u if ask else v]
    elif family=='availability':
        facts=[f'The shop has a {left}, but no {right}.',f'A {left} is in stock. A {right} is unavailable.',
               f'There is no {right} on sale today, though a {left} is available.',f'The seller can supply a {left}; the {right} is sold out.',
               f'Only the {left} remains in stock; the {right} has run out.',f'The shop has run out of the {right}. It still has a {left}.',
               f'Stock update: {left} available, {right} unavailable.',f'A customer can buy the {left} today, but cannot buy the {right}.',
               f'The shelf has a {left}. The {right} is not in stock.',f'A {right} cannot be supplied today. A {left} can be supplied.',
               f'The {left} is ready to purchase; the {right} is out of stock.',f'Today\'s stock includes a {left}, while a {right} is missing.']
        gold=right if ask else left;question='Which item is '+('unavailable?' if ask else 'available?');full=f'The {gold} is {"unavailable" if ask else "available"}.';foils=[left if ask else right]
    elif family=='updates':
        facts=[f'{a} put the {x} in the {first}. Later, {a} moved it to the {last}.',
               f'The {x} was in the {first} at first, but it is now in the {last}.',
               f'After finding the {x} in the {first}, {a} placed it in the {last}.',
               f'The old location of the {x} was the {first}. Its new location is the {last}.',
               f'{a} has moved the {x} from the {first} to the {last}.',
               f'The {x} is now kept in the {last}. Previously it was in the {first}.',
               f'A note originally listed the {first} as the place for the {x}. An update says it is in the {last} now.',
               f'First the {x} was placed in the {first}; afterward it was relocated to the {last}.',
               f'The {x} used to be in the {first}. {a} has since transferred it to the {last}.',
               f'An earlier record puts the {x} in the {first}; the most recent record puts it in the {last}.',
               f'{a} no longer keeps the {x} in the {first}. It has been moved to the {last}.',
               f'The latest location of the {x} is the {last}, replacing its former spot in the {first}.']
        gold=first if ask else last;question=f'Where {"was" if ask else "is"} the {x} {"before it was moved" if ask else "now"}?';full=f'The {x} {"was previously" if ask else "is now"} in the {gold}.';foils=[last if ask else first]
    elif family=='reference':
        facts=[f'{u} handed the {x} to {v}. The recipient put it away.',f'{v} received the {x} from {u}. The sender left afterward.',
               f'The person giving the {x} was {u}; the person receiving it was {v}.',f'{u} passed a {x} to {v}, who accepted it.',
               f'A {x} was delivered by {u} to {v}.',f'{v} accepted a {x} handed over by {u}.',
               f'{u} was the sender of a {x}. {v} was the recipient.',f'The {x} changed hands when {u} gave it to {v}.',
               f'After {u} gave the {x} to {v}, the recipient thanked the sender.',f'{v} thanked {u} for handing over the {x}.',
               f'The giver, {u}, handed the {x} to its recipient, {v}.',f'{u} supplied the {x}, and {v} took delivery of it.']
        gold=u if ask else v;question=f'Who {"gave" if ask else "received"} the {x}?';full=f'{gold} {"gave" if ask else "received"} the {x}.';foils=[v if ask else u]
    elif family=='missing':
        known=f'Its color is {color}.' if not flip else 'Its color was not recorded.'
        facts=[f'{a} found a {x} in the {p}. {known}',f'A {x} was found by {a} in the {p}. {known}',
               f'The report places a found {x} in the {p}. {known}',f'{a}\'s report concerns a {x} found in the {p}. {known}',
               f'A found {x} came from the {p}. {known}',f'The {p} is where {a} found the {x}. {known}',
               f'Discovery record: a {x}, found in the {p}. {known}',f'The location of the found {x} is the {p}. {known}',
               f'The account says the {x} was discovered in the {p}. {known}',f'A report describes a {x} discovered at the {p}. {known}',
               f'The {x} was located in the {p}, according to the record. {known}',f'{a} reported finding the {x} in the {p}. {known}']
        if ask:gold=p;full=f'The {x} was found in the {p}.';question=f'Where was the {x} found?';foils=[color]
        else:gold='Not stated' if flip else color;full='The passage does not state its color.' if flip else f'The {x} is {color}.';question=f'What color is the {x}?';foils=[color,p] if flip else [p]
    else:
        facts=[f'{a} bought a {left}, not a {right}.',f'{a} chose the {left} rather than the {right}.',
               f'The purchase was a {left}. {a} did not buy a {right}.',f'{a} left the {right} in the shop and purchased the {left}.',
               f'{a} decided against buying the {right} and bought a {left} instead.',f'The {right} was not purchased; {a} purchased the {left}.',
               f'{a}\'s receipt lists a {left}. No {right} was bought.',f'{a} bought only the {left}; the {right} was not bought.',
               f'{a} selected a {left} to buy, and left the {right} behind.',f'Instead of a {right}, {a} purchased a {left}.',
               f'The purchase made by {a} was the {left}, never the {right}.',f'{a} chose to buy a {left} and did not purchase a {right}.']
        if ask:
            question=f'Why did {a} buy the {right}?';gold=full=f'{a} did not buy the {right}; {a} bought the {left}.';foils=[]
        else:question=f'What did {a} buy?';gold=left;full=f'{a} bought a {left}.';foils=[right]
    if style>=8:
        alternate=style>=10
        if family=='contents':question=(f'To whom is the {x} being sent?' if alternate else f'Who should get the {x}?') if ask else (f'Which item will {b} receive?' if alternate else f'What has been packed for {b}?')
        elif family=='ownership':question=(f'Who has the {x} on loan?' if alternate else f'Who is the temporary user of the {x}?') if ask else (f'Whose property is the {x}?' if alternate else f'Who does the {x} belong to?')
        elif family=='availability':question=('Which item cannot be purchased?' if alternate else 'What is out of stock?') if ask else ('What can be bought today?' if alternate else 'Which item is still in stock?')
        elif family=='updates':question=(f'What was the earlier location of the {x}?' if alternate else f'Where did the {x} used to be?') if ask else (f'What is the current location of the {x}?' if alternate else f'Where would you find the {x} after the change?')
        elif family=='reference':question=(f'Who supplied the {x}?' if alternate else f'Who was the sender of the {x}?') if ask else (f'Who took delivery of the {x}?' if alternate else f'Who was the recipient of the {x}?')
        elif family=='missing':question=(f'What was the discovery location of the {x}?' if alternate else f'In what place was the {x} discovered?') if ask else (f'Which color does the account give for the {x}?' if alternate else f'What was the color of the discovered {x}?')
        else:question=(f'For what reason did {a} purchase the {right}?' if alternate else f'What made {a} buy the {right}?') if ask else (f'Which item did {a} purchase?' if alternate else f'What was bought by {a}?')
    alternatives=[gold,full]
    if gold=='Not stated':alternatives+=['Unknown','Not mentioned','The passage does not say.','The color is not stated.','Its color is unknown.','Its color was not recorded.']
    elif family!='false_premise' or not ask:
        alternatives += [f'It is {gold}.',f'The answer is {gold}.',f'The {gold}.']
        if family in ('updates','missing') and gold in (p,q):alternatives += [f'in the {gold}',f'in {gold}']
    return facts[style],question,gold,full,alternatives,foils

def records(split):
    n=96 if split=='train' else 16
    rows=[];base={'train':0,'dev':8,'test':10}[split];width=8 if split=='train' else 2
    rng=random.Random(20260926+base);seen=set()
    for family in FAMILIES:
        for i in range(n):
            for attempt in range(1000):
                a,b=rng.sample(NAMES[split],2);x,y=rng.sample(OBJECTS[split],2);p,q=rng.sample(PLACES[split],2);color=rng.choice(COLORS[split]);style=base+i%width
                group=[]
                for flip in (0,1):
                    for ask in (0,1):
                        context,question,gold,full,refs,foils=render(family,style,a,b,x,y,p,q,color,flip,ask)
                        directive=INSTRUCTIONS[split][i%len(INSTRUCTIONS[split])]
                        if i%3==0:prompt=f'{question}\n\n{context}\n\n{directive}'
                        elif i%3==1:prompt=f'{context}\n\n{question} {directive}'
                        else:prompt=f'{directive}\n\nText: {context}\nQuestion: {question}'
                        answer=gold if i%4==0 else full
                        group.append(dict(id=f'{split}-{family}-{i:03d}-{flip}-{ask}',family=family,origin='authored',group=f'{split}-{family}-{i:03d}',
                                         prompt=prompt,context=context,answer=answer,references=list(dict.fromkeys(refs)),foils=foils,unknown=gold=='Not stated',
                                         style=style,question=question,variant=flip,query=ask))
                keys={r['prompt'] for r in group}
                if len(keys)==4 and not keys&seen:
                    rows.extend(group);seen|=keys;break
            else:raise ValueError('Unable to build unique reading scenarios')
    return rows
