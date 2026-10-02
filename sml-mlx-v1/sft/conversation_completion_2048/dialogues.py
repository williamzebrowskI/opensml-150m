"""Fictional complete-answer dialogues; split whole contexts and wording.

These are constructed template examples, not independent human conversations.
No release-audit prompts or benchmark items are imported as training targets.
"""
import random
from sml_v1.common import fingerprint

FAMILIES = ('invitation','clarification','rewrite','summary','ownership','restriction','story','list_edit','greeting')
SETTINGS = {
    'train': [('badge swap','community room','name tag'),('clay workshop','art room','apron'),
              ('map club','meeting room','notebook'),('costume rehearsal','stage','scarf'),
              ('plant swap','patio','flowerpot'),('board game night','club room','snack'),
              ('drawing session','classroom','pencil case'),('baking meetup','kitchen','mixing bowl')],
    'dev': [('mosaic class','annex','tile box'),('puppet rehearsal','auditorium','fabric bag'),
            ('walking tour','plaza','water bottle'),('letter exchange','reading room','envelope')],
    'test': [('origami meetup','atrium','paper bundle'),('music circle','courtyard','tambourine'),
             ('book repair','workroom','cloth cover'),('flower arranging','sunroom','vase')]}
PEOPLE = {'train':['Nadia','Marco','Leila','Owen','Priya','Ellis','Tessa','Dante','Zara','Malik','Ruth','Caleb'],
          'dev':['Imogen','Rafael','Sonia','Wesley','Harper','Joel'],
          'test':['Amina','Bruno','Cleo','Darius','Elena','Finn']}


def history(first, reply, current):
    return f'Previous messages:\nUser: {first}\nAssistant: {reply}\n\nCurrent message: {current}'


def make(split,family,j,rng):
    event,place,item=rng.choice(SETTINGS[split]);a,b=rng.sample(PEOPLE[split],2)
    article='an' if item[0].lower() in 'aeiou' else 'a'
    day,newday=rng.sample(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],2)
    when=rng.choice(['morning','afternoon','evening']);color,newcolor=rng.sample(['red','blue','green','yellow','purple'],2)
    request=['Write','Draft','Compose','Make'][j%4] if split=='train' else ['Please put together','I would like'][j%2]
    first_terms=[];follow_terms=[];forbidden=[];fmt=None
    if family=='invitation':
        p=f'{request} a friendly invitation to {a} for the {event} at the {place} on {day} in the {when}. Ask them to bring {article} {item}. Include every detail.'
        ans=f'Hi {a}, please join us for the {event} at the {place} on {day} in the {when}. Please bring {article} {item}. I hope you can come!'
        q=f'It will be on {newday} instead. Give me the entire invitation again, with all the other details kept.'
        end=ans.replace(day,newday);first_terms=[a,event,place,day,when,item];follow_terms=[a,event,place,newday,when,item];forbidden=[day]
    elif family=='clarification':
        p=f'I need to send a message about the {event} at the {place} on {day}. Can you help me write it?'
        ans='Who is the message for, and what would you like to tell them?'
        q=f'It is for {a}. Say that I will be at the {place} on {day} in the {when}, and ask them to bring the {item}. Please write the full message.'
        end=f'Hi {a}, I will be at the {place} on {day} in the {when} for the {event}. Could you please bring the {item}? Thank you!'
        first_terms=['who','message'];follow_terms=[a,place,day,when,item]
    elif family=='rewrite':
        p=f'Please rewrite this as a polite request. Preserve all the information: "{a}, bring the {color} {item} to the {place} on {day}."'
        ans=f'{a}, could you please bring the {color} {item} to the {place} on {day}?'
        q=f'Add a separate thank-you sentence after the request. Return both sentences and keep all details.'
        end=ans+' Thank you for your help.';first_terms=[a,color,item,place,day];follow_terms=first_terms+['thank'];fmt='two_sentences'
    elif family=='summary':
        p=f'Give a complete short summary of these notes: {a} is organizing the {event}; it is at the {place} on {day}; {b} is bringing the {color} {item}. Keep all facts.'
        ans=f'{a} is organizing the {event} at the {place} on {day}. {b} is bringing the {color} {item}.'
        q=f'The {item} is {newcolor}, not {color}. Please return the full corrected summary.'
        end=ans.replace(color,newcolor);first_terms=[a,event,place,day,b,color,item];follow_terms=[a,event,place,day,b,newcolor,item];forbidden=[color]
    elif family=='ownership':
        p=f'{request} two clear sentences about these belongings: {a} owns a {color} {item}; {b} owns a {newcolor} bag. Do not mix up the owners.'
        ans=f'{a} owns a {color} {item}. {b} owns a {newcolor} bag.'
        q=f'Actually, the {color} {item} belongs to {b}. Keep the bag fact too, and give me the corrected description.'
        end=f'{b} owns a {color} {item}. {b} also owns a {newcolor} bag.';first_terms=[a,color,item,b,newcolor,'bag'];follow_terms=[b,color,item,newcolor,'bag'];forbidden=[a]
    elif family=='restriction':
        p=f'For the {event}, we can use {color} paper, pencils, and stickers. Suggest two simple ways to decorate a card, using those supplies.'
        ans=f'Draw a picture on the {color} paper with pencils. Add stickers to decorate the card.'
        q='We cannot use stickers anymore. Give two complete suggestions using only the remaining supplies.'
        end=f'Draw a picture on the {color} paper with pencils. Fold the paper to make a card and add a pencil border.';first_terms=[color,'pencils','stickers'];follow_terms=[color,'pencil','paper'];forbidden=['sticker'];fmt='two_sentences'
    elif family=='story':
        p=f'{request} a three-sentence miniature story: {a} loses a {color} {item} at the {place}; {b} finds it; {a} thanks {b}. Keep that order and include a complete ending.'
        ans=f'{a} lost a {color} {item} at the {place}. {b} found it and returned it to {a}. {a} thanked {b} for helping.'
        q=f'Change the item color to {newcolor}. Give the entire three-sentence story again; keep the people, place and ending.'
        end=ans.replace(color,newcolor);first_terms=[a,color,item,place,b,'thank'];follow_terms=[a,newcolor,item,place,b,'thank'];forbidden=[color];fmt='three_sentences'
    elif family=='greeting':
        greetings = {
            'train': ['hello','hi','hey','hello there','good morning','what can you help me with?',
                      'hi, can you help me write a note?',"what's up?"],
            'dev': ['Hey there!', 'Hello, how can you help today?', 'Good evening!', 'Hi! I have a writing task.'],
            'test': ['Hello again!', 'Good afternoon!', 'Hey, can you help with a message?', 'Hi there, what can we work on?']}
        p=greetings[split][j]
        ans='Hello! I can help you draft invitations, revise messages, summarize notes, or brainstorm ideas. What would you like to work on?'
        q=f'Please write a note to {a} asking them to bring the {item} to the {place} on {day}.'
        end=f'Hi {a}, could you please bring the {item} to the {place} on {day}? Thank you!';first_terms=['help'];follow_terms=[a,item,place,day]
    else:
        p=f'{request} a packing list for the {event}. Include exactly these three things: the {color} {item}, a towel, and a bottle. Use one bullet for each.'
        ans=f'- {color} {item}\n- towel\n- bottle'
        q='Replace the towel with a blanket. Return the whole updated list as three bullets.'
        end=f'- {color} {item}\n- blanket\n- bottle';first_terms=[color,item,'towel','bottle'];follow_terms=[color,item,'blanket','bottle'];forbidden=['towel'];fmt='three_bullets'
    group=fingerprint([family,p,ans,q,end])
    first=dict(id=group+':first',group=group,source='constructed',family=family,split=split,prompt=p,answer=ans,turn=0,
               first_prompt=p,first_answer=ans,required=first_terms,forbidden=[],format='three_bullets' if family=='list_edit' else 'three_sentences' if family=='story' else None)
    follow=dict(id=group+':follow',group=group,source='constructed',family=family,split=split,prompt=history(p,ans,q),answer=end,
                turn=1,first_prompt=p,first_answer=ans,current=q,required=follow_terms,forbidden=forbidden,format=fmt)
    return [first,follow]


def build(cfg):
    data={};seen=set()
    for split,n in [('train',32),('dev',4),('test',4)]:
        rng=random.Random(cfg['seed']+{'train':0,'dev':1,'test':2}[split]);rows=[]
        for family in FAMILIES:
            count=(8 if family=='greeting' else 31) if split=='train' else n
            j=0;attempts=0
            while j<count:
                attempts+=1
                if attempts>10000:raise ValueError(f'Cannot construct distinct {split}/{family} contexts')
                rr=make(split,family,j,rng)
                if rr[0]['prompt'] in seen:continue
                seen.add(rr[0]['prompt']);rows+=rr;j+=1
        data[split]=rows
    return data
