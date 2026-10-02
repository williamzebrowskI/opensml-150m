"""Original deterministic paired follow-ups; no benchmark or downloaded text."""
import random,re
from collections import Counter
from pathlib import Path
from sml_v1.common import read_json,fingerprint
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
FAMILIES=('replace','append','remove','recall','constraint','format','uncertainty','clarify')
NAMES=['Ada','Ben','Cara','Dylan','Esme','Felix','Grace','Hugo','Iris','Jonah','Keira','Luca','Mina','Noel','Opal','Perry','Quinn','Rosa','Seth','Tara','Uma','Victor','Willa','Xavier']
# Split whole settings, not individual follow-up rows. Content is fictional and local.
SETTINGS={
 'train': [('craft circle','studio','sketchbook'),('garden club','courtyard','watering can'),('choir rehearsal','hall','songbook'),('picnic','meadow','basket'),('puppet show','theater','puppet'),('walking group','park','raincoat'),('pottery class','workshop','apron'),('reading hour','library','bookmark')],
 'dev': [('photo club','gallery','camera'),('beach cleanup','shore','bucket'),('dance rehearsal','ballroom','ribbon'),('seed exchange','greenhouse','envelope')],
 'test': [('chess club','lounge','chessboard'),('kite festival','field','kite'),('weaving class','loft','yarn'),('bird walk','woodland','binoculars')]}
# Alternatives are genuinely usable without the rejected material, not keyword-only negatives.
ALTERNATIVES={
 'train':[('draw a picture','pencils','draw with a pencil','make a collage from torn paper','arrange pebbles into a picture'),('keep papers together','staples','staple the papers together','hold the papers with a paper clip','slide the papers into a folder'),('decorate a card','markers','decorate it with markers','attach paper shapes with glue','press shapes into the card with a blunt tool'),('carry loose apples','bags','put the apples in a bag','carry the apples in a basket','carry the apples in a bowl')],
 'dev':[('hold a book open','clips','clip the pages down','rest a small weight on each side','use a book stand'),('label a jar','stickers','put a sticker on the jar','tie a paper tag around the jar','write on the jar with a washable marker')],
 'test':[('organize postcards','rubber bands','wrap a rubber band around them','put the postcards in a small box','slide the postcards into an envelope'),('keep a door open','doorstops','use a doorstop','place a heavy box in front of the door','secure the door to a fixed hook')]}

def history(first,reply,current):
 return f'Previous messages:\nUser: {first}\nAssistant: {reply}\n\nCurrent message: {current}'

def scenario(split,family,j,rng):
 event,place,item=rng.choice(SETTINGS[split]);a,b,c=rng.sample(NAMES,3)
 day,day2=rng.sample(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],2)
 time,time2=rng.sample(['morning','afternoon','evening'],2)
 s1=f'{a} will bring the {item} to the {event}.';s2=f'{b} will wait at the {place}.'
 variant=j%3 if split=='train' else 3+(j%2)
 def pick(train,held):return (train+held)[variant]
 if family=='replace':
  reply=f'{a} will meet {b} at the {place} on {day}.'
  first=f'Write one sentence about this plan: {a} meets {b}; location: the {place}; day: {day}.'
  q1=pick([f'Change {a} to {c}. Keep everything else the same.',f'Replace the name {a} with {c} and give the full revised sentence.',f'Use {c} instead of {a}; leave the other details unchanged.'],[f'The first person should be {c}, not {a}. Please update the whole sentence.',f'Please revise that with {c} in place of {a}.'])
  q2=pick([f'Change {day} to {day2}. Keep everything else the same.',f'Replace {day} with {day2} and give the complete sentence.',f'Use {day2} instead of {day}; keep the people and place.'],[f'That meeting is now on {day2}. Please update the sentence.',f'Please revise the day to {day2}, preserving the remaining details.'])
  targets=[reply.replace(a,c),reply.replace(day,day2)]
 elif family=='append':
  reply=f'Thank you, {a}, for bringing the {item}.'
  first=f'Write a short thank-you sentence to {a} for bringing the {item}.'
  additions=[f'I will see you on {day}.',f'I will meet you at the {place}.']
  qs=[pick([f'Add this sentence at the end: "{x}" Return the whole note.',f'Keep the thank-you and append "{x}".',f'Include "{x}" after the existing sentence. Give both sentences.'],[f'Please keep your note and finish it with this: {x}',f'Extend that note with one more sentence: {x}']) for x in additions]
  q1,q2=qs;targets=[reply+' '+x for x in additions]
 elif family=='remove':
  reply=s1+' '+s2
  first=f'Write a two-sentence note. First say that {a} will bring the {item} to the {event}. Then say that {b} will wait at the {place}.'
  q1=pick(['Remove the second sentence and return the remaining text.','Keep only the first sentence.','Delete the last sentence without changing the first.'],['Please leave just the opening sentence.','Shorten that to its first sentence only.'])
  q2=pick(['Remove the first sentence and return the remaining text.','Keep only the second sentence.','Delete the opening sentence without changing the last.'],['Please leave just the closing sentence.','Shorten that to its second sentence only.'])
  targets=[s1,s2]
 elif family=='recall':
  reply=f'{a} has the {item}. {b} is at the {place}.'
  first=f'Summarize these facts in two short sentences: the {item} belongs to {a}, and {b} is at the {place}.'
  q1=pick([f'Who has the {item}? Answer with only the name.',f'Give only the name of the person with the {item}.',f'Who owns it? Reply with the name only.'],[f'Which person has that {item}? Just the name, please.',f'I forgot who has the {item}; give their name only.'])
  q2=pick([f'Where is {b}? Give only the place.',f'Give the place where {b} is, with no other words.',f'Where was {b} located? Reply with the place only.'],[f'Remind me of {b}\'s location. Only the place name.',f'Which place was mentioned for {b}? Just that place.'])
  targets=[a,place]
 elif family=='constraint':
  goal,material,old,alt1,alt2=rng.choice(ALTERNATIVES[split])
  first=f'{a} wants to {goal} for the {event}. Suggest a way using {material}.'
  reply=f'You could {old}.'
  # Both revisions require a different response to the same earlier dialogue.
  q1=pick([f'I have no {material}. Suggest a different way.',f'Can you suggest an alternative that does not need {material}?',f'That will not work because I lack {material}. What else can I do?'],[f'Please find a way around this: there are no {material} available.',f'We cannot use {material} after all. Give another workable idea.'])
  q2=pick([f'I do have {material}. Repeat your original suggestion.',f'{material.capitalize()} are available. Keep your first idea.',f'Actually I found the {material}; tell me the original method again.'],[f'The {material} are here now. Please restate your first suggestion.',f'Good news: I can use {material}. Remind me of your earlier idea.'])
  targets=[f'You could {alt1 if j%2==0 else alt2}.',reply]
 elif family=='format':
  reply=f'{a}, {b}, {c}'
  first=f'List these people on one line, separated by commas, in this order: {a}; {b}; {c}.'
  q1=pick(['Put those names on separate lines, with a dash before each.','Change the list to bullet points using "- ". Keep the order.','Return each name as its own dash bullet.'],['Please turn that into a dash-bulleted list in the same order.','Use one name per line, each starting with a dash and a space.'])
  q2=pick(['Return only the last name in the list.','Give just the final name, with nothing else.','Keep only the third name.'],['What was the last name you listed? Reply with that name alone.','Please reduce the list to its final entry only.'])
  targets=[f'- {a}\n- {b}\n- {c}',c]
 elif family=='uncertainty':
  reply=f'{a} brought the {item} to the {place} on {day}.'
  first=f'Write a one-sentence summary using only these facts: {a} brought the {item}; destination: the {place}; day: {day}.'
  q1=pick([f'What color was the {item}? If it was not stated, say so.',f'Do we know the color of the {item}? Do not guess.',f'Tell me its color only if our conversation gave it.'],[f'Was any color given for that {item}?',f'Can you determine the color of the {item} from what was said?'])
  q2=pick(['Which day was it? Answer with only the day.','Give only the day mentioned earlier.','When did that happen? Just the day.'],['Remind me of the day from that summary, with no extra words.','What day did you mention? Reply with its name only.'])
  targets=[f'The color of the {item} was not stated.',day]
 elif family=='clarify':
  first=f'I am {c}. I need help writing a message about the {event}.'
  reply='Who is the message for, and what should it say?'
  q1=f'It is for {a}. Tell them I will arrive on {day} in the {time}. Keep it friendly and brief.'
  q2=f'It is for {b}. Tell them I cannot attend and wish them a good time.'
  targets=[f'Hi {a}, I will arrive on {day} in the {time}. See you then!',f'Hi {b}, I cannot attend the {event}. I hope you have a good time!']
 group=fingerprint([family,first,reply,q1,q2])
 base=dict(group=group,family=family,split=split,source='original-local-constructed',first_prompt=first,first_answer=reply)
 return [dict(base,id=group+':first',turn=0,prompt=first,answer=reply,current=first)]+[dict(base,id=group+f':follow{i}',turn=1,branch=i,prompt=history(first,reply,q),current=q,answer=t) for i,(q,t) in enumerate(zip((q1,q2),targets))]

def original(cfg):
 data={};firsts=set()
 for split,count in [('train',32),('dev',4),('test',4)]:
  rng=random.Random(cfg['seed']+{'train':0,'dev':1,'test':2}[split]);rows=[]
  for family in FAMILIES:
   n=0;attempts=0
   while n<count:
    attempts+=1
    if attempts>10000:raise ValueError('Could not create unique contexts')
    g=scenario(split,family,n,rng)
    key=g[0]['prompt']
    if key in firsts:continue
    firsts.add(key);rows+=g;n+=1
  data[split]=rows
 return data

def build(cfg):
 from sft.clean_reply_ab.data import original as prior_original
 data=original(cfg);replay,_=prior_original()
 assert len(replay)==256
 # Exactly the original TRAIN phrasings consumed by 640, never prior dev/test prompts.
 data['replay']=[dict(r,source='prior-clean-reply-train',role='rehearsal') for r in replay]
 data['retention']=read_json(ROOT/'sft/clean_reply_ab/dev.json')
 data['new']=data.pop('train');data['train']=data['new']+data['replay']
 m=dict(counts={k:len(v) for k,v in data.items()},hashes={k:fingerprint(v) for k,v in data.items()},new_scenarios=256,new_targets=768,rehearsal_unique=256,rehearsal_exposures=768,updates=128,exposures=1536,limitations='Constructed template curriculum, not 256 independent human conversations. Held-out settings and follow-up phrasing, shared task families. Reused clean-reply development diagnostics are not a fresh test. No public benchmark content; test never evaluated during training.')
 return data,m

def rebuild(cfg):
 d,m=build(cfg)
 if m!=read_json(DIR/'selection.json'):raise ValueError('Data selection changed')
 return d

def batch_at(data,cfg,u):
 if not 0<=u<128:raise ValueError('Cursor')
 new=list(data['new']);random.Random(cfg['seed']).shuffle(new)
 replay=[]
 for epoch in range(3):
  rr=list(data['replay']);random.Random(cfg['seed']+100+epoch).shuffle(rr);replay+=rr
 rows=new[u*6:(u+1)*6]+replay[u*6:(u+1)*6]
 if len(rows)!=cfg['batch']:raise ValueError('Incomplete batch')
 return rows
