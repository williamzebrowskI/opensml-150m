"""Fixed fictional development scenarios; transparent checks, not semantic judging."""
import re
from sml_v1.common import fingerprint

def verify(text,c):
    t=text.strip();low=t.lower();words=re.findall(r"\b[\w'-]+\b",t)
    ok=bool(t)
    if 'exact' in c:ok &= low.rstrip('.!?')==c['exact'].lower()
    ok &= all(any(re.search(r'(?<!\w)'+re.escape(v.lower())+r'(?!\w)',low) for v in group) for group in c.get('include',[]))
    ok &= not any(re.search(r'(?<!\w)'+re.escape(v.lower())+r'(?!\w)',low) for v in c.get('exclude',[]))
    if 'min_words' in c:ok &= len(words)>=c['min_words']
    if 'max_words' in c:ok &= len(words)<=c['max_words']
    if c.get('no_question'):ok &= '?' not in t
    if 'bullets' in c:
        lines=[l for l in t.splitlines() if l.strip()];ok &= len(lines)==c['bullets'] and all(re.match(r'^\s*[-*] \S',l) is not None for l in lines)
    if 'paragraphs' in c:ok &= len(re.split(r'\n\s*\n',t))==c['paragraphs']
    if 'sentences' in c:ok &= len(re.findall(r'[.!?](?:\s|$)',t))==c['sentences']
    if 'ending' in c:ok &= t.endswith(c['ending'])
    if c.get('lowercase'):ok &= t==low and any(x.isalpha() for x in t)
    return bool(ok)

def suite(split):
    # Evaluation-only templates. Never used by the training practice generator.
    settings={'dev':[('Elara','Cedar','amber','otter'),('Bram','Willow','violet','fox')],
              'test':[('Soren','Maple','silver','badger'),('Veda','Aspen','teal','robin')]}[split]
    out=[]
    def add(category,prompts,expected,checks,rubric):
        def articles(t):return re.sub(r'\b([aA]) (amber|otter)\b',lambda m:('An' if m[1]=='A' else 'an')+' '+m[2],t)
        prompts=[articles(p) for p in prompts];expected=articles(expected)
        assert verify(expected,checks),(prompts,expected,checks)
        out.append(dict(id=fingerprint([split,category,prompts]),source='behavior-'+category,category=category,prompts=prompts,expected=expected,checks=checks,rubric=rubric))
    for name,place,color,animal in settings:
        add('history',[f'My pen name is {name}. Please remember it for this conversation.','What pen name did I give you? Reply with the name only.'],name,{'exact':name},'Recover a fact from history without adding a different name.')
        add('history',[f'I named my pottery stall {place}.','Correction: its name is Lantern now. Please use the corrected name.','What is the stall called now? Give only its name.'],'Lantern',{'exact':'Lantern'},'Apply a correction rather than repeat the superseded value.')
        add('history',[f'My favorite color is {color}, and my favorite animal is the {animal}.','What animal did I mention? Answer with only the animal.'],animal,{'exact':animal},'Retrieve the requested attribute, not the distractor.')
        add('history',[f'I have a scarf and a mug. The scarf belongs to {name}; the mug belongs to me.','Who owns the scarf? Give the name only.'],name,{'exact':name},'Preserve ownership across turns.')
        add('completion',[f'Write a warm birthday message for my sister {name}, who loves birdwatching. Write the message itself in no more than forty words.'],f'Happy birthday, {name}! I hope your day brings wonderful birds and happy moments.',{'include':[[name],['birthday'],['birds','birdwatching']],'min_words':8,'max_words':40,'no_question':True},'A usable birthday message; no request for unnecessary date or time.')
        add('completion',[f'Draft a short apology to {name} for returning their umbrella late. Promise to bring it tomorrow.'],f'{name}, I am sorry I returned your umbrella late. I will bring it tomorrow.',{'include':[[name],['sorry','apologize'],['umbrella'],['tomorrow']],'min_words':10,'no_question':True},'Actually write an apology with the stated promise.')
        add('completion',[f'Write a lost-property notice for a {color} notebook found at the {place} cafe. Tell its owner to ask at the counter.'],f'A {color} notebook was found at the {place} cafe. Please ask at the counter to collect it.',{'include':[[color],['notebook'],[place],['counter']],'min_words':10,'no_question':True},'Produce a notice containing all supplied details.')
        add('completion',[f'Write a two-sentence story about a {animal} finding a lantern. Give the story itself.'],f'A {animal} found a lantern beside a quiet stream. Its gentle light guided the animal home.',{'include':[[animal],['lantern']],'sentences':2,'min_words':12,'no_question':True},'Two narrative sentences, not instructions for writing a story.')
        add('grounded',[f'Fictional record: The {place} reading room opens on Friday. Its keeper is {name}. The door is {color}. Who is its keeper? Answer with the name only.'],name,{'exact':name},'Extract only the answer supported by the supplied record.')
        add('grounded',[f'Fictional record: {name} carries a red bag. Ivo carries a {color} bag. Who carries the {color} bag? Answer with the name only.'],'Ivo',{'exact':'Ivo'},'Use the correct entity rather than the first name.')
        add('grounded',[f'The notice says: The {place} club meets at the old mill. No meeting time is given. What time does it meet? Reply only with Unknown if the notice does not say.'],'Unknown',{'exact':'Unknown'},'Acknowledge absent information instead of inventing a time.')
        add('grounded',[f'Fictional notice: {name} cancelled the walk because the bridge was closed, not because of rain. Why was the walk cancelled?'], 'The bridge was closed.',{'include':[['bridge'],['closed']],'exclude':['rain'],'max_words':20},'Preserve the stated cause; avoid the explicitly excluded cause.')
        add('text_instructions',[f'List these supplies as exactly three bullet points and no other text: a {color} scarf, a lantern, a notebook.'],f'- {color} scarf\n- lantern\n- notebook',{'bullets':3,'include':[[color],['scarf'],['lantern'],['notebook']]},'Exactly three content-bearing bullets without a preamble.')
        add('text_instructions',[f'Write a short greeting for {name}. Use only lowercase letters and end with the exact words: welcome aboard'],f'hello {name.lower()}, welcome aboard',{'include':[[name]],'lowercase':True,'ending':'welcome aboard','max_words':20},'Satisfy name, casing and ending constraints together.')
        add('text_instructions',[f'Write exactly two short paragraphs. The first should mention {place}; the second should mention a {animal}.'],f'{place} is quiet today.\n\nA {animal} rests nearby.',{'paragraphs':2,'include':[[place],[animal]],'max_words':40},'Two paragraphs in the requested order; automatic check is only a proxy.')
        add('text_instructions',[f'Describe a {color} flower in exactly one sentence of at most twelve words.'],f'A {color} flower brightens the garden.',{'include':[[color],['flower']],'sentences':1,'max_words':12},'A descriptive sentence that respects the length limit.')
        add('relevance',[f'I am planning a work project called {place}.','Forget the project. Give me one name for a pet rabbit. Reply with the name only.'],'Clover',{'min_words':1,'max_words':2,'exclude':['project',place,'plan','schedule'],'no_question':True},'Follow the topic change and supply a plausible pet name; review semantic quality.')
        add('relevance',[f'I need a name for a quiet cafe in {place}.','Give me a name that includes Lantern. Only write the proposed name.'],'Quiet Lantern',{'include':[['Lantern']],'max_words':6,'exclude':['here','sure'],'no_question':True},'Produce the requested name directly.')
        add('relevance',[f'Write a gentle reminder to {name} to bring a jacket. Avoid scolding or blame.'],f'{name}, please remember your jacket so you can stay comfortable.',{'include':[[name],['jacket']],'exclude':['fault','careless','irresponsible'],'min_words':6,'no_question':True},'A relevant, gentle reminder; keyword check cannot fully judge tone.')
        add('relevance',[f'I prefer paper books to screens. Suggest a quiet activity involving a book for this evening.'], 'Read a favorite book in a comfortable chair.',{'include':[['read','reading'],['book','novel']],'exclude':['screen','video','phone'],'min_words':5,'no_question':True},'Respect the stated preference and give an activity rather than a generic offer.')
    return out

def tests():
    for split in ('dev','test'):
        for r in suite(split):
            assert verify(r['expected'],r['checks'])
            assert not verify('',r['checks'])
    assert not verify('Elara and someone else',{'exact':'Elara'})
    assert not verify('- scarf\n- lantern\nExtra introduction',{'bullets':2})
    assert not verify('The bridge was closed because of rain.',{'include':[['bridge']],'exclude':['rain']})
    assert not verify('hello there! Extra',{'ending':'hello there!'})
    assert not verify('one paragraph',{'paragraphs':2})
    return True
