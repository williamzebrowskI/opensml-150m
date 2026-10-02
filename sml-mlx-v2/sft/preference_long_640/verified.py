"""Original controlled preference pairs: correct complete answer vs one concrete error.

All scenes are fictional. These are auditable task examples, not new world facts
or human judgments. Exact checks apply to the specified transformations; natural
rewrites can have other valid phrasings and are not used to label model samples.
"""
import json,random
from sml_v2.common import fingerprint
from sft.grounded_rank_384.data import NAMES,DAYS,ROOMS,ITEMS,WORDS,norm,grams

TASKS=('json-update','sorted-json','two-bullets','reverse-numbered','corrected-message','two-sentence-summary','polite-complete','owner-not-borrower','conditional-choice','filter-records','small-arithmetic','required-ending','six-words','csv-columns','uppercase-prefix','followup-correction')


def partition(group):
    bucket=int(fingerprint(['verified-preference-640',group])[:8],16)%20
    return 'dev' if bucket==0 else 'test' if bucket==1 else 'train'


def passes(row,text):
    # Canonical equality is conservative for free wording; it is used only for
    # our constructed single-error negatives, never to reject sampled paraphrases.
    target=row['answer'];text=text.strip()
    if row['task'] in ('json-update','sorted-json'):
        def unique(pairs):
            result={}
            for k,v in pairs:
                if k in result:raise ValueError('Duplicate key')
                result[k]=v
            return result
        try:return json.loads(text,object_pairs_hook=unique)==json.loads(target)
        except (ValueError,TypeError):return False
    return text==target


def rows(split,count,seed,denied,dgrams):
    rng=random.Random(seed);result=[];seen=set();kind=0
    while len(result)<count:
        a,b,c=rng.sample(NAMES,3);day,old=rng.sample(DAYS,2);room=rng.choice([r for r in ROOMS if ' ' not in r]);item=rng.choice(ITEMS);words=rng.sample(WORDS,4)
        n,m=rng.sample(range(2,27),2);hour=rng.choice(['9 am','10 am','11 am','1 pm','3 pm','5 pm'])
        facts=dict(a=a,b=b,c=c,day=day,old=old,room=room,item=item,words=words,n=n,m=m,hour=hour,kind=kind)
        task=TASKS[kind];difficulty=1 if kind in (2,6,7,11,14) else 2 if kind in (0,4,5,8,10,15) else 3
        if kind==0:
            info=f'{a} booked the {room} for {old}. The day changed to {day}; the room stayed the same.'
            request='Return only a JSON object with keys "day" and "room" for the current booking. Lowercase both values.'
            answer=json.dumps(dict(day=day.lower(),room=room));wrong=json.dumps(dict(day=old.lower(),room=room));error='outdated day'
        elif kind==1:
            info=', '.join(words);request='Sort the supplied words alphabetically and uppercase them. Return only JSON with the key "words" containing the ordered array.'
            ordered=sorted(words);bad=list(ordered);bad[0],bad[1]=bad[1],bad[0]
            answer=json.dumps(dict(words=[w.upper() for w in ordered]));wrong=json.dumps(dict(words=[w.upper() for w in bad]));error='first two words in wrong order'
        elif kind==2:
            info=f'{a} has the {item}. {b} is in the {room}.';request='Copy both facts as exactly two bullet points starting with "- ", preserving their order. Add no introduction or closing text.'
            answer=f'- {a} has the {item}.\n- {b} is in the {room}.';wrong=f'- {b} has the {item}.\n- {a} is in the {room}.';error='people swapped'
        elif kind==3:
            info=', '.join(words);request='Return the four words in reverse input order, lowercase. Put one word on each line, numbered 1. through 4., and add nothing else.'
            answer='\n'.join(f'{i}. {w}' for i,w in enumerate(reversed(words),1));wrong='\n'.join(f'{i}. {w}' for i,w in enumerate(words,1));error='input order instead of reverse order'
        elif kind==4:
            info=f'Please join {a} in the {room} on {old} at {hour}. The corrected day is {day}.';request='Return only the corrected invitation sentence, changing the day and preserving every other detail.'
            answer=f'Please join {a} in the {room} on {day} at {hour}.';wrong=f'Please join {a} in the {room} on {old} at {hour}.';error='correction ignored'
        elif kind==5:
            info=f'{a} will meet {b} on {day} at {hour}. Their meeting is in the {room}. {c} will not attend.';request='Write exactly two sentences about the meeting. Begin with "Meeting:" and preserve both attendees, the day, time, and location. Do not mention the nonattendee.'
            answer=f'Meeting: {a} will meet {b} on {day} at {hour}. The meeting is in the {room}.';wrong=f'Meeting: {a} will meet {c} on {day} at {hour}. The meeting is in the {room}.';error='nonattendee substituted for attendee'
        elif kind==6:
            info=f'Send the {item} to {b} in the {room} by {day} at {hour}. Can you confirm when it has been sent?';request=f'Rewrite this as a polite two-sentence message addressed to {a}. Keep the deadline, recipient, location, item, and confirmation question.'
            answer=f'{a}, please send the {item} to {b} in the {room} by {day} at {hour}. Could you confirm when it has been sent?';wrong=f'{a}, please send the {item} to {b} in the {room} by {old} at {hour}. Could you confirm when it has been sent?';error='deadline changed'
        elif kind==7:
            info=f'{a} owns the {item}. {a} lent it to {b}, who carried it to the {room}. Lending changes possession but does not change ownership.';request='Name the owner and briefly explain using the stated rule. Answer in two sentences.'
            answer=f'{a} owns the {item}. Lending it to {b} did not change ownership.';wrong=f'{b} owns the {item}. Lending it to {b} changed ownership.';error='borrower confused with owner and stated rule reversed'
        elif kind==8:
            available=n%2==0
            info=f'The {room} is '+('available' if available else 'unavailable')+f'. {a} requested a meeting.';request='If the room is available, reply "Book the room." Otherwise reply "Ask for another room." Return only the selected sentence.'
            answer='Book the room.' if available else 'Ask for another room.';wrong='Ask for another room.' if available else 'Book the room.';error='wrong conditional branch'
        elif kind==9:
            counts=[n%4,m%4,(n+m)%4];people=[a,b,c]
            info='; '.join(f'{name}: {number} tickets' for name,number in zip(people,counts))+'.'
            request='List only the people who have at least two tickets, in their original order, separated by a comma and a space. If nobody qualifies, return "None".'
            eligible=[i for i,number in enumerate(counts) if number>=2]
            changed=sorted(set(eligible)^{0})
            answer=', '.join(people[i] for i in eligible) or 'None';wrong=', '.join(people[i] for i in changed) or 'None';error='first person eligibility inverted'
        elif kind==10:
            info=f'There are {n} trays. Each tray holds {m} stones.';request='Give the number of stones in one sentence, then a second sentence explaining that each tray contributes the same number.'
            answer=f'There are {n*m} stones. Each of the {n} trays contributes {m} stones.';wrong=f'There are {n*m+1} stones. Each of the {n} trays contributes {m} stones.';error='incorrect product'
        elif kind==11:
            info=f'{a} returned the {item} to {b} on {day}.';request='Copy the supplied sentence and then add exactly "Receipt confirmed." as a second sentence. Add nothing else.'
            answer=info+' Receipt confirmed.';wrong=info+' Receipt pending.';error='required ending changed'
        elif kind==12:
            info=f'Organizer: {a}; day: {day}; place: {room}.';request='Return exactly six lowercase words in this order: organizer, hosts, workshop, on, day, place. Use spaces only, with no punctuation.'
            answer=f'{a.lower()} hosts workshop on {day.lower()} {room}';wrong=f'{a.lower()} hosts workshop on {old.lower()} {room}';error='wrong day despite valid six-word format'
        elif kind==13:
            info=f'name,day\n{a},{day}\n{b},{old}';request='Swap the two CSV columns, including the header, while preserving the row order. Output only the resulting CSV.'
            answer=f'day,name\n{day},{a}\n{old},{b}';wrong=f'day,name\n{old},{a}\n{day},{b}';error='day values assigned to wrong people'
        elif kind==14:
            info=f'{a} has the {item}';request='Convert the supplied text to uppercase. Prefix it with "NOTICE: " and end with one period. Return nothing else.'
            answer='NOTICE: '+info.upper()+'.';wrong='NOTICE: '+f'{b} has the {item}'.upper()+'.';error='wrong person despite correct capitalization'
        else:
            # Native visible-prefix format for a genuine follow-up example.
            prompt=f'Previous messages:\nUser: Write one sentence saying {a} will meet {b} on {old} at {hour} in the {room}.\nAssistant: {a} will meet {b} on {old} at {hour} in the {room}.\n\nCurrent message: Change the day to {day}. Keep both names, the time and the place. Return only the revised sentence.'
            answer=f'{a} will meet {b} on {day} at {hour} in the {room}.';wrong=f'{a} will meet {b} on {old} at {hour} in the {room}.';error='follow-up correction ignored'
        group=fingerprint(['verified-pair-scene',task,prompt if kind==15 else info,answer])
        if group in seen or partition(group)!=split:continue
        if kind!=15:
            style=rng.randrange(3) if split=='train' else 3 if split=='dev' else 4
            if style==0:prompt=request+'\n\nInformation: '+info
            elif style==1:prompt='Information: '+info+'\n\n'+request
            elif style==2:prompt=request+'\n\n'+info
            elif style==3:prompt='Please complete this task.\n'+request+'\nDetails: '+info
            else:prompt='Details to use:\n'+info+'\nYour response must follow this request:\n'+request
        texts=(prompt,answer,wrong)
        if any((len(norm(t).split())>=4 and norm(t) in denied) or grams(t)&dgrams for t in texts):continue
        row=dict(id='verified:'+group,group=group,source='authored-verified-preference',split=split,task=task,difficulty=difficulty,prompt=prompt,answer=answer,rejected=wrong,error=error,facts=facts,verification='controlled facts and explicit output requirements; constructed negative contains the recorded concrete error')
        if not passes(row,answer) or passes(row,wrong):raise ValueError('Constructed preference failed verification')
        result.append(row);seen.add(group);kind=(kind+1)%len(TASKS)
    return result
