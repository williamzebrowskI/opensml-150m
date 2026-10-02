"""Original deterministic task examples. No downloaded corpus or model teacher.

This is a deliberately small controlled curriculum, not natural conversation
data. Each group has two changed-fact variants with different correct answers.
Entity pools and groups are disjoint across train/development/reserved test.
"""
import hashlib
import json
import random
import re

FAMILIES = ('location', 'owner', 'correction', 'negation', 'extract',
            'replace', 'summary', 'missing')
NAMES = {
    'train': 'Ada Ben Cora Dan Eva Finn Gail Hugo Iris Joel Kira Leon Mina Noel Orla Paul'.split(),
    'dev': 'Asha Bram Cleo Drew Esme Ford Gia Hale Iona Jude Kaia Lars Mira Nico Opal Penn'.split(),
    'test': 'Anya Beau Cian Dara Elsa Faye Greta Heath Ida Joss Keir Lena Moira Neil Oona Piers'.split(),
}
OBJECTS = 'basket camera coat drum envelope flask glove hat jacket kettle lamp notebook'.split()
PLACES = ['attic', 'barn', 'cellar', 'desk', 'drawer', 'garage', 'hallway',
          'kitchen', 'loft', 'porch', 'shed', 'wardrobe']
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
FRUIT = ['apples', 'bananas', 'cherries', 'dates', 'figs', 'grapes', 'lemons',
         'mangoes', 'oranges', 'peaches', 'pears', 'plums']
ACTIVITIES = ['drawing', 'gardening', 'hiking', 'knitting', 'painting', 'reading', 'singing', 'swimming']


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def normalized(x):
    return ' '.join(re.findall(r'\w+', x.casefold()))


def render(family, v, flip, style):
    a, b = v['names']; obj, obj2 = v['objects']; p, q = v['places']
    first, second = (a, b) if not flip else (b, a)
    left, right = (p, q) if not flip else (q, p)
    d, e = v['days'] if not flip else v['days'][::-1]
    f, g = v['fruit'] if not flip else v['fruit'][::-1]
    activity = v['activity']
    target = obj2 if v['query_second'] else obj
    def sentences(x, y):
        return f'{y} {x}' if v['reverse'] else f'{x} {y}'
    if family == 'location':
        text = sentences(f'The {obj} is in the {left}.', f'The {obj2} is in the {right}.')
        questions = [f'Where is the {target}? Give only its location.', f'Return just the place containing the {target}.',
                     f'Find the {target}. Answer with the place only.', f'Which place would I look in to find the {target}? Give the place alone.',
                     f'I need the {target}. Name its location, with no extra words.']
        answer, foil = (right, left) if v['query_second'] else (left, right)
        aliases = [answer, f'the {answer}', f'in the {answer}']
    elif family == 'owner':
        text = sentences(f'{first} owns the {obj}.', f'{second} owns the {obj2}.')
        questions = [f'Who owns the {target}? Give only the name.', f'Name the owner of the {target}.',
                     f'Whose {target} is it? Reply with the name alone.', f'Tell me just who the {target} belongs to.',
                     f'Identify the person to whom the {target} belongs. Name only.']
        answer, foil = (second, first) if v['query_second'] else (first, second)
        aliases = [answer]
    elif family == 'correction':
        text = sentences(f'The original notice schedules the visit to {a} for {d}.',
                         f'The replacement notice schedules the visit to {a} for {e}.')
        questions = ['What is the new day? Give only the day.', 'Return only the current day of the visit.',
                     'On which day will the visit now happen? Day only.', 'After the change, when should the visitors arrive? Name the day alone.',
                     'Ignoring the superseded notice, name the day the visit is scheduled for.']
        answer, foil, aliases = e, d, [e]
    elif family == 'negation':
        text = (f'{a} does not like {g}, but likes {f}.' if v['reverse']
                else f'{a} likes {f} but does not like {g}.')
        questions = [f'Which fruit does {a} dislike? Give only the fruit.', f'Return only the fruit {a} does not like.',
                     f'Name just the unwanted fruit for {a}.', f'Which fruit would {a} prefer to avoid? Fruit only.',
                     f'Give the name of the fruit that is not liked by {a}.']
        answer, foil, aliases = g, f, [g]
    elif family == 'extract':
        fields = [f'owner: {a}', f'stored in: {left}', f'object: {obj}', f'destination: {right}']
        text = 'Record: ' + '; '.join(fields[::-1] if v['reverse'] else fields) + '.'
        questions = ['Extract only the storage location.', 'Return just the stored in field.',
                     'Where is the object stored? Location only.', 'Read the record and name its present storage place, rather than its destination.',
                     'From this entry, copy the value for where the item is currently kept.']
        answer, foil, aliases = left, right, [left, f'the {left}']
    elif family == 'replace':
        text = f'{a} enjoys {activity}.'
        target = b if not flip else v['third']
        questions = [f'Replace {a} with {target}. Keep everything else unchanged.', f'Change only the name {a} to {target}.',
                     f'Rewrite with {target} instead of {a}, and make no other edits.', f'Substitute {target} for {a}; copy the rest exactly.',
                     f'Provide the sentence after changing its named person from {a} to {target}. Do not alter the other words.']
        answer, foil = f'{target} enjoys {activity}.', v['third'] if not flip else b
        aliases = [answer]
    elif family == 'summary':
        # Target proposition differs between paired variants; the second sentence
        # is explicitly irrelevant. Evaluate exact fidelity and preserve raw text.
        text = sentences(f'{first} found the {obj}.', f'{second} found the {obj2}.')
        other = obj if v['query_second'] else obj2
        questions = [f'Summarize only what happened to the {target}, in one sentence.',
                     f'Write one sentence about who found the {target}. Omit the other object.',
                     f'Keep only the information about the {target} in a short sentence.',
                     f'In a sentence, report the discovery of the {target}, leaving out the unrelated discovery.',
                     f'Give a one-sentence account of the {target} being found; exclude the {other}.']
        person, foil = (second, first) if v['query_second'] else (first, second)
        answer = f'{person} found the {target}.'
        aliases = [answer, f'The {target} was found by {person}.']
    else:
        # Same question: only one variant contains the requested owner. Mention
        # an unrelated person in both to make blind name copying fail.
        text = (sentences(f'The {obj} belongs to {a}.', f'{b} saw the {obj} in the {p}.') if not flip
                else sentences(f'The owner of the {obj} is not identified.', f'{b} saw the {obj} in the {p}.'))
        questions = [f'Who owns the {obj}? Give the name, or Not stated if the owner is missing.',
                     f'Name the owner of the {obj}. If the text does not identify the owner, reply Not stated.',
                     f'Whose {obj} is this? Use the name alone or Not stated when the passage does not tell you.',
                     f'Can you identify the owner of the {obj} from this account? Reply with their name; otherwise write Not stated.',
                     f'Return the named owner of the {obj}, using Not stated if the ownership information is absent.']
        answer, foil = (a, b) if not flip else ('Not stated', b)
        aliases = [answer]
    # Even training includes instruction-first and context-first phrasing.
    question = questions[style]
    prompt = f'{question}\n\n{text}' if style % 2 else f'{text}\n\n{question}'
    return dict(prompt=prompt, answer=answer, references=aliases, foil=foil)


def records(split, wording='unseen', seed=24092491):
    if split not in NAMES: raise ValueError(split)
    rng = random.Random(seed + {'train': 0, 'dev': 1000, 'test': 2000}[split])
    pairs = {'train': 64, 'dev': 8, 'test': 12}[split]
    result, groups, prompts = [], set(), set()
    for family in FAMILIES:
        i = 0
        while i < pairs:
            names = rng.sample(NAMES[split], 3)
            v = dict(names=names[:2], third=names[2], objects=rng.sample(OBJECTS, 2),
                     places=rng.sample(PLACES, 2), days=rng.sample(DAYS, 2),
                     fruit=rng.sample(FRUIT, 2), activity=rng.choice(ACTIVITIES),
                     reverse=bool(i % 2), query_second=bool((i // 2) % 2))
            style = i % 3 if split == 'train' else (0 if wording == 'familiar' else 3 if split == 'dev' else 4)
            rows = [render(family, v, flip, style) for flip in (0, 1)]
            # Reject duplicate semantic pairs, including those whose family
            # uses only a small subset of the randomized parameters.
            canonical = [render(family, v, f, 0)['prompt'] for f in (0, 1)]
            group = digest([family, canonical])
            if group in groups or any(p in prompts for p in canonical): continue
            groups.add(group)
            prompts.update(canonical)
            for flip, row in enumerate(rows):
                result.append(dict(id=f'{split}-{family}-{i:03d}-{flip}', pair=f'{split}-{family}-{i:03d}',
                                   group=group, family=family, variant=flip, **row))
            i += 1
    if split == 'train': random.Random(seed).shuffle(result)
    return result


# Qualitative, never trained or used for checkpoint selection. Review for meaning.
OPEN = [
    'Hi there, how are you?',
    'What makes a shadow appear? Explain briefly.',
    'Make this request more polite: Pass me the cup.',
    'Write a short invitation to a walk in the park on Sunday.',
    'I cannot decide which to buy. Ask one question to help clarify what I need.',
    'Write only the name of an animal that barks.',
    'Explain in one short sentence why people use umbrellas in the rain.',
    'Summarize: Heavy rain flooded the footpath. The library stayed open. Walkers used the road instead.',
]


def audit():
    splits = {s: records(s) for s in NAMES}
    seen = set()
    for split, rows in splits.items():
        local = {r['group'] for r in rows}
        if seen & local: raise ValueError('Semantic group leakage')
        seen |= local
        if len({r['prompt'] for r in rows}) != len(rows): raise ValueError('Duplicate prompt')
        pairs = {}
        for r in rows: pairs.setdefault(r['pair'], []).append(r)
        for rr in pairs.values():
            assert len(rr) == 2 and rr[0]['answer'] != rr[1]['answer']
    return dict(examples={s: len(r) for s, r in splits.items()},
                hashes={s: digest(r) for s, r in splits.items()},
                familiar_dev_hash=digest(records('dev', 'familiar')),
                provenance='Original authored templates with deterministic fact substitutions; no teacher or third-party corpus.',
                limitation='Related task structures; this diagnostic is not a general-chat benchmark.')
