"""Small fixed diagnostic, separate from corpus NLL and never training examples."""
import re

from .common import fingerprint

CASES = (
    ('location', 'Mira put her blue notebook in the kitchen drawer. Her red scarf stayed on the chair.',
     'Where is the blue notebook?', ('the kitchen drawer', 'in the kitchen drawer', 'kitchen drawer')),
    ('cause', 'The river path was flooded after heavy rain. Elena took the hill path to reach the village.',
     'Why did Elena take the hill path?', ('the river path was flooded', 'because the river path was flooded')),
    ('identity', 'Owen tends the orchard. Leah repairs bicycles. Iris teaches music.',
     'Who repairs bicycles?', ('leah',)),
    ('correction', 'The notice first said the meeting was on Monday. A correction changed it to Thursday.',
     'On which day will the meeting take place?', ('thursday', 'on thursday')),
    ('contrast', 'The wool coat belongs to Nora. The cotton jacket belongs to Sam.',
     'Which item belongs to Nora?', ('the wool coat', 'wool coat')),
    ('destination', 'Asha left the bakery and walked to the library to return a book.',
     'Where was Asha going?', ('the library', 'to the library', 'library')),
    ('sequence', 'Before planting the seeds, Tomas watered the soil. After planting, he covered the bed with straw.',
     'What did Tomas use to cover the bed?', ('straw', 'with straw')),
    ('absence', 'The shop sells apples and pears. The notice says that peaches are unavailable today.',
     'Which fruit is unavailable?', ('peaches',)),
)


def evaluate(generate):
    rows = []
    for key, passage, question, answers in CASES:
        prompt = f'Text: {passage}\nQuestion: {question}\nAnswer:'
        output = generate(prompt)
        normalized = re.sub(r'\s+', ' ', output.lower()).strip().strip('.!"\' ')
        rows.append(dict(id=key, passage=passage, question=question, output=output,
                         accepted_answers=list(answers), exact_match=normalized in answers))
    return dict(format='sml-v2-stage-b-reading-v1', fingerprint=fingerprint(CASES),
                exact_matches=sum(r['exact_match'] for r in rows), total=len(rows),
                note='Tiny authored diagnostic; exact-match may reject valid paraphrases. Review raw outputs; not a general benchmark.',
                cases=rows)
