"""Read-only fresh context/response-format audit; never use these cases as training data."""
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v2.common import atomic_json, file_sha256, lock
from sft.transfer_control.launch import preflight

CASES = [
    ('pottery', 'Mira teaches a pottery class on Tuesday at Harbor Hall. Admission is free.',
     'On which day does Mira teach the pottery class?', ['Tuesday'],
     'Write a short invitation that includes the teacher, activity, day, location, and admission cost.',
     ['Mira', 'pottery', 'Tuesday', 'Harbor Hall', 'free']),
    ('delivery', 'A parcel for Owen will arrive on Friday at the south entrance. His order contains a lamp.',
     'What does Owen\'s order contain?', ['lamp'],
     'Write Owen a delivery notification that includes the item, arrival day, and entrance.',
     ['Owen', 'lamp', 'Friday', 'south']),
    ('closure', 'Birch Pool will close on Monday for cleaning and reopen on Wednesday.',
     'Why is Birch Pool closing?', ['cleaning'],
     'Write a notice that includes the pool name, closure day, reason, and reopening day.',
     ['Birch Pool', 'Monday', 'cleaning', 'Wednesday']),
    ('rewrite', 'Please return the green drill to Sam by Thursday. The battery must be charged.',
     'Who should receive the drill?', ['Sam'],
     'Rewrite the note politely while keeping the color, item, recipient, deadline, and battery requirement.',
     ['green', 'drill', 'Sam', 'Thursday', 'charged']),
    ('correction', 'The old notice said the workshop was on Saturday in Room Cedar. The corrected notice says Sunday in Room Birch.',
     'Which room is named in the corrected notice?', ['Birch'],
     'Write a correction that gives the new day and room and makes clear that the old notice is outdated.',
     ['Sunday', 'Birch']),
    ('comparison', 'The orchard tour is outdoors and lasts one hour. The archive tour is indoors and lasts two hours.',
     'Which tour is indoors?', ['archive'],
     'Compare the two tours, including where each takes place and how long each lasts.',
     ['orchard', 'outdoors', 'one hour', 'archive', 'indoors', 'two hours']),
    ('request', 'Leah wants to move her appointment from Monday to Wednesday because her train was cancelled.',
     'Why does Leah want to move the appointment?', ['train', 'cancelled'],
     'Write a brief message from Leah requesting the change, including the old day, new day, and reason.',
     ['Monday', 'Wednesday', 'train', 'cancelled']),
    ('packing', 'For the field trip, children must bring a raincoat, a water bottle, and a notebook. Food is provided.',
     'What will be provided on the field trip?', ['Food'],
     'Write a parent reminder including all three required items and the information about food.',
     ['raincoat', 'water bottle', 'notebook', 'food']),
]


def main():
    from evaluation.full_benchmarks import spec, core
    model_specs = {
        'audit-parent-1920': dict(bundle='runs/sft_skill_balance_v1/step_0001920_628a0a65099b', step=1920,
                                 sha256='4cac7f733e6ef63230e26e2aa66418bf0d9661ef55f46fa1e01f2ac602b17407'),
        'audit-constraint-2048': dict(bundle='runs/sft_constraint_completion_1920_v1/step_0002048_1267d9f4cf7a', step=2048,
                                     sha256='e4e8a0de28e2141f6a580b49cc950aa64492c9f94d0d3aa98e6a26a2f439764b'),
    }
    rows = []
    for name, facts, question, short_facts, request, all_facts in CASES:
        for style, task, expected in [
            ('question', question, short_facts),
            ('plain', request, all_facts),
            ('formatted', request + ' Give exactly two complete sentences.', all_facts),
        ]:
            rows.append(dict(id=name + '-' + style, case=name, style=style,
                             prompt='Facts: ' + facts + '\n\n' + task, expected_facts=expected))
    out = ROOT / 'diagnostics/constraint_completion_2048_audit_20260927'
    preflight()
    with lock(ROOT / 'sft/.experiment.lock'):
        for name, model in model_specs.items():
            weights = ROOT / model['bundle'] / 'model.safetensors'
            if file_sha256(weights) != model['sha256']:
                raise ValueError('Unexpected model weights')
            spec.MODELS[name] = model
            backend = core.Backend(name)
            answers = []
            for row in rows:
                generation = core.generate(backend, row['prompt'], limit=128)
                answers.append(dict(**row, generation=generation))
                print(json.dumps(dict(model=name, id=row['id'], text=generation['text'],
                                      stop=generation['stop_reason']), ensure_ascii=False), flush=True)
            if file_sha256(weights) != model['sha256']:
                raise ValueError('Weights changed during read-only audit')
            atomic_json(out / (name + '.json'), dict(model=model, training=False, greedy=True,
                max_new_tokens=128, rows=answers,
                limitation='Eight authored contexts, three related prompt styles each; not independent public tests. Expected facts guide manual content review, not automatic semantic scoring. Do not train on these prompts.'))
            del backend
            gc.collect()


if __name__ == '__main__':
    main()
