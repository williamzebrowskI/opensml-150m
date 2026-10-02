"""Conclusion accuracy and completion are distinct; explanations need semantic review."""
import re
from collections import Counter
import mlx.core as mx
from sml_v2.common import read_json
from sft.skill_recovery_768.evaluate import assess as retention_assess, eligible
from sft.transfer_control.engine import encode, arrays, objective
from sft.reading_repair.generation import generate
from .data import DIR, LABELS, LEADS, norm


def grade(text, stop, label):
    text = text.strip(); prediction = None; remainder = ''
    for key, lead in LEADS.items():
        if text.lower().startswith(lead.lower()):
            prediction = key; remainder = text[len(lead):].strip(); break
    words = norm(remainder).split()
    triples = Counter(tuple(words[i:i+3]) for i in range(len(words)-2))
    repeated = any(n >= 3 for n in triples.values())
    # Surface screen ONLY. Neither token overlap nor a parsed label establishes
    # that the reason is true, sufficient, relevant, or logically valid.
    complete_surface = bool(stop == 'end' and 6 <= len(words) <= 60 and re.search(r'[.!?]$', remainder)
                            and not repeated)
    return dict(prediction=prediction, label_correct=prediction == label,
                complete_surface=complete_surface, repeated=repeated,
                correct_label_and_surface=prediction == label and complete_surface)


def assess(b, data, split, cfg, cancelled=lambda: False):
    b.model.eval(); answers = []; losses = []
    for index, row in enumerate(data[split]['explanation']):
        if cancelled(): raise InterruptedError('Stopped during explanation evaluation')
        x, y = arrays([encode(b, row, cfg['context'])], b.pad)
        nll = float(objective(b.model, x, y, False).item()); losses.append(nll)
        g = generate(b, row['prompt'], cfg['max_new_tokens'])
        answers.append(dict(id=row['id'], group=row['group'], prompt=row['prompt'],
            reference=row['answer'], label=row['label'], generation=g, **grade(g['text'], g['stop'], row['label'])))
        if (index+1) % 48 == 0: print('[explanation-eval]', split, index+1, flush=True)
    rates = {label: sum(r['label_correct'] for r in answers if r['label'] == label) /
             sum(r['label'] == label for r in answers) for label in LABELS}
    count = len(answers)
    metrics = dict(examples=count, conclusion_accuracy=sum(r['label_correct'] for r in answers)/count,
        by_label=rates, complete_surface=sum(r['complete_surface'] for r in answers)/count,
        correct_label_and_surface=sum(r['correct_label_and_surface'] for r in answers)/count,
        repeated=sum(r['repeated'] for r in answers)/count,
        stopped=sum(r['generation']['stop'] == 'end' for r in answers)/count,
        mean_generated_tokens=sum(r['generation']['tokens'] for r in answers)/count,
        assistant_nll=sum(losses)/count)
    manual = []
    for row in read_json(DIR/'probes.json')[split]:
        if cancelled(): raise InterruptedError('Stopped during manual probes')
        manual.append(dict(**row, generation=generate(b, row['prompt'], cfg['max_new_tokens'])))
    retained = retention_assess(b, data['replay'], split, cfg, cancelled)
    return dict(metrics=metrics, explanation_answers=answers, manual_answers=manual, retention=retained,
        limitation='Correct-label-and-surface is NOT semantic explanation quality. Read reasons and independent '
        'manual answers before selecting. Retention diagnostics are reused; no public benchmark gains assumed.')


def retention_ok(current, baseline, cfg):
    return eligible(current['retention']['metrics'], baseline['retention']['metrics'], cfg)
