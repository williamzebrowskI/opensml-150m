"""Separate development content, completion, retention and candidate likelihoods."""
from collections import Counter
import re
import mlx.core as mx
from sft.reading_repair.generation import generate
from sft.reading_repair.evaluate import grade
from sft.transfer_control.engine import prose_loss
from sft.intact_smoltalk_base_pilot.engine import generate as conversation_generate
from sft.grounded_rank_384.data import instruction_pass
from .engine import choice_arrays, choice_scores, conversation_arrays, supervised_arrays, supervised_loss
from sft.intact_smoltalk_base_pilot.data import visible_prefix


def repeated(text):
    words = text.lower().split()
    counts = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    return bool((counts and max(counts.values()) >= 4)
                or re.search(r'([A-Za-z]{2,12})\1{4,}', text))


def assess(b, data, cfg, cancelled=lambda: False, smoke=False):
    b.model.eval(); result = {}; all_generations = []; scores_by_source = {}
    for family in ('grounded', 'instruction'):
        answers = []
        rows = data['dev'][family][:2] if smoke else data['dev'][family]
        for row in rows:
            if cancelled(): raise InterruptedError('Paused during development evaluation')
            g = generate(b, row['prompt'], cfg['max_new_tokens'])
            grading = grade(row, g['text'], g['stop']) if family == 'grounded' else dict(passed=g['stop'] == 'end' and instruction_pass(row, g['text']))
            answers.append(dict(id=row['id'], source=row['source'], task=row.get('task'), unknown=row.get('unknown'), prompt=row['prompt'],
                                reference=row['answer'], generation=g, **grading))
            all_generations.append(g)
        result[family + '_answers'] = answers
    mc = []
    rows = data['dev']['ranking'][:4] if smoke else data['dev']['ranking']
    for row in rows:
        if cancelled(): raise InterruptedError('Paused during candidate evaluation')
        x, y, chars = choice_arrays(b, row, cfg['context'])
        raw, normalized = choice_scores(b.model, x, y, chars); mx.eval(raw, normalized)
        record = dict(id=row['id'], source=row['source'], prompt=row['ranking_prompt'], choices=row['choices'], gold=row['gold'],
                      raw_scores=raw.tolist(), normalized_scores=normalized.tolist(),
                      correct=int(mx.argmax(raw).item()) == row['gold'], correct_norm=int(mx.argmax(normalized).item()) == row['gold'])
        mc.append(record)
    for source in sorted({r['source'] for r in mc}):
        rr = [r for r in mc if r['source'] == source]
        scores_by_source[source] = dict(count=len(rr), acc=sum(r['correct'] for r in rr) / len(rr),
                                       acc_norm=sum(r['correct_norm'] for r in rr) / len(rr))
    result['multiple_choice'] = mc
    result['replay_answers'] = []; losses = []
    rows = data['dev']['replay'][:2] if smoke else data['dev']['replay']
    def native(messages):
        room = 2048 - len(b.encode(visible_prefix(messages)))
        if room <= 0:
            return dict(text='', tokens=0, token_ids=[], stop='context_limit', budget=0)
        budget = min(room, cfg['max_new_tokens'])
        return dict(**conversation_generate(b, messages, budget, cancelled), budget=budget)
    # Measure actual generated history for the full source conversation, rather
    # than giving the model an ideal earlier assistant answer on follow-ups.
    for row in rows:
        if cancelled(): raise InterruptedError('Paused during conversation evaluation')
        actual = []
        for message in row['messages'][:-1]:
            if message['role'] != 'assistant': actual.append(message); continue
            g = native(actual)
            all_generations.append(g)
            if g['stop'] == 'context_limit':
                # This intermediate failed generation is already counted; mark
                # the final requested turn separately without double-counting it.
                g = dict(text='', tokens=0, token_ids=[], stop='context_limit', budget=0)
                break
            actual.append(dict(role='assistant', content=g['text']))
        else:
            g = native(actual)
        parts = list(conversation_arrays(b, row, cfg['context']))
        losses.append(sum(float(supervised_loss(b.model, x, y, positions).item()) for x, y, positions in parts) / len(parts))
        result['replay_answers'].append(dict(id=row['id'], source=row['source'], actual_messages=actual,
                                            reference=row['answer'], generation=g,
                                            authored_exact=(g['stop'] == 'end' and g['text'].strip() == row['answer']) if row['source'] == 'authored-complete-dialogue' else None))
        all_generations.append(g)
    human = [r for r in result['grounded_answers'] if r['source'] == 'squad']
    relations = [r for r in result['grounded_answers'] if r['source'] == 'authored-relations']
    science = [r for r in result['grounded_answers'] if r['source'] == 'authored-science-evidence']
    answerable = [r for r in human if not r['unknown']]
    unknown = [r for r in human if r['unknown']]
    dialogue = [r for r in result['replay_answers'] if r['authored_exact'] is not None]
    rate = lambda rows: sum(r['passed'] for r in rows) / len(rows) if rows else None
    prose = prose_loss(b, data['prose_diagnostic'][:2] if smoke else data['prose_diagnostic'])
    result['metrics'] = dict(human_reading_exact=rate(human), human_answerable_exact=rate(answerable),
        human_unknown_exact=rate(unknown),
        answerable_false_abstention=sum(r['generation']['text'].strip().lower() == 'not stated' for r in answerable) / len(answerable) if answerable else None,
        relation_exact=rate(relations), science_evidence_exact=rate(science),
        authored_dialogue_exact=sum(r['authored_exact'] for r in dialogue) / len(dialogue) if dialogue else None,
        instruction_exact_proxy=rate(result['instruction_answers']), choice_by_source=scores_by_source,
        choice_macro_norm=sum(s['acc_norm'] for s in scores_by_source.values()) / len(scores_by_source),
        stopped=sum(g['stop'] == 'end' for g in all_generations) / len(all_generations),
        repeated=sum(repeated(g['text']) for g in all_generations) / len(all_generations), measured_generations=len(all_generations),
        replay_nll=sum(losses) / len(losses), prose_nll=prose)
    result['limitations'] = 'Development diagnostics, not official benchmark results. Authored tasks share families across disjoint scenes; instruction presentation variants are held out. '
    result['limitations'] += 'Instruction exact match conservatively rejects alternative valid wording. Replay stopping/NLL do not measure semantic quality; read saved answers.'
    return result
