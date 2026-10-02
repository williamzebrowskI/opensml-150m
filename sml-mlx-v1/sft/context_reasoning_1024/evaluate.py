"""Private development signals and saved answers; public benchmarks stay held out."""
from collections import Counter
import mlx.core as mx
from sft.skill_recovery_768.data import norm, unpack, stems, finite_sentence
from .engine import choice_arrays, choice_scores
from sft.reading_repair.generation import generate
from sft.reading_repair.evaluate import grade
from sft.transfer_control.engine import prose_loss
from sft.two_turn_640.data import history
from sft.transfer_control.launch import prose_texts


def instruction_grade(row, text, stop):
    content = unpack(text, row['format'])
    good_format = content is not None and bool(content.strip()) and stop == 'end'
    words = norm(content or '').split()
    covered = stems(' '.join(row['concepts'])) <= stems(content or '')
    ngrams = Counter(tuple(words[i:i+3]) for i in range(len(words)-2))
    repeated = bool(ngrams and max(ngrams.values()) >= 3)
    # This checks surface constraints/coverage, NOT truth or sensible relations.
    proxy = good_format and covered and not repeated and 6 <= len(words) <= 45 and finite_sentence(content or '')
    return dict(format_ok=good_format, concept_coverage=covered, repeated=repeated, proxy=proxy)


def grounded_grade(row,text,stop):
    content=unpack(text,row['format']);gold=unpack(row['answer'],row['format'])
    return stop=='end' and content is not None and content.strip()==gold.strip()


def assess(b, data, split, cfg, cancelled=lambda:False):
    b.model.eval(); mc=[]; generations=[]; instructions=[]; reading=[]
    for i,row in enumerate(data[split]['commonsense']):
        if cancelled(): raise InterruptedError('Stopped during evaluation')
        x,y = choice_arrays(b,row); scores = choice_scores(b.model,x,y).tolist()
        pred = max(range(len(scores)),key=lambda k:scores[k])
        mc.append(dict(id=row['id'],source=row['source'],gold=row['gold'],prediction=pred,scores=scores,correct=pred==row['gold']))
        if i%3 == 0:
            g=generate(b,row['prompt'],cfg['max_new_tokens'])
            generations.append(dict(id=row['id'],prompt=row['prompt'],reference=row['answer'],generation=g,
                exact=norm(g['text'])==norm(row['answer']) and g['stop']=='end'))
    for i,row in enumerate(data[split]['instruction']):
        if cancelled(): raise InterruptedError('Stopped during evaluation')
        g=generate(b,row['prompt'],cfg['max_new_tokens'])
        instructions.append(dict(id=row['id'],prompt=row['prompt'],reference=row['answer'],format=row['format'],
            generation=g,**instruction_grade(row,g['text'],g['stop'])))
        if (i+1)%32==0: print('[private-eval]',split,'instruction',i+1,flush=True)
    for row in data[split]['reading']:
        if cancelled(): raise InterruptedError('Stopped during evaluation')
        g=generate(b,row['prompt'],cfg['max_new_tokens'])
        reading.append(dict(id=row['id'],prompt=row['prompt'],references=row['references'],generation=g,
                            **grade(row,g['text'],g['stop'])))
    by_id={r['id']:r for r in reading}
    first={r['group']:by_id[r['id']]['generation']['text'] for r in data[split]['reading'] if r.get('turn')==0}
    own=[]
    for row in data[split]['reading']:
        if row.get('turn')!=1 or row['group'] not in first:continue
        if cancelled():raise InterruptedError('Stopped during own-history evaluation')
        prompt=history(row['first_prompt'],first[row['group']],row['current'])
        g=generate(b,prompt,cfg['max_new_tokens'])
        own.append(dict(id=row['id'],prompt=prompt,reference=row['answer'],generation=g,**grade(row,g['text'],g['stop'])))
    # Separately authored natural probes are a review aid, never a training target.
    from pathlib import Path
    from sml_v1.common import read_json
    probe_rows=read_json(Path(__file__).with_name('probes.json'))[split]
    natural=[]
    for row in probe_rows:
        if cancelled(): raise InterruptedError('Stopped during evaluation')
        natural.append(dict(**row,generation=generate(b,row['prompt'],cfg['max_new_tokens'])))
    all_g=[r['generation'] for rr in (generations,instructions,reading,natural) for r in rr]
    by_source={s:sum(r['correct'] for r in mc if r['source']==s)/sum(r['source']==s for r in mc) for s in {r['source'] for r in mc}}
    metrics=dict(commonsense_macro=sum(by_source.values())/len(by_source),commonsense_by_source=by_source,
        instruction_proxy=sum(r['proxy'] for r in instructions)/len(instructions),
        instruction_format=sum(r['format_ok'] for r in instructions)/len(instructions),
        reading_exact=sum(r['passed'] for r in reading)/len(reading),
        reading_f1=sum(r['f1'] for r in reading)/len(reading),
        stopped=sum(g['stop']=='end' for g in all_g)/len(all_g),prose_nll=prose_loss(b,prose_texts()))
    metrics['own_history_exact']=sum(r['passed'] for r in own)/len(own) if own else 0.
    grounded=[]
    for row in data[split].get('grounded',[]):
        if cancelled():raise InterruptedError('Stopped during exact-content evaluation')
        g=generate(b,row['prompt'],cfg['max_new_tokens'])
        passed=grounded_grade(row,g['text'],g['stop'])
        grounded.append(dict(id=row['id'],prompt=row['prompt'],reference=row['answer'],generation=g,passed=passed))
    if grounded:metrics['grounded_format_exact']=sum(r['passed'] for r in grounded)/len(grounded)
    knowledge=[]
    for i,row in enumerate(data[split].get('knowledge',[])):
        if cancelled():raise InterruptedError('Stopped during contextual reasoning evaluation')
        x,y=choice_arrays(b,row);scores=choice_scores(b.model,x,y).tolist()
        pred=max(range(len(scores)),key=lambda j:scores[j])
        record=dict(id=row['id'],source=row['source'],prompt=row['prompt'],choices=row['choices'],gold=row['gold'],scores=scores,correct=pred==row['gold'])
        if i%8==0:
            record['generation']=generate(b,row['prompt'],cfg['max_new_tokens'])
            record['reference']=row['answer']
        knowledge.append(record)
        if (i+1)%64==0:print('[context-eval]',split,i+1,flush=True)
    if knowledge:
        for source in sorted({r['source'] for r in knowledge}):
            rr=[r for r in knowledge if r['source']==source]
            metrics[source+'_acc']=sum(r['correct'] for r in rr)/len(rr)
        metrics['new_context_macro']=sum(metrics[s+'_acc'] for s in {r['source'] for r in knowledge})/len({r['source'] for r in knowledge})
    return dict(knowledge_answers=knowledge,grounded_answers=grounded,own_history=own,metrics=metrics,multiple_choice=mc,qa_answers=generations,instruction_answers=instructions,
        reading_answers=reading,natural_answers=natural,
        limitation='Private context-grouped new-source diagnostics; earlier skill/reading probes reused. Choice accuracy is not general conversational quality.')


def eligible(metrics, baseline, cfg):
    from sft.skill_recovery_768.evaluate import eligible as retained
    return retained(metrics,baseline,cfg) and metrics['new_context_macro']>=baseline['new_context_macro']


def score(metrics):
    return (metrics['new_context_macro']+metrics['commonsense_macro'])/2
