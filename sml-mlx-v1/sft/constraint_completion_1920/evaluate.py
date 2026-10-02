"""Keep surface compliance separate from semantic completion and retention."""
from sml_v1.common import fingerprint
from sft.reading_repair.generation import generate
from sft.skill_balance.evaluate import assess as retention_assess, eligible
from .constraints import verify, repetition, words
from .engine import encode,arrays,objective


def grade(row,text,stop):
    checks=verify(text,row['rules']);repeated=repetition(text)
    return dict(rule_checks=checks,all_rules=bool(text.strip()) and all(checks),repeated=repeated,
        surface_complete=stop=='end' and len(words(text))>=20 and not repeated,
        # This is deliberately NOT called task success. Truth/completeness require review.
        rules_and_surface=bool(text.strip()) and all(checks) and stop=='end' and len(words(text))>=20 and not repeated)


def assess(b,data,split,cfg,cancelled=lambda:False):
    b.model.eval();answers=[]
    for i,row in enumerate(data[split]['complete']):
        if cancelled():raise InterruptedError('Evaluation stopped')
        x,y=arrays([encode(b,row,cfg['context'])],b.pad)
        nll=float(objective(b.model,x,y,False).item());g=generate(b,row['prompt'],cfg['max_new_tokens'])
        answers.append(dict(id=row['id'],prompt=row['prompt'],reference=row['answer'],rules=row['rules'],generation=g,
            assistant_nll=nll,**grade(row,g['text'],g['stop'])))
        if (i+1)%16==0:print('[constraint-eval-progress]',split,i+1,flush=True)
    # Retention remains at its original generation budget for comparability.
    retained=retention_assess(b,data,split,dict(cfg,max_new_tokens=128),cancelled)
    from .protocol import DIR
    from sml_v1.common import read_json
    probes=[]
    for row in read_json(DIR/'probes.json')[split]:
        if cancelled():raise InterruptedError('Evaluation stopped')
        g=generate(b,row['prompt'],cfg['max_new_tokens'])
        probes.append(dict(**row,generation=g,repeated=repetition(g['text'])))
    n=len(answers)
    metrics=dict(count=n,assistant_nll=sum(r['assistant_nll'] for r in answers)/n,
        all_rules=sum(r['all_rules'] for r in answers)/n,rules_and_surface=sum(r['rules_and_surface'] for r in answers)/n,
        repeated=sum(r['repeated'] for r in answers)/n,stopped=sum(r['generation']['stop']=='end' for r in answers)/n,
        mean_generated_tokens=sum(r['generation']['tokens'] for r in answers)/n)
    return dict(metrics=metrics,answers=answers,probes=probes,retention=retained,
        limitation='Surface constraints and length do not establish correct or complete answers. Read actual outputs; compare semantic scores with the parent. Retention probes are reused, not fresh tests.')


def review_template(result):
    return dict(evaluation=fingerprint(result),status='unreviewed',reviewer=None,retention_answers_reviewed=False,
        criteria='For every response judge factual/content correctness, relevance, and completion of ALL requested parts. Filler, circular answers, missing items and unsupported claims fail. Do not use length or keyword presence as semantic proof.',
        answers=[dict(id=r['id'],content_correct=None,relevant=None,complete=None,notes='') for r in result['answers']+result['probes']])


def validate_review(review,result):
    expected={r['id'] for r in result['answers']+result['probes']}
    if review.get('evaluation')!=fingerprint(result) or review.get('status')!='reviewed' or not review.get('reviewer'):raise ValueError('Review saved answers first')
    if not review.get('retention_answers_reviewed'):raise ValueError('Review retention answers too')
    if {r['id'] for r in review['answers']}!=expected or len(review['answers'])!=len(expected):raise ValueError('Missing/duplicate reviewed answers')
    if any(type(r.get(k)) is not bool for r in review['answers'] for k in ('content_correct','relevant','complete')):raise ValueError('Unscored semantic review')
    return sum(all(r[k] for k in ('content_correct','relevant','complete')) for r in review['answers'])/len(expected)
