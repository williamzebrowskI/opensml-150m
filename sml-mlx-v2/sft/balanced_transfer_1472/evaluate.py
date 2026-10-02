"""Save content, completion, old-skill, and new-choice development answers."""
from sft.context_reasoning_1024.evaluate import assess as context_assess
from sft.skill_recovery_768.evaluate import eligible as retained_eligible
from sft.explanation_transfer_1024.evaluate import grade as explanation_grade
from sft.reading_repair.generation import generate


def assess(backend,data,cfg,cancelled=lambda:False):
    result=context_assess(backend,data,'dev',cfg,cancelled)
    explanations=[]
    for row in data['dev']['explanation']:
        if cancelled():raise InterruptedError('Stopped during explanation evaluation')
        g=generate(backend,row['prompt'],cfg['max_new_tokens'])
        explanations.append(dict(id=row['id'],group=row['group'],label=row['label'],
            prompt=row['prompt'],reference=row['answer'],generation=g,
            **explanation_grade(g['text'],g['stop'],row['label'])))
    result['explanation_answers']=explanations
    m=result['metrics'];m['explanation_verdict']=sum(r['label_correct'] for r in explanations)/len(explanations)
    m['explanation_surface']=sum(r['complete_surface'] for r in explanations)/len(explanations)
    m['explanation_verdict_and_surface']=sum(r['correct_label_and_surface'] for r in explanations)/len(explanations)
    m['explanation_by_label']={label:sum(r['label_correct'] for r in explanations if r['label']==label)/
        sum(r['label']==label for r in explanations) for label in sorted({r['label'] for r in explanations})}
    result['limitation']='Verdict and surface checks do not establish explanation truth. Read actual answers. '
    result['limitation']+='Old-skill probes were previously inspected; public benchmarks are untouched by training.'
    return result


def eligible(metrics,baseline,cfg):
    old=retained_eligible(metrics,baseline,cfg)
    return (old and metrics['explanation_verdict']>=baseline['explanation_verdict']-0.04
        and metrics['explanation_verdict_and_surface']>=baseline['explanation_verdict_and_surface']-0.04
        and metrics['new_context_macro']>=baseline['new_context_macro'])
