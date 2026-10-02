"""Development-only evaluation; broad scores, saved answers, explicit regression guard."""
import mlx.core as mx
from . import engine
from .diagnostics import assess as retained_assess,repeated
from sft.reading_repair.generation import generate
from .verified import passes


def assess(b,anchor,data,cfg,cancelled=lambda:False,smoke=False):
    result=retained_assess(b,data,cfg,cancelled,smoke)
    records=[];generations=[]
    rows=data['dev']['preference'][:2] if smoke else data['dev']['preference']
    for i,row in enumerate(rows):
        if cancelled():raise InterruptedError('Paused during preference evaluation')
        x,y=engine.pair_arrays(b,row,cfg['context'])
        ref,_=engine.sequence_logps(anchor.model,x,y);lp,counts=engine.sequence_logps(b.model,x,y)
        dpo,ce,margin=engine.preference_terms(lp,counts,ref,cfg);mx.eval(dpo,ce,margin,lp,counts)
        records.append(dict(id=row['id'],dpo_loss=float(dpo.item()),chosen_nll=float(ce.item()),margin=float(margin.item()),raw_chosen_logp=float(lp[0].item()),raw_rejected_logp=float(lp[1].item())))
        if i<(2 if smoke else 24):
            g=generate(b,row['prompt'],cfg['max_new_tokens'])
            generations.append(dict(id=row['id'],prompt=row['prompt'],reference=row['answer'],generation=g,passed=g['stop']=='end' and passes(row,g['text'])))
    m=result['metrics'];m.update(preference_dpo_loss=sum(r['dpo_loss'] for r in records)/len(records),preference_chosen_nll=sum(r['chosen_nll'] for r in records)/len(records),preference_relative_win=sum((r['margin']>0)+.5*(r['margin']==0) for r in records)/len(records))
    # No preference-score-only candidate promotion: inspect natural generations.
    m['preference_generation_exact_proxy']=sum(r['passed'] for r in generations)/len(generations)
    m['preference_generation_stopped']=sum(r['generation']['stop']=='end' for r in generations)/len(generations)
    m['preference_generation_repeated']=sum(repeated(r['generation']['text']) for r in generations)/len(generations)
    result['preference_scores']=records;result['preference_generations']=generations
    result['limitations']+=' Preferences are constructed single-error pairs with shared task families. Relative win has a 50% tie baseline by construction; it is not answer correctness. Reserved preference/test questions are not evaluated here.'
    return result


def failures(metrics,baseline,cfg):
    limits=cfg['retention'];bad=[]
    if metrics['human_reading_exact']<baseline['human_reading_exact']-limits['reading_drop']:bad.append('reading')
    if metrics['prose_nll']>baseline['prose_nll']+limits['prose_nll_increase']:bad.append('prose_nll')
    if metrics['instruction_exact_proxy']<baseline['instruction_exact_proxy']-limits['instruction_drop']:bad.append('instruction_completion')
    if metrics['stopped']<baseline['stopped']-limits['stop_drop']:bad.append('stopping')
    if metrics['repeated']>baseline['repeated']+limits['repeat_increase']:bad.append('repetition')
    for name,v in metrics['choice_by_source'].items():
        if v['acc_norm']<baseline['choice_by_source'][name]['acc_norm']-limits['source_accuracy_drop']:bad.append('choice:'+name)
    for key in ('preference_generation_stopped','preference_generation_repeated'):
        delta=metrics[key]-baseline[key]
        if (key.endswith('stopped') and delta < -limits['stop_drop']) or (key.endswith('repeated') and delta > limits['repeat_increase']):bad.append(key)
    return bad
