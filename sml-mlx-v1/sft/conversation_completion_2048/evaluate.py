"""Own-history dialogue evaluation; surface diagnostics cannot select a chat model."""
from collections import Counter
import re
from sml_v1.common import fingerprint
from .dialogues import history


def repeated(text):
    w=re.findall(r'\w+',text.lower());n=Counter(tuple(w[i:i+5]) for i in range(len(w)-4))
    return bool(n and max(n.values())>=3)


def surface(row,g):
    t=g['text'];lower=t.lower();required=all(x.lower() in lower for x in row['required'])
    forbidden=any(re.search(r'\b'+re.escape(x.lower())+r'\w*\b',lower) for x in row['forbidden'])
    fmt=row['format'];format_ok=True
    if fmt=='three_bullets':format_ok=len(t.splitlines())==3 and all(x.startswith('- ') for x in t.splitlines())
    if fmt in ['two_sentences','three_sentences']:format_ok=len(re.findall(r'[.!?](?:\s|$)',t))==(2 if fmt=='two_sentences' else 3)
    exact=' '.join(t.split()).lower()==' '.join(row['answer'].split()).lower()
    return dict(required_coverage=required,obsolete_fact=forbidden,format_ok=format_ok,
                repeated=repeated(t),exact_reference=exact,
                surface_complete=required and not forbidden and format_ok and not repeated(t) and g['stop']=='end')


def assess(b,data,cfg,split='dev',cancelled=lambda:False):
    from sft.constraint_completion_1920.engine import encode,arrays,objective
    from sft.reading_repair.generation import generate
    from sft.skill_balance.evaluate import assess as skill_assess
    b.model.eval();humans=[];dialogue_answers=[];scenario_scores=[]
    for i,row in enumerate(data['human'][split]):
        if cancelled():raise InterruptedError('Stopped in human evaluation')
        x,y=arrays([encode(b,row,cfg['context'])],b.pad)
        nll=float(objective(b.model,x,y,False).item());g=generate(b,row['prompt'],cfg['max_new_tokens'])
        humans.append(dict(id=row['id'],source='no-robots',prompt=row['prompt'],reference=row['answer'],generation=g,assistant_nll=nll,repeated=repeated(g['text'])))
        if (i+1)%16==0:print('[human-eval]',split,i+1,flush=True)
    rr=data['dialogues'][split]
    for i in range(0,len(rr),2):
        if cancelled():raise InterruptedError('Stopped in dialogue evaluation')
        first,follow=rr[i:i+2];assert first['group']==follow['group']
        g1=generate(b,first['prompt'],cfg['max_new_tokens']);s1=surface(first,g1)
        # The primary follow-up uses the model's actual preceding answer.
        actual=history(first['prompt'],g1['text'],follow['current'])
        g2=generate(b,actual,cfg['max_new_tokens']);s2=surface(follow,g2)
        gold=generate(b,follow['prompt'],cfg['max_new_tokens'])
        dialogue_answers += [dict(id=first['id'],source='constructed',family=first['family'],turn=0,prompt=first['prompt'],reference=first['answer'],generation=g1,**s1),
            dict(id=follow['id'],source='constructed',family=follow['family'],turn=1,prompt=actual,reference=follow['answer'],generation=g2,**s2,gold_history_generation=gold,gold_history_surface=surface(follow,gold))]
        scenario_scores.append(s1['surface_complete'] and s2['surface_complete'])
        if (i//2+1)%12==0:print('[dialogue-eval]',split,i//2+1,flush=True)
    retention=skill_assess(b,data,split,dict(cfg,max_new_tokens=128),cancelled)
    gens=[r['generation'] for r in humans+dialogue_answers]
    metrics=dict(heldout_nll=sum(r['assistant_nll'] for r in humans)/len(humans),human_answers=len(humans),
        dialogue_scenarios=len(scenario_scores),own_history_surface_complete=sum(scenario_scores)/len(scenario_scores),
        exact_reference=sum(r['exact_reference'] for r in dialogue_answers)/len(dialogue_answers),
        stopped=sum(g['stop']=='end' for g in gens)/len(gens),repeated=sum(repeated(g['text']) for g in gens)/len(gens),
        mean_tokens=sum(g['tokens'] for g in gens)/len(gens))
    return dict(metrics=metrics,answers=humans,probes=dialogue_answers,retention=retention,
        limitation='Surface coverage/exactness are diagnostics, not semantic correctness. Read all candidate replies for entity relations, unsupported claims, omissions, and full follow-up completion. Skill tests are reused; official No Robots test and release audit remain untouched.')


def review_template(result):
    return dict(evaluation=fingerprint(result),status='unreviewed',reviewer=None,retention_answers_reviewed=False,
        criteria='Pass only useful complete answers: relevant, factually faithful, all requested parts, coherent. Both turns of a dialogue must pass. Surface coverage alone is insufficient.',
        answers=[dict(id=r['id'],relevant=None,correct=None,complete=None,natural=None,notes='') for r in result['answers']+result['probes']])


def retention_gate(result,baseline,cfg):
    from sft.skill_balance.evaluate import eligible as skill_eligible
    return skill_eligible(result['retention']['metrics'],baseline['retention']['metrics'],cfg)
