"""Conversation SFT plus fixed development checks; public benchmarks remain separate."""
from collections import Counter,defaultdict
from sft.conversation_foundation_v1.engine import load,optimizer,update,generate,repeated
from sft.conversation_foundation_v1.engine import assess as public_assess
from .behavior import verify
from .data import PUBLIC

def assess(b,data,cfg):
    base_cfg=dict(cfg,training_counts={k:cfg['training_counts'][k] for k in PUBLIC})
    result=public_assess(b,data,base_cfg);rows=[];counts=defaultdict(list)
    for r in data['behavior_dev'][:cfg.get('behavior_limit',40)]:
        history=[];answers=[]
        for prompt in r['prompts']:
            history.append(dict(role='user',content=prompt));g=generate(b,history,cfg['max_new_tokens'])
            answers.append(dict(prompt=prompt,**g));history.append(dict(role='assistant',content=g['text']))
        passed=verify(answers[-1]['text'],r['checks']);counts[r['category']].append(passed)
        rows.append(dict(id=r['id'],source=r['source'],turns=answers,checks=r['checks'],proxy_passed=passed,rubric=r['rubric'],expected_example=r['expected']))
    turns=[t for r in rows for t in r['turns']]
    result['metrics']['behavior_checks']={k:dict(passed=sum(v),total=len(v),proxy_rate=sum(v)/len(v)) for k,v in counts.items()}
    result['metrics']['behavior_generated_turns']=len(turns)
    result['metrics']['behavior_stopped']=sum(t['stop']=='eos' for t in turns)/len(turns)
    result['metrics']['behavior_repeated']=sum(repeated(t['tokens']) for t in turns)/len(turns)
    result['metrics']['behavior_stop_counts']=dict(Counter(t['stop'] for t in turns))
    result['answers']+=rows
    result['note']+=' Behavior pass rates are explicit rule/keyword proxies, not semantic quality scores. History uses the model own generated replies. Review rubric and saved answers for relevance, tone, factual support, and actual completion. New fixed suite differs from old foundation metrics; compare only against same-suite reference results.'
    return result
