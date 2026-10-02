"""Matched legacy retention and separate unified development checks."""
from collections import Counter
from pathlib import Path
import re
from sml_v1.common import read_json
from sft.conversation_foundation_original import engine
from sft.text_followup_512_v1.engine import assess as retention_assess
from sft.text_followup_512_v1.data import verify_text
from data import ROOT,EXP,FOLLOWUP,recovery_score


def evaluate(b,data,cfg):
    legacy_cfg=read_json(ROOT/'sft/text_followup_512_v1/config.json')
    print('[evaluation] legacy retention: same prompts, greedy decode and limits as retained 768',flush=True)
    retention=retention_assess(b,read_json(FOLLOWUP)['data'],legacy_cfg)
    print('[retention-eval]',retention['metrics'],flush=True)
    # The public-source diagnostics score likelihood/stopping; not answer correctness.
    result=engine.assess(b,data,dict(cfg,validation_conversations=len(data['dev'])))
    print('[unified-eval]',result['metrics'],flush=True)
    probes=[]
    for category in ('format_followup','recovery'):
        for row in [r for r in data['dev'] if r['source']==category]:
            history=[];answers=[]
            for m in row['messages']:
                if m['role']!='user':continue
                history.append(m);g=engine.generate(b,history,128)
                history.append(dict(role='assistant',content=g['text']));answers.append(dict(prompt=m['content'],**g))
            score=verify_text(answers[-1]['text'],row['checks']) if category=='format_followup' else recovery_score(answers[-1]['text'],row['checks'])
            probes.append(dict(id=row['id'],source=category,family=row['checks']['family'],reference_messages=row['messages'],turns=answers,checks=score))
            if len(probes)%8==0:print('[behavior-probes]',len(probes),'/64 conversations',flush=True)
    def rates(rows):
        return dict(total=len(rows),joint_proxy=sum(r['checks']['joint_proxy'] for r in rows)/len(rows))
    metrics={category:rates([r for r in probes if r['source']==category]) for category in ('format_followup','recovery')}
    metrics['recovery_by_family']={family:rates([r for r in probes if r['source']=='recovery' and r['family']==family]) for family in sorted({r['family'] for r in probes if r['source']=='recovery'})}
    plain=[t for r in result['answers'] if r['source'] in ('conversation','recovery','writing') for t in r['turns']
           if not re.search(r'\b(bullets?|numbered|list|steps|points|outline)\b',t['prompt'],re.I)]
    metrics['heuristic_unrequested_list_turns']=len(plain)
    metrics['heuristic_unrequested_list_rate']=sum(bool(re.search(r'^\s*(?:[-*] |\d+[.)] )',t['text'],re.M)) for t in plain)/len(plain)
    print('[behavior-eval]',metrics,'; literal development proxies; no automatic selection',flush=True)
    return dict(retention=retention,development=result,behavior_metrics=metrics,behavior_answers=probes,public_benchmarks=False)
