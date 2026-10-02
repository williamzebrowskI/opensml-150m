"""Descriptive, matched-item analysis of the frozen base-model audit."""
import json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.base_audit.protocol import OUT, SEED, PRIMARY
from sml_v2.common import atomic_json

def interval(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]

def accuracy(rows,metric,mode='context'):
    k=sum(r[mode]['predictions'][metric]==r['gold'] for r in rows)
    return dict(correct=k,count=len(rows),accuracy=k/len(rows),wilson_95=interval(k,len(rows)))

def matched(a,b,metric):
    assert [r['id'] for r in a]==[r['id'] for r in b]
    aa=np.array([r['context']['predictions'][metric]==r['gold'] for r in a],dtype=np.int8)
    bb=np.array([r['context']['predictions'][metric]==r['gold'] for r in b],dtype=np.int8)
    delta=bb-aa;rng=np.random.default_rng(SEED)
    draws=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    gain=int(((aa==0)&(bb==1)).sum());loss=int(((aa==1)&(bb==0)).sum());n=gain+loss
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(gain,loss)+1))/2**n) if n else 1.
    return dict(delta=float(delta.mean()),paired_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),
                gained=gain,lost=loss,unchanged=len(delta)-n,mcnemar_exact_two_sided=p,
                caution='Descriptive multiple comparisons; no multiplicity-adjusted causal claims.')

def pairs(rows):
    groups=defaultdict(list)
    for r in rows: groups[r['pair_id']].append(r)
    result={}
    for metric in ('total','char_mean','token_mean'):
        both=neither=switched=directional=0
        for pair in groups.values():
            a,b=sorted(pair,key=lambda r:r['variant']);assert (a['gold'],b['gold'])==(0,1)
            pa=a['context']['predictions'][metric];pb=b['context']['predictions'][metric]
            both+=pa==0 and pb==1;neither+=pa==1 and pb==0;switched+=pa!=pb
            margin_a=a['context']['scores'][0][metric]-a['context']['scores'][1][metric]
            margin_b=b['context']['scores'][0][metric]-b['context']['scores'][1][metric]
            directional+=margin_a>margin_b
        result[metric]=dict(correct=sum(r['context']['predictions'][metric]==r['gold'] for r in rows),
            items=len(rows),pairs=len(groups),both_correct=both,neither_correct=neither,
            prediction_switched=switched,margin_moves_with_fact=directional)
    return result

def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    data={m:json.loads((OUT/f'scores_{m}.json').read_text()) for m in ['earlier','current','reference']}
    report=dict(protocol=protocol,models={},comparisons={})
    for model,source in data.items():
        rows=source['records'];stats={}
        for task in PRIMARY:
            rr=[r for r in rows if r['task']==task]
            stats[task]={m:accuracy(rr,m) for m in ('total','char_mean','token_mean')}
            if task.startswith('paired_'):
                stats[task]['paired']=pairs(rr)
                stats[task]['families']={f:pairs([r for r in rr if r['family']==f]) for f in sorted({r['family'] for r in rr})}
            if 'no_context' in rr[0]:
                stats[task]['no_context']=accuracy(rr,PRIMARY[task],'no_context')
                stats[task]['context_gained']=sum(r['context']['predictions'][PRIMARY[task]]==r['gold'] and r['no_context']['predictions'][PRIMARY[task]]!=r['gold'] for r in rr)
                stats[task]['context_lost']=sum(r['context']['predictions'][PRIMARY[task]]!=r['gold'] and r['no_context']['predictions'][PRIMARY[task]]==r['gold'] for r in rr)
            if task=='boolq':
                for mode in ('context','no_context'):
                    recall={str(label):accuracy([r for r in rr if r['gold']==label],'total',mode)['accuracy'] for label in [0,1]}
                    stats[task][mode+'_class_detail']=dict(recall=recall,balanced_accuracy=sum(recall.values())/2,
                        predicted_yes=sum(r[mode]['predictions']['total']==1 for r in rr)/len(rr))
        report['models'][model]=stats
    for first,second in [('earlier','current'),('current','reference')]:
        report['comparisons'][first+'_to_'+second]={}
        for task in ('piqa','hellaswag','boolq'):
            a=[r for r in data[first]['records'] if r['task']==task]
            b=[r for r in data[second]['records'] if r['task']==task]
            report['comparisons'][first+'_to_'+second][task]=matched(a,b,PRIMARY[task])
    atomic_json(OUT/'summary.json',report)
    for model,stats in report['models'].items():
        print(model)
        for task in ('piqa','hellaswag','boolq'):
            print(task,json.dumps(stats[task][PRIMARY[task]]))
        print('boolq context',json.dumps(stats['boolq']['context_class_detail']))
        print('boolq no-context',json.dumps(stats['boolq']['no_context']))
        for task in ('paired_qa','paired_completion'):
            print(task,json.dumps(stats[task]['paired']['total']))
    print('comparisons',json.dumps(report['comparisons'],indent=2))

if __name__=='__main__':main()
