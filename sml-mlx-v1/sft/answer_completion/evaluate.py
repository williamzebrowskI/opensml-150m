"""Teacher-forced target prediction and autonomous generation, kept distinct."""
from collections import Counter,defaultdict
import mlx.core as mx
import mlx.nn as nn
from .engine import encode,arrays
from sft.reading_repair.generation import generate

def normalized(s):return ' '.join(s.lower().split())
def teacher_stats(b,row,cfg):
    e=encode(b,row,cfg['context']);x,y=arrays([e],b.pad)
    logits=b.model(x)['logits'].astype(mx.float32);mask=y!=-100
    ce=nn.losses.cross_entropy(logits,mx.maximum(y,0),reduction='none')
    return dict(nll=float((ce*mask).sum().item())/e['targets'],targets=e['targets'],correct_tokens=int(((mx.argmax(logits,axis=-1)==y)*mask).sum().item()),eos_probability=float(mx.exp(-ce[0,-1]).item()))

def summary(records):
    n=len(records);targets=sum(r['teacher']['targets'] for r in records)
    return dict(examples=n,assistant_nll=sum(r['teacher']['nll']*r['teacher']['targets'] for r in records)/targets,teacher_forced_token_accuracy=sum(r['teacher']['correct_tokens'] for r in records)/targets,mean_gold_eos_probability=sum(r['teacher']['eos_probability'] for r in records)/n,free_exact_reference=sum(r['exact_reference'] for r in records)/n,stopped=sum(r['generation']['stop']=='end' for r in records),repeated=sum(r['repeated'] for r in records),mean_generated_tokens=sum(r['generation']['tokens'] for r in records)/n)

def assess(b,data,cfg,cancelled=lambda:False):
    b.model.eval();result={}
    for split in ('train','heldout','manual_review'):
        records=[];families=defaultdict(list)
        for row in (data['train_probes'] if split=='train' else data['dev'] if split=='heldout' else data.get('manual_review',[])):
            if cancelled():raise InterruptedError('Stopped during diagnostic evaluation')
            t=teacher_stats(b,row,cfg);g=generate(b,row['prompt'],cfg['max_new_tokens'])
            words=g['text'].lower().split();grams=Counter(tuple(words[i:i+5]) for i in range(len(words)-4))
            r=dict(**row,teacher=t,generation=g,exact_reference=normalized(g['text'])==normalized(row['answer']) and g['stop']=='end',repeated=bool(grams and max(grams.values())>=2))
            records.append(r);families[row['family']].append(r)
            if len(records)%32==0:print('[evaluation-progress]',split,len(records),flush=True)
        if not records:continue
        result[split]=dict(metrics=summary(records),families={k:summary(v) for k,v in families.items()},answers=records)
    from sft.transfer_control.engine import prose_loss
    from sft.transfer_control.launch import prose_texts
    result['prose_nll']=prose_loss(b,prose_texts())
    result['interpretation']='Train metrics cover fixed training probes only. Heldout means development examples; reserved test stays unevaluated. Exact reference matching is not semantic correctness for open responses. Review actual answers. Prose NLL uses the same fixed existing evaluation passages, never gradient targets.'
    return result
