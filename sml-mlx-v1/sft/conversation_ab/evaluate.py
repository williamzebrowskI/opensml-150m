"""Save actual answers; likelihood/repetition are diagnostics, not quality scores."""
from collections import Counter
from sft.reading_repair.generation import generate
from sft.reading_repair.evaluate import grade
from sft.skill_balance.engine import choice_arrays,choice_scores
from sft.skill_balance.evaluate import instruction_grade
from sft.transfer_control.engine import encode,arrays,objective,prose_loss
from sft.transfer_control.launch import prose_texts

def assess(b,data,cfg,split='dev',cancelled=lambda:False):
    b.model.eval();answers=[];loss=0;targets=0
    for row in data[split]:
        if cancelled():raise InterruptedError('Stopped during development evaluation')
        e=encode(b,row,cfg['context']);x,y=arrays([e],b.pad);v=float(objective(b.model,x,y).item());loss+=v*e['targets'];targets+=e['targets']
        g=generate(b,row['prompt'],cfg['max_new_tokens']);w=g['text'].lower().split();ngr=Counter(tuple(w[i:i+4]) for i in range(len(w)-3))
        answers.append(dict(id=row['id'],source=row['source'],prompt=row['prompt'],reference=row['answer'],generation=g,repeated=bool(ngr and max(ngr.values())>=3)))
    retain={};raw={}
    for family,rr in data['retention'].items():
        records=[]
        for r in rr:
            if cancelled():raise InterruptedError('Stopped during retention evaluation')
            if family=='commonsense':
                x,y=choice_arrays(b,r);s=choice_scores(b.model,x,y).tolist();correct=max(range(len(s)),key=lambda k:s[k])==r['gold']
                records.append(dict(id=r['id'],correct=correct,scores=s))
            else:
                g=generate(b,r['prompt'],cfg['max_new_tokens'])
                if family=='reading':correct=grade(r,g['text'],g['stop'])['passed']
                else:correct=instruction_grade(r,g['text'],g['stop'])['proxy']
                records.append(dict(id=r['id'],correct=correct,generation=g))
        raw[family]=records;retain[family]=sum(r['correct'] for r in records)/len(records)
    metrics=dict(natural_nll=loss/targets,answers=len(answers),stopped=sum(r['generation']['stop']=='end' for r in answers),repeated=sum(r['repeated'] for r in answers),mean_tokens=sum(r['generation']['tokens'] for r in answers)/len(answers),retention=retain,prose_nll=prose_loss(b,prose_texts()))
    return dict(metrics=metrics,answers=answers,retention_records=raw,note='Read correctness, relevance and completion. No keyword quality score or automatic best. Retention/prose checks reused; natural splits grouped before selection.')
