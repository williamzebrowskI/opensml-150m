"""Transparent rule coverage, with saved answers for human inspection."""
from collections import defaultdict
from sft.transfer_control.data import normalized


def contains(text, phrase):
    return f' {normalized(phrase)} ' in f' {normalized(text)} '


def grade(row, generation):
    text=generation['text'];norm=normalized(text);rubric=row['rubric']
    exact=norm in [normalized(s) for s in row['references']]
    requirements=all(any(contains(text,p) for p in group) for group in rubric['required'])
    forbidden=any(contains(text,p) for p in rubric['forbidden'])
    echo=not norm or norm==normalized(row['prompt'])
    length=rubric['min_words']<=len(norm.split())<=rubric['max_words']
    question=not rubric['question'] or text.count('?')==1
    structure=generation['stop']=='end' and not echo and length and question
    passed=structure and (exact if rubric['exact_only'] else requirements and not forbidden)
    return dict(rule_pass=bool(passed),exact=bool(exact and generation['stop']=='end'),
                prompt_echo=echo,forbidden_present=forbidden,requirements_met=requirements)


def audit_references(rows):
    for row in rows:
        for answer in row['references']:
            if not grade(row,dict(text=answer,stop='end'))['rule_pass']:
                raise ValueError(f'Reference fails its rubric: {row["id"]}: {answer}')


def evaluate(backend, rows, limit=64):
    from sft.transfer_control.engine import encode,arrays,objective,generate
    encoded=[encode(backend,r) for r in rows];losses=[];results=[];families=defaultdict(list)
    backend.model.eval()
    try:
        for i in range(0,len(encoded),2):
            batch=encoded[i:i+2];x,y=arrays(batch,backend.pad)
            losses.append((float(objective(backend.model,x,y,False).item()),len(batch)))
    finally:backend.model.train()
    for row in rows:
        generation=generate(backend,row['prompt'],limit)
        result=dict(**row,**generation,**grade(row,generation))
        results.append(result);families[row['family']].append(result)
    stats={f:dict(count=len(rr),rule_pass=sum(r['rule_pass'] for r in rr)/len(rr),
                  exact=sum(r['exact'] for r in rr)/len(rr)) for f,rr in families.items()}
    return dict(rule_macro=sum(s['rule_pass'] for s in stats.values())/len(stats),
                exact=sum(r['exact'] for r in results)/len(results),
                stopped=sum(r['stop']=='end' for r in results)/len(results),
                assistant_nll=sum(v*n for v,n in losses)/sum(n for _,n in losses),
                by_family=stats,records=results,
                limitation='Phrase/length rules are a diagnostic, not comprehensive semantic correctness; inspect saved answers.')
