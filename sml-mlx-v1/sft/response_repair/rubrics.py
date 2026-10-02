"""Separate constrained factual answers, requested form and open-task proxies."""
from collections import defaultdict
from sft.transfer_control.data import normalized
from .data import STRUCTURED


def has(text,phrase):return f' {normalized(phrase)} ' in f' {normalized(text)} '


def grade(row,generation):
    text=generation['text'].strip();norm=normalized(text);r=row['rubric'];kind=r['kind']
    echo=not norm or norm==normalized(row['prompt']) or bool(r['echo_text'] and norm==normalized(r['echo_text']))
    stopped=generation['stop']=='end'
    if kind=='copy':
        content=text==row['answer'];form=content
    else:
        if r['content_answers'] is not None:
            content=norm in {normalized(a) for a in r['content_answers']}
        elif kind=='greeting':
            content=any(has(text,x) for x in ('hello','hi','hey','hiya','good morning','good evening','good afternoon')) and any(has(text,x) for x in ('help','assist','discuss','ask','question','talk'))
        else:
            words=set(norm.split());keywords=set(r['keywords'] or [])
            content=bool(keywords) and len(words&keywords)/len(keywords)>=.6
        if kind=='brief':form=norm in {normalized(a) for a in r['format_answers']}
        else:
            form=3<=len(norm.split())<=45 and text.endswith(('.','!','?'))
            if kind in ('sentence','social','invitation','polite','question'):
                form=form and sum(text.count(p) for p in '.!?')==1
            if kind=='question':form=form and text.count('?')==1
            if kind=='polite':form=form and any(has(text,x) for x in ('please','could you','would you'))
            if kind=='invitation':form=form and any(has(text,x) for x in ('join','invited','come to')) and not any(has(text,x) for x in ('please invite','write an invitation','create an invitation'))
            if kind=='social':form=form and any(has(text,x) for x in ('welcome','congratulations','congrats','thank you','thanks'))
    return dict(content_match=bool(content and stopped and not echo),format_match=bool(form and stopped and not echo),
                task_pass=bool(content and form and stopped and not echo),prompt_echo=echo,
                exact=bool(stopped and norm==normalized(row['answer'])))


def audit_references(rows):
    for row in rows:
        for answer in row['references']:
            if not grade(row,dict(text=answer,stop='end'))['task_pass']:
                raise ValueError(f'Reference fails its rubric: {row["id"]}: {answer}')


def evaluate(backend,rows,limit=64):
    from sft.transfer_control.engine import encode,arrays,objective,generate
    encoded=[encode(backend,r) for r in rows];losses=[];results=[];families=defaultdict(list);groups=defaultdict(list)
    backend.model.eval()
    try:
        for i in range(0,len(encoded),2):
            batch=encoded[i:i+2];x,y=arrays(batch,backend.pad)
            losses.append((float(objective(backend.model,x,y,False).item()),len(batch)))
    finally:backend.model.train()
    for row in rows:
        generation=generate(backend,row['prompt'],limit)
        item=dict(**row,**generation,**grade(row,generation));results.append(item);families[row['family']].append(item)
        if row['family'] in STRUCTURED:groups[row['group']].append(item)
    if any(len(rr)!=4 for rr in groups.values()):raise ValueError('Incomplete factual/format contrast group')
    stats={f:dict(count=len(rr),**{k:sum(r[k] for r in rr)/len(rr) for k in ('content_match','format_match','task_pass','prompt_echo')}) for f,rr in families.items()}
    fact=[r for r in results if r['family'] in STRUCTURED]
    return dict(task_macro=sum(s['task_pass'] for s in stats.values())/len(stats),
                fact_content=sum(r['content_match'] for r in fact)/len(fact),
                fact_format=sum(r['format_match'] for r in fact)/len(fact),
                fact_joint=sum(r['task_pass'] for r in fact)/len(fact),
                contrast_groups=sum(all(r['task_pass'] for r in rr) for rr in groups.values())/len(groups),
                exact=sum(r['exact'] for r in results)/len(results),
                stopped=sum(r['stop']=='end' for r in results)/len(results),
                echo_rate=sum(r['prompt_echo'] for r in results)/len(results),
                assistant_nll=sum(v*n for v,n in losses)/sum(n for _,n in losses),
                by_family=stats,records=results,
                limitation='Factual content uses accepted text alternatives; sentence checks and open-task lexical coverage are proxies. Inspect saved answers.')
