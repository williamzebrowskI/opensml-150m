"""Content-first reading checks; never select on training loss or public benchmarks."""
from collections import Counter,defaultdict
from sft.transfer_control.data import normalized
from .generation import generate,verify_generation


def qa_normalize(text):
    return ' '.join(w for w in normalized(text).split() if w not in ('a','an','the'))


def f1_score(text,reference):
    p,g=qa_normalize(text).split(),qa_normalize(reference).split()
    if not p or not g:return float(p==g)
    common=sum((Counter(p)&Counter(g)).values())
    return 2*common/(len(p)+len(g))


def grade(row,text,stop):
    refs=row['references'];n=qa_normalize(text)
    exact=bool(n) and any(n==qa_normalize(r) for r in refs)
    words=normalized(text).split();triples=Counter(tuple(words[i:i+3]) for i in range(len(words)-2))
    loop=bool(triples and max(triples.values())>=4)
    echo=bool(text.strip()) and normalized(text)==normalized(row['prompt'])
    good_stop=stop=='end'
    return dict(passed=bool(exact and good_stop and not loop and not echo),
                f1=max(f1_score(text,r) for r in refs) if good_stop else 0.,
                repeated=loop,echo=echo,capped=not good_stop)


def score_rows(backend,rows,limit):
    records=[];families=defaultdict(list);groups=defaultdict(list)
    for row in rows:
        generation=generate(backend,row['prompt'],limit);grades=grade(row,generation['text'],generation['stop'])
        result=dict(**row,**generation,**grades);records.append(result)
        families[row['family']].append(result);groups[row['group']].append(result)
    n=len(records)
    family_scores={f:sum(r['passed'] for r in rr)/len(rr) for f,rr in families.items()}
    unknown=[r for r in records if r.get('unknown')]
    return dict(count=n,correct=sum(r['passed'] for r in records),exact=sum(r['passed'] for r in records)/n,
                family_macro=sum(family_scores.values())/len(family_scores),by_family=family_scores,
                all_group_correct=sum(all(r['passed'] for r in rr) for rr in groups.values())/len(groups),
                f1=sum(r['f1'] for r in records)/n,
                unknown_recall=sum(r['passed'] for r in unknown)/len(unknown) if unknown else None,
                repeated=sum(r['repeated'] for r in records)/n,echo=sum(r['echo'] for r in records)/n,
                capped=sum(r['capped'] for r in records)/n,records=records)


def assess_new(backend,data,split,limit):
    from sft.transfer_control.engine import encode,arrays,objective
    authored=score_rows(backend,data['authored'][split],limit)
    natural=score_rows(backend,data['natural'][split],limit)
    challenge=score_rows(backend,data['challenge'],limit) if split=='dev' else None
    encoded=[encode(backend,r) for r in data['natural'][split]]
    total=0.;count=0;was_training=backend.model.training;backend.model.eval()
    try:
        for i in range(0,len(encoded),2):
            rows=encoded[i:i+2];x,y=arrays(rows,backend.pad)
            total+=float(objective(backend.model,x,y,False).item())*len(rows);count+=len(rows)
    finally:backend.model.train(was_training)
    allrows=authored['records']+natural['records']+(challenge['records'] if challenge else [])
    return dict(task_macro=authored['family_macro'],paired=authored['all_group_correct'],natural_f1=natural['f1'],
                natural_exact=natural['exact'],natural_nll=total/count,
                challenge_correct=challenge['correct'] if challenge else None,
                repeat_rate=sum(r['repeated'] for r in allrows)/len(allrows),
                cap_rate=sum(r['capped'] for r in allrows)/len(allrows),
                authored=authored,natural=natural,challenge=challenge)


def parent_metrics(result):
    n=result['new_tasks']
    return dict(task_macro=n['task_macro'],challenge_correct=n['challenge_correct'],natural_f1=n['natural_f1'],
                repeat_rate=n['repeat_rate'],cap_rate=n['cap_rate'],legacy_paired=result['legacy']['both_correct'])


def eligible(result,parent,cfg):
    n=result['new_tasks']
    return (n['task_macro']>=parent['task_macro']+cfg['reading_min_improvement']-1e-12
            and n['challenge_correct']>=parent['challenge_correct']+cfg['challenge_min_additional_correct']
            and n['natural_f1']>=parent['natural_f1']-cfg['natural_f1_drop_limit']-1e-12
            and result['legacy']['both_correct']>=parent['legacy_paired']-cfg['legacy_pair_drop_limit']-1e-12
            and result['prose_change_from_parent']<=cfg['prose_drift_limit']
            and n['repeat_rate']<=max(parent['repeat_rate'],cfg['repeat_rate_limit'])+1e-12
            and n['cap_rate']<=max(parent['cap_rate'],cfg['repeat_rate_limit'])+1e-12)


def ranking(result):
    n=result['new_tasks']
    return [n['challenge_correct'],n['task_macro'],n['natural_f1'],n['paired'],result['legacy']['both_correct']]
