"""Pinned public training sources plus exact parent replay; no raw disk cache."""
import random
from collections import Counter,defaultdict
from pathlib import Path
from sml_v2.common import read_json,fingerprint
from sft.skill_balance.data import (SOURCES,stream,eligible,norm,grams,formats,unpack,stems,finite_sentence,encode_stats,language_tools)
from sft.skill_balance.data import partition as skill_partition, exclusions as prior_exclusions
from sft.base_curriculum_v2.data import partition as base_partition
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent

REVIEW_REJECTIONS={
 'commonsenseqa:e8307b441c208e73a3337a57fb7157d1bd4765f5fa4650efcc4404ea72f87321':'amusement_and_having_fun_both_valid',
 'socialiqa:997ce774f9f4e4d0e512bcc4d0b450c9215e96dcfeb918602e6851e33e15e266':'ambiguous_feeling_label',
 'commonsenseqa:56df66bd917ac1ece92f99f4823d4467be92816b8e7a89848de4a403fcef5c2e':'computer_and_cloud_both_valid',
 'commonsenseqa:434ec937ef8435375cb42831bdacbf2fa480e0c3d77086b6a34c72b3b35515e3':'ambiguous_location_and_typo',
 'commongen:532c70b397fc1a93ba7409ef716b5045259a3b13c678f6241606ebdfce26446c':'broken_subject_agreement',
 'commongen:74940ae8f82c85f1a90aa84544a445b3aec3bd61d18fc26c3c2ac6cb894bf410':'fragment_and_typo',
 'commongen:a5ba743c315888ede201dbb7557fda3288fcec631680e6cb92086eb919cf6cd2':'unnatural_situation',
}

def partition(group):
    # Intersect BOTH historical partitions; no group trained by either recipe
    # is allowed into this run's source dev/test pools.
    # CommonGen was absent from the 512 -> 640 -> 768 ancestry.
    if group.startswith('concepts:'):return skill_partition(group)
    a,b=skill_partition(group),base_partition(group)
    return a if a==b else 'excluded'

def exclusions():
    denied,dgrams,paths=prior_exclusions()
    path=DIR/'probes.json';paths.append(path)
    for rows in read_json(path).values():
        for r in rows:denied.add(norm(r['prompt']));dgrams|=grams(r['prompt'])
    return denied,dgrams,paths

def build(cfg,cancelled=lambda:False):
    from sml_v2.tokenization import Tokenizer
    from sft.two_turn_640.data import rebuild as parent_rebuild,batch_at as parent_batch
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');denied,dgrams,_=exclusions()
    pools=defaultdict(list);rejected=Counter()
    # Stream bounded raw files; keep at most the needed rows per source/split.
    quotas={s:cfg['updates']*3 for s in SOURCES}
    import heapq
    for source in SOURCES:
        groups=set();answers=set()
        for raw in stream(source,cancelled):
            row,why=eligible(raw,source,denied,dgrams)
            if row is None:rejected[source+':'+why]+=1;continue
            if row['id'] in REVIEW_REJECTIONS:
                rejected[source+':review_rejection']+=1;continue
            split=partition(row['group'])
            if split=='excluded':rejected[source+':historical_partition_conflict']+=1;continue
            if row['group'] in groups:continue
            groups.add(row['group'])
            if source=='commongen':
                if norm(row['answer']) in answers:continue
                answers.add(norm(row['answer']))
            row=dict(row,split=split,source_id=row['id'])
            quota=quotas[source] if split=='train' else cfg[split+'_per_source']
            rank=-int(fingerprint([cfg['seed'],row['id']]),16);entry=(rank,row['id'],row)
            h=pools[(source,split)]
            if len(h)<quota:heapq.heappush(h,entry)
            elif entry>h[0]:heapq.heapreplace(h,entry)
        for split in ('train','dev','test'):
            need=quotas[source] if split=='train' else cfg[split+'_per_source']
            if len(pools[source,split])!=need:raise ValueError(f'Insufficient {source}/{split}: {len(pools[source,split])}/{need}')
    data={s:dict(commonsense=[],instruction=[]) for s in ('train','dev','test')}
    kinds=('plain','lower','upper','quote','json','bullet','lower_bullet','lower_json')
    for (source,split),entries in sorted(pools.items()):
        for n,(_,_,r) in enumerate(sorted(entries,reverse=True)):
            row=dict(r)
            if source=='commongen':
                kind=kinds[n%len(kinds)];rule,answer=formats(row['answer'],kind)
                row.update(id=row['id']+':'+kind,base_answer=row['answer'],format=kind,answer=answer,prompt=row['prompt']+('\n'+rule if rule else ''))
                assert unpack(answer,kind) is not None
            data[split]['instruction' if source=='commongen' else 'commonsense'].append(row)
    pcfg=read_json(ROOT/'sft/two_turn_640/config.json');parent=parent_rebuild(pcfg)
    # Exact earlier exposures: 768 two-turn targets once + 256 authored targets three times.
    replay=[r for u in range(pcfg['updates']) for r in parent_batch(parent,pcfg,u)]
    assert len(replay)==1536 and len({r['id'] for r in replay})==1024
    data['train']['reading']=replay
    for split in ('dev','test'):
        data[split]['reading']=[dict(r,references=[r['answer']]) for r in parent[split]]
    rng=random.Random(cfg['seed'])
    for family in cfg['per_update']:rng.shuffle(data['train'][family])
    # Train-only social narratives/questions anchor ordinary text predictions.
    data['anchors']=[tok.encode(r['ranking_prompt'].removeprefix('Question: ').removesuffix('\nAnswer:'))[:cfg['anchor_context']] for r in data['train']['commonsense'] if r['source']=='socialiqa']
    rng.shuffle(data['anchors']);assert all(len(x)>1 for x in data['anchors'])
    data['two_turn']=parent
    stats={}
    for split in ('train','dev','test'):
        stats[split]={}
        for family,rows in data[split].items():
            sizes=[encode_stats(tok,r,cfg['context']) for r in rows]
            stats[split][family]=dict(exposures=len(rows),unique=len({r['id'] for r in rows}),answer_tokens=sum(t for _,t in sizes),maximum_tokens=max(n for n,_ in sizes),sources=dict(Counter(r.get('source','authored') for r in rows)),hash=fingerprint(rows))
    groups={s:{r['group'] for f in ('commonsense','instruction') for r in data[s][f]} for s in data if s in ('train','dev','test')}
    assert not(groups['train']&groups['dev'] or groups['train']&groups['test'] or groups['dev']&groups['test'])
    return data,dict(config=fingerprint(cfg),tokenizer=tok.fingerprint,sources=SOURCES,stats=stats,rejections=dict(rejected),anchor_count=len(data['anchors']),anchor_hash=fingerprint(data['anchors']),ids={s:{f:[r['id'] for r in rr] for f,rr in data[s].items()} for s in ('train','dev','test')},limitations='Public TRAIN split subpartitioned by intersection of historical group partitions; excludes known evaluation overlap heuristically, not semantic decontamination. Some source training rows may be repeat exposures from base SFT. Two-turn dev/test and natural probes are reused diagnostics. CommonGen format/coverage is not a semantic correctness score.')
