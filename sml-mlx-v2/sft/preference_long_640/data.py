"""Pinned TRAIN preferences; prompt groups and public evaluations stay separated."""
from collections import Counter
from pathlib import Path
import random
import re
from types import SimpleNamespace
from sml_v2.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v2.tokenization import Tokenizer
from sft.grounded_rank_384 import data as base
from sft.intact_smoltalk_base_pilot.data import turns

ROOT=Path(__file__).resolve().parents[2]
DIR=Path(__file__).resolve().parent

def encode_reply(tok,prompt,answer,context):
    prefix='User: '+prompt+'\nAssistant:';head=tok.encode(prefix);full=tok.encode(prefix+' '+answer)+[tok.eos]
    if not head or full[:len(head)]!=head or not answer.strip() or len(full)-1>context:
        raise ValueError('Invalid complete answer boundary/context')
    return dict(x=full[:-1],y=[-100]*(len(head)-1)+full[len(head):],targets=len(full)-len(head))


def cycling(rows,count,seed):
    if not rows:raise ValueError('Empty training pool')
    output=[];epoch=0
    while len(output)<count:
        part=list(rows);random.Random(seed+epoch).shuffle(part);output.extend(part);epoch+=1
    return output[:count]


def build(cfg):
    import pyarrow.parquet as pq
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    old_path=base.DIR/'prepared.json';old=read_json(old_path)
    denied,dgrams,exclude_paths,_=base.exclusions()
    # Existing private dev/test questions are exclusion-only inputs.
    for split in ('dev','test'):
        for rows in old[split].values():
            for row in rows:
                denied.add(base.norm(row['prompt']));dgrams|=base.grams(row['prompt'])
    from .verified import rows as verified_rows
    selected={split:verified_rows(split,cfg['preference_'+split+'_pairs'],cfg['seed'],denied,dgrams) for split in ('train','dev','test')}
    pools=selected;rejected=Counter()
    for rr in selected.values():
        for row in rr:
            row['chosen_tokens']=len(tok.encode(row['answer']));row['rejected_tokens']=len(tok.encode(row['rejected']))
            encode_reply(tok,row['prompt'],row['answer'],cfg['context']);encode_reply(tok,row['prompt'],row['rejected'],cfg['context'])
    # Preserve the old group partition; only source TRAIN rows enter gradients.
    rank_pools={};held={r['source_id'] for r in pq.read_table(ROOT/'evaluation/full_benchmarks/_cache/raw/hellaswag.parquet',columns=['source_id']).to_pylist()}
    for name in ('arc_easy','arc_challenge','piqa','hellaswag'):
        pp={s:[] for s in pools}
        for i,raw in enumerate(base.raw_rows(name)):
            row=base.ranking_row(name,raw,i)
            if name=='hellaswag' and row['origin_group'] in held:continue
            if base.overlaps([row['ranking_prompt']]+row['choices'],denied,dgrams):continue
            if len(set(row['choices']))!=len(row['choices']):continue
            head=tok.encode(row['ranking_prompt'])
            if not head or any(not c or tok.encode(row['ranking_prompt']+' '+c)[:len(head)]!=head or len(tok.encode(row['ranking_prompt']+' '+c))>cfg['context'] for c in row['choices']):continue
            pp[row['split']].append(row)
        rank_pools[name]=pp
    data={s:{} for s in pools};stats={}
    train=data['train']
    first=sorted(selected['train'],key=lambda r:(r['difficulty'],fingerprint([cfg['seed'],r['id']])))
    second=cycling(selected['train'],len(selected['train']),cfg['seed']+91)
    train['preference']=first+second
    train['instruction']=cycling(old['train']['instruction'],cfg['updates'],cfg['seed']+1)
    replay=[]
    for row in old['train']['replay']:
        r=dict(row,messages=row['messages']+[dict(role='assistant',content=row['answer'])])
        try:turns(tok,r,cfg['context'])
        except OverflowError:continue
        replay.append(r)
    train['replay']=cycling(replay,cfg['updates'],cfg['seed']+2)
    selected_ranking={name:cycling(base.choose(pp['train'],len({r['group'] for r in pp['train']}),cfg['seed']+3),cfg['updates'],cfg['seed']+4+i) for i,(name,pp) in enumerate(rank_pools.items())}
    train['ranking']=[r for batch in zip(*selected_ranking.values()) for r in batch]
    for split in ('dev','test'):
        data[split]['preference']=selected[split]
        for family in ('grounded','instruction'):
            data[split][family]=old[split][family]
        data[split]['ranking']=[r for pp in rank_pools.values() for r in base.choose(pp[split],64,cfg['seed']+5)]
        data[split]['replay']=[dict(r,messages=r['messages']+[dict(role='assistant',content=r['answer'])]) for r in old[split]['replay']]
        for row in data[split]['replay']:turns(tok,row,cfg['context'])
    # The ranking screen above excludes old dev prompts even from dev candidates,
    # so new choice diagnostics differ from earlier selected dev examples.
    data['anchors']=[tok.encode(r['ranking_prompt']+' '+r['answer'])[:cfg['anchor_context']] for r in rank_pools['hellaswag']['train'][:2048]]
    data['prose_diagnostic']=old['prose_diagnostic']
    for a,b in (('train','dev'),('train','test'),('dev','test')):
        aa={r['group'] for rows in data[a].values() for r in rows};bb={r['group'] for rows in data[b].values() for r in rows}
        if aa&bb:raise ValueError('Cross-split group overlap')
    for split in pools:
        stats[split]={f:dict(exposures=len(rows),unique_ids=len({r['id'] for r in rows}),sources=dict(Counter(r['source'] for r in rows))) for f,rows in data[split].items()}
    samples=dict(preference=[next(r for r in selected['train'] if r['task']==task) for task in sorted({r['task'] for r in selected['train']})],instruction=train['instruction'][:4],replay=train['replay'][:4],ranking=train['ranking'][:8])
    sources=[old_path]+[base.DIR/'_sources'/f'{n}.parquet' for n in rank_pools]+exclude_paths
    selection=dict(config=fingerprint(cfg),tokenizer=tok.fingerprint,preference_source=dict(type="original controlled tasks",families=16,verification="Correct complete targets and deliberately incorrect counterexamples; all constructed pairs checked",curriculum="First pass easier tasks first, second pass shuffled"),ranking_sources={n:base.SOURCES[n] for n in rank_pools},pool_counts={s:len(v) for s,v in pools.items()},ranking_pool_counts={n:{s:len(v) for s,v in pp.items()} for n,pp in rank_pools.items()},rejections=dict(rejected),stats=stats,inputs={str(p):file_sha256(p) for p in set(sources)},sample_sha256=fingerprint(samples),limitations='Original fictional scenes and deterministic transformations, not new world-knowledge facts. Negative responses are constructed errors rather than model samples; family/template transfer is not proof of broad instruction following. Two passes (curriculum then shuffle); instruction/conversation/ranking rehearsal intentionally repeats TRAIN rows. Private reading/instruction/chat diagnostics are reused. Public benchmark exclusion is exact/13-word heuristic, not proof of zero contamination. No public benchmark or reserved-test gradients. UltraFeedback was inspected and rejected for this experiment; none of it is used.')
    return data,selection,samples


def prepare(cfg):
    data,selection,samples=build(cfg)
    atomic_json(DIR/'prepared.json',data);atomic_json(DIR/'selection.json',selection);atomic_json(DIR/'review_samples.json',samples)
    return selection
