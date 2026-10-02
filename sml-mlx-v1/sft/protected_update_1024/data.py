"""Fixed human choice labels and previously consumed reply targets; identical arms."""
import random
from collections import Counter
from pathlib import Path
from sml_v1.common import read_json, fingerprint
from sft.skill_balance_768.data import build as previous_build, exclusions, partition, encode_stats
from sft.skill_recovery_768.data import TRAIN_REJECTIONS, format_views
ROOT=Path(__file__).resolve().parents[2]; DIR=Path(__file__).resolve().parent

def audit(data):
    groups={s:{r['group'] for rows in data[s].values() for r in rows} for s in ('train','dev','test')}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        if groups[a]&groups[b]:raise ValueError('Overlapping source groups: '+a+'/'+b)
    if any(r['split']!='train' for rr in data['train'].values() for r in rr):raise ValueError('Non-training exposure')

def build(cfg,cancelled=lambda:False):
    from sml_v1.tokenization import Tokenizer
    oldcfg=read_json(ROOT/'sft/skill_balance_768/config.json')
    # Same historical TRAIN and QA dev/test partition; these diagnostics are reused.
    oldcfg.update(dev_per_source=cfg['dev_per_source'],test_per_source=cfg['test_per_source'])
    data,old_selection=previous_build(oldcfg,cancelled)
    rng=random.Random(cfg['seed']); train=[]; rejections=dict(TRAIN_REJECTIONS)
    if (DIR/'sample_review_record.json').exists():
        rejections.update({k:v for k,v in read_json(DIR/'sample_review_record.json').items() if v!='accepted'})
    for source in ('commonsenseqa','socialiqa'):
        pool=[r for r in data['train']['commonsense'] if r['source']==source]
        rng.shuffle(pool); pool=[r for r in pool if r['id'] not in rejections]; need=cfg['updates']*2
        if len(pool)<need:raise ValueError('Insufficient accepted TRAIN '+source)
        train.append(pool[:need])
    # Every update sees two questions from each source, in fixed order across arms.
    data['train']['commonsense']=[r for i in range(cfg['updates']) for pool in train for r in pool[2*i:2*i+2]]
    formats=format_views(data['two_turn']['new'])
    original_replay=[dict(r,split='train') for r in data['train']['reading']]
    allowed_replay={r['id'] for key in ('new','replay') for r in data['two_turn'][key]}
    if any(r['id'] not in allowed_replay for r in original_replay):raise ValueError('Replay provenance mismatch')
    for family,pool in [('instruction',formats),('reading',original_replay)]:
        need=cfg['updates']*cfg['per_update'][family]; rows=[]
        while len(rows)<need:
            epoch=list(pool);rng.shuffle(epoch);rows+=epoch
        data['train'][family]=rows[:need]
    for s in ('dev','test'):
        data[s]['grounded']=format_views(data['two_turn'][s])
    audit(data)
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    data['anchors']=[tok.encode(r['ranking_prompt'].removeprefix('Question: ').removesuffix('\nAnswer:'))[:cfg['anchor_context']] for r in data['train']['commonsense'] if r['source']=='socialiqa']
    # Parent selection is reconstructed by its deterministic selection recipe only
    # in historical metadata; do not claim every QA row is new to the base lineage.
    stats={}
    for s in ('train','dev','test'):
        stats[s]={}
        for f,rows in data[s].items():
            sizes=[encode_stats(tok,r,cfg['context']) for r in rows]
            stats[s][f]=dict(exposures=len(rows),unique=len({r['id'] for r in rows}),groups=len({r['group'] for r in rows}),hash=fingerprint(rows),maximum_tokens=max(n for n,_ in sizes),answer_tokens=sum(t for _,t in sizes),sources=dict(Counter(r.get('source','authored') for r in rows)))
    neutral={k:v for k,v in cfg.items() if k!='arm'}
    return data,dict(config=fingerprint(cfg),data_config=fingerprint(neutral),tokenizer=tok.fingerprint,stats=stats,anchor_hash=fingerprint(data['anchors']),source_selection=fingerprint(old_selection),rejections=rejections,
      limitations='Same TRAIN data/order in both arms. 2048 QA rows; some previously consumed. 768 format views repeated twice; 1536 replay exposures repeated twice (1024 unique). Historical partition intersection and heuristic benchmark-overlap rejection, not complete decontamination. Reading/prose/CommonGen probes reused; all QA dev/test diagnostics are reused.')
