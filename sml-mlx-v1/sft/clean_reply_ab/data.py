"""Original reviewed tasks plus pinned, previously consumed skill rehearsal.
Public selections are rebuilt into RAM. No raw corpus or replay text is cached.
"""
from pathlib import Path
import random,re
from collections import Counter
from sml_v1.common import read_json,fingerprint
from sml_v1.tokenization import Tokenizer
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent

def original():
    groups=read_json(DIR/'authored.json');new=[];paraphrases=[]
    for g in groups:
        for i,p in enumerate(g['train_prompts']):
            new.append(dict(id=f"original:{g['group']}:{i}",group=g['group'],family=g['family'],prompt=p,answer=g['answer'],role='new'))
        paraphrases.append(dict(id=f"paraphrase:{g['group']}",group=g['group'],family=g['family'],prompt=g['paraphrase'],answer=g['answer']))
    return new,paraphrases

def build(cfg,cancelled=lambda:False,review=False):
    from sft.base_curriculum_v1 import data as parent
    from sft.natural_control import data as old
    previous,manifest=parent.build(read_json(ROOT/'sft/base_curriculum_v1/config.json'),cancelled)
    if fingerprint(manifest)!=fingerprint(read_json(ROOT/'sft/base_curriculum_v1/selection.json')):raise ValueError('Parent selection changed')
    new,para=original();dev=read_json(DIR/'dev.json');test=read_json(DIR/'test.json')
    original_texts={old.norm(r['prompt']) for r in new+para+dev+test};replay=[]
    for family,count in cfg['replay_counts'].items():
        pool=[r for r in previous['train'] if r['family']==family and len(r.get('content',r['prompt']).split())<=130 and len(r['prompt'].split())<=180 and len(r['answer'].split())<=32 and old.norm(r['prompt']) not in original_texts]
        pool.sort(key=lambda r:fingerprint([cfg['seed'],'replay',r['id']]))
        if (DIR/'replay_review.json').exists():
            chosen=read_json(DIR/'replay_review.json')['selected_ids'][family];by={r['id']:r for r in pool}
            if not set(chosen)<=by.keys():raise ValueError('Reviewed source IDs unavailable')
            rows=[by[i] for i in chosen]
        else:rows=pool[:count]
        if len(rows)!=count:raise ValueError(f'Not enough replay: {family} {len(rows)}')
        replay.extend(dict(r,role='rehearsal') for r in rows)
    if review:
        for r in replay:print('[replay-review]',__import__('json').dumps({k:r[k] for k in ('id','family','prompt','answer')},ensure_ascii=False),flush=True)
    parent_groups={r['group'] for split in ('dev','test') for r in previous[split]}
    assert not {r['group'] for r in replay}&parent_groups
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    for row in new+replay+para+dev+test:
        prefix='User: '+row['prompt']+'\nAssistant:';head=tok.encode(prefix);ids=tok.encode(prefix+' '+row['answer'])
        if ids[:len(head)]!=head or len(ids)>cfg['context']:raise ValueError('Token boundary or length')
        row['answer_targets']=len(ids)-len(head)+1
    data=dict(new=new,replay=replay,train=new+replay,train_probes=[new[i*2] for i in range(128)],dev=dev,test=test,paraphrases=para,
              replay_probes=[])
    data['replay_probes']=[r for fam in cfg['replay_counts'] for r in [r for r in replay if r['family']==fam][:8]]
    data['retention']=[r for fam in cfg['replay_counts'] for r in [r for r in previous['dev'] if r['family']==fam][:8]]
    stats={s:dict(examples=len(data[s]),hash=fingerprint(data[s]),targets=sum(r.get('answer_targets',0) for r in data[s])) for s in ('train','new','replay','dev','test','paraphrases','retention')}
    result=dict(stats=stats,parent_selection=fingerprint(manifest),replay_ids={f:[r['id'] for r in replay if r['family']==f] for f in cfg['replay_counts']},
        sources={k:manifest[k] for k in ('reading_source','skill_sources')},original_groups=128,original_training_phrasings=256,
        limitations='Authored assistant examples, not human-collected conversations. Paraphrase checks share task content with training and are transfer diagnostics, not independent tasks. 32 new development tasks and 32 reserved tasks. Rehearsal uses existing annotated parent TRAIN selections; no stronger teacher. No public benchmark items deliberately authored or copied.')
    parent._COLLECTION=None
    return data,result

def rebuild(cfg,cancelled=lambda:False):
    data,manifest=build(cfg,cancelled)
    if manifest!=read_json(DIR/'selection.json'):raise ValueError('Selection failed exact reproduction')
    return data

def batch_at(data,cfg,update):
    if not 0<=update<cfg['updates']:raise ValueError('Update outside run')
    epoch,offset=divmod(update,32)
    new=list(data['new']);replay=list(data['replay']);rng=random.Random(cfg['seed']+epoch)
    rng.shuffle(new);rng.shuffle(replay)
    rows=new[offset*8:offset*8+8]+replay[offset*4:offset*4+4];rng.shuffle(rows)
    if len(rows)!=12:raise ValueError('Incomplete balanced batch')
    return rows
