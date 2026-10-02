"""Reviewed public QA for ranking; exact-content formats; unchanged 768 replay."""
import random
from collections import Counter
from pathlib import Path
from sml_v1.common import read_json,fingerprint
from sft.skill_balance_768.data import (build as previous_build,partition,exclusions,norm,unpack,stems,finite_sentence,formats,encode_stats)
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
TRAIN_REJECTIONS={
 'socialiqa:d9fe5dc5eea8b3477618379624c7c9620193b0d64029d99325c34fdcbcc0d701': 'Question asks about Jordan, absent from context.',
 'socialiqa:007d5d5c825ca855b6a7fdf53f47bf770f8385eb8e701d548ea67805affae3dd': 'Poor alternatives do not establish the stated motivation.',
 'socialiqa:8d3e24725285b561f27ee65af8aea2e34ad3f3c8745a0b15ccf9783a3bfc6b17': 'Only grammatical choice assumes an unsupported reaction.',
 'socialiqa:39bdee084f205ec5465a61f431e761141188dcd8015bad90fe1bbf32cccd1db2': 'Audience sincerity judgment is unsupported by the forced apology.',

 'commonsenseqa:a92c8e76b128e2f73e366af7260622f2626ebc8dd0217ec36365b64223d619d6': 'Car parks need not be outside.',
 'socialiqa:8d353213f77cbb8be43404de1186474b882e5f40770b272b1f8dd3a3570f5674': 'Ambiguous reaction and mismatched feeling options.',
 'socialiqa:1c5cdfc28847ca0cc9a43ae123d14bb806ee55a2c2ca1563f1504448a0f7e1e5': 'Before-action label repeats the action.',
 'socialiqa:14dc6fc91a9c5b8ed2523a23b9b7b23392f9cf762408ed1dd2b75ca7e2760a06': 'Label does not establish the asked causal link.',
 'socialiqa:5872f7cb9a8ff28fd5f7ec86b5c991e168502339e75362bdde5086cccec6e1c9': 'Broken leg does not imply carelessness.',
}
KINDS=('plain','lower','upper','quote','json','bullet','lower_bullet','lower_json')

def format_views(rows):
    out=[]
    for n,row in enumerate(r for r in rows if r.get('turn')==0):
        for j in range(3):
            kind=KINDS[(n*3+j)%len(KINDS)]
            rule,answer=formats(row['answer'],kind)
            # Explicitly a text-preservation exercise. Not a test of world knowledge.
            prompt='Preserve all the content of the following text. Return only the requested version.\nText: '+row['answer']+'\n'
            prompt+=rule or 'Return the text exactly as written, without adding anything.'
            out.append(dict(id=row['id']+':verified-format:'+kind,group=row['group'],source='local-exact-format',
                split=row['split'],prompt=prompt,answer=answer,base_answer=row['answer'],format=kind))
    return out

def build(cfg,cancelled=lambda:False):
    from sml_v1.tokenization import Tokenizer
    oldcfg=read_json(ROOT/'sft/skill_balance_768/config.json')
    data,old_selection=previous_build(oldcfg,cancelled)
    if old_selection!=read_json(ROOT/'sft/skill_balance_768/selection.json'):
        raise ValueError('Original public-source selection changed')
    rng=random.Random(cfg['seed'])
    # Equal count per source, chosen only from the previously reviewed TRAIN pool.
    initial=[];reserve=[]
    for source in ('commonsenseqa','socialiqa'):
        rows=[r for r in data['train']['commonsense'] if r['source']==source]
        initial+=rows[:cfg['updates']*3];reserve+=rows[cfg['updates']*3:]
    rng.shuffle(initial)
    qa=[];counts=Counter()
    for row in initial+reserve:
        source=row['source']
        if row['id'] in TRAIN_REJECTIONS or counts[source]>=cfg['updates']*3:continue
        qa.append(row);counts[source]+=1
    data['train']['commonsense']=qa
    train_formats=format_views(data['two_turn']['new']);rng.shuffle(train_formats)
    data['train']['instruction']=train_formats
    # Replay order and multiplicity are identical to the previous run, consumed
    # in 6-example batches over 256 updates rather than 3 over 512.
    for split in ('dev','test'):
        data[split]['grounded']=format_views(data['two_turn'][split])
    fg={s:{r['group'] for r in (data['train']['instruction'] if s=='train' else data[s]['grounded'])} for s in ('train','dev','test')}
    assert not(fg['train']&fg['dev'] or fg['train']&fg['test'] or fg['dev']&fg['test'])
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    data['anchors']=[tok.encode(r['ranking_prompt'].removeprefix('Question: ').removesuffix('\nAnswer:'))[:cfg['anchor_context']] for r in qa if r['source']=='socialiqa']
    stats={}
    for split in ('train','dev','test'):
        stats[split]={}
        for family,rows in data[split].items():
            sizes=[encode_stats(tok,r,cfg['context']) for r in rows]
            stats[split][family]=dict(exposures=len(rows),unique=len({r['id'] for r in rows}),hash=fingerprint(rows),maximum_tokens=max(n for n,_ in sizes),answer_tokens=sum(t for _,t in sizes),sources=dict(Counter(r.get('source','authored') for r in rows)))
    for family,n in cfg['per_update'].items():
        if len(data['train'][family])!=n*cfg['updates']:raise ValueError('Incomplete '+family)
    groups={s:{r['group'] for f in ('commonsense','instruction') for r in data[s][f]} for s in ('train','dev','test')}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not groups[a]&groups[b]
    return data,dict(config=fingerprint(cfg),tokenizer=tok.fingerprint,previous_selection=fingerprint(old_selection),review_rejections=TRAIN_REJECTIONS,stats=stats,anchor_hash=fingerprint(data['anchors']),
        limitations='Existing development diagnostics are reused. New format views use held-out settings but shared templates. Format exactness is not conversational understanding. CommonGen is evaluation-only; QA train items may have appeared in base SFT. No target benchmark items intentionally trained; inherited overlap filtering is heuristic.')
