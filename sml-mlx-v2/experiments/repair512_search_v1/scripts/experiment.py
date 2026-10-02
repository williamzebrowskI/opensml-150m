"""User-authorized short experiments from immutable repair512.

Each candidate owns its data, rollouts, checkpoints, evaluations and benchmarks.
Unsuccessful candidate directories are removed with a compact comparison receipt.
Public benchmark answers never enter the training objective or training selection.
"""
import argparse
import gc
import importlib.util
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PUBLIC = ROOT/'experiments/public_repair_384_v1'
sys.path.insert(0, str(PUBLIC/'scripts'))
from sml_v2.common import atomic_json, file_sha256, fingerprint, read_json, lock

PARENT = PUBLIC/'runs/sft/step_0000512_6972bbf9be08'
PROTECTED = {
    str(PARENT/'model.safetensors'): '465ce42aad2e7c765f169aa75aa4124ec03fec17b46e6f5eabf75c2c4a8d4772',
    str(ROOT/'experiments/unified_text_v2_sft_v1/runs/sft/step_0000384_807628c9dbdc/model.safetensors'): 'fcc23cd8a204762f05ee43bd84c88846383dccfda40eec06cd390a59666656db',
    str(ROOT/'runs/sft_text_followup_512_v1/step_0000768_10f5ea2a3354/model.safetensors'): '309ad719dc8569e482444b47b05f3fd24e02e7b7470dad2af132a976bf11ca02',
    str(ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors'): 'ca9bd5d82005f28e77f319d3a5a29f29fb4e4f86e7ceea049f01162fd8aae80f',
}

def verify_parent():
    for p,h in PROTECTED.items():
        if file_sha256(p)!=h: raise ValueError('Protected weights changed: '+p)

def selective_repetition_mask(tokens,reference,special,ngram=4,allowance=2):
    """Mask occurrences beyond both the allowance and clean reference count."""
    allowed=Counter(tuple(reference[i:i+ngram]) for i in range(len(reference)-ngram+1))
    seen=Counter();mask=[False]*len(tokens)
    for end in range(ngram,len(tokens)+1):
        span=tuple(tokens[end-ngram:end])
        if any(t in special for t in span):continue
        seen[span]+=1
        if seen[span]>max(allowance,allowed[span]):
            for i in range(end-ngram,end):mask[i]=True
    return mask

def public_module():
    spec=importlib.util.spec_from_file_location('parent_public_launcher',PUBLIC/'scripts/launch.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def freeze(c,cfg):
    files=[Path(__file__),c/'config.json',c/'data.json',PUBLIC/'config.json',
           PUBLIC/'data/prepared.json',PUBLIC/'runs/sft/evaluations/update_00128.json']
    if (c/'negatives.json').exists():files.append(c/'negatives.json')
    if (c/'ranking.json').exists():files.append(c/'ranking.json')
    if (c/'format_pairs.json').exists():files += [c/'format_pairs.json',Path(__file__).with_name('format_preferences.py')]
    if cfg.get('balanced_instructions'):files.append(Path(__file__).with_name('balanced_instructions.py'))
    if cfg.get('data_focused'):
        files += [Path(__file__).with_name('data_focused.py'), c/'source_pins.json', c/'selection.json']
        files += list((c/'raw').glob('*.parquet'))
    if cfg.get('dataset_plan'):
        files += [Path(__file__).with_name('dataset_trials.py'), Path(__file__).with_name('short_constraints.py'), Path(__file__).with_name('human_dialogue.py'), Path(__file__).with_name('familiar_dialogue.py'), Path(__file__).with_name('data_focused.py'), c/'source_pins.json', c/'selection.json']
        files += list((c/'raw').glob('*'))
        if cfg['dataset_plan'] in ('science-replay','mixed-public'):
            files += [Path(__file__).with_name('science_dialogue.py'), ROOT/'experiments/unified_text_v2_sft_v1/data/prepared.json']
        if cfg['dataset_plan']=='mixed-public':files += [Path(__file__).with_name('mixed_public.py')]
        if cfg['dataset_plan']=='breadth-stable':files += [Path(__file__).with_name('breadth_stable.py')]
    for folder in ('sml_v2','sft/conversation_foundation_v1','sft/text_followup_512_v1',
                   'sft/skill_balance','sft/transfer_control','sft/intact_smoltalk_base_pilot',
                   'sft/conversation_b','sft/dpo_chat_768_v1'):
        files+=list((ROOT/folder).glob('*.py'))
    files+=list((PUBLIC/'scripts').glob('*.py'))
    base=ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json'
    files+=[base]+list((ROOT/'tokenizer/bytebpe32k_v1').glob('*'))
    return dict(config=cfg,architecture=file_sha256(base),
                protected={**PROTECTED,**{str(p):file_sha256(p) for p in files if p.is_file()}})

def verify_contract(frozen):
    for p,h in frozen['protected'].items():
        if file_sha256(p)!=h:raise ValueError('Experiment input changed: '+p)

def prepare(c,args):
    if c.exists() and not ((getattr(args,'data_focused',False) or getattr(args,'dataset_plan',None)) and not (c/'config.json').exists()):raise ValueError('Candidate already exists')
    c.mkdir(parents=True,exist_ok=True)
    cfg=read_json(PUBLIC/'config.json')
    cfg.update(source_bundle=str(PARENT.relative_to(ROOT)),source_step=512,
               source_model_sha256=PROTECTED[str(PARENT/'model.safetensors')],
               updates=args.updates,peak_lr=args.lr,final_lr=args.lr/10,
               weight_decay=0.,ul_weight=args.ul,dpo_weight=args.dpo,dpo_beta=.1,weighting=args.weighting,
               replay_only=args.replay_only,ranking_weight=args.ranking,kl_weight=args.kl,on_policy=args.on_policy,
               reference_aware=args.reference_aware,instruction_weight=args.instruction_weight,
               train_last_blocks=args.train_last_blocks,
               balanced_instructions=args.balanced_instructions,instruction_edge_weight=args.instruction_edge_weight,
               maximum_instruction_word_count=args.instruction_word_count,
               policy_kl_weight=args.policy_kl,
               ce_weight=args.ce_weight,
               train_final_norm=args.train_final_norm,
               train_all_norms=args.train_all_norms,
               freeze_embeddings=args.freeze_embeddings,
               generated_history=args.generated_history,
               repetition_ngram=args.repetition_ngram,repetition_allowance=args.repetition_allowance,
               format_preference_weight=args.format_preference,format_preference_beta=5.,
               seed=202610011,batch_conversations=16,evaluation_updates=[0,args.updates])
    if args.data_focused:
        assert args.updates==256 and args.ul==args.dpo==args.ranking==args.kl==args.policy_kl==args.format_preference==0
        from data_focused import prepare as prepare_fresh
        prepare_fresh(c,cfg)
        return
    if args.dataset_plan:
        assert args.dpo==args.ranking==args.policy_kl==args.format_preference==0
        if args.dataset_plan not in ('mixed-public','breadth-stable'):assert args.ul==args.kl==0
        from dataset_trials import prepare as prepare_dataset
        cfg.update(dataset_plan=args.dataset_plan, fresh_per_batch=args.fresh_per_batch)
        prepare_dataset(c,cfg)
        return
    raw=read_json(PUBLIC/'data/prepared.json')['data']
    sources=list(cfg['training_counts'])
    # Each update: two previously consumed and two unused examples per source.
    # Parent512 consumed only public-repair updates1..128 (first2048 rows).
    pools={s:([r for r in raw['train'][:2048] if r['source']==s],
              [r for r in raw['train'][2048:] if r['source']==s]) for s in sources}
    rng=random.Random(cfg['seed'])
    for pair in pools.values():
        for rows in pair:rng.shuffle(rows)
    instruction_rows=[];instruction_selection=None
    if args.balanced_instructions:
        assert args.replay_only
        from balanced_instructions import pool,select
        candidates,inventory=pool(768,args.instruction_word_count)
        instruction_rows,instruction_selection=select(candidates,args.updates*4,cfg['seed'])
        cfg['source_caps']['instructions']=768
        atomic_json(c/'instruction_selection.json',dict(inventory=inventory,selection=instruction_selection))
    train=[]
    for u in range(cfg['updates']):
        batch=[]
        for s in sources:
            if args.balanced_instructions and s=='instructions':batch+=instruction_rows[4*u:4*u+4]
            elif args.replay_only:
                offset=(4*u)%len(pools[s][0]);batch+=pools[s][0][offset:offset+4]
            else:
                for rows in pools[s]:batch+=rows[2*u:2*u+2]
        rng.shuffle(batch);train+=batch
    expected_unique=min(args.updates*16,2048) if args.replay_only else args.updates*16
    if args.balanced_instructions:expected_unique=min(args.updates*12,1536)+instruction_selection['unique_instruction_conversations']
    assert len(train)==args.updates*16 and len({r['group'] for r in train})==expected_unique
    assert not {r['group'] for r in train}&{r['group'] for split in ('dev','test') for r in raw[split]}
    from sml_v2.tokenization import Tokenizer
    from sft.conversation_foundation_v1.data import turns
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    encoded=[e for r in train for e in turns(tok,r,cfg['context'])]
    for e in encoded:
        assert len(e['x'])==len(e['y'])<=2048 and e['y'][-1]==tok.eos
        assert sum(t!=-100 for t in e['y'])==e['targets']
    atomic_json(c/'config.json',cfg)
    atomic_json(c/'data.json',dict(train=train,dev=raw['dev'],test=raw['test']))
    if args.format_preference:
        from format_preferences import build
        atomic_json(c/'format_pairs.json',build(train,args.updates,cfg['seed'],tok,cfg['context']))
    consumed_groups={r['group'] for r in raw['train'][:2048]}
    if args.ranking:
        from sft.conversation_foundation_v1.data import exclusions,grams,norm
        denied,dgrams=exclusions()
        held=set().union(*(grams(m['content']) for split in ('dev','test') for r in raw[split] for m in r['messages'] if m['role']=='user'))
        held.update(g for split in ('dev','test') for r in read_json(ROOT/'sft/text_followup_512_v1/prepared.json')['data'][split]
                    for m in r['messages'] if m['role']=='user' for g in grams(m['content']))
        original=ROOT/'sft/preference_long_640/prepared.json'
        ranks=read_json(original)['train']['ranking'];seen=set();selected={}
        for source in ('arc_easy','arc_challenge','piqa','hellaswag'):
            pool=[]
            for r in ranks:
                if r['source']!=source or r['source_split']!='train' or r['group'] in seen:continue
                if norm(r['prompt']) in denied or grams(r['prompt'])&(dgrams|held):continue
                assert r['choices'][r['gold']]==r['answer']
                pool.append(r);seen.add(r['group'])
            rng.shuffle(pool);assert len(pool)>=args.updates
            selected[source]=pool[:args.updates]
        rank_rows=[[selected[s][u] for s in selected] for u in range(args.updates)]
        atomic_json(c/'ranking.json',dict(updates=rank_rows,input=str(original),sha256=file_sha256(original),
                                      official_training_splits_only=True,benchmark_and_holdout_overlap_excluded=True))
    atomic_json(c/'selection.json',dict(training_conversations=len(train),sources=dict(Counter(r['source'] for r in train)),
        assistant_turns=len(encoded),unique_conversations=expected_unique,
        replay_passes=len(train)/expected_unique,parent_replay_fraction=sum(r['group'] in consumed_groups for r in train)/len(train),public_benchmarks_used_for_training=False,
        input_sha256=file_sha256(PUBLIC/'data/prepared.json'),config=fingerprint(cfg)))
    print('[prepared]',c.name,len(train),'conversations',flush=True)

def collect(c,cfg,data):
    if (c/'negatives.json').exists():return
    if cfg.get('on_policy') or (not cfg['ul_weight'] and not cfg.get('dpo_weight')):
        atomic_json(c/'negatives.json',dict(parent_sha256=cfg['source_model_sha256'],updates=[[] for _ in range(cfg['updates'])],negative_tokens=0))
        return
    from sft.conversation_foundation_v1 import engine
    from sft.conversation_foundation_v1.data import visible_prefix,turns
    from sft.conversation_b.engine import repetition_mask
    from sft.transfer_control.engine import arrays
    from sft.dpo_chat_768_v1.engine import sequence_logps
    from sft.dpo_chat_768_v1.data import encode_reply
    import mlx.core as mx
    b=engine.load(cfg);records=[]
    for u in range(cfg['updates']):
        batch=data['train'][u*16:(u+1)*16]
        # Training-only greedy rollouts; include zero-repeat rows in normalization.
        rows=[next(r for r in batch if r['source']==s) for s in ('chat','human')]
        negative=[]
        for r in rows:
            index=max(i for i,m in enumerate(r['messages']) if m['role']=='user')
            messages=r['messages'][:index+1]
            g=engine.generate(b,messages,192)
            tokens=g['tokens'];head=b.encode(visible_prefix(messages))
            special=(b.pad,b.eos,2,3)
            if cfg.get('reference_aware'):
                ref=[t for t in turns(b.tokenizer,r,cfg['context'])[-1]['y'] if t!=-100]
                mask=selective_repetition_mask(tokens,ref,special,cfg.get('repetition_ngram',4),cfg.get('repetition_allowance',2))
            else:mask=repetition_mask(tokens,4,special)
            # Select only later repeated spans; EOS, prompt and clean reference excluded.
            labels=[t if v else -100 for t,v in zip(tokens,mask)]
            e=dict(x=(head+tokens)[:-1],y=[-100]*(len(head)-1)+labels,targets=sum(mask))
            assert len(e['x'])==len(e['y'])<=cfg['context']
            item=dict(id=r['id'],source=r['source'],generation=g,encoded=e)
            if cfg.get('dpo_weight') and sum(mask)>0:
                pair=[encode_reply(b.tokenizer,messages,text,cfg['context']) for text in (r['messages'][-1]['content'],g['text'])]
                x,y=arrays(pair,b.pad);ref,_=sequence_logps(b.model,x,y);mx.eval(ref)
                item.update(pair=pair,reference_logps=ref.tolist())
            negative.append(item)
        records.append(negative)
        if u%8==0:print('[rollouts]',u+1,'/',cfg['updates'],flush=True)
    del b;gc.collect()
    import mlx.core as mx
    mx.clear_cache()
    atomic_json(c/'negatives.json',dict(parent_sha256=cfg['source_model_sha256'],updates=records,
                                     negative_tokens=sum(n['encoded']['targets'] for rr in records for n in rr)))
    verify_parent()

def history_reply(b,row,limit):
    """Roll training user messages forward with this model's own prior replies."""
    from sft.conversation_foundation_v1.engine import generate
    last=max(i for i,m in enumerate(row['messages']) if m['role']=='user')
    messages=[];reply=None
    for index,m in enumerate(row['messages'][:last+1]):
        if m['role']=='assistant':
            assert reply is not None
            messages.append(dict(role='assistant',content=reply['text']))
        else:
            messages.append(dict(m))
            if m['role']=='user':reply=generate(b,messages,limit if index==last else min(limit,64))
    assert messages[-1]['role']=='user' and reply is not None
    return messages,reply

def update(b,opt,rows,negative,cfg,step,anchor=None,ranking=None,format_pairs=None):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from sft.conversation_foundation_v1.data import turns
    from sft.transfer_control.engine import arrays
    from sft.conversation_b.engine import ul_gradients
    from sft.skill_balance.engine import learning_rate
    from sml_v2.pretrain import clip_gradients
    if cfg.get('on_policy'):
        from sft.conversation_foundation_v1.engine import generate
        from sft.conversation_foundation_v1.data import visible_prefix
        from sft.conversation_b.engine import repetition_mask
        negative=[]
        for source in ('chat','human'):
            row=next(r for r in rows if r['source']==source and (not cfg.get('negative_replay_only') or r.get('parent_replay')))
            i=max(i for i,m in enumerate(row['messages']) if m['role']=='user')
            if cfg.get('generated_history'):
                messages,g=history_reply(b,row,192)
            else:
                messages=row['messages'][:i+1];g=generate(b,messages,192)
            head=b.encode(visible_prefix(messages));tokens=g['tokens']
            special=(b.pad,b.eos,2,3)
            if cfg.get('reference_aware'):
                ref=[t for t in turns(b.tokenizer,row,cfg['context'])[-1]['y'] if t!=-100]
                mask=selective_repetition_mask(tokens,ref,special,cfg.get('repetition_ngram',4),cfg.get('repetition_allowance',2))
            else:mask=repetition_mask(tokens,4,special)
            labels=[t if v else -100 for t,v in zip(tokens,mask)]
            e=dict(x=(head+tokens)[:-1],y=[-100]*(len(head)-1)+labels,targets=sum(mask))
            assert len(e['x'])==len(e['y'])<=cfg['context']
            negative.append(dict(encoded=e))
    b.model.train();total=None;ce_total=0.;targets=0
    enc=[turns(b.tokenizer,r,cfg['context']) for r in rows]
    scales=[cfg.get('instruction_weight',1.) if r['source']=='instructions' else 1. for r in rows]
    total_targets=sum(s*e['targets'] for s,ee in zip(scales,enc) for e in ee)
    row_total=sum(scales)
    def supervised_objective(x,y,p,token_weights):
        losses=nn.losses.cross_entropy(b.model(x)['logits'][0,p].astype(mx.float32),y[0,p],reduction='none')
        return (losses*token_weights).sum()/token_weights.sum()
    fn=nn.value_and_grad(b.model,supervised_objective)
    weight_sum=0.
    for row,scale,ee in zip(rows,scales,enc):
        for e in ee:
            w=(scale*e['targets']/total_targets if cfg['weighting']=='token' else scale/(row_total*len(ee)))
            weight_sum+=w
            x,y=arrays([e],b.pad);p=mx.array([i for i,t in enumerate(e['y']) if t!=-100],dtype=mx.int32)
            weights=[1.]*e['targets']
            if row.get('fresh_balanced') and cfg.get('instruction_edge_weight',1.)!=1.:
                for index in set(range(min(16,len(weights))))|set(range(max(0,len(weights)-16),len(weights))):
                    weights[index]=cfg['instruction_edge_weight']
            token_weights=mx.array(weights,dtype=mx.float32)
            loss,g=fn(x,y,p,token_weights);g=tree_map(lambda t:t.astype(mx.float32)*w,g)
            total=g if total is None else tree_map(lambda a,z:a+z,total,g)
            mx.eval(loss,total);ce_total+=float(loss.item())*w;targets+=e['targets']
    assert abs(weight_sum-1.)<1e-8
    total=tree_map(lambda t:cfg.get('ce_weight',1.)*t,total)
    ul=0.
    if cfg['ul_weight']:
        v,g=ul_gradients(b.model,[n['encoded'] for n in negative],{'unlikelihood':{'epsilon':1e-6}})
        mx.eval(v);ul=float(v.item())
        if g is not None:total=tree_map(lambda a,z:a+cfg['ul_weight']*z,total,g)
    dpo=0.
    if cfg.get('dpo_weight'):
        from sft.dpo_chat_768_v1.engine import sequence_logps,preference_terms
        fn=nn.value_and_grad(b.model,lambda x,y,ref:preference_terms(
            *sequence_logps(b.model,x,y),ref,cfg['dpo_beta'])[0])
        for n in negative:
            if 'pair' not in n:continue
            x,y=arrays(n['pair'],b.pad);ref=mx.array(n['reference_logps'],dtype=mx.float32)
            loss,g=fn(x,y,ref)
            total=tree_map(lambda a,z:a+cfg['dpo_weight']*z/len(negative),total,g)
            mx.eval(loss,total);dpo+=float(loss.item())/len(negative)
    format_loss=0.
    if cfg.get('format_preference_weight'):
        from sft.dpo_chat_768_v1.engine import sequence_logps
        anchor.model.eval()
        beta=cfg['format_preference_beta']
        def objective(x,y,chars,ref):
            lp,_=sequence_logps(b.model,x,y)
            delta=lp/chars-mx.stop_gradient(ref)
            return mx.logaddexp(mx.array(0.),-beta*(delta[0]-delta[1]))
        fn=nn.value_and_grad(b.model,objective)
        for pair in format_pairs:
            x,y=arrays(pair['encoded'],b.pad)
            chars=mx.array([max(1,len(pair[k])) for k in ('chosen','rejected')],dtype=mx.float32)
            ref,_=sequence_logps(anchor.model,x,y);ref=mx.stop_gradient(ref/chars);mx.eval(ref)
            loss,g=fn(x,y,chars,ref)
            total=tree_map(lambda a,z:a+cfg['format_preference_weight']*z/len(format_pairs),total,g)
            mx.eval(loss,total);format_loss+=float(loss.item())/len(format_pairs)
    rank_loss=0.
    if cfg.get('ranking_weight'):
        from sft.skill_balance.engine import choice_arrays
        # Character-normalized likelihood, matching the primary MC metric.
        def rank_objective(x,y,chars,gold):
            z=b.model(x)['logits'].astype(mx.float32)
            ce=nn.losses.cross_entropy(z,mx.maximum(y,0),reduction='none')
            scores=-(ce*(y!=-100)).sum(axis=1)/chars/.25
            return mx.logsumexp(scores)-scores[gold]
        fn=nn.value_and_grad(b.model,rank_objective)
        for r in ranking:
            x,y=choice_arrays(b,r);chars=mx.array([len(t) for t in r['choices']],dtype=mx.float32)
            loss,g=fn(x,y,chars,r['gold'])
            total=tree_map(lambda a,z:a+cfg['ranking_weight']*z/len(ranking),total,g)
            mx.eval(loss,total);rank_loss+=float(loss.item())/len(ranking)
    kl=0.;policy_kl=0.
    if cfg.get('kl_weight') or cfg.get('policy_kl_weight'):
        anchor.model.eval()
        def kl_objective(x,p,ref):
            z=b.model(x)['logits'][0,p].astype(mx.float32)
            lp=z-mx.logsumexp(z,axis=-1,keepdims=True)
            ref=mx.stop_gradient(ref)
            return (mx.exp(ref)*(ref-lp)).sum(axis=-1).mean()
        fn=nn.value_and_grad(b.model,kl_objective)
        for source in ('chat','instructions'):
            eligible=[r for r in rows if r['source']==source and (not cfg.get('anchor_replay_only') or r.get('parent_replay'))]
            assert eligible;row=eligible[0]
            for mode,weight in [('reference',cfg.get('kl_weight',0)),('policy',cfg.get('policy_kl_weight',0))]:
                if not weight:continue
                if mode=='reference':
                    e=turns(b.tokenizer,row,cfg['context'])[-1]
                else:
                    from sft.conversation_foundation_v1.engine import generate
                    from sft.conversation_foundation_v1.data import visible_prefix
                    i=max(i for i,m in enumerate(row['messages']) if m['role']=='user')
                    if cfg.get('generated_history'):
                        messages,generation=history_reply(anchor,row,64)
                    else:
                        messages=row['messages'][:i+1];generation=generate(anchor,messages,64)
                    head=anchor.encode(visible_prefix(messages));tokens=list(generation['tokens'])
                    if generation['stop']=='eos':tokens.append(anchor.eos)
                    assert tokens
                    full=head+tokens
                    e=dict(x=full[:-1],y=[-100]*(len(head)-1)+tokens,targets=len(tokens))
                    assert len(e['x'])==len(e['y'])<=cfg['context']
                    assert e['x'][:len(head)]==head
                x,_=arrays([e],b.pad)
                indices=[i for i,t in enumerate(e['y']) if t!=-100]
                if len(indices)>64:indices=[indices[round(i*(len(indices)-1)/63)] for i in range(64)]
                p=mx.array(indices,dtype=mx.int32)
                z=anchor.model(x)['logits'][0,p].astype(mx.float32)
                ref=mx.stop_gradient(z-mx.logsumexp(z,axis=-1,keepdims=True));mx.eval(ref)
                loss,g=fn(x,p,ref);total=tree_map(lambda a,z:a+weight*z/2,total,g)
                mx.eval(loss,total)
                if mode=='reference':kl+=float(loss.item())/2
                else:policy_kl+=float(loss.item())/2
    if step==1 and cfg.get('policy_kl_weight'):assert abs(policy_kl)<1e-6
    total,norm=clip_gradients(total,cfg['clip_norm']);mx.eval(total,norm)
    assert math.isfinite(ce_total+ul+dpo+rank_loss+kl+format_loss+policy_kl) and math.isfinite(float(norm.item()))
    rate=learning_rate(step,cfg);opt.learning_rate=rate;opt.update(b.model,total);mx.eval(b.model.parameters(),opt.state)
    return dict(update=step,loss=cfg.get('ce_weight',1.)*ce_total+cfg['ul_weight']*ul+cfg.get('dpo_weight',0)*dpo+cfg.get('ranking_weight',0)*rank_loss+cfg.get('kl_weight',0)*kl+cfg.get('format_preference_weight',0)*format_loss+cfg.get('policy_kl_weight',0)*policy_kl,
                ce=ce_total,ul=ul,dpo=dpo,format_preference=format_loss,ranking=rank_loss,kl=kl,policy_kl=policy_kl,lr=rate,
                grad_norm=float(norm.item()),targets=targets,negative_tokens=sum(n['encoded']['targets'] for n in negative))

def evaluate(c,b,cfg,data,u,bundle,frozen):
    module=public_module();module.RUN=c/'runs/sft'
    module.evaluate(b,data,cfg,u,bundle,frozen)
    p=module.RUN/f'evaluations/update_{u:05d}.json';r=read_json(p)
    r.update(parent_step=512,step=512+u)
    atomic_json(p,r)
    if cfg.get('data_focused'):
        from data_focused import assess
        assess(c,b,data,u)
    if cfg.get('dataset_plan'):
        from dataset_trials import assess
        assess(c,b,data,u)
    if u==0:
        parent=read_json(PUBLIC/'runs/sft/evaluations/update_00128.json')
        for k in ('retention','public_development'):
            for field,value in parent[k]['metrics'].items():
                assert r[k]['metrics'][field]==value,(k,field,r[k]['metrics'][field],value)
        print('[baseline-parity] Every retained metric matches repair512',flush=True)

def run(c,args):
    import mlx.core as mx
    from mlx.utils import tree_unflatten,tree_flatten
    from sft.conversation_foundation_v1 import engine
    from sml_v2.checkpoint_bundle import save_bundle,resolve_bundle
    cfg=read_json(c/'config.json');data=read_json(c/'data.json')
    # Existing tests exercise negative-label alignment, padding and gradient direction.
    from sft.conversation_b.test_unlikelihood import run_unit_checks
    run_unit_checks()
    if cfg.get('reference_aware'):
        a=[10,11,12,13]
        assert selective_repetition_mask(a*2,a,(0,1,2,3))==[False]*8
        assert selective_repetition_mask(a*3,a,(0,1,2,3))==[False]*8+[True]*4
        assert not any(selective_repetition_mask(a*3,a*3,(0,1,2,3)))
        assert not any(selective_repetition_mask([0,10,11,12]*4,[],(0,1,2,3)))
        long=list(range(10,18))
        assert selective_repetition_mask(long*2,long,(0,1,2,3),8,1)==[False]*8+[True]*8
        assert not any(selective_repetition_mask(long*2,long*2,(0,1,2,3),8,1))
        print('[selective-ul-check] harmless and reference-requested repetition preserved',flush=True)
    if cfg.get('dpo_weight'):
        from sft.dpo_chat_768_v1.engine import preference_terms
        ref=mx.array([-20.,-40.]);counts=mx.array([10.,20.])
        f=lambda lp:preference_terms(lp,counts,ref,.1)[0]
        loss,g=mx.value_and_grad(f)(ref);mx.eval(loss,g)
        assert abs(float(loss.item())-math.log(2))<1e-6
        assert float(g[0].item())<0 and float(g[1].item())>0
        assert float(f(ref-.1*g).item())<float(loss.item())
        print('[dpo-check] baseline cancellation and preferred/rejected gradient directions passed',flush=True)
    if cfg.get('format_preference_weight'):
        ref=mx.array([-1.,-2.]);beta=cfg['format_preference_beta']
        f=lambda lp:mx.logaddexp(mx.array(0.),-beta*((lp-ref)[0]-(lp-ref)[1]))
        loss,g=mx.value_and_grad(f)(ref);mx.eval(loss,g)
        assert abs(float(loss.item())-math.log(2))<1e-6
        assert g[0].item()<0<g[1].item() and f(ref-.01*g).item()<loss.item()
        print('[format-preference-check] reference cancellation and gradient direction passed',flush=True)
    collect(c,cfg,data)
    frozen=freeze(c,cfg);run_dir=c/'runs/sft';run_dir.mkdir(parents=True,exist_ok=True)
    contract=run_dir/'contract.json'
    if contract.exists():assert read_json(contract)==frozen
    else:atomic_json(contract,frozen)
    negatives=read_json(c/'negatives.json')['updates']
    mx.random.seed(cfg['seed']);b=engine.load(cfg)
    frozen_parameters={}
    if cfg.get('train_last_blocks') or cfg.get('train_final_norm') or cfg.get('train_all_norms'):
        n=0 if cfg.get('train_final_norm') or cfg.get('train_all_norms') else cfg['train_last_blocks']
        assert 0<=n<len(b.model.blocks)
        b.model.freeze()
        if n:
            for block in b.model.blocks[-n:]:block.unfreeze()
        b.model.norm.unfreeze()
        if cfg.get('train_all_norms'):
            for block in b.model.blocks:
                block.norm1.unfreeze()
                block.norm2.unfreeze()
        trainable=dict(tree_flatten(b.model.trainable_parameters()))
        for key in trainable:
            if cfg.get('train_all_norms'):
                assert key=='norm.weight' or (key.startswith('blocks.') and key.split('.')[2] in ('norm1','norm2') and key.endswith('.weight')),key
            else:
                assert key.startswith('norm.') or (key.startswith('blocks.') and int(key.split('.')[1])>=len(b.model.blocks)-n),key
        if cfg.get('train_all_norms'):
            assert len(trainable)==2*len(b.model.blocks)+1
        frozen_parameters={k:v for k,v in tree_flatten(b.model.parameters()) if k not in trainable}
        assert 'embed.weight' in frozen_parameters
        atomic_json(c/'trainable_parameters.json',dict(trainable=sum(v.size for v in trainable.values()),
                    frozen=sum(v.size for v in frozen_parameters.values()),trainable_keys=sorted(trainable)))
        print('[partial-training]', 'all normalization weights; attention, feed-forward and embedding weights frozen' if cfg.get('train_all_norms') else f'{n} top blocks; embedding and lower layers frozen',flush=True)
    elif cfg.get('freeze_embeddings'):
        b.model.embed.freeze()
        trainable=dict(tree_flatten(b.model.trainable_parameters()))
        frozen_parameters={k:v for k,v in tree_flatten(b.model.parameters()) if k not in trainable}
        assert set(frozen_parameters)=={'embed.weight'}
        assert any(k.startswith('blocks.0.') for k in trainable) and 'norm.weight' in trainable
        atomic_json(c/'trainable_parameters.json',dict(trainable=sum(v.size for v in trainable.values()),
                    frozen=sum(v.size for v in frozen_parameters.values()),trainable_keys=sorted(trainable)))
        print('[partial-training] all transformer blocks trainable; tied embedding/output weights frozen',flush=True)
    opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    anchor=engine.load(cfg) if cfg.get('kl_weight') or cfg.get('format_preference_weight') or cfg.get('policy_kl_weight') else None
    if anchor is not None:anchor.model.freeze();assert not tree_flatten(anchor.model.trainable_parameters())
    ranking=read_json(c/'ranking.json')['updates'] if cfg.get('ranking_weight') else [None]*cfg['updates']
    format_pairs=read_json(c/'format_pairs.json')['updates'] if cfg.get('format_preference_weight') else [None]*cfg['updates']
    u=0;bundle=cfg['source_bundle']
    if (run_dir/'latest.json').exists():
        weights=resolve_bundle(run_dir/'latest.json');meta=read_json(weights+'.json')
        assert meta['contract']==fingerprint(frozen) and meta['source_model_sha256']==cfg['source_model_sha256']
        u=meta['additional_updates'];b.model.load_weights(weights,strict=True)
        opt.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()));mx.eval(b.model.parameters(),opt.state)
        assert int(opt.state['step'].item())==u
        bundle=Path(weights).parent.name
    evaluate(c,b,cfg,data,0 if u==0 else u,bundle,frozen) if u in cfg['evaluation_updates'] else None
    with (run_dir/'training.jsonl').open('a',buffering=1) as log:
        for cursor in range(u,cfg['updates']):
            if (c/'STOP').exists():break
            start=time.monotonic();r=update(b,opt,data['train'][cursor*16:(cursor+1)*16],negatives[cursor],cfg,cursor+1,anchor,ranking[cursor],format_pairs[cursor]);u=cursor+1
            if frozen_parameters and u==1:
                current=dict(tree_flatten(b.model.parameters()))
                assert all(bool(mx.array_equal(v,current[k]).item()) for k,v in frozen_parameters.items())
                print('[frozen-check] first update preserved every frozen parameter',flush=True)
            log.write(json.dumps(r)+'\n')
            if u==1 or u%8==0:print(f"[experiment {c.name} {u}/{cfg['updates']}] CE={r['ce']:.4f} UL={r['ul']:.4f} DPO={r['dpo']:.4f} format={r['format_preference']:.4f} rank={r['ranking']:.4f} KL={r['kl']:.4f} policyKL={r['policy_kl']:.4f} seconds={time.monotonic()-start:.1f}",flush=True)
            if u in cfg['evaluation_updates'] and u<cfg['updates']:
                meta=dict(step=512+u,additional_updates=u,source_step=512,source_bundle=cfg['source_bundle'],source_model_sha256=cfg['source_model_sha256'],contract=fingerprint(frozen),training_format='plain-user-assistant-eos-v1',tokenizer=b.tokenizer.fingerprint,experimental=True,automatic_promotion=False)
                bundle=Path(save_bundle(run_dir,b.model,opt,meta,None,keep=cfg.get('checkpoint_keep',2),reserve_gib=3)).name
                evaluate(c,b,cfg,data,u,bundle,frozen)
    saved_update=0 if bundle==cfg['source_bundle'] else read_json(run_dir/bundle/'model.safetensors.json')['additional_updates']
    if u>saved_update:
        meta=dict(step=512+u,additional_updates=u,source_step=512,source_bundle=cfg['source_bundle'],
                  source_model_sha256=cfg['source_model_sha256'],contract=fingerprint(frozen),
                  training_format='plain-user-assistant-eos-v1',tokenizer=b.tokenizer.fingerprint,
                  experimental=True,automatic_promotion=False)
        bundle=Path(save_bundle(run_dir,b.model,opt,meta,None,keep=cfg.get('checkpoint_keep',2 if len(cfg['evaluation_updates'])>2 else 1),reserve_gib=3)).name
    if frozen_parameters:
        current=dict(tree_flatten(b.model.parameters()))
        assert all(bool(mx.array_equal(v,current[k]).item()) for k,v in frozen_parameters.items())
        atomic_json(c/'frozen_parameter_check.json',dict(exactly_equal=True,parameters_checked=len(frozen_parameters),
                    embedding_unchanged=True,additional_updates=u))
    evaluate(c,b,cfg,data,u,bundle,frozen)
    verify_contract(frozen);verify_parent()
    atomic_json(run_dir/'finished.json',dict(status='complete' if u==cfg['updates'] else 'paused',updates=u,bundle=bundle))
    print('[finished]',c.name,u,flush=True)

def benchmark(c,args):
    from evaluation.full_benchmarks import core,launch,spec
    run_dir=c/'runs/sft';f=read_json(run_dir/'finished.json');assert f['status']=='complete'
    bundle=run_dir/f['bundle'];meta=read_json(bundle/'model.safetensors.json')
    if args.checkpoint_updates:
        matches=[p.parent for p in run_dir.glob('step_*/model.safetensors.json') if read_json(p)['additional_updates']==args.checkpoint_updates]
        assert len(matches)==1;bundle=matches[0];meta=read_json(bundle/'model.safetensors.json')
    name='repair512-search-'+c.name+(f'-u{args.checkpoint_updates}' if args.checkpoint_updates else '');results=c/'benchmarks'
    spec.MODELS[name]=dict(bundle=str(bundle.relative_to(ROOT)),step=meta['step'],
                           sha256=file_sha256(bundle/'model.safetensors'),decode_mode='cached')
    core.RESULTS=launch.RESULTS=results;launch.DIR=results/'readiness';launch.DIR.mkdir(parents=True,exist_ok=True)
    suites=['multiple-choice','ifeval'] if args.suite=='all' else [args.suite]
    for suite in suites:
        frozen=core.frozen(name,suite);frozen['code'][str(Path(__file__).resolve())]=file_sha256(__file__)
        launch.check(name,suite,frozen);launch.run(name,suite,frozen)
    verify_parent()

def delete(c):
    assert c.parent==EXP/'candidates' and not c.is_symlink()
    assert c.is_dir() and not any(p.is_symlink() for p in c.rglob('*'))
    verify_parent()
    report=dict(candidate=c.name,config=read_json(c/'config.json'),removed=True)
    if (c/'comparison.json').exists():report['comparison']=read_json(c/'comparison.json')
    if (c/'content_review.json').exists():report['qualitative_content_review']=read_json(c/'content_review.json')
    if (c/'frozen_parameter_check.json').exists():report['parameter_freezing']=read_json(c/'frozen_parameter_check.json')
    if (c/'trainable_parameters.json').exists():
        counts=read_json(c/'trainable_parameters.json')
        report['parameter_counts']={k:counts[k] for k in ('trainable','frozen')}
    ep=c/'runs/sft/evaluations'
    if ep.exists():
        report['development']={p.stem:{k:read_json(p)[k]['metrics'] for k in ('retention','public_development')} for p in ep.glob('update_*.json')}
    report['benchmarks']={str(p.relative_to(c/'benchmarks')):read_json(p) for p in (c/'benchmarks').rglob('summary.json')}
    size=sum(p.stat().st_size for p in c.rglob('*') if p.is_file())
    shutil.rmtree(c);verify_parent()
    report.update(freed_gib=round(size/1024**3,2),protected_weights_unchanged=True)
    atomic_json(EXP/'comparisons'/f'{c.name}.json',report)
    print('[removed]',c.name,report['freed_gib'],'GiB; protected parents intact',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run','benchmark','delete'])
    p.add_argument('--candidate',required=True)
    p.add_argument('--updates',type=int,choices=[128,256,512],default=128)
    p.add_argument('--lr',type=float,default=2e-6);p.add_argument('--ul',type=float,default=.1)
    p.add_argument('--dpo',type=float,default=0.)
    p.add_argument('--ranking',type=float,default=0.);p.add_argument('--kl',type=float,default=0.)
    p.add_argument('--replay-only',action='store_true')
    p.add_argument('--on-policy',action='store_true')
    p.add_argument('--reference-aware',action='store_true')
    p.add_argument('--instruction-weight',type=float,default=1.)
    p.add_argument('--train-last-blocks',type=int,default=0)
    p.add_argument('--balanced-instructions',action='store_true')
    p.add_argument('--data-focused',action='store_true')
    p.add_argument('--dataset-plan',choices=['squad-replay','constraints-replay','human-replay','familiar-replay','science-replay','mixed-public','breadth-stable'])
    p.add_argument('--fresh-per-batch',type=int,choices=[4,8],default=4)
    p.add_argument('--checkpoint-updates',type=int,choices=[128,256,384,512])
    p.add_argument('--instruction-edge-weight',type=float,default=1.)
    p.add_argument('--format-preference',type=float,default=0.)
    p.add_argument('--instruction-word-count',type=int,choices=[120,400],default=120)
    p.add_argument('--policy-kl',type=float,default=0.)
    p.add_argument('--ce-weight',type=float,default=1.)
    p.add_argument('--train-final-norm',action='store_true')
    p.add_argument('--train-all-norms',action='store_true')
    p.add_argument('--freeze-embeddings',action='store_true')
    p.add_argument('--generated-history',action='store_true')
    p.add_argument('--repetition-ngram',type=int,choices=[4,8],default=4)
    p.add_argument('--repetition-allowance',type=int,choices=[1,2],default=2)
    p.add_argument('--weighting',choices=['conversation','token'],default='conversation')
    p.add_argument('--suite',choices=['all','multiple-choice','ifeval'],default='all')
    a=p.parse_args()
    assert sum(bool(x) for x in (a.train_final_norm,a.train_all_norms,a.train_last_blocks,a.freeze_embeddings))<=1
    assert a.candidate.replace('-','').isalnum() and a.lr>0 and a.ul>=0 and a.instruction_weight>0 and a.train_last_blocks>=0 and a.instruction_edge_weight>0 and a.format_preference>=0 and a.policy_kl>=0 and a.ce_weight>0
    c=EXP/'candidates'/a.candidate
    with lock(ROOT/'sft/.experiment.lock'):
        verify_parent()
        if a.mode=='prepare':prepare(c,a)
        elif a.mode=='run':run(c,a)
        elif a.mode=='benchmark':benchmark(c,a)
        else:delete(c)

if __name__=='__main__':main()
