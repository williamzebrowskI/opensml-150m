"""One public-data repair pilot from unified 384. Training requires --run."""
import argparse
import gc
import importlib.metadata
import json
import shutil
import signal
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
DIR=Path(__file__).resolve().parent
EXP=DIR.parent
RUN=EXP/'runs/sft'
BASE=ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json'
from sml_v2.common import atomic_json, file_sha256, fingerprint, read_json, lock

def validate(cfg):
    assert cfg['source_step']==384 and cfg['context']==2048 and cfg['epochs']==1
    assert cfg['updates']*cfg['batch_conversations']==sum(cfg['training_counts'].values())==4096
    assert sum(cfg['batch_counts'].values())==cfg['batch_conversations']==16
    assert all(cfg['batch_counts'][s]*cfg['updates']==n for s,n in cfg['training_counts'].items())
    bundle=ROOT/cfg['source_bundle'];meta=read_json(bundle/'model.safetensors.json')
    if meta['step']!=384 or meta['training_format']!='plain-user-assistant-eos-v1':raise ValueError('Expected unified-384 parent')
    if file_sha256(bundle/'model.safetensors')!=cfg['source_model_sha256']:raise ValueError('Parent weights changed')

def prepared(cfg):
    p=read_json(EXP/'data/prepared.json')
    if p['receipt']['config']!=fingerprint(cfg):raise ValueError('Preparation/config mismatch')
    for path,digest in p['receipt']['inputs'].items():
        if file_sha256(path)!=digest:raise ValueError('Preparation input changed: '+path)
    for split,rows in p['data'].items():
        if fingerprint(rows)!=p['receipt']['stats'][split]['hash']:raise ValueError('Prepared rows changed')
    return p['data']

def contract(cfg):
    files=list(DIR.glob('*.py'))+[EXP/'config.json',EXP/'data/prepared.json',EXP/'source_pins.json',BASE]
    for folder in ('sml_v2','sft/conversation_foundation_v1','sft/text_followup_512_v1','sft/skill_balance','sft/transfer_control','sft/intact_smoltalk_base_pilot','sft/natural_control','sft/complete_transfer_1024'):
        files+=list((ROOT/folder).glob('*.py'))
    files+=list((ROOT/'tokenizer/bytebpe32k_v1').glob('*'))
    files+=list((ROOT/'evaluation/full_benchmarks/vendor').rglob('*.py'))
    files+=[ROOT/'evaluation/full_benchmarks/core.py',ROOT/'sft/text_followup_512_v1/config.json',ROOT/'sft/text_followup_512_v1/prepared.json']
    for bundle in (cfg['source_bundle'],'runs/sft_text_followup_512_v1/step_0000768_10f5ea2a3354'):
        files+=[ROOT/bundle/n for n in ('model.safetensors','model.safetensors.json','manifest.json')]
    return dict(config=cfg,architecture=file_sha256(BASE),protected={str(p):file_sha256(p) for p in sorted(set(files)) if p.is_file()},runtime={p:importlib.metadata.version(p) for p in ('mlx','numpy','tokenizers','pyarrow','requests')})

def verify(frozen,cfg):
    if contract(cfg)!=frozen:raise ValueError('Frozen experiment inputs changed')

def inspect(cfg,dataset):
    from data import turns,visible_prefix,verify as verify_constraints
    from sml_v2.tokenization import Tokenizer
    from collections import Counter
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');seen=set();count=0
    for split,rows in dataset.items():
        for r in rows:
            if r['group'] in seen:raise ValueError('Duplicate or split overlap')
            seen.add(r['group'])
            for e in turns(tok,r,cfg['context']):
                head=tok.encode(visible_prefix(r['messages'][:e['message_index']]))
                assert e['y'][:len(head)-1]==[-100]*(len(head)-1) and e['y'][-1]==tok.eos
                assert len(e['x'])==len(e['y'])<=cfg['context']
                assert e['targets']==sum(t!=-100 for t in e['y'])<=cfg['source_caps'][r['source']]
                count+=1
            if r['source']=='instructions' and not all(verify_constraints(r['messages'][-1]['content'],r['checks'])):raise ValueError('Invalid parsed-constraint target')
    train=dataset['train'];assert len(train)==4096 and len({r['id'] for r in train})==4096
    for u in range(cfg['updates']):assert Counter(r['source'] for r in train[u*16:(u+1)*16])==cfg['batch_counts']
    return dict(assistant_turns_checked=count,training_conversations=4096,whole_answers=True,assistant_only_eos_masks=True,batch_coverage=True,production_updates=0)

def check(cfg,dataset,frozen):
    import mlx.core as mx
    import mlx.nn as nn
    from sft.conversation_foundation_v1 import engine
    from sft.transfer_control.engine import arrays
    from data import turns
    result=inspect(cfg,dataset);b=engine.load(cfg)
    row=min(dataset['train'],key=lambda r:r['max_context']);e=turns(b.tokenizer,row,cfg['context'])[0];x,y=arrays([e],b.pad)
    logits=b.model(x)['logits'].astype(mx.float32);pos=mx.array([i for i,t in enumerate(e['y']) if t!=-100])
    gathered=nn.losses.cross_entropy(logits[0,pos],y[0,pos],reduction='mean')
    masked=(nn.losses.cross_entropy(logits,mx.maximum(y,0),reduction='none')*(y!=-100)).sum()/e['targets']
    mx.eval(gathered,masked);assert abs(gathered.item()-masked.item())<1e-5
    messages=[dict(role='user',content='Hello!')]
    a=engine.generate(b,messages,8);z=engine.generate(b,messages,8,full=True);assert a==z
    result.update(masked_ce_parity=True,cached_full_decode_parity=True,parent_loaded=True,synthetic_probe=a,optimizer_updates=0)
    del b,logits,gathered,masked;gc.collect();mx.clear_cache();verify(frozen,cfg)
    atomic_json(EXP/'readiness.json',dict(status='passed',contract=frozen,checks=result))
    print('[check-passed]',json.dumps(result),flush=True)

def evaluate(b,dataset,cfg,u,bundle,frozen):
    import mlx.core as mx
    import mlx.nn as nn
    from sft.transfer_control.engine import arrays
    from sft.conversation_foundation_v1 import engine
    from sft.text_followup_512_v1.engine import assess as assess_retention
    from data import verify as verify_constraints
    from collections import Counter
    p=RUN/f'evaluations/update_{u:05d}.json'
    if p.exists():
        result=read_json(p)
        if result['contract']!=fingerprint(frozen):raise ValueError('Evaluation contract changed')
        return
    retention=assess_retention(b,read_json(ROOT/'sft/text_followup_512_v1/prepared.json')['data'],read_json(ROOT/'sft/text_followup_512_v1/config.json'))
    from data import turns
    b.model.eval();weighted=0.;targets=0;conv_losses=[]
    for row in dataset['dev'][:cfg['validation_conversations']]:
        losses=[]
        for e in turns(b.tokenizer,row,cfg['context']):
            x,y=arrays([e],b.pad)
            positions=mx.array([i for i,t in enumerate(e['y']) if t!=-100],dtype=mx.int32)
            loss=nn.losses.cross_entropy(b.model(x)['logits'][0,positions].astype(mx.float32),y[0,positions],reduction='mean')
            mx.eval(loss);value=float(loss.item());losses.append(value);weighted+=value*e['targets'];targets+=e['targets']
        conv_losses.append(sum(losses)/len(losses))
    public=dict(metrics=dict(validation_token_nll=weighted/targets,validation_conversation_nll=sum(conv_losses)/len(conv_losses),validation_conversations=len(conv_losses),validation_targets=targets),public_benchmarks=False)
    # Generate every turn of a fixed source-balanced public dev sample. Constraints
    # are formal checks only; general QA answers are saved beside references.
    answers=[];all_turns=[];formal=[];family=Counter();family_total=Counter()
    for source in cfg['training_counts']:
        for row in [r for r in dataset['dev'] if r['source']==source][:cfg['generation_per_source'][source]]:
            history=[];out=[]
            for m in row['messages']:
                if m['role']=='system':history.append(m)
                elif m['role']=='user':
                    history.append(m);g=engine.generate(b,history,cfg['max_new_tokens']);history.append(dict(role='assistant',content=g['text']));out.append(dict(prompt=m['content'],**g));all_turns.append(g)
            item=dict(id=row['id'],source=source,reference_messages=row['messages'],turns=out)
            if row.get('checks'):
                ok=verify_constraints(out[-1]['text'],row['checks']);item.update(checks=row['checks'],constraint_results=ok);formal.append(all(ok))
                for c,v in zip(row['checks'],ok):family[c['family']]+=v;family_total[c['family']]+=1
            answers.append(item)
    public['answers']=answers
    public['metrics'].update(generated_turns=len(all_turns),stopped=sum(t['stop']=='eos' for t in all_turns)/len(all_turns),repeated=sum(engine.repeated(t['tokens']) for t in all_turns)/len(all_turns),empty=sum(not t['text'].strip() for t in all_turns)/len(all_turns),mean_generated_tokens=sum(len(t['tokens']) for t in all_turns)/len(all_turns),stop_counts=dict(Counter(t['stop'] for t in all_turns)),verified_named_constraint_passes=sum(formal),verified_named_constraint_total=len(formal),constraint_by_family={f:dict(passed=family[f],total=n) for f,n in family_total.items()})
    public['note']='All generated turns saved. Parsed named constraints are formal checks, not correctness/completeness scores. General and grounded outputs require content review. Reserved test not evaluated.'
    result=dict(update=u,parent_step=384,step=384+u,bundle=bundle,contract=fingerprint(frozen),retention=retention,public_development=public,automatic_selection=False)
    atomic_json(p,result)
    print('[retention-eval]',u,json.dumps(retention['metrics']),flush=True)
    print('[public-dev-eval]',u,json.dumps(public['metrics']),'; formal checks only; review saved answers',flush=True)

def train(cfg,dataset,frozen,stop):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sft.conversation_foundation_v1 import engine
    from sml_v2.checkpoint_bundle import resolve_bundle,save_bundle
    RUN.mkdir(parents=True,exist_ok=True);cp=RUN/'contract.json'
    if cp.exists() and read_json(cp)!=frozen:raise ValueError('Existing run contract differs')
    if not cp.exists():atomic_json(cp,frozen)
    mx.random.seed(cfg['seed']);b=engine.load(cfg);opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    u=saved=0;bundle=cfg['source_bundle']
    if (RUN/'latest.json').exists():
        w=resolve_bundle(RUN/'latest.json');meta=read_json(w+'.json');u=meta['additional_updates']
        if meta['contract']!=fingerprint(frozen) or meta['source_model_sha256']!=cfg['source_model_sha256'] or meta['step']!=384+u or not 0<=u<=cfg['updates']:raise ValueError('Resume ancestry mismatch')
        b.model.load_weights(w,strict=True);opt.state=tree_unflatten(list(mx.load(w+'.optimizer.safetensors').items()));mx.eval(b.model.parameters(),opt.state)
        assert int(opt.state['step'].item())==u and meta['conversation_exposures']==u*16
        saved=u;bundle=Path(w).parent.name;print('[resume]',u,'/256 with saved optimizer and cursor',flush=True)
        lp=RUN/'training.jsonl'
        if lp.exists():
            committed=[];tail=[]
            for line in lp.read_text().splitlines(keepends=True):
                try:r=json.loads(line)
                except json.JSONDecodeError:tail.append(line);continue
                (committed if r['update']<=u else tail).append(line)
            if tail:
                (RUN/f'uncommitted_after_{u:05d}.jsonl').write_text(''.join(tail));lp.write_text(''.join(committed))
    def save():
        nonlocal saved,bundle
        if u==saved:return
        meta=dict(step=384+u,additional_updates=u,source_step=384,source_bundle=cfg['source_bundle'],source_model_sha256=cfg['source_model_sha256'],conversation_exposures=u*16,contract=fingerprint(frozen),training_format='plain-user-assistant-eos-v1',tokenizer=b.tokenizer.fingerprint,experimental=True,automatic_promotion=False)
        bundle=Path(save_bundle(RUN,b.model,opt,meta,None,keep=1000,reserve_gib=3)).name;saved=u
        print('[checkpoint]',384+u,'retained;',u,'additional updates',flush=True)
    def cancelled():return stop['requested'] or (RUN/'STOP').exists()
    print('[ready]',u,'/256 updates; public repair from unified 384; one pass',flush=True)
    with (RUN/'training.jsonl').open('a',buffering=1) as journal:
        while u<cfg['updates'] and not cancelled():
            if u in cfg['evaluation_updates']:evaluate(b,dataset,cfg,u,bundle,frozen)
            if cancelled():break
            start=__import__('time').monotonic();r=engine.update(b,opt,dataset['train'][u*16:(u+1)*16],cfg,u+1);u+=1
            journal.write(json.dumps(r)+'\n')
            if u==1 or u%8==0:print(f"[public-repair {u}/256] loss={r['loss']:.4f} lr={r['lr']:.3e} turns={r['assistant_turns']} seconds={__import__('time').monotonic()-start:.1f}",flush=True)
            if u in cfg['evaluation_updates']:save()
        save()
        if not cancelled() and u in cfg['evaluation_updates']:evaluate(b,dataset,cfg,u,bundle,frozen)
    verify(frozen,cfg)
    status='complete' if u==cfg['updates'] else 'paused';atomic_json(RUN/'finished.json',dict(status=status,updates=u,source_step=384,automatic_promotion=False))
    print('[finished]',status,u,'updates; review development responses before public benchmarks',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    for m in ('prepare','check','run'):g.add_argument('--'+m,action='store_true')
    p.add_argument('--clear-stop',action='store_true');a=p.parse_args();cfg=read_json(EXP/'config.json');validate(cfg)
    print(json.dumps(dict(source=cfg['source_bundle'],source_step=384,mix=cfg['training_counts'],passes=1,updates=256,learning_rate=cfg['peak_lr'],method='Assistant-only conversation-balanced SFT + EOS; complete short public targets',output=str(RUN),training=bool(a.run),automatic_selection=False),indent=2),flush=True)
    if not(a.prepare or a.check or a.run):return
    with lock(ROOT/'sft/.experiment.lock'),lock(EXP/'.launcher.lock'):
        if a.prepare:
            if (RUN/'contract.json').exists():raise ValueError('Cannot reprepare a started run')
            from data import prepare
            prepare(cfg);return
        dataset=prepared(cfg);frozen=contract(cfg)
        if a.check:check(cfg,dataset,frozen);return
        ready=read_json(EXP/'readiness.json')
        if ready['status']!='passed' or ready['contract']!=frozen:raise ValueError('Missing/stale readiness; use --check first')
        if a.clear_stop:(RUN/'STOP').unlink(missing_ok=True)
        if (RUN/'STOP').exists():raise ValueError('STOP exists; review results before --clear-stop')
        if not (RUN/'latest.json').exists() and shutil.disk_usage(ROOT).free<cfg['minimum_free_gib']*1024**3:raise OSError('Need 12 GiB free')
        stop={'requested':False}
        def request_stop(*_):stop['requested']=True;print('[stop-requested] Saving after current update/evaluation.',flush=True)
        signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
        from sft.transfer_control.launch import Tee
        RUN.mkdir(parents=True,exist_ok=True)
        with (RUN/'training.log').open('a',buffering=1) as log:
            old=sys.stdout;sys.stdout=Tee(old,log)
            try:train(cfg,dataset,frozen,stop)
            finally:sys.stdout=old

if __name__=='__main__':main()
