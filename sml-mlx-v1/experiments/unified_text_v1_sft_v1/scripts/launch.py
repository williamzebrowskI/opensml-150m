"""Prepare, verify, and run a fresh conversation SFT from the preserved V1 base."""
import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json,file_sha256,fingerprint,read_json,lock
DIR=Path(__file__).resolve().parent
EXP=DIR.parent
OUTPUT=EXP/'runs/sft'


def accepts_contract(digest,frozen):return digest==fingerprint(frozen)


def validate(cfg):
    assert cfg['source_step']==73243 and cfg['context']==2048 and cfg['epochs']==1
    assert sum(cfg['training_counts'].values())==cfg['updates']*cfg['batch_conversations']==12288
    assert 0<cfg['warmup_updates']<cfg['updates'] and cfg['evaluation_updates'][-1]==cfg['updates']
    source=ROOT/cfg['source_bundle'];meta=read_json(source/'model.safetensors.json')
    assert meta['step']==73243 and 'recipe' in meta and not meta.get('training_format')
    if file_sha256(source/'model.safetensors')!=cfg['source_model_sha256']:raise ValueError('Pretrained base changed')
    for path,digest in (('runs/sft_conversation_foundation_v1/step_0000512_156a108611bc','9f9574d7ba0cae82720e139f7889c1ea3a4b8ac81e9664fd59883839c669c5f2'),('runs/sft_text_followup_512_v1/step_0000768_10f5ea2a3354','309ad719dc8569e482444b47b05f3fd24e02e7b7470dad2af132a976bf11ca02')):
        if file_sha256(ROOT/path/'model.safetensors')!=digest:raise ValueError('Preserved reference model changed: '+path)


def prepared(cfg):
    p=read_json(EXP/'data/prepared.json')
    if p['receipt']['config']!=fingerprint(cfg):raise ValueError('Preparation/config mismatch')
    for path,digest in p['receipt']['inputs'].items():
        if file_sha256(path)!=digest:raise ValueError('Preparation source changed: '+path)
    for split,rows in p['data'].items():
        if fingerprint(rows)!=p['receipt']['stats'][split]['hash']:raise ValueError('Prepared data changed')
    return p['data']


def contract(cfg):
    from data import INPUTS,exclusion_files
    files=list(DIR.glob('*.py'))+[EXP/'config.json',EXP/'data/prepared.json']
    files+=list((ROOT/'sml_v1').glob('*.py'))
    for directory in ('conversation_foundation_v1','text_followup_512_v1','skill_balance','transfer_control','natural_control','intact_smoltalk_base_pilot','complete_transfer_1024'):
        files+=list((ROOT/'sft'/directory).glob('*.py'))
    files+=list((ROOT/'tokenizer/bytebpe32k_v1').glob('*'))
    files+=INPUTS+exclusion_files()+[ROOT/'sft/text_followup_512_v1/config.json']
    for bundle in (cfg['source_bundle'],'runs/sft_conversation_foundation_v1/step_0000512_156a108611bc','runs/sft_text_followup_512_v1/step_0000768_10f5ea2a3354'):
        files+=[ROOT/bundle/n for n in ('model.safetensors','model.safetensors.json','manifest.json')]
    files+=[ROOT/'scripts/completion_playground.py']
    files+=list((ROOT/'diagnostics/text_followup_512_benchmarks_v1/text-followup-512-768').glob('*/summary.json'))
    return dict(config=cfg,protected={str(p):file_sha256(p) for p in sorted(set(files)) if p.is_file()},
                runtime={p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','numpy','tokenizers','pyarrow','requests')})


def inspect_data(cfg,data):
    from data import turns,visible_prefix,batch_at,group_id
    from sml_v1.tokenization import Tokenizer
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    seen=set();checked=0
    for split,rows in data.items():
        groups={r.get('scenario_group',group_id(r['messages'])) for r in rows}
        if len(groups)!=len(rows) or seen&groups:raise ValueError('Group leakage/duplicates')
        seen|=groups
        for row in rows:
            for e in turns(tok,row,cfg['context']):
                head=tok.encode(visible_prefix(row['messages'][:e['message_index']]))
                assert len(e['x'])==len(e['y'])<=2048
                assert e['y'][:len(head)-1]==[-100]*(len(head)-1) and e['y'][-1]==tok.eos
                assert e['targets']==sum(t!=-100 for t in e['y'])
                checked+=1
    exposed=[r['id'] for u in range(cfg['updates']) for r in batch_at(data,cfg,u)]
    assert len(exposed)==len(set(exposed))==12288
    from collections import Counter
    assert Counter(r['source'] for r in data['train'])==cfg['training_counts']
    for u in range(cfg['updates']):
        c=Counter(r['source'] for r in batch_at(data,cfg,u))
        assert c['conversation']==10 and c['format_followup']==1 and c['recovery']==1
    return dict(assistant_turns_checked=checked,distinct_training_conversations=12288,epochs=1,unified_batch_coverage=True)


def check(cfg,data,frozen):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten,tree_unflatten
    from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
    from sft.conversation_foundation_original import engine
    from sft.conversation_foundation_original.data import turns,visible_prefix
    from sft.transfer_control.engine import arrays
    from evaluation.generation_comparison.run import playground
    print('[check] dataset masks, coverage and split isolation',flush=True)
    result=inspect_data(cfg,data);print('[check] native prompt, loss, decode and optimizer restore',flush=True)
    b=engine.load(cfg);pg=playground()
    short=min(data['train'],key=lambda r:r['max_context'])
    long=max(data['train'],key=lambda r:r['max_context'])
    multi=next(r for r in data['train'] if r['assistant_turns']>1)
    for row in (short,long,multi):
        for e in turns(b.tokenizer,row,cfg['context']):
            messages=row['messages'][:e['message_index']]
            instruction=messages[0]['content'] if messages[0]['role']=='system' else ''
            body=messages[1:] if instruction else messages
            opts=pg.GenerationRequest(messages=[pg.Message(**m) for m in body],instruction=instruction,max_tokens=1)
            actual=pg.prepare_plain_sft_prompt(b.tokenizer,opts,2048)
            assert actual['ids']==b.encode(visible_prefix(messages)) and not actual['trimmed_tokens']
    e=turns(b.tokenizer,multi,cfg['context'])[0];x,y=arrays([e],b.pad)
    logits=b.model(x)['logits'].astype(mx.float32);pos=mx.array([i for i,t in enumerate(e['y']) if t!=-100])
    gathered=nn.losses.cross_entropy(logits[0,pos],y[0,pos],reduction='mean')
    masked=(nn.losses.cross_entropy(logits,mx.maximum(y,0),reduction='none')*(y!=-100)).sum()/e['targets']
    mx.eval(gathered,masked);assert abs(gathered.item()-masked.item())<1e-5
    for messages in ([dict(role='user',content='Hello!')],
                     [dict(role='user',content='My name is Maya.'),dict(role='assistant',content='Hello, Maya!'),
                      dict(role='user',content='What is my name?')]):
        if messages[-1]['role']!='user':continue
        assert engine.generate(b,messages,12)==engine.generate(b,messages,12,full=True)
    del logits,gathered,masked
    opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    first=engine.update(b,opt,[short,long],cfg,1)
    assert first['grad_norm']>0 and int(opt.state['step'].item())==1
    with tempfile.TemporaryDirectory(prefix='opensml-conversation-check-') as scratch:
        bundle=save_bundle(scratch,b.model,opt,dict(step=1),None,keep=100,reserve_gib=1)
        weights=resolve_bundle(bundle);restored=engine.load(cfg,weights)
        opt2=engine.optimizer(cfg);opt2.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()));mx.eval(opt2.state)
        before=dict(tree_flatten(opt.state));after=dict(tree_flatten(opt2.state))
        assert before.keys()==after.keys() and all(bool(mx.array_equal(v,after[k]).item()) for k,v in before.items())
        engine.update(b,opt,[short],cfg,2);engine.update(restored,opt2,[short],cfg,2)
        a=dict(tree_flatten(b.model.parameters()));z=dict(tree_flatten(restored.model.parameters()))
        assert a.keys()==z.keys() and all(bool(mx.array_equal(v,z[k]).item()) for k,v in a.items())
        del restored,opt2,before,after,a,z
    # Exercise the actual evaluation path on disposable updated weights, never public benchmark prompts.
    tiny=dict(cfg,validation_conversations=2,generation_conversations=6,max_new_tokens=8)
    smoke=engine.assess(b,data,tiny);assert smoke['metrics']['generated_turns']>=6
    del b,opt;gc.collect();mx.clear_cache()
    assert contract(cfg)==frozen
    result.update(native_history_parity=True,masked_ce_parity=True,cached_decode_parity=True,
                  exact_optimizer_restore=True,exact_next_update_weights=True,evaluation_smoke=True,
                  disposable_update=first,production_updates=0,source_unchanged=True)
    atomic_json(EXP/'readiness.json',dict(status='passed',contract=frozen,checks=result))
    print('[check-passed]',json.dumps(result),flush=True)


def train(cfg,data,frozen,stop):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
    from sft.conversation_foundation_original import engine
    from data import batch_at
    OUTPUT.mkdir(parents=True,exist_ok=True)
    path=OUTPUT/'contract.json'
    if path.exists() and read_json(path)!=frozen:raise ValueError('Existing run contract changed')
    if not path.exists():atomic_json(path,frozen)
    # Archive the small recipe files; prepared data remain fingerprinted at their original path.
    for p in list(DIR.glob('*.py'))+[EXP/'config.json']:
        dest=OUTPUT/'inputs'/p.name;dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copy2(p,dest)
    mx.random.seed(cfg['seed']);b=engine.load(cfg);opt=engine.optimizer(cfg)
    opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    u=0;saved=0;bundle=None
    if (OUTPUT/'latest.json').exists():
        weights=resolve_bundle(OUTPUT/'latest.json');meta=read_json(weights+'.json');u=meta['step']
        assert accepts_contract(meta['contract'],frozen) and meta['source_step']==73243 and 0<=u<=cfg['updates']
        b.model.load_weights(weights,strict=True)
        opt.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()));mx.eval(opt.state,b.model.parameters())
        assert int(opt.state['step'].item())==u and meta['conversation_exposures']==u*cfg['batch_conversations']
        saved=u;bundle=Path(weights).parent
        print('[resume]',u,'with saved optimizer and data cursor',flush=True)
        log_path=OUTPUT/'training.jsonl'
        if log_path.exists():
            lines=log_path.read_text().splitlines(keepends=True);committed=[];tail=[]
            for line in lines:
                try:row=json.loads(line)
                except json.JSONDecodeError:tail.append(line);continue
                (committed if row['update']<=u else tail).append(line)
            if tail:
                (OUTPUT/f'training_uncommitted_{time.time_ns()}.jsonl').write_text(''.join(tail))
                log_path.write_text(''.join(committed))
    def cancelled():return stop['requested'] or (OUTPUT/'STOP').exists()
    def save():
        nonlocal saved,bundle
        if u==saved:return
        meta=dict(step=u,additional_updates=u,source_step=73243,source_bundle=cfg['source_bundle'],
                  source_model_sha256=cfg['source_model_sha256'],conversation_exposures=u*cfg['batch_conversations'],
                  contract=fingerprint(frozen),tokenizer=b.tokenizer.fingerprint,training_format='plain-user-assistant-eos-v1',
                  experimental=True,automatic_promotion=False,repetition_auto_stop=False)
        bundle=Path(save_bundle(OUTPUT,b.model,opt,meta,None,keep=1000000,reserve_gib=5));saved=u
        print('[checkpoint]',u,'retained',flush=True)
    def evaluate():
        p=OUTPUT/f'evaluations/update_{u:05d}.json'
        if p.exists():
            old=read_json(p)
            assert accepts_contract(old['contract'],frozen)
            return
        from evaluate import evaluate as assess_unified
        result=assess_unified(b,data,cfg)
        result.update(update=u,contract=fingerprint(frozen),bundle=bundle.name if bundle else cfg['source_bundle'])
        baseline=read_json(EXP/'baselines/retained_768.json')
        result['change_from_retained_768']={k:result['retention']['metrics'][k]-baseline['retention']['metrics'][k] for k in ('validation_conversation_nll','chat_stopped','chat_repeated','text_followup_joint_proxy')}
        result['change_from_retained_768']['new_recovery_joint_proxy']=result['behavior_metrics']['recovery']['joint_proxy']-baseline['behavior_metrics']['recovery']['joint_proxy']
        print('[change-from-retained-768]',json.dumps(result['change_from_retained_768']),'; reporting only',flush=True)
        atomic_json(p,result)
        lines=[f'# Unified text SFT — update {u}', '', 'Development responses; content review required.', '']
        for answer in result['development']['answers']+result['behavior_answers']:
            lines += [f"## {answer['source']} — {answer['id'][:12]}", '']
            for t in answer['turns']:
                lines += ['**User**', '', t['prompt'], '', '**Model**', '', t['text'], '', f"Stop: {t['stop']}", '']
        p.with_suffix('.md').write_text('\n'.join(lines))
    status='stopped'
    try:
        if u in cfg['evaluation_updates'] and not cancelled():evaluate()
        print('[ready]',u,'/',cfg['updates'],'updates; unified mixture from pretrained v1',flush=True)
        while u<cfg['updates'] and not cancelled():
            batch=batch_at(data,cfg,u);started=time.monotonic()
            metrics=engine.update(b,opt,batch,cfg,u+1);u+=1
            metrics.update(seconds=time.monotonic()-started,batch_hash=fingerprint([r['id'] for r in batch]))
            with (OUTPUT/'training.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            if u==1 or u%8==0:print(f'[unified-sft {u}/{cfg["updates"]}] loss={metrics["loss"]:.4f} lr={metrics["lr"]:.3e} turns={metrics["assistant_turns"]} seconds={metrics["seconds"]:.1f}',flush=True)
            if u in cfg['evaluation_updates']:
                save()
                if not cancelled():evaluate()
        save();status='complete' if u==cfg['updates'] else 'stopped'
    except Exception as exc:
        atomic_json(OUTPUT/'report.json',dict(status='error',update=u,last_saved_update=saved,error=repr(exc)))
        raise
    finally:
        del b,opt;gc.collect();mx.clear_cache()
    assert contract(cfg)==frozen
    atomic_json(OUTPUT/'report.json',dict(status=status,update=u,selected=None,source_step=73243,
                automatic_promotion=False,all_checkpoints_retained=True,public_benchmarks_run=False))
    print('[finished]',status,u,'; review development responses before public benchmarks',flush=True)


def baselines(cfg,data,frozen):
    import mlx.core as mx
    from sft.conversation_foundation_original import engine
    from evaluate import evaluate
    p=EXP/'baselines/retained_768.json'
    if p.exists():
        old=read_json(p)
        digest=old.pop('result_hash',None)
        if old['contract']!=fingerprint(frozen) or fingerprint(old)!=digest:raise ValueError('Baseline contract/results changed')
        print('[baseline] retained 768 already complete',flush=True);return
    b=engine.load(cfg,str(ROOT/'runs/sft_text_followup_512_v1/step_0000768_10f5ea2a3354/model.safetensors'))
    result=evaluate(b,data,cfg);result.update(contract=fingerprint(frozen),model='retained-768')
    result['result_hash']=fingerprint(result)
    atomic_json(p,result);del b;gc.collect();mx.clear_cache()
    assert contract(cfg)==frozen
    print('[baseline-finished] retained 768, inference only',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    for mode in ('prepare','check','baselines','run'):g.add_argument('--'+mode,action='store_true')
    p.add_argument('--clear-stop',action='store_true');a=p.parse_args()
    cfg=read_json(EXP/'config.json');validate(cfg)
    print(json.dumps(dict(source=cfg['source_bundle'],updates=cfg['updates'],conversations=12288,
                         method='Unified conversation-balanced assistant-only CE + EOS; one optimizer and schedule',
                         mix=cfg['training_counts'],output=str(OUTPUT),training=bool(a.run)),indent=2),flush=True)
    if not (a.prepare or a.check or a.baselines or a.run):return
    from sft.transfer_control.launch import preflight,Tee
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        if a.prepare:
            if (OUTPUT/'contract.json').exists():raise ValueError('Cannot reprepare a started run')
            from data import build
            build(cfg);return
        data=prepared(cfg);frozen=contract(cfg)
        if a.check:check(cfg,data,frozen);return
        if a.baselines:baselines(cfg,data,frozen);return
        ready=read_json(EXP/'readiness.json')
        if ready['status']!='passed' or ready['contract']!=frozen:raise ValueError('Run --check before training')
        baselines(cfg,data,frozen)
        if not (OUTPUT/'latest.json').exists() and shutil.disk_usage(ROOT).free<cfg['minimum_free_gib']*1024**3:
            raise OSError('Need 35 GiB free for retained checkpoint bundles')
        if a.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists():raise ValueError('Stopped run: review saved responses, then use --clear-stop to explicitly resume')
        stop={'requested':False}
        def request_stop(*_):
            stop['requested']=True;print('[stop-requested] Saving after the current update/evaluation.',flush=True)
        signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
        OUTPUT.mkdir(parents=True,exist_ok=True)
        with (OUTPUT/'training.log').open('a',buffering=1) as log:
            old=sys.stdout;sys.stdout=Tee(old,log)
            try:train(cfg,data,frozen,stop)
            finally:sys.stdout=old

if __name__=='__main__':main()
