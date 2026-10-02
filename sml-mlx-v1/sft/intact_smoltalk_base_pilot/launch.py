"""Prepare/check/run a bounded intact-message SFT pilot from the Stage B base."""
import argparse
import gc
import importlib.metadata
import json
import math
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json, file_sha256, fingerprint, lock, read_json
from sft.transfer_control.launch import Tee, preflight

DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_intact_smoltalk_base_pilot_v1'
BASE=ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json'


def validate(cfg):
    if cfg['source_step']!=73243 or cfg['checkpoint_start_step']!=0 or sum(cfg['training_counts'].values())!=12000:
        raise ValueError('Expected pretrained 73243, local SFT steps and 12,000 conversations')
    if sum(cfg['validation_counts'].values())!=500 or sum(cfg['reserved_counts'].values())!=500:
        raise ValueError('Expected 500 development and 500 reserved conversations')
    if cfg['updates']!=128 or cfg['batch_conversations']!=32 or cfg['epochs']!=2:
        raise ValueError('Expected a 128-update pilot, batch 32 and the existing curriculum order')
    if cfg['lr_schedule_updates']!=750 or not 0<cfg['warmup_updates']<cfg['updates']:
        raise ValueError('Keep the original LR schedule while limiting the pilot budget')
    if cfg['evaluation_updates']!=[0,16,32,64,128] or cfg['stop_on_retention_failure'] is not True:
        raise ValueError('Expected early evaluation and a checkpointed retention stop')
    if 12000%cfg['batch_conversations'] or cfg['context']!=2048:
        raise ValueError('Batch/context mismatch')
    if cfg['evaluation_updates'][0]!=0 or cfg['evaluation_updates'][-1]!=cfg['updates']:
        raise ValueError('Incomplete evaluation schedule')
    source=ROOT/cfg['source_bundle']
    meta=read_json(source/'model.safetensors.json')
    if meta['step']!=cfg['source_step'] or 'recipe' not in meta or meta.get('training_format'):
        raise ValueError('Expected a pretrained source, not an SFT checkpoint')
    if file_sha256(source/'model.safetensors')!=cfg['source_model_sha256']:
        raise ValueError('Preserved source changed')


def code_files():
    files=list(DIR.glob('*.py'))+list((ROOT/'sml_v1').glob('*.py'))
    for name in ('natural_control/data.py','skill_balance/engine.py','transfer_control/engine.py',
                 'transfer_control/launch.py','transfer_control/data.py','reading_repair/evaluate.py',
                 'reading_repair/generation.py','constraint_completion_1920/constraints.py',
                 'conversation_completion_2048/evaluate.py','conversation_completion_2048/dialogues.py'):
        files.append(ROOT/'sft'/name)
    files.append(ROOT/'evaluation/generation_comparison/run.py')
    files.append(ROOT/'scripts/completion_playground.py')
    return sorted(set(files))


def input_files():
    return code_files()+[DIR/n for n in ('config.json','selection.json','prepared.json','review_samples.json','review.json','review_rejections.json','data_review.md')]


def contract(cfg):
    validate(cfg); selection=read_json(DIR/'selection.json'); review=read_json(DIR/'review.json')
    if selection['config']!=fingerprint(cfg) or review.get('status')!='sample-reviewed':
        raise ValueError('Prepare and review the frozen training sample first')
    if review['sample_sha256']!=file_sha256(DIR/'review_samples.json'):
        raise ValueError('Reviewed sample changed')
    from .data import PARENT_EVAL
    paths=input_files()+[BASE,PARENT_EVAL,ROOT/'diagnostics/base_capability_20260924/items.json']
    paths += [ROOT/cfg['source_bundle']/n for n in ('model.safetensors','model.safetensors.json','manifest.json')]
    paths += [p for p in (ROOT/'tokenizer/bytebpe32k_v1').iterdir() if p.is_file()]
    return dict(config=fingerprint(cfg),selection=fingerprint(selection),architecture=file_sha256(BASE),
                protected={str(p):file_sha256(p) for p in paths},
                runtime={p:importlib.metadata.version(p) for p in ('mlx','numpy','tokenizers','pyarrow','fsspec')})


def verify(frozen):
    for name,sha in frozen['protected'].items():
        if file_sha256(name)!=sha:raise ValueError('Frozen input changed: '+name)
    if frozen['runtime']!={p:importlib.metadata.version(p) for p in frozen['runtime']}:
        raise ValueError('Runtime changed')


def rebuild(cfg):
    p=read_json(DIR/'prepared.json');receipt=read_json(DIR/'selection.json')
    if p['receipt']!=receipt or receipt['config']!=fingerprint(cfg):
        raise ValueError('Prepared selection identity changed')
    for split,rows in p['data'].items():
        if fingerprint(rows)!=receipt['stats'][split]['hash']:
            raise ValueError('Prepared messages changed: '+split)
    return p['data']


def snapshot(directory):
    for src in input_files():
        dest=Path(directory)/src.resolve().relative_to(ROOT.parent)
        if dest.exists():
            if file_sha256(src)!=file_sha256(dest):raise ValueError('Snapshot conflict: '+str(dest))
        else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)


def checks(cfg,frozen):
    import mlx.core as mx
    from . import engine
    from .data import turns,batch_at,visible_prefix
    from .evaluate import assess
    from evaluation.generation_comparison.run import playground
    data=rebuild(cfg); b=engine.load(cfg);pg=playground();checked=0;target_count=0
    for split,rows in data.items():
        for row in rows:
            for e in turns(b.tokenizer,row,cfg['context']):
                i=e['message_index'];messages=row['messages'][:i]
                instruction=messages[0]['content'] if messages[0]['role']=='system' else ''
                body=messages[1:] if messages[0]['role']=='system' else messages
                opts=pg.GenerationRequest(messages=[pg.Message(**m) for m in body],instruction=instruction,max_tokens=1)
                actual=pg.prepare_plain_sft_prompt(b.tokenizer,opts,2048)
                expected=b.encode(visible_prefix(messages))
                if actual['ids']!=expected or actual['trimmed_tokens']:
                    raise ValueError('Training/playground full-history mismatch')
                head=len(expected)
                if e['y'][:head-1]!=[-100]*(head-1) or e['y'][-1]!=b.eos or len(e['x'])!=len(e['y']):
                    raise ValueError('Assistant-only mask/EOS mismatch')
                if any(v!=-100 for v in e['y'][:head-1]):raise ValueError('History supervised twice')
                if e['targets']!=sum(t!=-100 for t in e['y']):raise ValueError('Target count mismatch')
                checked+=1;target_count+=e['targets']
                if checked%8000==0:
                    print('[check] full-history masks/playground parity',checked,'assistant turns',flush=True)
    ids=[r['id'] for u in range(cfg['updates']) for r in batch_at(data,cfg,u)]
    from collections import Counter
    if len(ids)!=4096 or len(set(ids))!=4096 or set(Counter(ids).values())!={1}:
        raise ValueError('Pilot must expose exactly 4,096 distinct conversations once')
    probes=[[dict(role='user',content='Hello!')],
            [dict(role='user',content='Write a polite one-sentence invitation to a book club.'),
             dict(role='assistant',content='Please join our book club this Saturday.'),
             dict(role='user',content='Move it to Sunday and return the complete invitation.')]]
    for messages in probes:
        cached=engine.generate(b,messages,24);full=engine.generate(b,messages,24,full=True)
        if cached!=full:raise ValueError('Cached/full-context generation mismatch')
    print('[check] cached/full-context generation agrees',flush=True)
    # Independent full-vocabulary masked CE must agree with selected-position CE.
    import mlx.nn as nn
    row=min(data['train'],key=lambda r:r['max_context'])
    e=turns(b.tokenizer,row,cfg['context'])[0];x,y=engine.arrays([e],b.pad)
    logits=b.model(x)['logits'].astype(mx.float32);mask=y!=-100
    reference=(nn.losses.cross_entropy(logits,mx.maximum(y,0),reduction='none')*mask).sum()/mask.sum()
    selected=engine.objective(b.model,x,y)
    if abs(float(reference.item())-float(selected.item()))>1e-5:
        raise ValueError('Selected-position CE differs from independent masked CE')
    smoke=assess(b,data,cfg,smoke=True)
    if not smoke['answers'] or 'own_history' not in smoke['answers'][0]:
        raise ValueError('Own-history evaluation failed')
    print('[check] own-history evaluation smoke passed; checking disposable gradient update',flush=True)
    opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    # Disposable two-conversation update; validates the exact production gradient path.
    ordered=sorted(data['train'],key=lambda r:r['max_context'])
    rr=[ordered[0],ordered[-1]]
    result=engine.update(b,opt,rr,cfg,1)
    if result['grad_norm']<=0 or int(opt.state['step'].item())!=1:
        raise ValueError('Disposable optimizer step failed')
    print('[check] disposable shortest/longest-conversation update passed',flush=True)
    import tempfile
    from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
    from mlx.utils import tree_unflatten
    with tempfile.TemporaryDirectory(prefix='opensml-intact-check-') as scratch:
        meta=dict(step=1,source_step=cfg['source_step'],additional_updates=1,contract=fingerprint(frozen),
                  training_format='plain-user-assistant-eos-v1')
        bundle=save_bundle(scratch,b.model,opt,meta,None,keep=1000000,reserve_gib=1.)
        weights=resolve_bundle(bundle)
        restored=engine.optimizer(cfg)
        restored.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
        mx.eval(restored.state)
        if int(restored.state['step'].item())!=1:raise ValueError('Optimizer checkpoint round-trip failed')
        resumed=engine.load(cfg,weights)
        from mlx.utils import tree_flatten
        before=dict(tree_flatten(opt.state));after=dict(tree_flatten(restored.state))
        if before.keys()!=after.keys() or any(not bool(mx.array_equal(t,after[k]).item()) for k,t in before.items()):
            raise ValueError('Saved optimizer state was not restored exactly')
        expected_next=batch_at(data,cfg,1)
        if [r['id'] for r in expected_next]!=[r['id'] for r in batch_at(data,cfg,int(restored.state['step'].item()))]:
            raise ValueError('Restored cursor differs from next pilot batch')
        next_uninterrupted=engine.update(b,opt,rr,cfg,2)
        next_resumed=engine.update(resumed,restored,rr,cfg,2)
        uninterrupted=dict(tree_flatten(b.model.parameters()))
        interrupted=dict(tree_flatten(resumed.model.parameters()))
        if uninterrupted.keys()!=interrupted.keys() or any(not bool(mx.array_equal(t,interrupted[k]).item()) for k,t in uninterrupted.items()):
            raise ValueError('Next model update differs after save/resume')
        first=dict(tree_flatten(opt.state));second=dict(tree_flatten(restored.state))
        if first.keys()!=second.keys():raise ValueError('Restored optimizer structure differs')
        differences={k:float(mx.max(mx.abs(t-second[k])).item()) for k,t in first.items() if not bool(mx.array_equal(t,second[k]).item())}
        # Save/load is exact above. Subsequent FP32 arithmetic can differ in the
        # final bits of Adam moments; require exact weights/master/counter and
        # cap moment differences rather than claiming bitwise moment identity.
        if any(not k.endswith(('.m','.v')) or not math.isfinite(v) or v>1e-9 for k,v in differences.items()):
            raise ValueError('Next optimizer update exceeds FP32 moment tolerance: '+repr(differences))
        optimizer_max_abs=max(differences.values(),default=0.)
        print('[check] exact state restore/next weights; next Adam moment max difference',optimizer_max_abs,flush=True)
        del resumed
        snapshot(Path(scratch)/'inputs');snapshot(Path(scratch)/'inputs')
        del restored
    del b,opt;gc.collect();mx.clear_cache();verify(frozen)
    return dict(masked_assistant_turns=checked,assistant_targets=target_count,
                available_training_conversations=12000,pilot_conversation_exposures=4096,
                native_history_parity=True,cached_generation_parity=True,evaluation_smoke=True,
                independent_masked_ce_parity=True,
                disposable_update=result,checkpoint_optimizer_roundtrip=True,exact_next_update_weights=True,
                next_update_optimizer_max_abs=optimizer_max_abs,next_update_moment_tolerance=1e-9,snapshot_roundtrip=True,
                source_unchanged=True,production_updates=0)


def run(cfg,frozen,data,stop):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
    from . import engine
    from .data import batch_at
    from .evaluate import assess,gate,review_template
    ep=OUTPUT/'evaluations';ep.mkdir(parents=True,exist_ok=True)
    mx.random.seed(cfg['seed']);b=engine.load(cfg);opt=engine.optimizer(cfg)
    opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    u=0;saved=-1;bundle=None
    if (OUTPUT/'latest.json').exists():
        weights=resolve_bundle(OUTPUT/'latest.json');meta=read_json(weights+'.json');u=meta['additional_updates']
        if (meta['contract']!=fingerprint(frozen) or not 0<=u<=cfg['updates']
                or meta['step']!=u or meta['source_step']!=cfg['source_step']):
            raise ValueError('Resume identity mismatch')
        if meta['conversation_exposures']!=u*cfg['batch_conversations']:raise ValueError('Resume cursor mismatch')
        b.model.load_weights(weights,strict=True)
        opt.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()));mx.eval(opt.state,b.model.parameters())
        if int(opt.state['step'].item())!=u:raise ValueError('Optimizer resume mismatch')
        saved=u;bundle=Path(weights).parent
        print('[resume]',u,'with exact optimizer and conversation cursor',flush=True)
    cancelled=lambda:stop['requested'] or (OUTPUT/'STOP').exists()
    def save():
        nonlocal saved,bundle
        meta=dict(step=u,source_step=cfg['source_step'],additional_updates=u,optimizer_local_step=u,
                  conversation_exposures=u*cfg['batch_conversations'],contract=fingerprint(frozen),
                  training_format='plain-user-assistant-eos-v1',experiment=cfg['format'],
                  experimental=True,automatic_promotion=False)
        bundle=Path(save_bundle(OUTPUT,b.model,opt,meta,None,keep=1000000,reserve_gib=10.))
        saved=u;print('[checkpoint]',u,'retained; no pruning',flush=True)
    def evaluate():
        dest=ep/f'update_{u:05d}.json'
        newly_evaluated=not dest.exists()
        if dest.exists():
            result=read_json(dest)
            if result['bundle']!=bundle.name or result['contract']!=fingerprint(frozen):
                raise ValueError('Saved evaluation identity mismatch')
        else:
            result=assess(b,data,cfg,cancelled=cancelled)
            result.update(update=u,step=u,bundle=bundle.name,contract=fingerprint(frozen))
            base=result if u==0 else read_json(ep/'update_00000.json')
            result['retention_gate_passed']=gate(result['metrics'],base['metrics'],cfg)
            atomic_json(dest,result)
        print('[intact-eval]',u,json.dumps(dict(**result['metrics'],retention_gate=result['retention_gate_passed'])),
              '; content/completion review required, no automatic best',flush=True)
        if not (ep/f'review_{u:05d}.json').exists():atomic_json(ep/f'review_{u:05d}.json',review_template(result))
        if u and newly_evaluated and not result['retention_gate_passed'] and cfg['stop_on_retention_failure']:
            stop['requested']=True;(OUTPUT/'STOP').touch()
            print('[retention-stop] Checkpoint saved. Review answers before explicitly resuming.',flush=True)
    try:
        if saved<0:save()
        if u in cfg['evaluation_updates'] and not cancelled():evaluate()
        print(f'[ready] Stage B base 73243; SFT {u}/{cfg["updates"]}; 4,096-conversation pilot; all assistant turns',flush=True)
        while u<cfg['updates'] and not cancelled():
            rr=batch_at(data,cfg,u);started=time.monotonic()
            result=engine.update(b,opt,rr,cfg,u+1);u+=1
            result.update(seconds=time.monotonic()-started,batch_hash=fingerprint([r['id'] for r in rr]))
            with (OUTPUT/'training.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if u==1 or u%8==0:
                print(f'[intact-base {u}/{cfg["updates"]}] step={u} loss={result["loss"]:.4f} lr={result["lr"]:.3e} turns={result["assistant_turns"]} targets={result["assistant_targets"]:,} update_seconds={result["seconds"]:.1f}',flush=True)
            if u%cfg['checkpoint_every']==0 or u in cfg['evaluation_updates']:save()
            if u in cfg['evaluation_updates'] and not cancelled():evaluate()
        if saved!=u:save()
        status='stopped' if cancelled() else 'complete'
    except (KeyboardInterrupt,InterruptedError):
        stop['requested']=True;(OUTPUT/'STOP').touch()
        if saved!=u:save()
        status='stopped'
    except Exception as exc:
        atomic_json(OUTPUT/'report.json',dict(status='error',update=u,last_saved_update=saved,error=repr(exc)))
        raise
    finally:del b,opt;gc.collect();mx.clear_cache()
    verify(frozen)
    atomic_json(OUTPUT/'report.json',dict(status=status,update=u,step=u,source_step=cfg['source_step'],contract=fingerprint(frozen),
                selected=None,all_checkpoints_retained=True,automatic_promotion=False,reserved_test_evaluated=False))
    print('[finished]',status,'SFT step',u,'; review saved answers before choosing or benchmarking',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group()
    for mode in ('prepare','check','run'):g.add_argument('--'+mode,action='store_true')
    p.add_argument('--clear-stop',action='store_true');a=p.parse_args();cfg=read_json(DIR/'config.json');validate(cfg)
    mode='prepare' if a.prepare else 'check' if a.check else 'run' if a.run else 'plan'
    print(json.dumps(dict(mode=mode,source=str(ROOT/cfg['source_bundle']),training_conversations=12000,
          validation_conversations=500,reserved_conversations=500,epochs=cfg['epochs'],
          updates=cfg['updates'],final_step=cfg['updates'],peak_lr=cfg['peak_lr'],lr_schedule_updates=cfg['lr_schedule_updates'],
          algorithm='Ordinary SFT; every assistant content/EOS target with complete native history',
          intact_messages=True,truncation=False,injected_system=False,teacher=False,
          replay=False,output=str(OUTPUT),stop_on_retention_failure=True,automatic_promotion=False),indent=2),flush=True)
    if mode=='plan':return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        if mode=='prepare':
            if (OUTPUT/'contract.json').exists():raise ValueError('Cannot change a started run')
            from .data import build,review_rows
            data,receipt=build(cfg)
            atomic_json(DIR/'selection.json',receipt);atomic_json(DIR/'prepared.json',dict(data=data,receipt=receipt))
            atomic_json(DIR/'review_samples.json',review_rows(data,cfg,'train'))
            atomic_json(DIR/'review.json',dict(status='unreviewed',sample_sha256=file_sha256(DIR/'review_samples.json')))
            print('[prepared]',json.dumps(receipt['stats']),'; review the training sample before --check',flush=True);return
        frozen=contract(cfg)
        if mode=='check':
            result=checks(cfg,frozen)
            atomic_json(DIR/'readiness.json',dict(status='passed',contract=frozen,checks=result))
            print('[check-passed]',json.dumps(result),flush=True);return
        readiness=read_json(DIR/'readiness.json')
        if readiness.get('status')!='passed' or readiness['contract']!=frozen:
            raise ValueError('Readiness stale; run --check')
        if (OUTPUT/'contract.json').exists() and read_json(OUTPUT/'contract.json')!=frozen:
            raise ValueError('Run contract changed')
        if (OUTPUT/'report.json').exists() and read_json(OUTPUT/'report.json')['status']=='complete':
            print('[already-complete] No automatic extension.',flush=True);return
        if shutil.disk_usage(ROOT).free<35*1024**3:raise OSError('Need 35 GiB free for retained pilot checkpoints and reserve')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        if a.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists():raise ValueError('STOP exists; use --clear-stop')
        atomic_json(OUTPUT/'contract.json',frozen);atomic_json(OUTPUT/'config.json',cfg);snapshot(OUTPUT/'inputs')
        stop={'requested':False}
        def request_stop(*_):
            stop['requested']=True;(OUTPUT/'STOP').touch();print('[stop] Saving after current operation',flush=True)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
        with (OUTPUT/'training.log').open('a',buffering=1) as log:
            previous=sys.stdout;sys.stdout=Tee(previous,log)
            try:run(cfg,frozen,rebuild(cfg),stop)
            finally:sys.stdout=previous


if __name__=='__main__':
    # Direct script execution retains the package context for relative imports.
    __package__='sft.intact_smoltalk_base_pilot'
    main()
