"""Prepare, validate, and run the 1472 balanced-transfer continuation."""
import argparse
import gc
import json
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sml_v2.common import atomic_json,read_json,fingerprint,lock
from sft.balanced_transfer_1472.protocol import DIR,OUTPUT,contract,code_files,plan,validate,verify
from sft.transfer_control.launch import preflight,Tee


def restore(b,opt,path,frozen,cfg):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v2.checkpoint_bundle import resolve_bundle
    weights=resolve_bundle(path);meta=read_json(weights+'.json');u=meta['additional_updates']
    if meta['contract']!=fingerprint(frozen) or not 0<=u<=cfg['updates']:
        raise ValueError('Resume contract/cursor changed')
    if meta['step']!=cfg['source_step']+u or meta['next_examples']!={k:u*n for k,n in cfg['per_update'].items()}:
        raise ValueError('Resume exposure mismatch')
    b.model.load_weights(weights,strict=True)
    opt.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
    mx.eval(b.model.parameters(),opt.state)
    if int(opt.state['step'].item())!=u:
        raise ValueError('Optimizer step/cursor mismatch')
    return u,meta['training']


def run(cfg,frozen,data,stop):
    import mlx.core as mx
    from sml_v2.checkpoint_bundle import save_bundle
    from sft.balanced_transfer_1472 import engine
    from sft.balanced_transfer_1472.evaluate import assess,eligible
    OUTPUT.mkdir(parents=True,exist_ok=True);ep=OUTPUT/'evaluations';ep.mkdir(exist_ok=True)
    mx.random.seed(cfg['seed'])
    b=engine.load(cfg);anchor=engine.load(cfg);opt=engine.optimizer(cfg)
    opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    u=0;training=[];saved=-1;bundle=None
    if (OUTPUT/'latest.json').exists():
        u,training=restore(b,opt,OUTPUT/'latest.json',frozen,cfg);saved=u
        bundle=OUTPUT/read_json(OUTPUT/'latest.json')['bundle']
        print('[resume]',u,'exact FP32 optimizer and family cursors',flush=True)
    cancelled=lambda:stop['requested'] or (OUTPUT/'STOP').exists()

    def save():
        nonlocal saved,bundle
        meta=dict(step=cfg['source_step']+u,source_step=cfg['source_step'],additional_updates=u,
            optimizer_local_step=u,next_examples={k:u*n for k,n in cfg['per_update'].items()},
            contract=fingerprint(frozen),training=training,training_format='plain-user-assistant-eos-v1',
            experimental=True,automatic_promotion=False,tokenizer=frozen['tokenizer'],
            objective='balanced-context-ce-choice-ranking-prose-kl-v1')
        bundle=Path(save_bundle(OUTPUT,b.model,opt,meta,None,keep=1000000,reserve_gib=10.))
        saved=u
        print('[checkpoint]',meta['step'],'retained for review',flush=True)

    def evaluation():
        path=ep/f'update_{u:05d}.json'
        if path.exists():
            if read_json(path)['contract']!=fingerprint(frozen):
                raise ValueError('Evaluation contract changed')
            return
        result=assess(b,data,cfg,cancelled);m=result['metrics']
        baseline=m if u==0 else read_json(ep/'update_00000.json')['metrics']
        result.update(update=u,step=cfg['source_step']+u,contract=fingerprint(frozen),
            bundle=bundle.name,eligible_for_review=eligible(m,baseline,cfg),
            semantic_review_required=True)
        atomic_json(path,result)
        print('[balanced-eval]',u,json.dumps(dict(context=m['new_context_macro'],
            old_choice=m['commonsense_macro'],instruction=m['instruction_proxy'],
            reading=m['reading_exact'],explanation_verdict=m['explanation_verdict'],
            verdict_and_surface=m['explanation_verdict_and_surface'],
            stopped=m['stopped'],prose_nll=m['prose_nll'],
            eligible_for_review=result['eligible_for_review'])),
            '; inspect saved answers before choosing',flush=True)

    try:
        if saved<0:save()
        if u in cfg['evaluation_updates'] and not cancelled():evaluation()
        print(f"[ready] 1472 + {u}/512; 2,048 new independent situations; one pass; main Mac only",flush=True)
        while u<cfg['updates'] and not cancelled():
            rows=engine.batch_at(data,cfg,u);start=time.monotonic()
            result=engine.update(b,anchor,opt,rows,data['anchors'][u%len(data['anchors'])],cfg,u+1)
            u+=1;result['seconds']=time.monotonic()-start
            result['batch_hash']=fingerprint({k:[r['id'] for r in rr] for k,rr in rows.items()})
            training.append(result)
            if u==1 or u%16==0:
                print(f"[balanced {u}/512] step={1472+u} loss={result['loss']:.4f} "
                      f"rank={result['ranking_ce']:.4f} KL={result['anchor_kl']:.5f} lr={result['lr']:.3e}",flush=True)
            if u%cfg['checkpoint_every']==0:save()
            if u in cfg['evaluation_updates'] and not cancelled():evaluation()
        if saved!=u:save()
        status='stopped' if cancelled() else 'complete'
    except (InterruptedError,KeyboardInterrupt):
        stop['requested']=True;(OUTPUT/'STOP').touch()
        if saved!=u:save()
        status='stopped'
    except Exception as exc:
        atomic_json(OUTPUT/'report.json',dict(status='error',update=u,error=repr(exc),last_saved_update=saved))
        raise
    finally:
        del b,anchor,opt;gc.collect();mx.clear_cache()
    verify(frozen)
    atomic_json(OUTPUT/'report.json',dict(status=status,update=u,step=1472+u,
        contract=fingerprint(frozen),automatic_promotion=False,
        test_evaluated=False,public_benchmarks_evaluated=False))
    print('[finished]',status,'review development answers; no automatic promotion',flush=True)


def check(cfg,frozen,data):
    import mlx.core as mx
    import tempfile
    from sft.balanced_transfer_1472 import engine
    from sft.balanced_transfer_1472.evaluate import assess
    from sft.transfer_control.engine import encode
    from sml_v2.checkpoint_bundle import save_bundle
    for family,n in cfg['per_update'].items():
        if len(data['train'][family])!=cfg['updates']*n:raise ValueError('Incomplete family '+family)
    for family,rows in data['train'].items():
        for row in rows[:2]:
            if not row.get('prompt') or not row.get('answer'):
                raise ValueError('Missing prompt/answer in '+family)
    backend=engine.load(cfg);anchor=engine.load(cfg)
    for family in ('new_answer','instruction','reading','explanation'):
        for row in data['train'][family][:2]:
            encoded=encode(backend,row,cfg['context'])
            if encoded['targets']<1 or encoded['y'][-1]!=backend.eos:
                raise ValueError('Answer/EOS masking failed: '+family)
    for family in ('new_answer','new_choice','old_choice'):
        row=data['train'][family][0];engine.choice_arrays(backend,row)
    opt=engine.optimizer(cfg);opt.init(backend.model.trainable_parameters());mx.eval(opt.state)
    result=engine.update(backend,anchor,opt,engine.batch_at(data,cfg,0),data['anchors'][0],cfg,1)
    if int(opt.state['step'].item())!=1:raise ValueError('Disposable update did not advance')
    with tempfile.TemporaryDirectory(prefix='opensml-balanced-check-') as temp:
        meta=dict(step=1473,source_step=1472,additional_updates=1,contract=fingerprint(frozen),
            next_examples=dict(cfg['per_update']),training=[result])
        saved=save_bundle(Path(temp)/'resume',backend.model,opt,meta,None,keep=1)
        restored=engine.load(cfg);restored_opt=engine.optimizer(cfg)
        step,_=restore(restored,restored_opt,saved,frozen,cfg)
        if step!=1:raise ValueError('Disposable checkpoint did not resume at step 1')
        tiny={'dev':{family:rows[:4] for family,rows in data['dev'].items()}}
        smoke=assess(restored,tiny,cfg)
        if not smoke['explanation_answers'] or not smoke['knowledge_answers']:
            raise ValueError('Development evaluation smoke check did not generate answers')
        del restored,restored_opt
    del backend,anchor,opt;gc.collect();mx.clear_cache();verify(frozen)
    return dict(prepared_selection_verified=True,source_1472_preserved=True,
        first_disposable_update=result,checkpoint_resume_step=step,
        evaluation_smoke=dict(knowledge=len(smoke['knowledge_answers']),
            explanation=len(smoke['explanation_answers'])),production_updates=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--check',action='store_true')
    group.add_argument('--run',action='store_true')
    parser.add_argument('--clear-stop',action='store_true')
    args=parser.parse_args()
    cfg=read_json(DIR/'config.json');validate(cfg)
    print(json.dumps(dict(mode='prepare' if args.prepare else 'check' if args.check else 'run' if args.run else 'plan',
        **plan(cfg)),indent=2),flush=True)
    if not(args.prepare or args.check or args.run):return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        if args.prepare:
            if (OUTPUT/'contract.json').exists():raise ValueError('Existing run cannot be re-prepared')
            from sft.balanced_transfer_1472.data import build
            data,selection=build(cfg)
            atomic_json(DIR/'prepared.json',data);atomic_json(DIR/'selection.json',selection)
            atomic_json(DIR/'review_samples.json',{family:rows[:12] for family,rows in data['train'].items()})
            print('[prepared]',json.dumps(selection['stats']['train']),flush=True)
            return
        frozen=contract(cfg);data=read_json(DIR/'prepared.json')
        if args.check:
            result=check(cfg,frozen,data)
            if contract(cfg)!=frozen:raise ValueError('Inputs changed during checks')
            atomic_json(DIR/'readiness.json',dict(status='passed',contract=frozen,checks=result,
                production_training_started=False))
            print('[check-passed]',json.dumps(result),flush=True)
            return
        receipt=DIR/'readiness.json'
        if not receipt.exists() or read_json(receipt).get('status')!='passed' or read_json(receipt)['contract']!=frozen:
            raise ValueError('Run --check before production')
        if (OUTPUT/'contract.json').exists() and read_json(OUTPUT/'contract.json')!=frozen:
            raise ValueError('Existing run contract changed')
        if (OUTPUT/'report.json').exists() and read_json(OUTPUT/'report.json').get('status')=='complete':
            print('[already-complete] No more training',flush=True);return
        if shutil.disk_usage(ROOT).free<50*1024**3:raise OSError('Need 50 GiB free for retained checkpoints')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        if args.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists():raise ValueError('STOP exists; use --clear-stop to resume')
        atomic_json(OUTPUT/'contract.json',frozen);atomic_json(OUTPUT/'config.json',cfg)
        for src in code_files()+[DIR/n for n in ('config.json','selection.json','prepared.json')]:
            dest=OUTPUT/'inputs'/src.relative_to(ROOT)
            if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        stop={'requested':False}
        def request_stop(*_):
            stop['requested']=True;(OUTPUT/'STOP').touch()
            print('[stop] Finish current update/evaluation and save; no automatic restart',flush=True)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
        with (OUTPUT/'training.log').open('a',buffering=1) as log:
            original=sys.stdout;sys.stdout=Tee(original,log)
            try:run(cfg,frozen,data,stop)
            finally:sys.stdout=original


if __name__=='__main__':main()
