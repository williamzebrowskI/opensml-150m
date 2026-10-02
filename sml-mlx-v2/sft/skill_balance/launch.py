"""Prepare/check a balanced 896 continuation; --run starts the user-owned job."""
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
from sft.skill_balance.protocol import DIR,OUTPUT,contract,code_files,plan,validate,verify
from sft.transfer_control.launch import preflight,Tee


def rebuild(cfg,cancelled=lambda:False):
    from sft.skill_balance.data import build
    data,selection=build(cfg,cancelled)
    if selection!=read_json(DIR/'selection.json'): raise ValueError('Rebuilt source selection differs from reviewed data')
    return data


def restore(b,opt,path,frozen,cfg):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v2.checkpoint_bundle import resolve_bundle
    weights=resolve_bundle(path); meta=read_json(weights+'.json')
    u=meta['additional_updates']
    if meta['contract']!=fingerprint(frozen) or not 0<=u<=cfg['updates']: raise ValueError('Resume contract/cursor changed')
    if meta['step']!=cfg['source_step']+u or meta['next_examples']!={k:u*n for k,n in cfg['per_update'].items()}:
        raise ValueError('Resume exposure mismatch')
    b.model.load_weights(weights,strict=True)
    opt.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
    mx.eval(b.model.parameters(),opt.state)
    if int(opt.state['step'].item())!=u: raise ValueError('Fresh optimizer step/cursor mismatch')
    return u,meta['training']


def prune(out,cfg):
    bundles=[]
    for p in out.glob('step_*_*'):
        if p.is_symlink() or not p.is_dir() or not (p/'manifest.json').exists(): continue
        m=read_json(p/'manifest.json')
        if m.get('format')=='sml-pretrain-bundle-v1': bundles.append((m['created_ns'],m['step'],p))
    bundles.sort(reverse=True); keep={p for _,_,p in bundles[:2]}; seen=set()
    candidates={cfg['source_step']+u for u in cfg['evaluation_updates']}
    for _,step,p in bundles:
        if step in candidates and step not in seen: keep.add(p); seen.add(step)
    for pointer in ('candidate.json','selection_frozen.json'):
        if (out/pointer).exists():
            value=read_json(out/pointer)
            if value.get('bundle'): keep.add(out/value['bundle'])
    for _,_,p in bundles:
        if p not in keep: shutil.rmtree(p)


def run(cfg,frozen,data,stop,output=OUTPUT):
    import mlx.core as mx
    from sml_v2.checkpoint_bundle import save_bundle
    from sft.skill_balance import engine
    from sft.skill_balance.evaluate import assess,eligible,score
    output.mkdir(parents=True,exist_ok=True); ep=output/'evaluations'; ep.mkdir(exist_ok=True)
    b=engine.load(cfg); anchor=engine.load(cfg); opt=engine.optimizer(cfg); opt.init(b.model.trainable_parameters())
    mx.eval(opt.state); u=0; training=[]; saved=-1; bundle=None
    if (output/'latest.json').exists():
        u,training=restore(b,opt,output/'latest.json',frozen,cfg); saved=u
        bundle=output/read_json(output/'latest.json')['bundle']
        print('[resume]',u,'exact saved optimizer, family cursors, and frozen 896 anchor',flush=True)
    cancelled=lambda:stop['requested'] or (output/'STOP').exists()

    def save():
        nonlocal saved,bundle
        meta=dict(step=cfg['source_step']+u,source_step=cfg['source_step'],additional_updates=u,
            optimizer_local_step=u,next_examples={k:u*n for k,n in cfg['per_update'].items()},
            contract=fingerprint(frozen),training=training,training_format='plain-user-assistant-eos-v1',
            experimental=True,automatic_promotion=False,tokenizer=frozen.get('tokenizer'),
            objective='balanced-supervision-ranking-parent-KL-v1')
        bundle=Path(save_bundle(output,b.model,opt,meta,None,keep=1000000,reserve_gib=10.)); saved=u
        prune(output,cfg); print('[checkpoint]',meta['step'],'development candidate' if u in cfg['evaluation_updates'] else 'latest',flush=True)

    def evaluation():
        path=ep/f'update_{u:05d}.json'
        if path.exists():
            value=read_json(path)
            if value['contract']!=fingerprint(frozen): raise ValueError('Evaluation contract changed')
            return
        result=assess(b,data,'dev',cfg,cancelled); metrics=result['metrics']
        result.update(update=u,step=cfg['source_step']+u,contract=fingerprint(frozen),bundle=bundle.name)
        if u==0:
            baseline=metrics
        else:
            baseline=read_json(ep/'update_00000.json')['metrics']
        ok=eligible(metrics,baseline,cfg); result['eligible_for_review']=ok
        atomic_json(path,result)
        pointer=output/'candidate.json'
        previous=read_json(pointer) if pointer.exists() else None
        if u==0 or (ok and score(metrics)>=score(baseline)+cfg['candidate_min_score_gain']
                       and (previous is None or score(metrics)>previous['proxy_score'])):
            atomic_json(pointer,dict(bundle=bundle.name,step=cfg['source_step']+u,update=u,
                proxy_score=score(metrics),evaluation=path.name,review_required=True,promoted=False))
        print('[skill-eval]',u,json.dumps(dict(**metrics,eligible_for_review=ok)),flush=True)

    try:
        if saved<0: save()
        if u in cfg['evaluation_updates'] and not cancelled(): evaluation()
        print(f"[ready] 896 + {u}/{cfg['updates']}; main Mac only; 4 commonsense + 4 instruction + 2 reading per update",flush=True)
        while u<cfg['updates'] and not cancelled():
            rows=engine.batch_at(data,cfg,u); t=time.monotonic()
            v=engine.update(b,anchor,opt,rows,data['anchors'][u%len(data['anchors'])],cfg,u+1)
            u+=1; v['seconds']=time.monotonic()-t; v['batch_hash']=fingerprint({k:[r['id'] for r in rr] for k,rr in rows.items()}); training.append(v)
            if u==1 or u%16==0:
                print(f"[skill-balance {u}/{cfg['updates']}] step={896+u} loss={v['loss']:.4f} rank={v['ranking_ce']:.4f} KL={v['anchor_kl']:.5f} lr={v['lr']:.3e}",flush=True)
            if u%cfg['checkpoint_every']==0 or u in cfg['evaluation_updates']: save()
            if u in cfg['evaluation_updates'] and not cancelled(): evaluation()
        if saved!=u: save()
        if not cancelled() and u==cfg['updates']: evaluation()
        status='stopped' if cancelled() else 'complete'
    except (InterruptedError,KeyboardInterrupt):
        stop['requested']=True; (output/'STOP').touch()
        if saved!=u: save()
        status='stopped'
    except Exception as exc:
        atomic_json(output/'report.json',dict(status='error',update=u,error=repr(exc),last_saved_update=saved))
        raise
    finally:
        del b,anchor,opt; gc.collect(); mx.clear_cache()
    verify(frozen)
    candidate=read_json(output/'candidate.json') if (output/'candidate.json').exists() else None
    atomic_json(output/'report.json',dict(status=status,update=u,step=896+u,contract=fingerprint(frozen),
        candidate=candidate,test_evaluated=False,public_benchmarks_evaluated=False,automatic_promotion=False))
    print('[finished]',status,'candidate=',candidate,'; review answers before testing/promotion.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group(); action.add_argument('--prepare',action='store_true'); action.add_argument('--check',action='store_true'); action.add_argument('--run',action='store_true')
    p.add_argument('--clear-stop',action='store_true'); args=p.parse_args()
    cfg=read_json(DIR/'config.json'); validate(cfg)
    print(json.dumps(dict(mode='run' if args.run else 'check' if args.check else 'prepare' if args.prepare else 'plan',**plan(cfg)),indent=2),flush=True)
    if not (args.run or args.check or args.prepare): return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        if args.prepare:
            if (OUTPUT/'contract.json').exists(): raise ValueError('Cannot reprepare an existing experiment')
            from sft.skill_balance.data import build
            data,selection=build(cfg); atomic_json(DIR/'selection.json',selection)
            atomic_json(DIR/'review_samples.json',{s:{f:rr[:12] for f,rr in data[s].items()} for s in ('train','dev')})
            print('[prepared]',json.dumps(selection['stats']['train']),flush=True); return
        frozen=contract(cfg)
        if args.check:
            from sft.skill_balance.check import run_checks
            result=run_checks(cfg,frozen)
            if contract(cfg)!=frozen: raise ValueError('Inputs changed during checks')
            atomic_json(DIR/'readiness.json',dict(status='passed',contract=frozen,checks=result,production_training_started=False)); return
        receipt=DIR/'readiness.json'
        if not receipt.exists() or read_json(receipt).get('status')!='passed' or read_json(receipt).get('contract')!=frozen:
            raise ValueError('Readiness absent/stale; run with --check')
        if (OUTPUT/'contract.json').exists() and read_json(OUTPUT/'contract.json')!=frozen: raise ValueError('Existing run contract differs')
        if shutil.disk_usage(ROOT).free<35*1024**3: raise OSError('Need 35 GiB available for retained candidates and save reserve')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        if args.clear_stop: (OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists(): raise ValueError('STOP exists; use --clear-stop to resume')
        atomic_json(OUTPUT/'contract.json',frozen); atomic_json(OUTPUT/'config.json',cfg)
        for src in code_files()+[DIR/'config.json',DIR/'selection.json',DIR/'probes.json']:
            dest=OUTPUT/'inputs'/src.relative_to(ROOT)
            if not dest.exists(): dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)
        stop={'requested':False}
        def request_stop(*_):
            stop['requested']=True; (OUTPUT/'STOP').touch(); print('[stop] Finish current operation and save. No automatic restart.',flush=True)
        for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,request_stop)
        with (OUTPUT/'training.log').open('a',buffering=1) as log:
            original=sys.stdout; sys.stdout=Tee(original,log)
            try:
                data=rebuild(cfg,lambda:stop['requested'])
                if not stop['requested']:
                    if contract(cfg)!=frozen: raise ValueError('Inputs changed during preparation')
                    run(cfg,frozen,data,stop)
            except InterruptedError: print('[stopped] No training during cancelled data preparation.',flush=True)
            finally: sys.stdout=original


if __name__=='__main__': main()
