"""One-pass complete-answer constraint SFT from preserved Balanced Skills 1920."""
import argparse
import gc
import json
import shutil
import signal
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v2.common import read_json,atomic_json,fingerprint,lock
from sft.constraint_completion_1920.protocol import DIR,OUTPUT,INPUTS,validate,contract,verify,code_files,plan
from sft.transfer_control.launch import preflight,Tee


def rebuild(cfg,cancelled=lambda:False):
    from sft.constraint_completion_1920.data import build
    data,receipt=build(cfg,cancelled)
    if receipt!=read_json(DIR/'selection.json'):raise ValueError('Rebuilt data differs from frozen selection')
    print('[data] verified',receipt['stats']['train']['complete']['count'],'distinct new tasks and 1,024 replay exposures',flush=True)
    return data


def metadata(cfg,frozen,u,training):
    return dict(step=1920+u,source_step=1920,additional_updates=u,optimizer_local_step=u,
        next_examples={f:u*n for f,n in cfg['per_update'].items()},training=training,contract=fingerprint(frozen),
        training_format='plain-user-assistant-eos-v1',tokenizer=frozen['tokenizer'],
        objective='constraint-completion-1920-v1',experimental=True,automatic_promotion=False)


def restore(b,opt,path,frozen,cfg):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v2.checkpoint_bundle import resolve_bundle
    weights=resolve_bundle(path);meta=read_json(weights+'.json');u=meta['additional_updates']
    if meta['contract']!=fingerprint(frozen) or not 0<=u<=cfg['updates']:raise ValueError('Resume contract/cursor mismatch')
    if meta['step']!=1920+u or meta['next_examples']!={f:u*n for f,n in cfg['per_update'].items()}:raise ValueError('Resume exposure mismatch')
    b.model.load_weights(weights,strict=True);opt.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
    mx.eval(b.model.parameters(),opt.state)
    if int(opt.state['step'].item())!=u:raise ValueError('Optimizer step mismatch')
    return u,meta['training']


def run(cfg,frozen,data,stop,root=OUTPUT):
    import mlx.core as mx
    from sml_v2.checkpoint_bundle import save_bundle
    from sft.constraint_completion_1920 import engine
    from sft.constraint_completion_1920.evaluate import assess,eligible,review_template
    ep=root/'evaluations';ep.mkdir(parents=True,exist_ok=True)
    mx.random.seed(cfg['seed']);b=engine.load(cfg);anchor=engine.load(cfg);opt=engine.optimizer(cfg)
    opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    u=0;training=[];saved=-1;bundle=None
    if (root/'latest.json').exists():
        u,training=restore(b,opt,root/'latest.json',frozen,cfg);saved=u
        bundle=root/read_json(root/'latest.json')['bundle'];print('[resume]',u,'exact optimizer and data cursors',flush=True)
    cancelled=lambda:stop['requested'] or (root/'STOP').exists()
    def save():
        nonlocal saved,bundle
        bundle=Path(save_bundle(root,b.model,opt,metadata(cfg,frozen,u,training),None,keep=1000000,reserve_gib=10.))
        saved=u;print('[checkpoint]',1920+u,'retained; no pruning',flush=True)
    def evaluate():
        dest=ep/f'update_{u:05d}.json'
        if dest.exists():
            result=read_json(dest)
            if result['contract']!=fingerprint(frozen) or result['bundle']!=bundle.name:raise ValueError('Evaluation identity changed')
        else:
            result=assess(b,data,'dev',cfg,cancelled)
            result.update(update=u,step=1920+u,bundle=bundle.name,contract=fingerprint(frozen))
            baseline=result if u==0 else read_json(ep/'update_00000.json')
            result['retention_gate_passed']=eligible(result['retention']['metrics'],baseline['retention']['metrics'],cfg)
            atomic_json(dest,result)
            print('[constraint-eval]',u,json.dumps(dict(**result['metrics'],retention=result['retention']['metrics'],
                retention_gate_passed=result['retention_gate_passed'])),'; content/completion review required, no automatic best',flush=True)
        review=ep/f'review_{u:05d}.json'
        if not review.exists():atomic_json(review,review_template(result))
    try:
        if saved<0:save()
        if u in cfg['evaluation_updates'] and not cancelled():evaluate()
        print(f'[ready] 1920 + {u}/128; one pass; 8 complete replies + 8 replay per update',flush=True)
        while u<cfg['updates'] and not cancelled():
            rows=engine.batch_at(data,cfg,u);t=time.monotonic()
            result=engine.update(b,anchor,opt,rows,data['anchors'][u%len(data['anchors'])],cfg,u+1)
            result['seconds']=time.monotonic()-t;result['batch_hash']=fingerprint({f:[r['id'] for r in rr] for f,rr in rows.items()})
            u+=1;training.append(result)
            if u==1 or u%8==0:print(f"[constraint-completion {u}/128] step={1920+u} complete={result['complete_ce']:.4f} rank={result['ranking_ce']:.4f} KL={result['anchor_kl']:.5f} lr={result['lr']:.3e}",flush=True)
            if u%cfg['checkpoint_every']==0 or u in cfg['evaluation_updates']:save()
            if u in cfg['evaluation_updates'] and not cancelled():evaluate()
        if saved!=u:save()
        if u==cfg['updates'] and not cancelled():evaluate()
        status='stopped' if cancelled() else 'complete'
    except (InterruptedError,KeyboardInterrupt):
        stop['requested']=True;(root/'STOP').touch()
        if saved!=u:save()
        status='stopped'
    except Exception as exc:
        atomic_json(root/'report.json',dict(status='error',update=u,last_saved_update=saved,error=repr(exc)));raise
    finally:
        del b,anchor,opt;gc.collect();mx.clear_cache()
    verify(frozen)
    atomic_json(root/'report.json',dict(status=status,update=u,step=1920+u,contract=fingerprint(frozen),
        all_checkpoints_retained=True,candidate=None,automatic_promotion=False,reserved_test_evaluated=False))
    print('[finished]',status,'step',1920+u,'; review saved answers against 1920 before selecting. No automatic restart.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    for name in ('prepare','check','run'):g.add_argument('--'+name,action='store_true')
    p.add_argument('--clear-stop',action='store_true');a=p.parse_args();cfg=read_json(DIR/'config.json');validate(cfg)
    print(json.dumps(dict(mode='run' if a.run else 'check' if a.check else 'prepare' if a.prepare else 'plan',**plan(cfg)),indent=2),flush=True)
    if not(a.prepare or a.check or a.run):return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        if a.prepare:
            if (OUTPUT/'contract.json').exists():raise ValueError('Cannot reprepare an existing run')
            from sft.constraint_completion_1920.data import build
            _,receipt=build(cfg);atomic_json(DIR/'selection.json',receipt)
            print('[prepared]',json.dumps(receipt['stats']),flush=True);return
        frozen=contract(cfg)
        if a.check:
            from sft.constraint_completion_1920.check import run_checks
            result=run_checks(cfg,frozen)
            if contract(cfg)!=frozen:raise ValueError('Inputs changed during checks')
            atomic_json(DIR/'readiness.json',dict(status='passed',contract=frozen,checks=result,production_training_started=False))
            print('[check-passed] masks, generation parity, disposable learning, exact resume and benchmark compatibility',flush=True);return
        receipt=DIR/'readiness.json'
        if not receipt.exists() or read_json(receipt).get('status')!='passed' or read_json(receipt).get('contract')!=frozen:raise ValueError('Readiness absent/stale; run --check')
        if (OUTPUT/'contract.json').exists() and read_json(OUTPUT/'contract.json')!=frozen:raise ValueError('Existing run contract changed')
        if (OUTPUT/'report.json').exists() and read_json(OUTPUT/'report.json').get('status')=='complete':print('[already-complete] No additional training.',flush=True);return
        if shutil.disk_usage(ROOT).free<35*1024**3:raise OSError('Need 35 GiB free for retained checkpoints and reserve')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        if a.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists():raise ValueError('STOP exists; use --clear-stop to resume')
        atomic_json(OUTPUT/'contract.json',frozen);atomic_json(OUTPUT/'config.json',cfg)
        for src in code_files()+[DIR/n for n in INPUTS]:
            dest=OUTPUT/'inputs'/src.relative_to(ROOT)
            if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        stop={'requested':False}
        def stop_now(*_):
            stop['requested']=True;(OUTPUT/'STOP').touch();print('[stop] Save after current operation; no automatic restart.',flush=True)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop_now)
        with (OUTPUT/'training.log').open('a',buffering=1) as log:
            original=sys.stdout;sys.stdout=Tee(original,log)
            try:
                data=rebuild(cfg,lambda:stop['requested'])
                if contract(cfg)!=frozen:raise ValueError('Inputs changed while rebuilding data')
                run(cfg,frozen,data,stop)
            except InterruptedError:print('[stopped] During preparation; no automatic restart.',flush=True)
            finally:sys.stdout=original


if __name__=='__main__':main()
