"""Matched natural-response SFT. Default is a plan; --run is explicit."""
import argparse,gc,json,os,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json,file_sha256,fingerprint,lock,read_json
from sft.natural_control.protocol import DIR,OUTPUT,arms,contract,plan,code_files,validate
from sft.transfer_control.launch import preflight,Tee


def rebuild(cfg,cancelled=lambda:False):
    from sft.natural_control.data import build
    data,manifest=build(cfg,cancelled=cancelled)
    if manifest!=read_json(DIR/'selection.json'):raise ValueError('Streamed selection differs from reviewed manifest; refusing training')
    print('[data] reviewed selections reproduced:',json.dumps(manifest['stats']),flush=True)
    return data


def restore(b,opt,path,frozen,arm,cfg):
    import mlx.core as mx
    from mlx.utils import tree_map,tree_unflatten
    from sml_v1.checkpoint_bundle import resolve_bundle
    weights=resolve_bundle(path);meta=read_json(weights+'.json')
    if meta['contract']!=fingerprint(frozen) or meta['arm']!=arm:raise ValueError('Resume input contract changed')
    step=meta['step']
    if not 0<=step<=cfg['updates'] or meta['next_example']!=step*cfg['batch']:raise ValueError('Invalid resume cursor')
    b.model.load_weights(weights)
    b.model.update(tree_map(lambda x:x.astype(mx.float32),b.model.parameters()))
    opt.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
    mx.eval(b.model.parameters(),opt.state)
    if int(opt.state['step'].item())!=step:raise ValueError('Optimizer step does not match data cursor')
    return step,meta['training']


def prune_checkpoints(out,cfg):
    """Keep every evaluation candidate plus two latest saves; only this fit's bundles."""
    import shutil
    bundles=[]
    for p in out.glob('step_*_*'):
        if p.is_symlink() or not p.is_dir() or not (p/'manifest.json').exists():continue
        m=read_json(p/'manifest.json')
        if m.get('format')=='sml-pretrain-bundle-v1':bundles.append((m['created_ns'],m['step'],p))
    bundles.sort(reverse=True);keep={p for _,_,p in bundles[:2]};seen=set()
    for _,step,p in bundles:
        if step in cfg['evaluation_updates'] and step not in seen:keep.add(p);seen.add(step)
    for _,_,p in bundles:
        if p not in keep:shutil.rmtree(p)


def run_arm(arm,cfg,frozen,data,stop,output=OUTPUT):
    import mlx.core as mx
    from sft.natural_control import engine
    from sft.natural_control.evaluate import assess
    from sml_v1.checkpoint_bundle import save_bundle
    out=output/arm['name'];out.mkdir(parents=True,exist_ok=True)
    report_path=out/'report.json'
    if report_path.exists():
        old=read_json(report_path)
        if old.get('contract')!=fingerprint(frozen):raise ValueError('Existing fit belongs to another contract')
        if old.get('status')=='complete':print('[skip-completed]',arm['name'],flush=True);return
    mx.random.seed(cfg['seed']);b=engine.load(arm['model']);opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters())
    step=0;training=[];saved_step=-1;evals=out/'evaluations';evals.mkdir(exist_ok=True)
    report=dict(status='running',arm=arm,contract=fingerprint(frozen),promoted=False)
    if (out/'latest.json').exists():
        step,training=restore(b,opt,out/'latest.json',frozen,arm,cfg);saved_step=step
        print('[resume]',arm['name'],'step',step,'FP32 weights, optimizer and exact next example',flush=True)

    def save():
        nonlocal saved_step
        meta=dict(step=step,arm=arm,contract=fingerprint(frozen),next_example=step*cfg['batch'],training=training,
            experimental=True,training_format='plain-user-assistant-eos-v1',playground_compatible=False,
            note='Step is local to this new base-model fit; no automatic best pointer.')
        path=save_bundle(out,b.model,opt,meta,None,keep=1000000,reserve_gib=10.)
        saved_step=step;prune_checkpoints(out,cfg)
        report.update(step=step,latest=path);atomic_json(report_path,report)
        print('[checkpoint]',arm['name'],step,'candidate' if step in cfg['evaluation_updates'] else 'latest',flush=True)

    def evaluate():
        path=evals/f'step_{step:07d}.json'
        if path.exists():
            if read_json(path)['contract']!=fingerprint(frozen):raise ValueError('Evaluation contract mismatch')
            return
        print('[evaluation]',arm['name'],step,'saving development answers; no keyword-based selection',flush=True)
        value=assess(b,data,'dev',cfg);value.update(step=step,arm=arm,contract=fingerprint(frozen))
        atomic_json(path,value)
        print('[eval]',arm['name'],step,json.dumps({k:value[k] for k in ('natural_likelihood','prose_nll','structure','reading_exact_diagnostic')}),flush=True)

    try:
        if saved_step<0:save()
        if step in cfg['evaluation_updates']:evaluate()
        print(f"[ready] {arm['name']}: {step}/{cfg['updates']}; one pass; {len(data['train']):,} examples",flush=True)
        while step<cfg['updates'] and not stop['requested'] and not (output/'STOP').exists():
            start=step*cfg['batch'];rows=data['train'][start:start+cfg['batch']]
            if len(rows)!=cfg['batch']:raise ValueError('Incomplete batch/data exposure mismatch')
            t=time.monotonic();v=engine.update(b,opt,rows,cfg,step+1,arm['peak_lr']);step+=1
            v.update(examples=step*cfg['batch'],batch_hash=fingerprint([r['id'] for r in rows]),seconds=time.monotonic()-t)
            training.append(v)
            if step==1 or step%10==0:print(f"[natural {arm['name']} {step}/{cfg['updates']}] loss={v['loss']:.4f} lr={v['lr']:.3e} examples={v['examples']:,}",flush=True)
            if step in cfg['evaluation_updates'] or step%cfg['checkpoint_every']==0:save()
            if stop['requested'] or (output/'STOP').exists():break
            if step in cfg['evaluation_updates']:evaluate()
        # If stopped at a candidate, leave its evaluation for a later resume.
        interrupted=stop['requested'] or (output/'STOP').exists()
        if saved_step!=step:save()
        if not interrupted and step==cfg['updates']:
            evaluate();report['status']='complete'
        else:report['status']='stopped';stop['requested']=True
        report.update(step=step,training=training,evaluations=sorted(p.name for p in evals.glob('step_*.json')))
        atomic_json(report_path,report)
        print('[finished-fit]',arm['name'],report['status'],'step',step,'no automatic promotion',flush=True)
    except Exception as exc:
        report.update(status='error',error=repr(exc),step=step);atomic_json(report_path,report);raise
    finally:
        del b,opt;gc.collect();mx.clear_cache()


def readiness(cfg,frozen):
    p=DIR/'readiness.json'
    if not p.exists() or read_json(p).get('contract')!=frozen or read_json(p).get('status')!='passed':
        raise ValueError('Readiness evidence is absent/stale; run this launcher with --check')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group();g.add_argument('--prepare',action='store_true');g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true')
    ap.add_argument('--clear-stop',action='store_true');args=ap.parse_args()
    cfg=read_json(DIR/'config.json');validate(cfg);p=plan(cfg);p['mode']='run' if args.run else 'check' if args.check else 'prepare' if args.prepare else 'plan';print(json.dumps(p,indent=2),flush=True)
    if not (args.run or args.prepare or args.check):return
    preflight()
    with lock(DIR/'.launcher.lock'):
        if args.prepare:
            if (OUTPUT/'contract.json').exists():raise ValueError('An experiment exists; its data selection is frozen')
            from sft.natural_control.data import build
            _,manifest=build(cfg,show_samples=True);atomic_json(DIR/'selection.json',manifest);return
        frozen=contract(cfg)
        if args.check:
            from sft.natural_control.check import run_checks
            result=run_checks(cfg,frozen)
            if contract(cfg)!=frozen:raise ValueError('Inputs changed during checks')
            atomic_json(DIR/'readiness.json',dict(status='passed',contract=frozen,checks=result,production_training_started=False));return
        readiness(cfg,frozen)
        if (OUTPUT/'contract.json').exists() and read_json(OUTPUT/'contract.json')!=frozen:raise ValueError('Run contract differs; do not overwrite this experiment')
        # Bounded retention: five candidates plus latest saves for each fit.
        if shutil.disk_usage(ROOT).free<90*1024**3:raise OSError('Need 90 GiB free for all four fits and checkpoint reserve')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        if args.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists():raise ValueError('STOP exists; use --clear-stop to resume')
        atomic_json(OUTPUT/'contract.json',frozen);atomic_json(OUTPUT/'config.json',cfg)
        if not (OUTPUT/'inputs').exists():
            for src in code_files()+[DIR/'config.json',DIR/'selection.json',DIR/'probes.json']:
                dest=OUTPUT/'inputs'/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        stop={'requested':False}
        def request_stop(signum,frame):
            stop['requested']=True;(OUTPUT/'STOP').touch();print('[stop] Finish current operation, save, and stop; no next fit.',flush=True)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
        with (OUTPUT/'training.log').open('a',buffering=1) as log:
            original=sys.stdout;sys.stdout=Tee(original,log)
            try:
                print('[preparation] Rebuilding bounded RAM selections from pinned stream.',flush=True)
                try:data=rebuild(cfg,lambda:stop['requested'] or (OUTPUT/'STOP').exists())
                except InterruptedError:
                    print('[stopped] Stream preparation stopped. Rerun the same command when ready.',flush=True);return
                if contract(cfg)!=frozen:raise ValueError('Inputs changed during streaming')
                for arm in arms(cfg):
                    if stop['requested'] or (OUTPUT/'STOP').exists():break
                    run_arm(arm,cfg,frozen,data,stop)
                complete=all((OUTPUT/a['name']/'report.json').exists() and read_json(OUTPUT/a['name']/'report.json')['status']=='complete' for a in arms(cfg))
                if complete:
                    from sft.natural_control.evaluate import blind_packet
                    count=blind_packet(OUTPUT);atomic_json(OUTPUT/'summary.json',dict(status='complete',arms=arms(cfg),review_answers=count,selected=None,test_evaluated=False))
                    print('[finished] All four fits complete. Review review_blinded.json before choosing candidates. Test remains unevaluated.',flush=True)
                else:print('[stopped] Rerun the same command to resume exact saved cursors.',flush=True)
            finally:sys.stdout=original

if __name__=='__main__':main()
