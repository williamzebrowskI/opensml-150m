"""Independent clean-reply learning-rate comparison from preserved checkpoint 512. Default plan; --run explicitly trains."""
import argparse,gc,json,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v2.common import read_json,atomic_json,fingerprint,lock
from sft.clean_reply_ab.protocol import DIR,OUTPUT,contract,plan,code_files,arm_config
from sft.transfer_control.launch import preflight,Tee

def restore(b,opt,path,frozen,cfg):
    import mlx.core as mx
    from mlx.utils import tree_unflatten,tree_map
    from sml_v2.checkpoint_bundle import resolve_bundle
    weights=resolve_bundle(path);meta=read_json(weights+'.json');n=meta['additional_updates']
    if meta['contract']!=fingerprint(frozen) or not 0<=n<=cfg['updates'] or meta['step']!=cfg['source_step']+n or meta['next_exposure']!=n*cfg['batch']:raise ValueError('Invalid resume contract/cursor')
    b.model.load_weights(weights,strict=True);b.model.update(tree_map(lambda t:t.astype(mx.float32),b.model.parameters()))
    opt.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()));mx.eval(b.model.parameters(),opt.state)
    if int(opt.state['step'].item())!=n:raise ValueError('Optimizer step mismatch')
    return n

def metadata(n,frozen,cfg):
    return dict(step=cfg['source_step']+n,source_step=cfg['source_step'],additional_updates=n,next_exposure=n*cfg['batch'],optimizer_local_step=n,contract=fingerprint(frozen),experiment='clean-reply-ab-v1',arm=cfg['arm'],experimental=True,training_format='plain-user-assistant-eos-v1',tokenizer='1194292c6d906ac19e6f01c6ad2b42825143fad426a22aa8856d903776701ecb',automatic_promotion=False)

def train(cfg,frozen,data,stop):
    import mlx.core as mx
    from sml_v2.checkpoint_bundle import save_bundle
    from sft.clean_reply_ab import engine
    from sft.clean_reply_ab.evaluate import assess
    mx.random.seed(cfg['seed']);b=engine.load(cfg);opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());n=0;saved=-1
    if (OUTPUT/'latest.json').exists():n=restore(b,opt,OUTPUT/'latest.json',frozen,cfg);saved=n
    def cancelled():return stop['requested'] or (OUTPUT/'STOP').exists() or (OUTPUT.parent/'STOP').exists()
    def save():
        nonlocal saved
        p=save_bundle(OUTPUT,b.model,opt,metadata(n,frozen,cfg),None,keep=100000,reserve_gib=10);saved=n
        print('[checkpoint]',cfg['source_step']+n,'review candidate',flush=True);return p
    def evaluate():
        p=OUTPUT/'evaluations'/f'update_{n:05d}.json'
        if p.exists():
            if read_json(p)['contract']!=fingerprint(frozen):raise ValueError('Stale evaluation')
            return
        print('[evaluation]',n,'generating complete development answers',flush=True)
        result=assess(b,data,cfg,cancelled=cancelled);result.update(update=n,step=cfg['source_step']+n,contract=fingerprint(frozen));atomic_json(p,result)
        print('[curriculum-eval]',n,json.dumps(dict(**{s:result[s]['metrics'] for s in ('trained','paraphrase','new_tasks','retention')},prose_nll=result['prose_nll'])),'; review answers, no automatic best',flush=True)
    try:
        if saved<0:save()
        if n in cfg['evaluation_updates'] and not cancelled():evaluate()
        print(f"[ready] Clean reply {cfg['arm']}: {n}/{cfg['updates']}; 384 examples, four passes; 8 new + 4 rehearsal per update",flush=True)
        while n<cfg['updates'] and not cancelled():
            rows=engine.batch_at(data,cfg,n);t=time.monotonic();result=engine.update(b,opt,rows,cfg,n+1);n+=1
            result.update(step=cfg['source_step']+n,batch_hash=fingerprint([r['id'] for r in rows]),seconds=time.monotonic()-t)
            with (OUTPUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if n==1 or n%8==0:print(f"[clean-reply {cfg['arm']} {n}/{cfg['updates']}] step={cfg['source_step']+n} loss={result['loss']:.4f} lr={result['lr']:.3e}",flush=True)
            if n%cfg['checkpoint_every']==0:save()
            if n in cfg['evaluation_updates'] and not cancelled():evaluate()
    except InterruptedError:
        stop['requested']=True
    finally:
        if saved!=n:save()
    complete=n==cfg['updates'] and not cancelled()
    atomic_json(OUTPUT/'report.json',dict(status='complete' if complete else 'stopped',updates=n,step=cfg['source_step']+n,contract=fingerprint(frozen),selected=None,arm=cfg['arm'],heldout_is_development=True,automatic_promotion=False))
    print('[finished]' if complete else '[stopped]',f"Clean reply {cfg['arm']} updates={n}; review saved development answers. No automatic promotion.",flush=True)
    del b,opt;gc.collect();mx.clear_cache()

def main():
    global OUTPUT
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true')
    p.add_argument('--arm',choices=['low','high','both'],default='both');p.add_argument('--clear-stop',action='store_true');a=p.parse_args()
    cfg=read_json(DIR/'config.json');print(json.dumps(dict(mode='run' if a.run else 'check' if a.check else 'plan',selected_arm=a.arm,**plan(cfg)),indent=2),flush=True)
    if not(a.check or a.run):return
    preflight();base=OUTPUT;names=['low','high'] if a.arm=='both' else [a.arm]
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        configs={name:arm_config(cfg,name) for name in names};frozen={name:contract(configs[name]) for name in names}
        if a.check:
            from sft.clean_reply_ab.data import rebuild
            from sft.clean_reply_ab.check import run_checks
            data=rebuild(cfg);results={}
            for name in names:
                OUTPUT=base/name;results[name]=run_checks(configs[name],frozen[name],data)
                if contract(configs[name])!=frozen[name]:raise ValueError('Inputs changed during checks')
            receipt=DIR/'readiness.json';ready=read_json(receipt) if receipt.exists() else {'arms':{}}
            for name in names:ready['arms'][name]=dict(status='passed',contract=frozen[name],checks=results[name])
            ready['production_training_started']=False;atomic_json(receipt,ready)
            print('[check-passed] Disposable updates and resume passed; no production training started.',flush=True);return
        ready=read_json(DIR/'readiness.json')
        for name in names:
            if ready['arms'].get(name,{}).get('contract')!=frozen[name] or ready['arms'][name]['status']!='passed':raise ValueError('Readiness stale; run --check')
        if all((base/n/'report.json').exists() and read_json(base/n/'report.json')['status']=='complete' for n in names):
            print('[already-complete] Selected arms are complete; no extension.');return
        if shutil.disk_usage(ROOT).free<40*1024**3:raise OSError('Need 40 GiB free')
        base.mkdir(parents=True,exist_ok=True)
        if a.clear_stop:
            (base/'STOP').unlink(missing_ok=True)
            for name in names:(base/name/'STOP').unlink(missing_ok=True)
        if (base/'STOP').exists():raise ValueError('STOP exists; use --clear-stop')
        stop={'requested':False}
        def request_stop(*_):
            stop['requested']=True;(base/'STOP').touch();print('[stop] Save current arm, then exit without starting another.',flush=True)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
        from sft.clean_reply_ab.data import rebuild
        print('[preparation] Reconstructing identical pinned rehearsal for both arms.',flush=True)
        try:data=rebuild(cfg,lambda:stop['requested'])
        except InterruptedError:print('[stopped] During data preparation.');return
        for name in names:
            if stop['requested']:break
            OUTPUT=base/name;OUTPUT.mkdir(exist_ok=True)
            if (OUTPUT/'STOP').exists():raise ValueError('Arm STOP exists; use --clear-stop')
            old=OUTPUT/'contract.json'
            if old.exists() and read_json(old)!=frozen[name]:raise ValueError('Existing arm has different inputs')
            if (OUTPUT/'report.json').exists() and read_json(OUTPUT/'report.json')['status']=='complete':
                print('[already-complete]',name,flush=True);continue
            if contract(configs[name])!=frozen[name]:raise ValueError('Inputs changed during reconstruction')
            atomic_json(old,frozen[name]);atomic_json(OUTPUT/'config.json',configs[name])
            if not (OUTPUT/'inputs').exists():
                for file in code_files()+list(DIR.glob('*.json')):
                    if file.name=='readiness.json':continue
                    dest=OUTPUT/'inputs'/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
            with (OUTPUT/'training.log').open('a',buffering=1) as log:
                original=sys.stdout;sys.stdout=Tee(original,log)
                try:train(configs[name],frozen[name],data,stop)
                finally:sys.stdout=original
        print('[done] Selected arms finished or stopped. Review answers; no automatic best or promotion.',flush=True)
if __name__=='__main__':main()
