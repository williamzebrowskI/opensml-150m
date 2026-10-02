"""Reviewed conversational branch B. Default plan; --run explicitly trains."""
import argparse,gc,json,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v2.common import read_json,atomic_json,fingerprint,lock
from sft.conversation_b.protocol import DIR,OUTPUT,contract,plan,code_files
from sft.transfer_control.launch import preflight,Tee

def restore(b,opt,path,frozen,cfg):
    import mlx.core as mx
    from mlx.utils import tree_unflatten,tree_map
    from sml_v2.checkpoint_bundle import resolve_bundle
    weights=resolve_bundle(path);meta=read_json(weights+'.json');n=meta['additional_updates']
    if meta['contract']!=fingerprint(frozen) or not 0<=n<=cfg['updates'] or meta['step']!=1920+n or meta['next_exposure']!=n*cfg['batch']:raise ValueError('Invalid resume contract/cursor')
    b.model.load_weights(weights,strict=True);b.model.update(tree_map(lambda t:t.astype(mx.float32),b.model.parameters()))
    opt.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()));mx.eval(b.model.parameters(),opt.state)
    if int(opt.state['step'].item())!=n:raise ValueError('Optimizer step mismatch')
    return n

def metadata(n,frozen,cfg):
    return dict(step=1920+n,source_step=1920,additional_updates=n,next_exposure=n*cfg['batch'],optimizer_local_step=n,contract=fingerprint(frozen),branch='B',experimental=True,training_format='plain-user-assistant-eos-v1',tokenizer='1194292c6d906ac19e6f01c6ad2b42825143fad426a22aa8856d903776701ecb',automatic_promotion=False)

def train(cfg,frozen,data,stop):
    import mlx.core as mx
    from sml_v2.checkpoint_bundle import save_bundle
    from sft.conversation_b import engine
    from sft.conversation_ab.evaluate import assess
    mx.random.seed(cfg['seed']);b=engine.load(cfg);opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());n=0;saved=-1
    if (OUTPUT/'latest.json').exists():n=restore(b,opt,OUTPUT/'latest.json',frozen,cfg);saved=n
    def cancelled():return stop['requested'] or (OUTPUT/'STOP').exists()
    def save():
        nonlocal saved
        p=save_bundle(OUTPUT,b.model,opt,metadata(n,frozen,cfg),None,keep=100000,reserve_gib=10);saved=n
        print('[checkpoint]',1920+n,'review candidate',flush=True);return p
    def evaluate():
        p=OUTPUT/'evaluations'/f'update_{n:05d}.json'
        if p.exists():
            if read_json(p)['contract']!=fingerprint(frozen):raise ValueError('Stale evaluation')
            return
        print('[evaluation]',n,'generating complete development answers',flush=True)
        result=assess(b,data,cfg,cancelled=cancelled);result.update(update=n,step=1920+n,contract=fingerprint(frozen));atomic_json(p,result)
        print('[conversation-B-eval]',n,json.dumps(result['metrics']),'; review answers, no automatic best',flush=True)
    try:
        if saved<0:save()
        if n in cfg['evaluation_updates'] and not cancelled():evaluate()
        print(f"[ready] Branch B: {n}/{cfg['updates']}; 160 examples, three passes; preserved 1920 parent",flush=True)
        while n<cfg['updates'] and not cancelled():
            rows=engine.batch_at(data,cfg,n);t=time.monotonic();result=engine.update(b,opt,rows,cfg,n+1);n+=1
            result.update(step=1920+n,batch_hash=fingerprint([r['id'] for r in rows]),seconds=time.monotonic()-t)
            with (OUTPUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if n==1 or n%8==0:print(f"[conversation-B {n}/{cfg['updates']}] step={1920+n} CE={result['ce']:.4f} UL={result['ul']:.4f} repeat_tokens={result['negative_tokens']} lr={result['lr']:.3e}",flush=True)
            if n%cfg['checkpoint_every']==0:save()
            if n in cfg['evaluation_updates'] and not cancelled():evaluate()
    except InterruptedError:
        stop['requested']=True
    finally:
        if saved!=n:save()
    complete=n==cfg['updates'] and not cancelled()
    atomic_json(OUTPUT/'report.json',dict(status='complete' if complete else 'stopped',updates=n,step=1920+n,contract=fingerprint(frozen),selected=None,test_evaluated=False,automatic_promotion=False))
    print('[finished]' if complete else '[stopped]',f'Branch B updates={n}; review saved development answers. No automatic promotion or further training.',flush=True)
    del b,opt;gc.collect();mx.clear_cache()

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true');p.add_argument('--clear-stop',action='store_true');a=p.parse_args()
    cfg=read_json(DIR/'config.json');print(json.dumps(dict(mode='run' if a.run else 'check' if a.check else 'plan',**plan(cfg)),indent=2),flush=True)
    if not(a.check or a.run):return
    preflight()
    with lock(DIR/'.launcher.lock'):
        frozen=contract(cfg)
        if a.check:
            from sft.conversation_b.check import run_checks
            results=run_checks(cfg,frozen)
            if contract(cfg)!=frozen:raise ValueError('Inputs changed during readiness checks')
            atomic_json(DIR/'readiness.json',dict(status='passed',contract=frozen,checks=results,production_training_started=False));print('[check-passed] Branch B ready; production training has not started.',flush=True);return
        ready=read_json(DIR/'readiness.json')
        if ready.get('status')!='passed' or ready['contract']!=frozen:raise ValueError('Readiness stale; run --check')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        old=OUTPUT/'contract.json'
        if old.exists() and read_json(old)!=frozen:raise ValueError('Existing output belongs to different inputs')
        if (OUTPUT/'report.json').exists() and read_json(OUTPUT/'report.json')['status']=='complete':print('[already-complete] No further training.',flush=True);return
        if shutil.disk_usage(ROOT).free<25*1024**3:raise OSError('Need 25 GiB free for saved candidates and reserve')
        if a.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists():raise ValueError('STOP exists; use --clear-stop to resume')
        atomic_json(old,frozen);atomic_json(OUTPUT/'config.json',cfg)
        if not (OUTPUT/'inputs').exists():
            for src in code_files()+[DIR/'config.json']+[ROOT/'sft/conversation_ab'/n for n in ('config.json','selection.json','originals.json','review.json')]:
                dest=OUTPUT/'inputs'/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        stop={'requested':False}
        def request_stop(*_):
            stop['requested']=True;(OUTPUT/'STOP').touch();print('[stop] Finish current update, save and exit. No automatic restart.',flush=True)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
        with (OUTPUT/'training.log').open('a',buffering=1) as log:
            original=sys.stdout;sys.stdout=Tee(original,log)
            try:
                from sml_v2.tokenization import Tokenizer
                from sft.conversation_ab.data import rebuild
                try:data=rebuild(Tokenizer(ROOT/'tokenizer/bytebpe32k_v1'),lambda:stop['requested'] or (OUTPUT/'STOP').exists())
                except InterruptedError:print('[stopped] Data preparation cancelled.');return
                if contract(cfg)!=frozen:raise ValueError('Inputs changed during data preparation')
                if not stop['requested']:train(cfg,frozen,data,stop)
            finally:sys.stdout=original
if __name__=='__main__':main()
