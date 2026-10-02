"""Conversational SFT continuation from preserved High LR 640. Default plan; --run explicitly trains."""
import argparse,gc,json,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import read_json,atomic_json,fingerprint,lock
from sft.two_turn_640.protocol import DIR,OUTPUT,contract,plan,code_files
from sft.transfer_control.launch import preflight,Tee

def metadata(n,frozen,cfg):
    return dict(step=640+n,source_step=640,additional_updates=n,next_exposure=n*cfg['batch'],optimizer_local_step=n,contract=fingerprint(frozen),experiment='two-turn-640-v1',experimental=True,method='full-parameter assistant-only SFT',parent=cfg['source_bundle'],training_format='plain-user-assistant-eos-v1',tokenizer='1194292c6d906ac19e6f01c6ad2b42825143fad426a22aa8856d903776701ecb',automatic_promotion=False)

def train(cfg,frozen,data,stop):
    import mlx.core as mx
    from sft.two_turn_640 import engine
    from sft.two_turn_640.evaluate import assess
    mx.random.seed(cfg['seed']);b=engine.load(cfg);opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());n=0;saved=-1
    if (OUTPUT/'latest.json').exists():n=engine.restore(b,opt,OUTPUT/'latest.json',frozen,cfg);saved=n
    def cancelled():return stop['requested'] or (OUTPUT/'STOP').exists()
    def save():
        nonlocal saved
        from sml_v1.checkpoint_bundle import save_bundle
        p=save_bundle(OUTPUT,b.model,opt,metadata(n,frozen,cfg),None,keep=100000,reserve_gib=10);saved=n
        print('[checkpoint]',640+n,'review candidate',flush=True);return p
    def evaluate():
        p=OUTPUT/'evaluations'/f'update_{n:05d}.json'
        if p.exists():
            if read_json(p)['contract']!=fingerprint(frozen):raise ValueError('Stale evaluation')
            return
        print('[evaluation]',n,'generating complete development answers',flush=True)
        result=assess(b,data,cfg,cancelled=cancelled);result.update(update=n,step=640+n,contract=fingerprint(frozen));atomic_json(p,result)
        print('[conversation-eval]',n,json.dumps(result['metrics']),'; review answers, no automatic best',flush=True)
    try:
        if saved<0:save()
        if n in cfg['evaluation_updates'] and not cancelled():evaluate()
        print(f"[ready] Conversation: {n}/{cfg['updates']}; 768 new targets once + 256 rehearsal targets three times; High LR 640",flush=True)
        while n<cfg['updates'] and not cancelled():
            rows=engine.batch_at(data,cfg,n);t=time.monotonic();result=engine.update(b,opt,rows,cfg,n+1);n+=1
            result.update(step=640+n,batch_hash=fingerprint([r['id'] for r in rows]),seconds=time.monotonic()-t)
            with (OUTPUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if n==1 or n%8==0:print(f"[two-turn-640 {n}/{cfg['updates']}] step={640+n} loss={result['loss']:.4f} lr={result['lr']:.3e}",flush=True)
            if n%cfg['checkpoint_every']==0 or n in cfg['evaluation_updates']:save()
            if n in cfg['evaluation_updates'] and not cancelled():evaluate()
    except InterruptedError:
        stop['requested']=True
    finally:
        if saved!=n:save()
    complete=n==cfg['updates'] and not cancelled()
    atomic_json(OUTPUT/'report.json',dict(status='complete' if complete else 'stopped',updates=n,step=640+n,contract=fingerprint(frozen),selected=None,test_evaluated=False,automatic_promotion=False))
    print('[finished]' if complete else '[stopped]',f'Conversation updates={n}; review saved development answers. No automatic promotion.',flush=True)
    del b,opt;gc.collect();mx.clear_cache()

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true');p.add_argument('--clear-stop',action='store_true');a=p.parse_args()
    cfg=read_json(DIR/'config.json');print(json.dumps(dict(mode='run' if a.run else 'check' if a.check else 'plan',**plan(cfg)),indent=2),flush=True)
    if not(a.check or a.run):return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        frozen=contract(cfg)
        if a.check:
            from sft.two_turn_640.check import run_checks
            results=run_checks(cfg,frozen)
            if contract(cfg)!=frozen:raise ValueError('Inputs changed during readiness checks')
            atomic_json(DIR/'readiness.json',dict(status='passed',contract=frozen,checks=results,production_training_started=False));print('[check-passed] Two-turn continuation ready; production training has not started.',flush=True);return
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
            for src in code_files()+[DIR/n for n in ('config.json','selection.json','review.json')]:
                dest=OUTPUT/'inputs'/src.relative_to(ROOT.parent);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        stop={'requested':False}
        def request_stop(*_):
            stop['requested']=True;(OUTPUT/'STOP').touch();print('[stop] Finish current update, save and exit. No automatic restart.',flush=True)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
        with (OUTPUT/'training.log').open('a',buffering=1) as log:
            original=sys.stdout;sys.stdout=Tee(original,log)
            try:
                from sml_v1.tokenization import Tokenizer
                from sft.two_turn_640.data import rebuild
                try:data=rebuild(cfg)
                except InterruptedError:print('[stopped] Data preparation cancelled.');return
                if contract(cfg)!=frozen:raise ValueError('Inputs changed during data preparation')
                if not stop['requested']:train(cfg,frozen,data,stop)
            finally:sys.stdout=original
if __name__=='__main__':main()
