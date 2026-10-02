"""Prepare/check/run a resumable 2,048-update preference branch from preserved 640."""
import argparse,gc,json,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v2.common import atomic_json,read_json,fingerprint,lock
from sft.transfer_control.launch import preflight,Tee
from sft.corrective_640.launch import restore
from sft.preference_long_640.protocol import DIR,OUTPUT,validate,contract,code_files,verify,plan


def run(cfg,frozen,data,stop):
    import mlx.core as mx
    from sft.preference_long_640 import engine
    from sft.preference_long_640.evaluate import assess,failures
    from sml_v2.checkpoint_bundle import save_bundle
    mx.random.seed(cfg['seed']);b=engine.load(cfg);anchor=engine.load(cfg)
    opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    u=0;training=[];saved=-1;bundle=None;streak=0;guarded=False
    if (OUTPUT/'latest.json').exists():
        u,training=restore(b,opt,OUTPUT/'latest.json',frozen,cfg);saved=u
        bundle=OUTPUT/read_json(OUTPUT/'latest.json')['bundle']
        print('[resume]',u,'weights, optimizer, and deterministic batch cursor restored',flush=True)
    evaluations=OUTPUT/'evaluations';evaluations.mkdir(exist_ok=True)
    if not stop.get('reset_guard', False):
        prior=[read_json(p) for p in sorted(evaluations.glob('update_*.json')) if int(p.stem.split('_')[1])<=u]
        streak=prior[-1].get('consecutive_failures',0) if prior else 0
    cancelled=lambda:stop['requested'] or (OUTPUT/'STOP').exists()
    def save():
        nonlocal saved,bundle
        meta=dict(step=640+u,source_step=640,additional_updates=u,next_examples={f:u*n for f,n in cfg['per_update'].items()},contract=fingerprint(frozen),training=training,training_format='plain-user-assistant-eos-v1',tokenizer=frozen['tokenizer'],source_model_sha256=cfg['source_model_sha256'],objective='reference-dpo-chosen-ce-rehearsal-chat-prose-kl-v1',fresh_optimizer=True,experimental=True,automatic_promotion=False)
        bundle=Path(save_bundle(OUTPUT,b.model,opt,meta,None,keep=1000000,reserve_gib=10.));saved=u
        print('[checkpoint]',640+u,'retained',flush=True)
    def evaluate():
        nonlocal streak,guarded
        path=evaluations/f'update_{u:05d}.json'
        if path.exists():
            result=read_json(path)
            if result['contract']!=fingerprint(frozen) or result['step']!=640+u:raise ValueError('Evaluation identity mismatch')
            return
        result=assess(b,anchor,data,cfg,cancelled)
        baseline=result['metrics'] if u==0 else read_json(evaluations/'update_00000.json')['metrics']
        failed=failures(result['metrics'],baseline,cfg)
        streak=streak+1 if failed else 0
        result.update(update=u,step=640+u,bundle=bundle.name,contract=fingerprint(frozen),retention_failures=failed,consecutive_failures=streak,content_review_required=True)
        atomic_json(path,result)
        print('[preference-eval]',u,json.dumps(result['metrics']),'retention_failures=',failed,flush=True)
        if streak>=cfg['retention']['consecutive_failures']:
            guarded=True;(OUTPUT/'STOP').touch();atomic_json(OUTPUT/'regression_stop.json',dict(update=u,failures=failed,consecutive_failures=streak,requires_review=True))
            print('[regression-stop] Two consecutive development checks regressed. Checkpoints retained; review before --clear-stop.',flush=True)
    try:
        if saved<0:save()
        if u in cfg['evaluation_updates'] and not cancelled():evaluate()
        print(f'[ready] preserved 640 + {u}/{cfg["updates"]}; preference continuation, no public benchmark training',flush=True)
        while u<cfg['updates'] and not cancelled():
            start=time.monotonic();rows=engine.batch_at(data,cfg,u)
            result=engine.update(b,anchor,opt,rows,data['anchors'][u%len(data['anchors'])],cfg,u+1)
            u+=1;result['seconds']=time.monotonic()-start;result['batch_hash']=fingerprint({f:[r['id'] for r in rr] for f,rr in rows.items()});training.append(result)
            if u==1 or u%8==0:print(f'[preference {u}/{cfg["updates"]}] step={640+u} loss={result["loss"]:.4f} dpo={result["dpo_loss"]:.4f} chosen_nll={result["chosen_nll"]:.4f} rank={result["ranking_ce"]:.4f} lr={result["lr"]:.3e} seconds={result["seconds"]:.1f}',flush=True)
            if u%cfg['checkpoint_every']==0 or u in cfg['evaluation_updates']:save()
            if u in cfg['evaluation_updates'] and not cancelled():evaluate()
        if saved!=u:save()
        status='regression_stopped' if guarded else 'stopped' if cancelled() else 'complete'
    except (InterruptedError,KeyboardInterrupt):
        stop['requested']=True;(OUTPUT/'STOP').touch()
        if saved!=u:save()
        status='stopped'
    except Exception as exc:
        atomic_json(OUTPUT/'report.json',dict(status='error',update=u,last_saved_update=saved,error=repr(exc)));raise
    finally:
        del b,anchor,opt;gc.collect();mx.clear_cache()
    verify(frozen)
    atomic_json(OUTPUT/'report.json',dict(status=status,update=u,step=640+u,contract=fingerprint(frozen),automatic_promotion=False,public_benchmarks_evaluated=False,reserved_test_evaluated=False,content_review_required=True))
    print('[finished]',status,'review saved answers and development scores before selecting a checkpoint',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    for name in ('prepare','check','run'):actions.add_argument('--'+name,action='store_true')
    parser.add_argument('--clear-stop',action='store_true');args=parser.parse_args();cfg=read_json(DIR/'config.json');validate(cfg)
    print(json.dumps(plan(cfg),indent=2),flush=True)
    if not (args.prepare or args.check or args.run):return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        if args.prepare:
            if (OUTPUT/'contract.json').exists():raise ValueError('Cannot change a started experiment')
            from sft.preference_long_640.data import prepare
            print('[prepared]',json.dumps(prepare(cfg)['stats']),flush=True);return
        frozen=contract(cfg);data=read_json(DIR/'prepared.json')
        if args.check:
            from sft.preference_long_640.check import check
            checks=check(cfg,frozen,data)
            if contract(cfg)!=frozen:raise ValueError('Readiness inputs changed')
            atomic_json(DIR/'readiness.json',dict(status='passed',contract=frozen,checks=checks,production_training_started=False))
            print('[check-passed]',json.dumps(checks),flush=True);return
        receipt=read_json(DIR/'readiness.json')
        if receipt.get('status')!='passed' or receipt['contract']!=frozen:raise ValueError('Readiness missing/stale; run --check')
        if (OUTPUT/'contract.json').exists() and read_json(OUTPUT/'contract.json')!=frozen:raise ValueError('Run contract changed')
        if (OUTPUT/'report.json').exists() and read_json(OUTPUT/'report.json').get('status')=='complete':print('[already-complete]');return
        if shutil.disk_usage(ROOT).free<60*1024**3:raise OSError('Need 60 GiB free for retained checkpoints and reserve')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        if args.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists():raise ValueError('STOP exists; review cause and use --clear-stop to resume')
        atomic_json(OUTPUT/'contract.json',frozen);atomic_json(OUTPUT/'config.json',cfg)
        for src in code_files()+[DIR/n for n in ('config.json','prepared.json','selection.json','review.json','review_samples.json')]:
            dest=OUTPUT/'inputs'/src.relative_to(ROOT)
            if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        stop={'requested':False,'reset_guard':args.clear_stop}
        def request_stop(*_):
            stop['requested']=True;(OUTPUT/'STOP').touch();print('[stop] Finish current update and save',flush=True)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
        with (OUTPUT/'training.log').open('a',buffering=1) as log:
            original=sys.stdout;sys.stdout=Tee(original,log)
            try:run(cfg,frozen,data,stop)
            finally:sys.stdout=original

if __name__=='__main__':main()
