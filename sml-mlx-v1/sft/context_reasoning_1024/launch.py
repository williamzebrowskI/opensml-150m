"""Contextual commonsense and reference resolution from unchanged Recovery 1024."""
import argparse,json,sys,signal,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import read_json,atomic_json,fingerprint,lock
from sft.context_reasoning_1024.protocol import DIR,OUTPUT,contract,code_files,validate,plan
from sft.transfer_control.launch import preflight,Tee

def main():
 p=argparse.ArgumentParser(description=__doc__)
 g=p.add_mutually_exclusive_group();g.add_argument('--prepare',action='store_true');g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true')
 p.add_argument('--arm',choices=['all','main'],default='all');p.add_argument('--clear-stop',action='store_true');args=p.parse_args()
 base=read_json(DIR/'config.json');arms=base['arms'] if args.arm=='all' else [args.arm]
 print(json.dumps({'mode':'run' if args.run else 'check' if args.check else 'prepare' if args.prepare else 'plan','independent_fits':[plan(dict(base,arm=a)) for a in arms],'total_updates':base['updates']*len(arms)},indent=2),flush=True)
 if not(args.prepare or args.check or args.run):return
 preflight()
 with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
  if args.prepare:
   if any((OUTPUT/a/'contract.json').exists() for a in arms):raise ValueError('Existing run cannot be re-prepared')
   from sft.context_reasoning_1024.data import build
   for a in arms:
    cfg=dict(base,arm=a);validate(cfg);data,selection=build(cfg)
    atomic_json(DIR/f'selection_{a}.json',selection)
    atomic_json(DIR/'review_samples.json',{s:{source:[r for r in data['new_split'][s] if r['source']==source][:32 if s=='train' else 8] for source in ('winogrande','cosmosqa')} for s in ('train','dev')})
    print('[prepared]',a,json.dumps(selection['stats']['train']),flush=True)
   return
  stop={'requested':False}
  def request_stop(*_):
   stop['requested']=True;OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'STOP').touch();print('[stop] Finish current operation and save; do not start another arm.',flush=True)
  for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
  if args.run:
   if args.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
   if (OUTPUT/'STOP').exists():raise ValueError('STOP exists; use --clear-stop')
  for a in arms:
   if stop['requested'] or (args.run and (OUTPUT/'STOP').exists()):break
   cfg=dict(base,arm=a);validate(cfg);frozen=contract(cfg);out=OUTPUT/a
   if args.check:
    from sft.context_reasoning_1024.check import run_checks
    result=run_checks(cfg,frozen)
    if frozen!=contract(cfg):raise ValueError('Inputs changed during checks')
    atomic_json(DIR/f'readiness_{a}.json',dict(status='passed',contract=frozen,checks=result,production_training_started=False));continue
   receipt=DIR/f'readiness_{a}.json'
   if not receipt.exists() or read_json(receipt).get('status')!='passed' or read_json(receipt)['contract']!=frozen:raise ValueError('Run --check before production')
   if (out/'contract.json').exists() and read_json(out/'contract.json')!=frozen:raise ValueError('Existing contract changed')
   if (out/'report.json').exists() and read_json(out/'report.json').get('status')=='complete':print('[already-complete]',a,flush=True);continue
   if shutil.disk_usage(ROOT).free<45*1024**3:raise OSError('Need 45 GiB free')
   out.mkdir(parents=True,exist_ok=True)
   if args.clear_stop:(out/'STOP').unlink(missing_ok=True)
   if (out/'STOP').exists():raise ValueError('Arm STOP exists; --clear-stop')
   atomic_json(out/'contract.json',frozen);atomic_json(out/'config.json',cfg)
   for src in code_files()+[DIR/'config.json',DIR/f'selection_{a}.json',DIR/'review.json',DIR/'probes.json']:
    dest=out/'inputs'/src.relative_to(ROOT)
    if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
   from sft.context_reasoning_1024.runner import run,rebuild
   with (out/'training.log').open('a',buffering=1) as log:
    orig=sys.stdout;sys.stdout=Tee(orig,log)
    try:
     print('[arm]',a,'source is preserved Recovery 1024; fresh optimizer',flush=True)
     data=rebuild(cfg,lambda:stop['requested'])
     if not stop['requested']:
      if frozen!=contract(cfg):raise ValueError('Inputs changed')
      run(cfg,frozen,data,stop,output=out)
    except InterruptedError:print('[stopped] during preparation',flush=True)
    finally:sys.stdout=orig
   if stop['requested'] or (out/'STOP').exists():break

if __name__=='__main__':main()
