"""Complete-answer transfer from preserved Recovery 1024; main Mac only."""
import argparse,json,signal,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import read_json,atomic_json,lock
from sft.complete_transfer_1024.protocol import DIR,OUTPUT,contract,code_files
from sft.transfer_control.launch import preflight,Tee

def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true');p.add_argument('--clear-stop',action='store_true');a=p.parse_args();cfg=read_json(DIR/'config.json')
 print(json.dumps(dict(mode='run' if a.run else 'check' if a.check else 'plan',source=cfg['source_bundle'],source_step=1024,additional_updates=192,final_step=1216,independent_natural_examples=576,independent_grounded_contexts=192,ranked_qa_examples=384,reading_replay=384,per_update='3 natural + 1 complete-source reply + 2 ranked choices + 2 reading replay',data='Pinned MIT UltraChat train_sft first replies; synthetic. Source streamed to bounded RAM and discarded. No local teacher or raw dataset disk cache.',objective='Assistant-only SFT including EOS, labeled choice ranking, small frozen-parent prose KL',peak_lr=cfg['peak_lr'],final_lr=cfg['final_lr'],checkpoints='All saved checkpoints retained, evaluated every 32 updates',output=str(OUTPUT),automatic_promotion=False),indent=2),flush=True)
 if not(a.check or a.run):return
 preflight()
 with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
  frozen=contract(cfg)
  if a.check:
   from sft.complete_transfer_1024.check import run_checks
   result=run_checks(cfg,frozen)
   if frozen!=contract(cfg):raise ValueError('Contract changed')
   atomic_json(DIR/'readiness.json',dict(status='passed',contract=frozen,checks=result));return
  receipt=read_json(DIR/'readiness.json')
  if receipt['status']!='passed' or receipt['contract']!=frozen:raise ValueError('Run --check for current code/data first')
  if (OUTPUT/'report.json').exists() and read_json(OUTPUT/'report.json')['status']=='complete':print('[already-complete]');return
  if (OUTPUT/'contract.json').exists() and read_json(OUTPUT/'contract.json')!=frozen:raise ValueError('Resume contract changed')
  if shutil.disk_usage(ROOT).free<35*1024**3:raise OSError('Need 35 GiB free to retain checkpoints')
  OUTPUT.mkdir(parents=True,exist_ok=True)
  if a.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
  if (OUTPUT/'STOP').exists():raise ValueError('Use --clear-stop')
  stop={'requested':False}
  def request_stop(*_):stop['requested']=True;(OUTPUT/'STOP').touch();print('[stop] Saving at the next update boundary.',flush=True)
  for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
  atomic_json(OUTPUT/'config.json',cfg);atomic_json(OUTPUT/'contract.json',frozen)
  for src in code_files()+list(DIR.glob('*.json')):
   dest=OUTPUT/'inputs'/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():shutil.copy2(src,dest)
  from sft.complete_transfer_1024.runner import run,rebuild
  with (OUTPUT/'training.log').open('a',buffering=1) as log:
   orig=sys.stdout;sys.stdout=Tee(orig,log)
   try:
    data=rebuild(cfg,lambda:stop['requested'])
    if not stop['requested']:run(cfg,frozen,data,stop,output=OUTPUT)
   except InterruptedError:print('[stopped] during data preparation')
   finally:sys.stdout=orig
if __name__=='__main__':main()
