"""Freeze a development-reviewed candidate before comparing on reserved source groups."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v2.common import read_json,atomic_json,file_sha256,fingerprint,lock
from sft.complete_transfer_1024.protocol import OUTPUT,DIR,contract,verify
from sft.transfer_control.launch import preflight

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--step',type=int,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();cfg=read_json(DIR/'config.json');u=a.step-cfg['source_step']
 if u not in cfg['evaluation_updates']:raise ValueError('Choose a saved evaluated step')
 ev=read_json(OUTPUT/'evaluations'/f'update_{u:05d}.json');bundle=OUTPUT/ev['bundle'];frozen=contract(cfg)
 if read_json(OUTPUT/'report.json')['status']!='complete' or ev['contract']!=fingerprint(frozen):raise ValueError('Incomplete or changed run')
 identity=dict(step=a.step,bundle=str(bundle),sha256=file_sha256(bundle/'model.safetensors'),contract=fingerprint(frozen));print(json.dumps(identity,indent=2))
 if not a.run:return
 preflight()
 with lock(ROOT/'sft/.experiment.lock'):
  pin=OUTPUT/'reserved_selection.json'
  if pin.exists() and read_json(pin)!=identity:raise ValueError('Reserved selection already frozen to another step')
  atomic_json(pin,identity)
  from sft.complete_transfer_1024.runner import rebuild
  from sft.complete_transfer_1024.engine import load
  from sft.complete_transfer_1024.evaluate import assess
  import mlx.core as mx,gc
  data=rebuild(cfg)
  for name,weights in [('parent',ROOT/cfg['source_bundle']/'model.safetensors'),('selected',bundle/'model.safetensors')]:
   path=OUTPUT/f'reserved_{name}.json'
   if path.exists():continue
   b=load(cfg,weights);result=assess(b,data,'test',cfg);result['selection']=identity;atomic_json(path,result);del b;gc.collect();mx.clear_cache()
  verify(frozen);print('[reserved-complete] Review both saved answer files; no automatic promotion.')
if __name__=='__main__':main()
