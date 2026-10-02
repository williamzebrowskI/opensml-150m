"""Pin a reviewed arm/step for public diagnostic benchmarking; never trains."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v2.common import read_json,file_sha256,lock,atomic_json
from sft.transfer_control.launch import preflight
from sft.explanation_transfer_1024.protocol import OUTPUT,DIR

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--step',type=int,required=True)
 p.add_argument('--suite',choices=['all','multiple-choice','ifeval'],default='all');g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args()
 cfg=read_json(DIR/'config.json');u=a.step-cfg['source_step'];run=OUTPUT
 if u not in cfg['evaluation_updates']:raise ValueError('Select an evaluated checkpoint')
 report=read_json(run/'report.json');ev=read_json(run/'evaluations'/f'update_{u:05d}.json');bundle=run/ev['bundle'];sha=file_sha256(bundle/'model.safetensors');meta=read_json(bundle/'model.safetensors.json')
 if report['status']!='complete' or meta['contract']!=report['contract'] or (meta['step'],meta['source_step'],meta['additional_updates'])!=(a.step,1024,u):raise ValueError('Checkpoint/run identity mismatch')
 model=f'explanation-transfer-1024-{a.step}';out=ROOT/'diagnostics/explanation_transfer_1024_benchmarks_v1';suites=['multiple-choice','ifeval'] if a.suite=='all' else [a.suite]
 print(json.dumps(dict(model=model,source=str(bundle),sha256=sha,suites=suites,output=str(out/model),training=False,automatic_promotion=False,review_required=True),indent=2),flush=True)
 if not(a.check or a.run):return
 preflight()
 with lock(ROOT/'sft/.experiment.lock'):
  identity=dict(bundle=str(bundle),sha256=sha,step=a.step)
  pin=out/model/'selection.json'
  if pin.exists() and read_json(pin)!=identity:raise ValueError('Benchmark checkpoint changed')
  if not pin.exists():atomic_json(pin,identity)
  from evaluation.full_benchmarks import spec,core,launch
  spec.MODELS[model]=dict(bundle=str(bundle.relative_to(ROOT)),step=a.step,sha256=sha)
  core.RESULTS=launch.RESULTS=out
  for suite in suites:
   frozen=core.frozen(model,suite);frozen['code'][str(Path(__file__).resolve())]=file_sha256(__file__)
   for path in [run/'report.json',run/'contract.json',pin]:frozen['protected'][str(path)]=file_sha256(path)
   if a.check:launch.check(model,suite,frozen);continue
   receipt=spec.DIR/f'readiness_{model}_{suite}.json'
   if not receipt.exists() or read_json(receipt)['contract']!=frozen:raise ValueError('Run --check first')
   launch.run(model,suite,frozen)
   r=read_json(out/model/suite/'summary.json')
   if r['completed']!=r['expected']:break
if __name__=='__main__':main()
