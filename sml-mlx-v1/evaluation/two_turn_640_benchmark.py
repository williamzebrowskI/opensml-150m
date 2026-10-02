"""Benchmark the fixed two-turn step 768 with the same protocol as High LR 640."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sml_v1.common import read_json,file_sha256,lock
from sft.transfer_control.launch import preflight
RUN=ROOT/'runs/sft_two_turn_640_v1'
BUNDLE=RUN/'step_0000768_49364940d7ce'
SHA='db6b16b45153f814de3c0799c49e6460e6b638982168ffcb7bfc2b6b1878bce1'
MODEL='two-turn-high640-768'
RESULTS=ROOT/'diagnostics/two_turn_640_benchmarks_v1'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--suite',choices=['all','multiple-choice','ifeval'],default='all')
 g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true')
 args=p.parse_args();suites=['multiple-choice','ifeval'] if args.suite=='all' else [args.suite]
 if file_sha256(BUNDLE/'model.safetensors')!=SHA:raise ValueError('Selected step 768 weights changed')
 meta=read_json(BUNDLE/'model.safetensors.json');report=read_json(RUN/'report.json')
 if (meta['step'],meta['source_step'],meta['additional_updates'])!=(768,640,128) or report['status']!='complete' or report['contract']!=meta['contract']:raise ValueError('Completed two-turn checkpoint identity mismatch')
 print(json.dumps(dict(model=MODEL,source=str(BUNDLE),suites=suites,output=str(RESULTS/MODEL),training=False),indent=2),flush=True)
 if not(args.check or args.run):return
 preflight()
 with lock(ROOT/'sft/.experiment.lock'):
  from evaluation.full_benchmarks import spec,core,launch
  spec.MODELS[MODEL]=dict(bundle=str(BUNDLE.relative_to(ROOT)),step=768,sha256=SHA)
  launch.RESULTS=core.RESULTS=RESULTS
  for suite in suites:
   frozen=core.frozen(MODEL,suite)
   frozen['code'][str(Path(__file__).resolve())]=file_sha256(__file__)
   for name in ('report.json','contract.json'):frozen['protected'][str(RUN/name)]=file_sha256(RUN/name)
   if args.check:launch.check(MODEL,suite,frozen);continue
   receipt=spec.DIR/f'readiness_{MODEL}_{suite}.json'
   if not receipt.exists() or read_json(receipt)['contract']!=frozen:raise ValueError('Run --check for this suite first')
   launch.run(MODEL,suite,frozen)
   result=read_json(RESULTS/MODEL/suite/'summary.json')
   if result['completed']!=result['expected']:break

if __name__=='__main__':main()
