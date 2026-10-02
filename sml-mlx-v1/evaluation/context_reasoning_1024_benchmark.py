"""Benchmark Context Reasoning 1536 from Recovery 1024; experimental, not promoted."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sml_v1.common import read_json,file_sha256,lock
from sft.transfer_control.launch import preflight
RUN=ROOT/'runs/sft_context_reasoning_1024_v1/main'
BUNDLE=RUN/'step_0001536_e2c87e54b25b'
SHA='e3fc639b5c2a8403377116b01bf0bc35a41f4ccdc2db13af54569d7623f79834'
MODEL='context-reasoning-1024-1536'
RESULTS=ROOT/'diagnostics/context_reasoning_1024_benchmarks_v1'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--suite',choices=['all','multiple-choice','ifeval'],default='all')
 g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true')
 args=p.parse_args();suites=['multiple-choice','ifeval'] if args.suite=='all' else [args.suite]
 if file_sha256(BUNDLE/'model.safetensors')!=SHA:raise ValueError('Selected step 1536 weights changed')
 meta=read_json(BUNDLE/'model.safetensors.json');report=read_json(RUN/'report.json')
 if (meta['step'],meta['source_step'],meta['additional_updates'])!=(1536,1024,512) or report['status']!='complete' or report['contract']!=meta['contract']:raise ValueError('Completed context-reasoning checkpoint identity mismatch')
 print(json.dumps(dict(model=MODEL,source=str(BUNDLE),suites=suites,output=str(RESULTS/MODEL),training=False,experimental=True,automatic_promotion=False,selection_note='Step 1536 pinned for diagnostic benchmarking after development-answer review; failed formatting retention gate; original Recovery 1024 remains selected. Reserved private test not yet evaluated.'),indent=2),flush=True)
 if not(args.check or args.run):return
 preflight()
 with lock(ROOT/'sft/.experiment.lock'):
  from evaluation.full_benchmarks import spec,core,launch
  spec.MODELS[MODEL]=dict(bundle=str(BUNDLE.relative_to(ROOT)),step=1536,sha256=SHA)
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
