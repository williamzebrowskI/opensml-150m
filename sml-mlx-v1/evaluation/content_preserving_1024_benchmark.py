"""Benchmark content-preserving candidate 1216 from Recovery 1024; not promoted."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sml_v1.common import read_json,file_sha256,lock
from sft.transfer_control.launch import preflight
RUN=ROOT/'runs/sft_content_preserving_1024_v1'
BUNDLE=RUN/'step_0001216_1244746bf21e'
SHA='3684383e91522211c0b29e15d9789e01212438e8336ff41850d927dffee052a2'
MODEL='content-preserving-1024-1216'
RESULTS=ROOT/'diagnostics/content_preserving_1024_benchmarks_v1'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--suite',choices=['all','multiple-choice','ifeval'],default='all')
 g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true')
 args=p.parse_args();suites=['multiple-choice','ifeval'] if args.suite=='all' else [args.suite]
 if file_sha256(BUNDLE/'model.safetensors')!=SHA:raise ValueError('Selected step 1216 weights changed')
 meta=read_json(BUNDLE/'model.safetensors.json');report=read_json(RUN/'report.json')
 if (meta['step'],meta['source_step'],meta['additional_updates'])!=(1216,1024,192) or report['status']!='complete' or report['contract']!=meta['contract']:raise ValueError('Completed content-preserving checkpoint identity mismatch')
 if report['candidate']['step']!=1216 or report['candidate']['bundle']!=BUNDLE.name:raise ValueError('Reviewed candidate mismatch')
 print(json.dumps(dict(model=MODEL,source=str(BUNDLE),suites=suites,output=str(RESULTS/MODEL),training=False,experimental=True,automatic_promotion=False,selection_note='Candidate 1216 reviewed on development answers and pinned for public benchmarking; original Recovery 1024 preserved. Reserved private test not yet evaluated.'),indent=2),flush=True)
 if not(args.check or args.run):return
 preflight()
 with lock(ROOT/'sft/.experiment.lock'):
  from evaluation.full_benchmarks import spec,core,launch
  spec.MODELS[MODEL]=dict(bundle=str(BUNDLE.relative_to(ROOT)),step=1216,sha256=SHA)
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
