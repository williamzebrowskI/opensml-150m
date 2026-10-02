"""Evaluate final concise curriculum step 512 using the existing full public benchmark protocol. No promotion or training."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sml_v2.common import read_json,file_sha256,lock
OUTPUT=ROOT/'runs/sft_base_curriculum_v2'
from sft.transfer_control.launch import preflight


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['multiple-choice','ifeval'],required=True)
    g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true')
    args=p.parse_args();selected=dict(step=512,bundle='step_0000512_fd767fb7545a',weights_sha256='b54676d218eb44bcefdffdda59144373db5e1596b9a0282ca75a91e14c8d35b5')
    path=OUTPUT/selected['bundle'];model="base-curriculum-v2-512"
    if file_sha256(path/'model.safetensors')!=selected['weights_sha256']:raise ValueError('Selected weights changed')
    from evaluation.full_benchmarks import spec,core,launch
    spec.MODELS[model]=dict(bundle=str(path.relative_to(ROOT)),step=selected['step'],sha256=selected['weights_sha256'])
    launch.RESULTS=core.RESULTS=ROOT/'diagnostics/base_curriculum_v2_benchmarks_v1'
    print(json.dumps(dict(model=model,suite=args.suite,output=str(launch.RESULTS/model/args.suite),training=False),indent=2))
    if not (args.check or args.run):return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'):
        frozen=core.frozen(model,args.suite)
        frozen['code'][str(Path(__file__).resolve())]=file_sha256(__file__)
        if args.check:launch.check(model,args.suite,frozen);return
        receipt=spec.DIR/f'readiness_{model}_{args.suite}.json'
        if not receipt.exists() or read_json(receipt)['contract']!=frozen:raise ValueError('Run --check for this suite first')
        launch.run(model,args.suite,frozen)


if __name__=='__main__':main()
