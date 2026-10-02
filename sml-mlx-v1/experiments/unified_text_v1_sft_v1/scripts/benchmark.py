"""Full unchanged ARC/PIQA/HellaSwag and official IFEval for this experiment."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from sml_v1.common import read_json,file_sha256,fingerprint,atomic_json,lock
EXP=Path(__file__).resolve().parents[1]
RUN=EXP/'runs/sft'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--suite',choices=['all','multiple-choice','ifeval'],default='all')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true')
    a=p.parse_args();cfg=read_json(EXP/'config.json')
    if a.step not in cfg['evaluation_updates'] or a.step==0:p.error('Choose a saved nonzero evaluation checkpoint')
    ev_path=RUN/f'evaluations/update_{a.step:05d}.json';ev=read_json(ev_path);bundle=RUN/ev['bundle']
    frozen_run=read_json(RUN/'contract.json');meta=read_json(bundle/'model.safetensors.json')
    if bundle.parent!=RUN or not bundle.name.startswith(f'step_{a.step:07d}_'):raise ValueError('Wrong checkpoint path')
    if meta['step']!=a.step or meta['source_step']!=73243 or meta['source_model_sha256']!=cfg['source_model_sha256'] or meta['source_bundle']!=cfg['source_bundle'] or meta['contract']!=fingerprint(frozen_run) or ev['contract']!=meta['contract']:raise ValueError('Checkpoint ancestry/contract mismatch')
    digest=file_sha256(bundle/'model.safetensors');manifest=read_json(bundle/'manifest.json')
    if digest!=manifest['files']['model.safetensors']['sha256']:raise ValueError('Weights changed')
    model=f'unified-text-v1-{a.step}';results=EXP/'benchmarks';suites=['multiple-choice','ifeval'] if a.suite=='all' else [a.suite]
    print(json.dumps(dict(model=model,source=str(bundle),sha256=digest,suites=suites,output=str(results/model),training=False,automatic_promotion=False),indent=2),flush=True)
    if not (a.check or a.run):return
    from sft.transfer_control.launch import preflight
    preflight()
    with lock(ROOT/'sft/.experiment.lock'):
        from evaluation.full_benchmarks import core,launch,spec
        spec.MODELS[model]=dict(bundle=str(bundle.relative_to(ROOT)),step=a.step,sha256=digest,decode_mode='cached')
        core.RESULTS=launch.RESULTS=results
        launch.DIR=results/'readiness'
        for suite in suites:
            frozen=core.frozen(model,suite)
            frozen['code'][str(Path(__file__).resolve())]=file_sha256(__file__)
            for f in (RUN/'contract.json',ev_path,EXP/'config.json'):frozen['protected'][str(f)]=file_sha256(f)
            receipt=launch.DIR/f'readiness_{model}_{suite}.json'
            if a.check or not receipt.exists() or read_json(receipt)['contract']!=frozen:launch.check(model,suite,frozen)
            if a.run:launch.run(model,suite,frozen)


if __name__=='__main__':main()
