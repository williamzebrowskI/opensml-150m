"""Unchanged full public benchmarks for a saved public-repair checkpoint."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from sml_v2.common import read_json,file_sha256,fingerprint,lock
EXP=Path(__file__).resolve().parents[1]
RUN=EXP/'runs/sft'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates',type=int,choices=(64,128,192,256),required=True)
    p.add_argument('--suite',choices=('all','multiple-choice','ifeval'),default='all')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true')
    a=p.parse_args();cfg=read_json(EXP/'config.json')
    ep=RUN/f'evaluations/update_{a.updates:05d}.json';ev=read_json(ep)
    bundle=RUN/ev['bundle'];step=384+a.updates;frozen_run=read_json(RUN/'contract.json');meta=read_json(bundle/'model.safetensors.json')
    if bundle.parent!=RUN or not bundle.name.startswith(f'step_{step:07d}_'):raise ValueError('Wrong checkpoint path')
    if (meta['step']!=step or meta['additional_updates']!=a.updates or meta['source_step']!=384 or
        meta['source_bundle']!=cfg['source_bundle'] or meta['source_model_sha256']!=cfg['source_model_sha256'] or
        meta['contract']!=fingerprint(frozen_run) or ev['contract']!=meta['contract']):raise ValueError('Checkpoint ancestry/contract mismatch')
    digest=file_sha256(bundle/'model.safetensors');manifest=read_json(bundle/'manifest.json')
    if digest!=manifest['files']['model.safetensors']['sha256']:raise ValueError('Weights changed')
    name=f'public-repair-384-plus-{a.updates}';results=EXP/'benchmarks';suites=['multiple-choice','ifeval'] if a.suite=='all' else [a.suite]
    print(json.dumps(dict(model=name,source=str(bundle),step=step,additional_updates=a.updates,sha256=digest,suites=suites,output=str(results/name),training=False,automatic_promotion=False),indent=2),flush=True)
    if not(a.check or a.run):return
    with lock(ROOT/'sft/.experiment.lock'):
        from evaluation.full_benchmarks import core,launch,spec
        spec.MODELS[name]=dict(bundle=str(bundle.relative_to(ROOT)),step=step,sha256=digest,decode_mode='cached')
        core.RESULTS=launch.RESULTS=results;launch.DIR=results/'readiness'
        for suite in suites:
            frozen=core.frozen(name,suite)
            frozen['code'][str(Path(__file__).resolve())]=file_sha256(__file__)
            for f in (RUN/'contract.json',ep,EXP/'config.json'):frozen['protected'][str(f)]=file_sha256(f)
            receipt=launch.DIR/f'readiness_{name}_{suite}.json'
            if a.check or not receipt.exists() or read_json(receipt)['contract']!=frozen:launch.check(name,suite,frozen)
            if a.run:launch.run(name,suite,frozen)

if __name__=='__main__':main()
