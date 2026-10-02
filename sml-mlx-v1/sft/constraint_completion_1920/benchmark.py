"""Public benchmark wrapper for the frozen, reviewed step. Never trains."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import read_json,file_sha256,atomic_json,lock,fingerprint
from sft.constraint_completion_1920.protocol import DIR,OUTPUT,contract
from sft.transfer_control.launch import preflight


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',type=int,required=True);p.add_argument('--suite',choices=['all','multiple-choice','ifeval'],default='all')
    g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args()
    cfg=read_json(DIR/'config.json');u=a.step-1920
    if u not in cfg['evaluation_updates']:raise ValueError('Select an evaluated checkpoint')
    pin=read_json(OUTPUT/'selection_frozen.json')
    if pin['step']!=a.step:raise ValueError('Use the frozen development choice, finalized before testing')
    selected=read_json(OUTPUT/'test_selected.json')
    if selected['selection']!=pin:raise ValueError('Reserved test not complete for this selection')
    run=OUTPUT;ev=read_json(run/'evaluations'/f'update_{u:05d}.json');bundle=run/ev['bundle']
    sha=file_sha256(bundle/'model.safetensors');frozen_run=contract(cfg)
    if sha!=pin['sha256'] or pin['contract']!=fingerprint(frozen_run):raise ValueError('Frozen identity changed')
    model=f'constraint-completion-1920-{a.step}';out=ROOT/'diagnostics/constraint_completion_1920_benchmarks_v1'
    suites=['multiple-choice','ifeval'] if a.suite=='all' else [a.suite]
    print(json.dumps(dict(model=model,source=str(bundle),suites=suites,output=str(out/model),training=False),indent=2),flush=True)
    if not(a.check or a.run):return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'):
        from evaluation.full_benchmarks import spec,core,launch
        spec.MODELS[model]=dict(bundle=str(bundle.relative_to(ROOT)),step=a.step,sha256=sha)
        core.RESULTS=launch.RESULTS=out
        identity=dict(step=a.step,bundle=str(bundle),sha256=sha)
        path=out/model/'selection.json'
        if path.exists() and read_json(path)!=identity:raise ValueError('Benchmark selection changed')
        atomic_json(path,identity)
        for suite in suites:
            frozen=core.frozen(model,suite)
            frozen['code'][str(Path(__file__).resolve())]=file_sha256(__file__)
            for f in [path,run/'report.json',run/'contract.json',OUTPUT/'selection_frozen.json']:
                frozen['protected'][str(f)]=file_sha256(f)
            if a.check:launch.check(model,suite,frozen);continue
            receipt=spec.DIR/f'readiness_{model}_{suite}.json'
            if not receipt.exists() or read_json(receipt)['contract']!=frozen:raise ValueError('Run --check first')
            launch.run(model,suite,frozen)
            summary=read_json(out/model/suite/'summary.json')
            if summary['completed']!=summary['expected']:break


if __name__=='__main__':main()
