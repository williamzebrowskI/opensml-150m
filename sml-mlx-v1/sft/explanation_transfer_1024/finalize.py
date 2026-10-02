"""After development-answer review, freeze a candidate before the new test."""
import argparse
import gc
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import read_json,atomic_json,fingerprint,lock,file_sha256
from sft.explanation_transfer_1024.protocol import DIR,OUTPUT,contract,verify
from sft.transfer_control.launch import preflight


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--step',type=int,required=True);p.add_argument('--run',action='store_true');args=p.parse_args()
    cfg=read_json(DIR/'config.json');u=args.step-cfg['source_step']
    if u not in cfg['evaluation_updates']:raise ValueError('Select a saved development candidate')
    evaluation=read_json(OUTPUT/'evaluations'/f'update_{u:05d}.json')
    path=OUTPUT/evaluation['bundle']
    print(json.dumps(dict(step=args.step,source=str(path),mode='test' if args.run else 'plan',
        purpose='Freeze development choice, then compare it with unchanged 1024 on new source groups. Reading diagnostics are reused.',training=False),indent=2))
    if not args.run:return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        frozen=contract(cfg)
        if read_json(OUTPUT/'contract.json')!=frozen or read_json(OUTPUT/'report.json')['status']!='complete':raise ValueError('Complete unchanged training first')
        from sml_v1.checkpoint_bundle import resolve_bundle
        weights=resolve_bundle(path)
        selected=dict(step=args.step,bundle=path.name,weights_sha256=file_sha256(weights),contract=fingerprint(frozen))
        choice=OUTPUT/'selection_frozen.json'
        if choice.exists() and read_json(choice)!=selected:raise ValueError('Test selection already frozen; do not tune on inspected test answers')
        atomic_json(choice,selected)  # MUST precede generating any test answer.
        from sft.explanation_transfer_1024.launch import rebuild
        from sft.skill_recovery_768.engine import load
        from sft.explanation_transfer_1024.evaluate import assess
        import mlx.core as mx
        data=rebuild(cfg)
        for name,source in [('parent',None),('selected',weights)]:
            out=OUTPUT/f'test_{name}.json'
            if out.exists():
                if read_json(out)['selection']!=selected:raise ValueError('Test identity changed')
                continue
            b=load(cfg,source);result=assess(b,data,'test',cfg);result['selection']=selected
            atomic_json(out,result);del b;gc.collect();mx.clear_cache()
        verify(frozen)
        print('[test-complete] Read test_parent.json and test_selected.json; no automatic promotion.')


if __name__=='__main__':main()
