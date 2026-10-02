"""Freeze one reviewed step, then compare reserved answers with 1920."""
import argparse
import gc
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import read_json,atomic_json,file_sha256,fingerprint,lock
from sft.constraint_completion_1920.protocol import DIR,OUTPUT,contract,verify
from sft.transfer_control.launch import preflight


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',type=int,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    cfg=read_json(DIR/'config.json');u=a.step-1920
    if u not in cfg['evaluation_updates']:raise ValueError('Choose an evaluated checkpoint')
    run=OUTPUT;ev=read_json(run/'evaluations'/f'update_{u:05d}.json');bundle=run/ev['bundle']
    print(json.dumps(dict(step=a.step,source=str(bundle),mode='test' if a.run else 'plan',training=False,
        purpose='Freeze reviewed choice before reserved model generations; compare with unchanged 1920.'),indent=2),flush=True)
    if not a.run:return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        frozen=contract(cfg)
        if read_json(OUTPUT/'contract.json')!=frozen:raise ValueError('Experiment changed')
        if read_json(OUTPUT/'report.json').get('status')!='complete':raise ValueError('Finish the run first')
        from sft.constraint_completion_1920.evaluate import validate_review,assess
        review_path=run/'evaluations'/f'review_{u:05d}.json'
        score=validate_review(read_json(review_path),ev)
        if not ev['retention_gate_passed']:raise ValueError('Retention guard failed; do not promote')
        baseline=read_json(run/'evaluations'/'update_00000.json')
        parent_score=validate_review(read_json(run/'evaluations'/'review_00000.json'),baseline)
        if score<=parent_score:raise ValueError('No semantic improvement over parent; review before further testing')
        from sml_v1.checkpoint_bundle import resolve_bundle
        weights=resolve_bundle(bundle)
        selected=dict(step=a.step,bundle=str(bundle.relative_to(OUTPUT)),
            sha256=file_sha256(weights),contract=fingerprint(frozen),semantic_joint_score=score,
            review_sha256=file_sha256(review_path))
        pin=OUTPUT/'selection_frozen.json'
        if pin.exists() and read_json(pin)!=selected:raise ValueError('Reserved selection already frozen; do not retune on test')
        atomic_json(pin,selected)
        from sft.constraint_completion_1920.launch import rebuild
        from sft.constraint_completion_1920.engine import load
        import mlx.core as mx
        data=rebuild(cfg)
        for name,path in [('parent',None),('selected',weights)]:
            out=OUTPUT/f'test_{name}.json'
            if out.exists():
                if read_json(out)['selection']!=selected:raise ValueError('Reserved test identity mismatch')
                continue
            b=load(cfg,path);result=assess(b,data,'test',cfg);result.update(selection=selected,training=False)
            atomic_json(out,result);del b;gc.collect();mx.clear_cache()
        verify(frozen)
        print('[test-complete] Inspect test_parent.json and test_selected.json; no automatic promotion.',flush=True)


if __name__=='__main__':main()
