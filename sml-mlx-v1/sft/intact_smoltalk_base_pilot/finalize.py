"""Freeze an explicitly chosen development checkpoint, then open reserved conversations."""
import argparse
import gc
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json,file_sha256,fingerprint,lock,read_json
from sft.intact_smoltalk_base_pilot.launch import DIR,OUTPUT,contract,rebuild,verify
from sft.transfer_control.launch import preflight


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--step',required=True,type=int)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    cfg=read_json(DIR/'config.json');u=a.step-cfg['checkpoint_start_step']
    if u not in cfg['evaluation_updates'] or u==0:raise ValueError('Choose a trained development checkpoint')
    ev=read_json(OUTPUT/'evaluations'/f'update_{u:05d}.json');bundle=OUTPUT/ev['bundle']
    if not a.run:
        print('Would freeze',bundle,'then compare parent/selected on reserved conversations.');return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'):
        frozen=contract(cfg);saved=read_json(OUTPUT/'contract.json')
        if frozen!=saved or ev['contract']!=fingerprint(frozen):raise ValueError('Checkpoint contract mismatch')
        from sml_v1.checkpoint_bundle import resolve_bundle
        resolve_bundle(bundle)
        meta=read_json(bundle/'model.safetensors.json')
        if meta['step']!=a.step or meta['contract']!=fingerprint(frozen):raise ValueError('Bundle identity mismatch')
        selection=dict(step=a.step,bundle=bundle.name,sha256=file_sha256(bundle/'model.safetensors'),
                       development_evaluation=fingerprint(ev),contract=fingerprint(frozen))
        path=OUTPUT/'frozen_selection.json'
        if path.exists() and read_json(path)!=selection:raise ValueError('Selection was already frozen; do not choose using reserved results')
        if not path.exists():atomic_json(path,selection)
        from sft.intact_smoltalk_base_pilot.engine import load
        from sft.intact_smoltalk_base_pilot.evaluate import assess
        import mlx.core as mx
        data=rebuild(cfg)
        for name,weights in [('parent',ROOT/cfg['source_bundle']/'model.safetensors'),('selected',bundle/'model.safetensors')]:
            dest=OUTPUT/f'test_{name}.json'
            if dest.exists():
                if read_json(dest)['selection']!=selection:raise ValueError('Reserved result identity changed')
                continue
            b=load(cfg,weights);r=assess(b,data,cfg,split='test');r['selection']=selection
            atomic_json(dest,r);del b;gc.collect();mx.clear_cache()
        verify(frozen);print('[test-complete] Read test_parent.json and test_selected.json; no automatic promotion.')


if __name__=='__main__':main()
