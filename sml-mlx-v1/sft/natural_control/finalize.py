"""Freeze explicit development selections, then evaluate the untouched test split."""
import argparse,gc,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json,file_sha256,fingerprint,lock,read_json
from sft.natural_control.protocol import DIR,OUTPUT,arms,contract
from sft.transfer_control.launch import preflight


def candidate(spec,model,cfg,frozen):
    name,step_string=spec.split(':');step=int(step_string)
    allowed={a['name']:a for a in arms(cfg) if a['model']==model}
    if name not in allowed or step not in cfg['evaluation_updates']:raise ValueError('Choose an existing model-specific development candidate as arm:step')
    folder=OUTPUT/name
    if read_json(folder/'report.json')['status']!='complete':raise ValueError('Finish the development fits first')
    matches=[p for p in folder.glob(f'step_{step:07d}_*') if p.is_dir() and not p.is_symlink()]
    if len(matches)!=1:raise ValueError('Candidate is absent or ambiguous')
    from sml_v1.checkpoint_bundle import resolve_bundle
    weights=resolve_bundle(matches[0])
    meta=read_json(weights+'.json')
    if meta['contract']!=fingerprint(frozen) or meta['arm']!=allowed[name] or meta['step']!=step:
        raise ValueError('Candidate metadata does not match this experiment')
    return dict(arm=allowed[name],step=step,weights=weights,sha256=file_sha256(weights))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--opensml',required=True,help='Example: opensml-lr5e-05:640')
    ap.add_argument('--reference',required=True,help='Example: reference-lr2e-04:960')
    args=ap.parse_args();preflight();cfg=read_json(DIR/'config.json')
    with lock(DIR/'.launcher.lock'):
        frozen=contract(cfg)
        if read_json(OUTPUT/'contract.json')!=frozen:raise ValueError('Contract changed')
        if read_json(OUTPUT/'summary.json')['status']!='complete':raise ValueError('All fits must finish before selection')
        selected={m:candidate(getattr(args,m),m,cfg,frozen) for m in cfg['models']}
        choice=dict(contract=frozen,selected=selected,basis='Explicit development-answer review; frozen before test generation')
        path=OUTPUT/'frozen_test_selection.json'
        if path.exists() and read_json(path)!=choice:raise ValueError('Selection is already frozen; the inspected test cannot select another checkpoint')
        atomic_json(path,choice)
        from sft.natural_control.launch import rebuild
        from sft.natural_control.engine import load
        from sft.natural_control.evaluate import assess
        import mlx.core as mx
        from mlx.utils import tree_map
        data=rebuild(cfg)
        for model in cfg['models']:
            for label in ('parent','selected'):
                target=OUTPUT/'heldout'/f'{model}_{label}.json'
                if target.exists():
                    if read_json(target)['selection']!=choice:raise ValueError('Heldout report belongs to another selection')
                    continue
                b=load(model)
                if label=='selected':
                    b.model.load_weights(selected[model]['weights'])
                    b.model.update(tree_map(lambda x:x.astype(mx.float32),b.model.parameters()));mx.eval(b.model.parameters())
                print('[heldout]',model,label,flush=True)
                value=assess(b,data,'test',cfg)
                atomic_json(target,dict(selection=choice,model=model,label=label,evaluation=value))
                del b;gc.collect();mx.clear_cache()
        print('[finished] Test answers saved. Compare meaning and completion within each model; cross-tokenizer NLL is not comparable.',flush=True)

if __name__=='__main__':main()
