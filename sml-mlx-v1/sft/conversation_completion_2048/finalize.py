"""Freeze an explicitly chosen development candidate before reserved evaluation."""
import argparse,gc,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json,read_json,file_sha256,fingerprint,lock
from sft.conversation_completion_2048.launch import DIR,OUTPUT,contract,verify,rebuild
from sft.transfer_control.launch import preflight


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--step',type=int,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    cfg=read_json(DIR/'config.json');u=a.step-cfg['source_step']
    if u not in cfg['evaluation_updates'] or u==0:raise ValueError('Choose an evaluated derivative')
    ev=read_json(OUTPUT/'evaluations'/f'update_{u:05d}.json');bundle=OUTPUT/ev['bundle']
    frozen=contract(cfg)
    if read_json(OUTPUT/'contract.json')!=frozen or ev['contract']!=fingerprint(frozen):raise ValueError('Changed run identity')
    print(json.dumps(dict(step=a.step,source=str(bundle),training=False,reserved_test=True,automatic_promotion=False),indent=2))
    if not a.run:return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'):
        choice=dict(step=a.step,bundle=bundle.name,sha256=file_sha256(bundle/'model.safetensors'),dev_evaluation=file_sha256(OUTPUT/'evaluations'/f'update_{u:05d}.json'))
        pin=OUTPUT/'frozen_selection.json'
        if pin.exists() and read_json(pin)!=choice:raise ValueError('Reserved test already opened for another choice')
        atomic_json(pin,choice)
        from sft.constraint_completion_1920 import engine
        from sft.conversation_completion_2048.evaluate import assess
        import mlx.core as mx
        data=rebuild(cfg)
        for name,weights in [('parent',None),('selected',bundle/'model.safetensors')]:
            dest=OUTPUT/f'test_{name}.json'
            if dest.exists():continue
            b=engine.load(cfg,weights)
            try:
                result=assess(b,data,cfg,'test');result.update(selection=choice,contract=fingerprint(frozen));atomic_json(dest,result)
            finally:del b;gc.collect();mx.clear_cache()
        verify(frozen)
        print('[test-complete] Read saved whole-dialogue and human answers; no automatic promotion.')


if __name__=='__main__':main()
