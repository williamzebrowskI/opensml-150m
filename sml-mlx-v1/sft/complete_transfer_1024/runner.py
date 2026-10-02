"""Prepare/check a content-preserving 1024 continuation; --run starts the user-owned job."""
import argparse
import gc
import json
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json,read_json,fingerprint,lock
from sft.complete_transfer_1024.protocol import DIR,OUTPUT,contract,code_files,verify
from sft.transfer_control.launch import preflight,Tee


def rebuild(cfg,cancelled=lambda:False):
    from sft.complete_transfer_1024.data import build
    data,selection=build(cfg,cancelled)
    if selection!=read_json(DIR/'selection.json'): raise ValueError('Rebuilt source selection differs from reviewed data')
    return data


def restore(b,opt,path,frozen,cfg):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v1.checkpoint_bundle import resolve_bundle
    weights=resolve_bundle(path); meta=read_json(weights+'.json')
    u=meta['additional_updates']
    if meta['contract']!=fingerprint(frozen) or not 0<=u<=cfg['updates']: raise ValueError('Resume contract/cursor changed')
    if meta['step']!=cfg['source_step']+u or meta['next_examples']!={k:u*n for k,n in cfg['per_update'].items()}:
        raise ValueError('Resume exposure mismatch')
    b.model.load_weights(weights,strict=True)
    opt.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
    mx.eval(b.model.parameters(),opt.state)
    if int(opt.state['step'].item())!=u: raise ValueError('Fresh optimizer step/cursor mismatch')
    return u,meta['training']


def prune(out,cfg):
    # Keep every saved checkpoint, including interrupted saves. User chooses cleanup.
    return


def run(cfg,frozen,data,stop,output=OUTPUT):
    import mlx.core as mx
    from sml_v1.checkpoint_bundle import save_bundle
    from sft.complete_transfer_1024 import engine
    from sft.complete_transfer_1024.evaluate import assess,eligible
    output.mkdir(parents=True,exist_ok=True); ep=output/'evaluations'; ep.mkdir(exist_ok=True)
    mx.random.seed(cfg['seed'])
    b=engine.load(cfg); anchor=engine.load(cfg); opt=engine.optimizer(cfg); opt.init(b.model.trainable_parameters())
    mx.eval(opt.state); u=0; training=[]; saved=-1; bundle=None
    if (output/'latest.json').exists():
        u,training=restore(b,opt,output/'latest.json',frozen,cfg); saved=u
        bundle=output/read_json(output/'latest.json')['bundle']
        print('[resume]',u,'exact saved optimizer, family cursors, and frozen 1024 anchor',flush=True)
    cancelled=lambda:stop['requested'] or (output/'STOP').exists()

    def save():
        nonlocal saved,bundle
        meta=dict(step=cfg['source_step']+u,source_step=cfg['source_step'],additional_updates=u,
            optimizer_local_step=u,next_examples={k:u*n for k,n in cfg['per_update'].items()},
            contract=fingerprint(frozen),training=training,training_format='plain-user-assistant-eos-v1',
            experimental=True,automatic_promotion=False,tokenizer=frozen.get('tokenizer'),
            objective='complete-transfer-sft-v1')
        bundle=Path(save_bundle(output,b.model,opt,meta,None,keep=1000000,reserve_gib=10.)); saved=u
        prune(output,cfg); print('[checkpoint]',meta['step'],'development candidate' if u in cfg['evaluation_updates'] else 'latest',flush=True)

    def evaluation():
        path=ep/f'update_{u:05d}.json'
        if path.exists():
            value=read_json(path)
            if value['contract']!=fingerprint(frozen): raise ValueError('Evaluation contract changed')
            return
        result=assess(b,data,'dev',cfg,cancelled); metrics=result['metrics']
        result.update(update=u,step=cfg['source_step']+u,contract=fingerprint(frozen),bundle=bundle.name)
        if u==0:
            baseline=metrics
        else:
            baseline=read_json(ep/'update_00000.json')['metrics']
        ok=eligible(metrics,baseline,cfg); result['eligible_for_review']=ok
        atomic_json(path,result)
        # Every checkpoint is reviewable. No proxy-only automatic best selection.
        print('[complete-transfer-eval]',u,json.dumps(dict(**metrics,eligible_for_review=ok)),flush=True)

    try:
        if saved<0: save()
        if u in cfg['evaluation_updates'] and not cancelled(): evaluation()
        print(f"[ready] {cfg['source_step']} + {u}/{cfg['updates']}; main Mac only; 3 natural + 1 complete-source reply + 2 ranked choices + 2 replay per update; all checkpoints retained",flush=True)
        while u<cfg['updates'] and not cancelled():
            rows=engine.batch_at(data,cfg,u); t=time.monotonic()
            v=engine.update(b,anchor,opt,rows,data['anchors'][u%len(data['anchors'])],cfg,u+1)
            u+=1; v['seconds']=time.monotonic()-t; v['batch_hash']=fingerprint({k:[r['id'] for r in rr] for k,rr in rows.items()}); training.append(v)
            if u==1 or u%16==0:
                print(f"[complete-transfer {u}/{cfg['updates']}] step={cfg['source_step']+u} loss={v['loss']:.4f} rank={v['ranking_ce']:.4f} KL={v['anchor_kl']:.5f} lr={v['lr']:.3e}",flush=True)
            if u%cfg['checkpoint_every']==0 or u in cfg['evaluation_updates']: save()
            if u in cfg['evaluation_updates'] and not cancelled(): evaluation()
        if saved!=u: save()
        if not cancelled() and u==cfg['updates']: evaluation()
        status='stopped' if cancelled() else 'complete'
    except (InterruptedError,KeyboardInterrupt):
        stop['requested']=True; (output/'STOP').touch()
        if saved!=u: save()
        status='stopped'
    except Exception as exc:
        atomic_json(output/'report.json',dict(status='error',update=u,error=repr(exc),last_saved_update=saved))
        raise
    finally:
        del b,anchor,opt; gc.collect(); mx.clear_cache()
    verify(frozen)
    atomic_json(output/'report.json',dict(status=status,update=u,step=cfg['source_step']+u,contract=fingerprint(frozen),
        selection_requires_answer_review=True,test_evaluated=False,public_benchmarks_evaluated=False,automatic_promotion=False))
    print('[finished]',status,'; all checkpoints retained. Review development answers before choosing a step for testing; no automatic best or promotion.',flush=True)

