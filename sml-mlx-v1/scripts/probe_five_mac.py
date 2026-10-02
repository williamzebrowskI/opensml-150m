#!/usr/bin/env python3
"""Bounded five-rank native RDMA correctness; only disposable tiny checkpoints."""
import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path
import threading

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from sml_v1.jaccl_control import ControlChannel
from sml_v1.model import TransformerConfig, TransformerLM
from sml_v1.precision import MasterAdamW
from sml_v1.checkpoint_bundle import resolve_bundle, save_bundle


def canonical_digest(model, optimizer):
    digest=hashlib.sha256()
    for prefix,state in [('model',model.parameters()),('optimizer',optimizer.state)]:
        for name,value in sorted(tree_flatten(state)):
            digest.update((prefix+name+str(value.shape)+str(value.dtype)).encode())
            digest.update(np.asarray(value.astype(mx.float32) if value.dtype==mx.bfloat16 else value).tobytes())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-dir',type=Path,required=True)
    args=p.parse_args()
    job=args.job_dir
    job.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['MLX_RANK'])
    (job/f'worker_rank{rank}.json').write_text(json.dumps(dict(pid=os.getpid(),output=str(job))))
    watchdog=threading.Timer(300,lambda:os._exit(70));watchdog.daemon=True;watchdog.start()
    if os.environ.get('MLX_JACCL_RING')!='1':raise RuntimeError('Native JACCL ring required')
    mx.set_default_device(mx.cpu)
    group=mx.distributed.init(backend='jaccl',strict=True)
    if group.size()!=5 or group.rank()!=rank:raise RuntimeError('Five ranks required; no fallback')
    control=ControlChannel(mx,group,ring_mode='native')
    for _ in range(25):
        values=control.exchange(dict(rank=rank,payload='five-mac-proof'),'framed-probe')
        if [v['rank'] for v in values]!=list(range(5)):raise RuntimeError('Control rank order differs')
    for dtype in (mx.int32,mx.float32,mx.bfloat16):
        for size in (1,17,4096,8193,262147,8388608):
            for repeat in range(2):
                # Small exactly representable integers exercise BF16 as well.
                data=mx.full((size,),rank+repeat,dtype=dtype)
                result=mx.distributed.all_sum(data,group=group,stream=mx.cpu)
                mx.eval(result)
                passed=bool(mx.all(result==(10+5*repeat)).item())
                if not all(control.exchange(passed,'all-sum-verdict')):raise RuntimeError('Native all_sum mismatch')
    batches=[4,3,2,2,2]
    # An actual autodiff gradient over unequal local sample counts, compared to
    # the same loss over the complete global batch.
    global_x=mx.arange(13*7,dtype=mx.float32).reshape(13,7)/100
    start,end=sum(batches[:rank]),sum(batches[:rank+1])
    params=mx.arange(7,dtype=mx.float32)/10
    grad=mx.grad(lambda w,x:mx.mean((x@w-1)**2))
    local=grad(params,global_x[start:end])
    reduced=mx.distributed.all_sum(local*batches[rank],group=group,stream=mx.cpu)/sum(batches)
    expected=grad(params,global_x)
    err=float(mx.max(mx.abs(reduced-expected)).item())
    if not all(control.exchange(bool(mx.allclose(reduced,expected,rtol=1e-5,atol=1e-6).item()),'weighted-gradient')):
        raise RuntimeError('Token-weighted gradient differs from global reference')
    # Roundtrip real BF16 compute weights and FP32 AdamW master/moments on every
    # rank, then verify replica equality. Each rank uses its own scratch files.
    mx.random.seed(7337)
    model=TransformerLM(TransformerConfig(vocab_size=32,d_model=16,n_heads=2,n_kv_heads=1,n_layers=1,
                         max_seq_len=16,mlp_multiple_of=16,attention_impl='vanilla'))
    model.update(tree_map(lambda x:x.astype(mx.bfloat16),model.parameters()))
    optimizer=MasterAdamW(learning_rate=5e-5)
    optimizer.init(model.trainable_parameters())
    x=(mx.arange(13*16,dtype=mx.int32).reshape(13,16)%32)[start:end]
    loss,grads=nn.value_and_grad(model,lambda m:m(x,targets=(x+1)%32)['loss'])(model)
    grads=tree_map(lambda g:mx.distributed.all_sum(g.astype(mx.float32)*batches[rank],group=group,stream=mx.cpu)/13,grads)
    mx.eval(grads)
    optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state)
    before=canonical_digest(model,optimizer)
    if len(set(control.exchange(before,'tiny-replicas')))!=1:raise RuntimeError('Tiny model replicas diverged')
    scratch=job/f'tiny_rank{rank}'
    metadata=dict(step=1,tokens=208,scheduler=dict(cap=5e-5,bad_evals=0))
    bundle=save_bundle(scratch,model,optimizer,metadata,dict(offsets={'fixture':208}),keep=1,reserve_gib=0)
    weights=resolve_bundle(bundle)
    model.load_weights(weights)
    optimizer.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
    mx.eval(model.parameters(),optimizer.state)
    after=canonical_digest(model,optimizer)
    if json.loads(Path(weights+'.json').read_text())!=metadata:raise RuntimeError('Checkpoint metadata changed')
    with gzip.open(weights+'.rank0.data_state.json.gz','rt') as f:
        if json.load(f)!=dict(step=1,stream_state=dict(offsets={'fixture':208})):raise RuntimeError('Checkpoint cursor changed')
    if not all(control.exchange(after==before,'tiny-resume')):raise RuntimeError('Checkpoint roundtrip changed replica state')
    receipt=dict(rank=rank,world=5,backend='jaccl',ring=True,native_gather=True,all_sum=True,
                 weighted_gradient_max_error=err,checkpoint_digest=after,status='passed')
    (job/f'probe_rank{rank}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    control.exchange(True,'probe-finished')
    watchdog.cancel()
    print(f'[probe-passed] rank={rank} native gather/sum, weighted gradients, BF16/FP32 checkpoint roundtrip',flush=True)


if __name__=='__main__':main()
