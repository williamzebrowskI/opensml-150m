"""Validate masks, exact resume, source separation and real-model gradients on scratch copies."""
import json,gc,math,tempfile,unittest
from pathlib import Path
from sml_v2.common import read_json,fingerprint
from sft.complete_transfer_1024.protocol import DIR,ROOT

def run_checks(cfg,frozen,prepared=None):
 import mlx.core as mx
 import numpy as np
 from mlx.utils import tree_flatten
 from sft.complete_transfer_1024 import engine
 from sft.complete_transfer_1024.runner import rebuild,restore
 from sft.complete_transfer_1024.evaluate import grade
 from sft.complete_transfer_1024.test_completion import CompletionTests
 from sft.reading_repair.generation import verify_generation
 from sml_v2.checkpoint_bundle import save_bundle
 tests=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(CompletionTests))
 if not tests.wasSuccessful():raise ValueError('Completion verifier tests failed')
 # Reconstruct by default; setup may pass the hash-verified selection already streamed.
 data=rebuild(cfg) if prepared is None else prepared
 if fingerprint(data)!=read_json(DIR/'selection.json')['data_hash']:raise ValueError('Prepared data hash mismatch')
 b=engine.load(cfg);anchor=engine.load(cfg);opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());mx.eval(opt.state)
 before={k:mx.array(v) for k,v in tree_flatten(anchor.model.parameters())};mx.eval(before)
 split_groups=[];encodings=0
 for s in ('train','dev','test'):
  groups=set()
  for family,rows in data['new'][s].items():
   for r in rows:
    if r['group'] in groups:raise ValueError('Duplicate source group')
    groups.add(r['group']);enc=engine.encode(b,r,cfg['context']);encodings+=1
    head=b.encode('User: '+r['prompt']+'\nAssistant:')
    if enc['y'][:len(head)-1]!=[-100]*(len(head)-1) or enc['y'][-1]!=b.eos:raise ValueError('Prompt/EOS mask error')
    if family=='complete' and not grade(r,r['answer'],'end')['passed']:raise ValueError('Gold verifier mismatch')
  split_groups.append(groups)
 if any(split_groups[i]&split_groups[j] for i in range(3) for j in range(i)):raise ValueError('Train/eval source overlap')
 for family in cfg['per_update']:
  if len(data['train'][family])!=cfg['updates']*cfg['per_update'][family]:raise ValueError('Exposure count')
 rows=engine.batch_at(data,cfg,0);v1=engine.update(b,anchor,opt,rows,data['anchors'][0],cfg,1)
 if not math.isfinite(v1['loss']):raise ValueError('Nonfinite update')
 with tempfile.TemporaryDirectory(prefix='complete-transfer-check-') as td:
  meta=dict(step=1025,source_step=1024,additional_updates=1,next_examples={k:n for k,n in cfg['per_update'].items()},contract=fingerprint(frozen),training=[v1])
  bundle=save_bundle(Path(td),b.model,opt,meta,None,keep=10,reserve_gib=3.)
  v2=engine.update(b,anchor,opt,engine.batch_at(data,cfg,1),data['anchors'][1],cfg,2)
  expected={k:np.array(v) for k,v in tree_flatten(b.model.parameters())};estate={k:np.array(v) for k,v in tree_flatten(opt.state)}
  u,t=restore(b,opt,Path(td)/'latest.json',frozen,cfg)
  if u!=1 or len(t)!=1:raise ValueError('Resume cursor')
  engine.update(b,anchor,opt,engine.batch_at(data,cfg,u),data['anchors'][u],cfg,u+1)
  delta=max(float(np.max(np.abs(np.array(v)-expected[k]))) for k,v in tree_flatten(b.model.parameters()))
  state_delta=max(float(np.max(np.abs(np.array(v)-estate[k]))) for k,v in tree_flatten(opt.state))
  if delta>1e-7 or state_delta>1e-6:raise ValueError(f'Resume mismatch {delta}, {state_delta}')
 if any(float(mx.max(mx.abs(v-before[k])).item())!=0 for k,v in tree_flatten(anchor.model.parameters())):raise ValueError('Frozen parent changed')
 generation=verify_generation(b,['Hello!','Explain how to organize a desk in two sentences.'],16)
 from sft.complete_transfer_1024.evaluate import assess
 small={'dev':{k:rr[:2] for k,rr in data['dev'].items()},'new':{'dev':{k:rr[:2] for k,rr in data['new']['dev'].items()}}}
 assessment=assess(b,small,'dev',dict(cfg,max_new_tokens=64))
 result=dict(tests=tests.testsRun,encoded_new_examples=encodings,source_groups_disjoint=True,distinct_updates_checked=2,disposable_optimizer_calls=3,resume_weight_max_error=delta,resume_state_max_error=state_delta,frozen_parent_unchanged=True,generation=generation,evaluation_smoke=assessment['metrics'],production_training_started=False)
 del b,anchor,opt;gc.collect();mx.clear_cache();return result
