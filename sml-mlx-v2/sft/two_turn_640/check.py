"""Data, masking, generation and exact optimizer-resume checks; disposable updates."""
import tempfile,gc
from pathlib import Path
from collections import Counter
import mlx.core as mx
from mlx.utils import tree_flatten
from sml_v2.common import fingerprint
from sml_v2.checkpoint_bundle import save_bundle
from . import engine
from .data import rebuild,batch_at,FAMILIES,history

def run_checks(cfg,frozen):
 data=rebuild(cfg);b=engine.load(cfg);mx.random.seed(cfg['seed'])
 allrows=data['new']+data['dev']+data['test']+data['replay']
 maxlen=0
 for r in allrows:
  e=engine.encode(b,r,cfg['context']);h=b.encode('User: '+r['prompt']+'\nAssistant:');full=b.encode('User: '+r['prompt']+'\nAssistant: '+r['answer'])+[b.eos]
  assert e['y'][:len(h)-1]==[-100]*(len(h)-1) and e['y'][len(h)-1:]==full[len(h):] and e['y'][-1]==b.eos
  maxlen=max(maxlen,len(e['x']))
  if r.get('turn')==1:assert r['prompt']==history(r['first_prompt'],r['first_answer'],r['current'])
 groups=[{r['group'] for r in rows} for rows in (data['new'],data['dev'],data['test'])]
 assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
 assert len({r['prompt'] for r in data['new']+data['dev']+data['test']})==960
 for rows in (data['new'],data['dev'],data['test']):
  for group in {r['group'] for r in rows}:
   rr=[r for r in rows if r['group']==group];assert len(rr)==3 and rr[1]['answer']!=rr[2]['answer']
 batchrows=[r for i in range(cfg['updates']) for r in batch_at(data,cfg,i)]
 counts=Counter(r['id'] for r in batchrows)
 assert len(batchrows)==1536 and all(counts[r['id']]==1 for r in data['new']) and all(counts[r['id']]==3 for r in data['replay'])
 # Compare microbatch accumulation and joint batch gradients.
 rr=batch_at(data,cfg,0)[:2];x,y=engine.arrays([engine.encode(b,r,cfg['context']) for r in rr],b.pad)
 v1,g1=engine.gradients(b.model,x,y,microbatch=1);v2,g2=engine.gradients(b.model,x,y,microbatch=2)
 d1=dict(tree_flatten(g1));d2=dict(tree_flatten(g2));err=max(float(mx.max(mx.abs(d1[k]-d2[k])).item()) for k in d1)
 assert abs(v1.item()-v2.item())<1e-4 and err<3e-4
 del g1,g2,d1,d2;gc.collect();mx.clear_cache()
 opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters())
 engine.update(b,opt,batch_at(data,cfg,0),cfg,1)
 from .launch import metadata
 with tempfile.TemporaryDirectory(prefix='two-turn-640-check-') as tmp:
  save_bundle(tmp,b.model,opt,metadata(1,frozen,cfg),None,keep=10,reserve_gib=0)
  restored=engine.load(cfg);op2=engine.optimizer(cfg);op2.init(restored.model.trainable_parameters())
  assert engine.restore(restored,op2,Path(tmp)/'latest.json',frozen,cfg)==1
  ra=engine.update(b,opt,batch_at(data,cfg,1),cfg,2);rb=engine.update(restored,op2,batch_at(data,cfg,1),cfg,2)
  compare=dict(tree_flatten(restored.model.parameters()));resume=max(float(mx.max(mx.abs(v-compare[k])).item()) for k,v in tree_flatten(b.model.parameters()))
  assert resume==0 and ra==rb
  del restored,op2,compare;gc.collect();mx.clear_cache()
 from sft.reading_repair.generation import verify_generation
 parity=verify_generation(b,[data['dev'][0]['prompt'],data['dev'][1]['prompt']],12)
 from .evaluate import assess
 sample=assess(b,dict(data,dev=data['dev'][:3],retention=data['retention'][:2]),dict(cfg,max_new_tokens=12))
 assert len(sample['own_history'])==2 and sample['metrics']['paired_groups']==1
 return dict(source_step=640,max_training_tokens=maxlen,mask_checks=len(allrows),microbatch_gradient_error=err,resume_parameter_error=resume,disposable_updates=2,training_exposures=1536,unique_new_targets=768,rehearsal_unique=256,split_groups_disjoint=True,paired_targets_different=True,cached_full_context_checks=parity,evaluation_smoke=True,reserved_test_generated=False,production_training_started=False)
