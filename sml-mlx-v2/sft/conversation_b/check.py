"""Data, masking, numerical and real-model resume checks; temporary outputs only."""
import gc
import math
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from sml_v2.common import read_json
from sml_v2.checkpoint_bundle import save_bundle
from sml_v2.tokenization import Tokenizer
from . import engine
from sft.conversation_ab.data import ROOT,DIR,rebuild,norm
from .launch import restore,metadata

def gap(a,b):
    assert a.keys()==b.keys()
    return max(float(mx.max(mx.abs(a[k]-b[k])).item()) for k in a)

def run_checks(cfg,frozen):
    from .test_unlikelihood import run_unit_checks
    unit_checks=run_unit_checks()
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    data=rebuild(tok)
    print('[check] Frozen selections rebuilt; checking splits, masks and epoch cursors.',flush=True)
    view=SimpleNamespace(encode=tok.encode,eos=tok.eos,pad=tok.pad)
    for split in ('train','dev','test'):
        for row in data[split]:
            e=engine.encode(view,row,cfg['context'])
            head=tok.encode('User: '+row['prompt']+'\nAssistant:')
            assert all(y==-100 for y in e['y'][:len(head)-1])
            assert e['y'][-1]==tok.eos
            assert all(y not in (tok.pad,2,3) for y in e['y'][len(head)-1:])
            assert e['targets']==len(e['y'])-len(head)+1
        for other in ('train','dev','test'):
            if split==other:continue
            # Shared source trees and exact prompt duplicates are disallowed.
            groups={r['group'] for r in data[split] if 'group' in r}
            assert not groups & {r['group'] for r in data[other] if 'group' in r}
            assert not {norm(r['prompt']) for r in data[split]} & {norm(r['prompt']) for r in data[other]}
    assert len(data['retention']['commonsense'])==32
    assert Counter(r['source'] for r in data['retention']['commonsense'])=={'commonsenseqa':16,'socialiqa':16}
    for epoch in range(3):
        rows=[r for n in range(epoch*40,(epoch+1)*40) for r in engine.batch_at(data,cfg,n)]
        assert len({r['id'] for r in rows})==160
        assert Counter(r['source'] for r in rows)=={'oasst2':96,'original':24,'rehearsal':40}
        assert [r['id'] for r in engine.batch_at(data,cfg,epoch*40+17)]==[r['id'] for r in rows[68:72]]
    assert engine.learning_rate(12,cfg)==cfg['peak_lr']
    assert abs(engine.learning_rate(120,cfg)-cfg['final_lr'])<1e-12
    # Padding and unequal-length examples must not change example weighting.
    class Toy(nn.Module):
        def __init__(self):super().__init__();self.embedding=nn.Embedding(8,8)
        def __call__(self,x):return {'logits':self.embedding(x)}
    toy=Toy();x=mx.array([[1,2,3],[4,0,0]]);y=mx.array([[2,3,4],[5,-100,-100]])
    direct=float(engine.objective(toy,x,y).item())
    separate=(float(engine.objective(toy,x[:1],y[:1]).item())+float(engine.objective(toy,x[1:,:1],y[1:,:1]).item()))/2
    assert abs(direct-separate)<1e-6
    a,ga=engine.gradients(toy,x,y,microbatch=1);b,gb=engine.gradients(toy,x,y,microbatch=2)
    assert abs(float(a.item())-float(b.item()))<1e-6
    assert gap(dict(tree_flatten(ga)),dict(tree_flatten(gb)))<1e-6
    del toy,ga,gb
    print('[check] Data and objective checks passed; checking real-model generation and resume.',flush=True)
    b=engine.load(cfg);opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters())
    from sft.reading_repair.generation import verify_generation
    parity=verify_generation(b,['Hello!','Please politely ask someone to close a door.',data['dev'][0]['prompt']],16)
    from sft.conversation_ab.evaluate import assess
    small={'dev':data['dev'][:1],'retention':{k:v[:1] for k,v in data['retention'].items()}}
    assessment=assess(b,small,dict(cfg,max_new_tokens=16))
    assert math.isfinite(assessment['metrics']['natural_nll']) and math.isfinite(assessment['metrics']['prose_nll'])
    with tempfile.TemporaryDirectory(prefix='opensml-conversation-B-check-') as temp:
        p=Path(temp)
        first=engine.update(b,opt,engine.batch_at(data,cfg,0),cfg,1)
        saved=save_bundle(p/'resume',b.model,opt,metadata(1,frozen,cfg),None,keep=1)
        second=engine.update(b,opt,engine.batch_at(data,cfg,1),cfg,2)
        mx.save_safetensors(str(p/'expected.safetensors'),dict(tree_flatten(b.model.parameters())))
        mx.save_safetensors(str(p/'state.safetensors'),dict(tree_flatten(opt.state)))
        del opt;gc.collect();mx.clear_cache();opt=engine.optimizer(cfg)
        assert restore(b,opt,saved,frozen,cfg)==1
        repeated=engine.update(b,opt,engine.batch_at(data,cfg,1),cfg,2)
        model_gap=gap(mx.load(str(p/'expected.safetensors')),dict(tree_flatten(b.model.parameters())))
        state_gap=gap(mx.load(str(p/'state.safetensors')),dict(tree_flatten(opt.state)))
        if max(model_gap,state_gap,abs(second['loss']-repeated['loss']))>1e-6:raise ValueError('Resume differs from uninterrupted update')
    del b,opt,data;gc.collect();mx.clear_cache()
    return dict(unlikelihood_unit_checks=unit_checks,selection_rebuilt=True,split_and_mask_checks=True,epoch_cursor_checks=True,padding_and_gradient_checks=True,cache_parity=parity,evaluation_smoke=assessment['metrics'],first_disposable_update=first,second_disposable_update=second,resume_model_gap=model_gap,resume_optimizer_gap=state_gap,disposable_updates=3,production_updates=0,test_generations=0)
