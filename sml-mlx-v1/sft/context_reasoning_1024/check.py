"""Real 1024 smoke test in disposable memory, plus numerical/resume unit tests."""
import gc
import math
import tempfile
import time
import unittest
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from sml_v1.common import read_json
from sml_v1.checkpoint_bundle import save_bundle
from . import engine
from .protocol import DIR,verify
from .runner import rebuild,restore


def compare(a,b):
    aa=dict(tree_flatten(a));bb=dict(tree_flatten(b))
    if aa.keys()!=bb.keys(): raise ValueError('Tensor keys differ')
    return max(float(mx.max(mx.abs(aa[k]-bb[k])).item()) for k in aa)


def run_checks(cfg,frozen):
    from . import test_context
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_context)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise ValueError('Unit checks failed')
    data=rebuild(cfg)
    b=engine.load(cfg); anchor=engine.load(cfg); opt=engine.optimizer(cfg); opt.init(b.model.trainable_parameters())
    from collections import Counter
    for family,n in cfg['per_update'].items():
        assert len(data['train'][family])==n*cfg['updates']
    from .data import audit
    audit(data['new_split'])
    for split in ('train','dev','test'):
        for family in ('commonsense','instruction','reading'):
            for row in data[split][family]:
                e=engine.encode(b,row,cfg['context']);head=b.encode('User: '+row['prompt']+'\nAssistant:')
                assert e['y'][:len(head)-1]==[-100]*(len(head)-1) and e['y'][-1]==b.eos
    for row in data['train']['commonsense']:
        engine.choice_arrays(b,row)
    rows=engine.batch_at(data,cfg,0); frozen_anchor={k:mx.array(v) for k,v in tree_flatten(anchor.model.parameters())}
    from sft.reading_repair.generation import verify_generation
    parity=verify_generation(b,['Hello!','The carton contains a red scarf. What does it contain?','Please politely ask someone to close a door.'],16)
    from .evaluate import assess
    small={'dev':{family:rows[:2] for family,rows in data['dev'].items()}}
    assessment=assess(b,small,'dev',dict(cfg,max_new_tokens=16))
    for value in assessment['metrics'].values():
        numbers=value.values() if isinstance(value,dict) else [value]
        if not all(math.isfinite(v) for v in numbers): raise ValueError('Nonfinite evaluation smoke check')
    rows=engine.batch_at(data,cfg,0)
    t=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='opensml-skill-check-') as temp:
        first=engine.update(b,anchor,opt,rows,data['anchors'][0],cfg,1)
        meta=dict(step=cfg['source_step']+1,additional_updates=1,contract=__import__('sml_v1.common',fromlist=['fingerprint']).fingerprint(frozen),
                  next_examples={k:n for k,n in cfg['per_update'].items()},training=[first])
        saved=save_bundle(Path(temp)/'resume',b.model,opt,meta,None,keep=1)
        second=engine.update(b,anchor,opt,engine.batch_at(data,cfg,1),data['anchors'][1],cfg,2)
        mx.save_safetensors(str(Path(temp)/'expected.safetensors'),dict(tree_flatten(b.model.parameters())))
        mx.save_safetensors(str(Path(temp)/'state.safetensors'),dict(tree_flatten(opt.state)))
        del opt;gc.collect();mx.clear_cache();opt=engine.optimizer(cfg)
        step,_=restore(b,opt,saved,frozen,cfg)
        if step!=1: raise ValueError('Wrong restored cursor')
        repeated=engine.update(b,anchor,opt,engine.batch_at(data,cfg,1),data['anchors'][1],cfg,2)
        expected=mx.load(str(Path(temp)/'expected.safetensors'));actual=dict(tree_flatten(b.model.parameters()))
        gap=max(float(mx.max(mx.abs(expected[k]-actual[k])).item()) for k in expected)
        expected_state=mx.load(str(Path(temp)/'state.safetensors'));actual_state=dict(tree_flatten(opt.state))
        state_gap=max(float(mx.max(mx.abs(expected_state[k]-actual_state[k])).item()) for k in expected_state)
        if gap>1e-6 or state_gap>1e-6 or abs(second['loss']-repeated['loss'])>1e-6: raise ValueError('Non-exact optimizer resume')
        anchor_after=dict(tree_flatten(anchor.model.parameters()))
        if any(not bool(mx.array_equal(frozen_anchor[k],anchor_after[k]).item()) for k in frozen_anchor): raise ValueError('Frozen anchor mutated')
    elapsed=time.monotonic()-t
    del b,anchor,opt,data;gc.collect();mx.clear_cache();verify(frozen)
    return dict(unit_tests=result.testsRun,selection_rebuilt=True,cache_parity=parity,
        evaluation_smoke=assessment['metrics'],
        first_disposable_update=first,second_disposable_update=second,resume_model_max_abs=gap,
        resume_state_max_abs=state_gap,anchor_unchanged=True,disposable_updates=3,
        check_update_seconds=elapsed,production_updates=0)
