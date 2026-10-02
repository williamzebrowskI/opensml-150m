"""Real disposable updates, EOS/mask tests, evaluation and exact bundle-resume."""
import gc
import math
import tempfile
import time
from collections import Counter
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from sml_v2.common import fingerprint, atomic_json, file_sha256
from sml_v2.checkpoint_bundle import save_bundle
from sft.skill_recovery_768 import engine
from sft.reading_repair.generation import verify_generation
from .protocol import verify
from .data import LABELS, split_of
from .evaluate import grade, assess


def run_checks(cfg, frozen):
    from .launch import rebuild, restore
    data = rebuild(cfg)
    assert grade('Supported.', 'end', 'entailment')['correct_label_and_surface'] is False
    assert grade('Contradicted. The boy is running, whereas the claim says he is still.', 'end', 'contradiction')['correct_label_and_surface']
    assert not grade('Supported. The boy is running in the park.', 'length', 'entailment')['complete_surface']
    assert not grade('Supported. The boy is running in the park.', 'end', 'neutral')['label_correct']
    b = engine.load(cfg); anchor = engine.load(cfg); opt = engine.optimizer(cfg)
    opt.init(b.model.trainable_parameters()); mx.eval(opt.state)
    for split in ('train','dev','test'):
        for family, rows in data[split].items():
            for row in rows:
                enc = engine.encode(b, row, cfg['context']); head=b.encode(engine.prefix(row['prompt']))
                assert enc['y'][:len(head)-1] == [-100]*(len(head)-1)
                assert enc['y'][-1] == b.eos and len(enc['x']) == len(enc['y']) <= cfg['context']
                assert enc['targets'] == sum(i != -100 for i in enc['y'])
                if family == 'explanation': assert split_of(row['group']) == split
    groups = [{r['group'] for r in data[s]['explanation']} for s in ('train','dev','test')]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert len(groups[0]) == 3072
    for u in range(cfg['updates']):
        batch=engine.batch_at(data,cfg,u)
        assert Counter(r['label'] for r in batch['explanation']) == Counter({k:2 for k in LABELS})
        assert {r['source'] for r in batch['commonsense']} == {'commonsenseqa','socialiqa'}
    parity=verify_generation(b,['Hello!',data['dev']['explanation'][0]['prompt']],24)
    small=dict(data)
    small['dev']={'explanation':[next(r for r in data['dev']['explanation'] if r['label']==label) for label in LABELS]}
    small['replay']=dict(data['replay'],dev={f:rr[:2] for f,rr in data['replay']['dev'].items()})
    smoke=assess(b,small,'dev',dict(cfg,max_new_tokens=24))
    assert math.isfinite(smoke['metrics']['assistant_nll'])
    before={k:mx.array(v) for k,v in tree_flatten(anchor.model.parameters())};mx.eval(before)
    t=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='explanation-transfer-check-') as tmp:
        tmp=Path(tmp)
        first=engine.update(b,anchor,opt,engine.batch_at(data,cfg,0),data['anchors'][0],cfg,1)
        assert first['grad_norm'] > 0 and first['explanation_targets'] > 6
        delta=max(float(mx.max(mx.abs(v-before[k])).item()) for k,v in tree_flatten(b.model.parameters()))
        assert delta > 0
        meta=dict(step=1025,source_step=1024,additional_updates=1,
                  next_examples=cfg['per_update'],contract=fingerprint(frozen),training=[first],
                  training_format='plain-user-assistant-eos-v1')
        saved=save_bundle(tmp/'resume',b.model,opt,meta,None,keep=1000000)
        atomic_json(tmp/'resume'/'contract.json',frozen)
        # Exercise the shared benchmark architecture/format contract using a
        # disposable bundle. No benchmark questions are generated or scored.
        from evaluation.full_benchmarks import spec, core
        nickname='explanation-transfer-disposable-check'
        spec.MODELS[nickname]=dict(bundle=saved,step=1025,sha256=file_sha256(Path(saved)/'model.safetensors'))
        try:
            for suite in ('multiple-choice','ifeval'):
                manifest=core.frozen(nickname,suite)
                assert manifest['step']==1025 and manifest['training'] is False
        finally: spec.MODELS.pop(nickname,None)
        second=engine.update(b,anchor,opt,engine.batch_at(data,cfg,1),data['anchors'][1],cfg,2)
        mx.save_safetensors(str(tmp/'expected.safetensors'),dict(tree_flatten(b.model.parameters())))
        mx.save_safetensors(str(tmp/'state.safetensors'),dict(tree_flatten(opt.state)))
        del opt;gc.collect();mx.clear_cache();opt=engine.optimizer(cfg)
        u,_=restore(b,opt,saved,frozen,cfg);assert u==1
        repeated=engine.update(b,anchor,opt,engine.batch_at(data,cfg,1),data['anchors'][1],cfg,2)
        expected=mx.load(str(tmp/'expected.safetensors'));actual=dict(tree_flatten(b.model.parameters()))
        gap=max(float(mx.max(mx.abs(expected[k]-actual[k])).item()) for k in expected)
        es=mx.load(str(tmp/'state.safetensors'));ss=dict(tree_flatten(opt.state))
        sg=max(float(mx.max(mx.abs(es[k].astype(mx.float32)-ss[k].astype(mx.float32))).item()) for k in es)
        assert gap <= 1e-7 and sg <= 1e-7 and abs(second['loss']-repeated['loss']) <= 1e-7
        assert int(opt.state['step'].item()) == 2
        assert all(bool(mx.array_equal(v,before[k]).item()) for k,v in tree_flatten(anchor.model.parameters()))
    verify(frozen)
    return dict(selection_rebuilt=True,all_masks_eos_and_lengths_checked=True,independent_train_scenes=3072,
        labels_per_update=dict(Counter(r['label'] for r in engine.batch_at(data,cfg,0)['explanation'])),
        cached_generation_parity=parity,evaluation_smoke=smoke['metrics'],
        first_disposable_update=first,second_disposable_update=second,parameter_change=delta,
        resume_model_max_error=gap,resume_optimizer_max_error=sg,frozen_anchor_unchanged=True,
        benchmark_contracts_validated=True,public_benchmark_items_scored=0,
        check_update_seconds=time.monotonic()-t,production_updates=0)
