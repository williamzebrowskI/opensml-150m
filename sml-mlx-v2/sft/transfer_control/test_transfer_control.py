"""Meaningful data-isolation, objective and exact-resume tests on tiny models."""
import json
import re
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
mx.set_default_device(mx.cpu)
from sft.transfer_control import data, engine
from sml_v2.checkpoint_bundle import save_bundle, resolve_bundle
from sml_v2.common import read_json
from sml_v2.precision import MasterAdamW


class CharacterTokenizer:
    eos = 0
    def encode(self, text): return [ord(c) for c in text]


class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = nn.Embedding(128, 8); self.projection = nn.Linear(8, 128)
    def __call__(self, x): return self.projection(self.embedding(x))


class TransferTests(unittest.TestCase):
    def test_pairs_splits_and_coverage(self):
        a = data.audit(); self.assertEqual(a['examples'], dict(train=1024, dev=128, test=192))
        train, dev = data.records('train'), data.records('dev')
        self.assertEqual(len({r['id'] for r in train}), 1024)
        self.assertFalse({r['prompt'] for r in train} & {r['prompt'] for r in dev})
        familiar = data.records('dev', 'familiar')
        self.assertEqual([r['answer'] for r in dev], [r['answer'] for r in familiar])
        self.assertTrue(all(a['prompt'] != b['prompt'] for a,b in zip(dev, familiar)))
        batches = [train[i:i+8] for i in range(0,1024,8)]
        resumed = [r for batch in batches[:37] for r in batch] + data.records('train')[37*8:]
        self.assertEqual(train, resumed)

    def test_constant_predictions_cannot_pass_pairs(self):
        rr = data.records('dev')
        for i in range(0,len(rr),2):
            prediction = dict(text=rr[i]['answer'], stop='end')
            self.assertTrue(data.normalized(rr[i]['answer']) != data.normalized(rr[i+1]['answer']))
            self.assertFalse(all(engine.grade(r,prediction)['exact'] for r in rr[i:i+2]))

    def test_answers_are_not_always_in_one_sentence_position(self):
        for split in ('train','dev','test'):
            rows=[r for r in data.records(split) if r['family']=='location']
            positions=Counter()
            for r in rows:
                locations=re.findall(r'is in the (\w+)',r['prompt'])
                self.assertEqual(len(locations),2)
                positions[locations.index(r['answer'])]+=1
            self.assertEqual(positions[0],positions[1])

    def test_mask_and_prefix_boundary(self):
        tok = CharacterTokenizer(); row = dict(prompt='Find Ada.', answer='Ada')
        r = engine.encode(tok,row); prefix = tok.encode(engine.prefix(row['prompt']))
        targets = [v for v in r['y'] if v != -100]
        self.assertEqual(targets, tok.encode(' Ada')+[tok.eos])
        self.assertEqual(r['y'][:len(prefix)-1], [-100]*(len(prefix)-1))
        self.assertEqual(len(r['x']),len(r['y']))
        with self.assertRaises(ValueError): engine.encode(tok,row,context=3)

    def test_padding_microbatch_and_optimizer_resume(self):
        mx.random.seed(192); model=Tiny()
        rows=[engine.encode(CharacterTokenizer(),dict(prompt='Where?',answer=a)) for a in ('shed','the attic','hall')]
        x,y=engine.arrays(rows)
        full,grad=engine.gradients(model,x,y,True,3)
        small,micro=engine.gradients(model,x,y,True,1)
        self.assertAlmostEqual(full.item(),small.item(),places=5)
        for (_,a),(_,b) in zip(tree_flatten(grad),tree_flatten(micro)):
            self.assertLess(mx.max(mx.abs(a-b)).item(),1e-6)
        # Masking prompt targets must make their assigned token IDs irrelevant.
        manual=nn.losses.cross_entropy(model(x).astype(mx.float32),mx.maximum(y,0),reduction='none')
        expected=((manual*(y!=-100)).sum(axis=1)/(y!=-100).sum(axis=1)).mean()
        self.assertAlmostEqual(full.item(),expected.item(),places=6)
        opt=MasterAdamW(learning_rate=3e-5,betas=(.9,.95),weight_decay=.01)
        opt.update(model,grad);mx.eval(model.parameters(),opt.state)
        with tempfile.TemporaryDirectory() as d:
            save_bundle(d,model,opt,dict(step=1),None,keep=1,reserve_gib=0)
            path=resolve_bundle(Path(d)/'latest.json')
            clone=Tiny();clone.load_weights(path)
            restored=MasterAdamW(learning_rate=3e-5,betas=(.9,.95),weight_decay=.01)
            restored.state=tree_unflatten(list(mx.load(path+'.optimizer.safetensors').items()))
            for m,o in ((model,opt),(clone,restored)):
                _,g=engine.gradients(m,x,y,True,2);o.update(m,g);mx.eval(m.parameters(),o.state)
            for (_,a),(_,b) in zip(tree_flatten(model.parameters()),tree_flatten(clone.parameters())):
                self.assertEqual(mx.max(mx.abs(a-b)).item(),0.)

    def test_lr_schedule(self):
        cfg=read_json(Path(__file__).parent/'config.json')
        self.assertEqual(engine.learning_rate(8,cfg),3e-5)
        self.assertEqual(engine.learning_rate(128,cfg),3e-6)
        self.assertGreater(engine.learning_rate(7,cfg),engine.learning_rate(1,cfg))

    def test_launcher_interruption_matches_uninterrupted(self):
        # Exercise the actual launcher save/restore path, not just the optimizer.
        # All model loads and evaluation calls are replaced with tiny random
        # models/stubs; this cannot load or train either real pretrained model.
        from sft.transfer_control import launch
        cfg=read_json(Path(__file__).parent/'config.json')
        cfg.update(updates=4,warmup_updates=1,evaluation_steps=[0,2,4])
        def toy_load(arm):
            mx.random.seed(981)
            return SimpleNamespace(model=Tiny(),encode=CharacterTokenizer().encode,eos=0,pad=0,name='reference')
        metrics=dict(exact=.25,both_correct=.125,same_answer_pairs=.5,stopped=1.,assistant_nll=2.)
        original=engine.gradients
        with tempfile.TemporaryDirectory() as d, patch.object(engine,'load',toy_load), \
             patch.object(engine,'evaluate',return_value=metrics),patch.object(engine,'prose_loss',return_value=4.):
            root=Path(d);stop=dict(requested=False);calls=[]
            def interrupted(*args,**kwargs):
                value=original(*args,**kwargs);calls.append(1)
                if len(calls)==2:stop['requested']=True
                return value
            with patch.object(launch,'OUTPUT',root/'resumed'),patch.object(engine,'gradients',interrupted):
                launch.run_arm('reference',cfg,{},stop)
                report=read_json(root/'resumed/reference/report.json')
                self.assertEqual((report['status'],report['step']),('stopped',2))
            with patch.object(launch,'OUTPUT',root/'resumed'):
                launch.run_arm('reference',cfg,{},dict(requested=False))
            with patch.object(launch,'OUTPUT',root/'continuous'):
                launch.run_arm('reference',cfg,{},dict(requested=False))
            paths=[root/n/'reference' for n in ('resumed','continuous')]
            states=[mx.load(resolve_bundle(p/'latest.json')) for p in paths]
            for k in states[0]:self.assertEqual(mx.max(mx.abs(states[0][k]-states[1][k])).item(),0.)
            reports=[read_json(p/'report.json') for p in paths]
            self.assertEqual(reports[0]['training'],reports[1]['training'])
            self.assertEqual([r['step'] for r in reports[0]['evaluations']],[0,2,4])
            self.assertEqual(reports[0]['status'],'complete')


if __name__=='__main__':unittest.main()
