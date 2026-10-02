"""Prove optimizer inheritance, exact stopping/resume and parent-best retention."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
mx.set_default_device(mx.cpu)
from sml_v1.checkpoint_bundle import resolve_bundle,save_bundle
from sml_v1.common import atomic_json,read_json
from sml_v1.precision import MasterAdamW
from sml_v1.pretrain import clip_gradients
from sft.transfer_control import engine
from sft.transfer_control.data import records
from sft.response_expansion import launch
from sft.response_expansion.data import audit,new_records,training_records
from sft.response_expansion.rubrics import grade,audit_references


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(128,8);self.projection=nn.Linear(8,128)
    def __call__(self,x):return {'logits':self.projection(self.embedding(x))}


def toy_load(name):
    if name!='opensml':raise AssertionError('Reference model must never load')
    return SimpleNamespace(model=Tiny(),encode=lambda t:[ord(c) for c in t],eos=0,pad=0,name='current')


def metrics(score=.5):
    return dict(exact=score,both_correct=score,rule_macro=score,stopped=1.,assistant_nll=1.)


class ContinuationTests(unittest.TestCase):
    def source(self,root):
        mx.random.seed(713);backend=toy_load('opensml')
        optimizer=MasterAdamW(learning_rate=3e-5,betas=(.9,.95),weight_decay=.01)
        rows=records('train')[:8]
        x,y=engine.arrays([engine.encode(backend,r) for r in rows[:2]])
        for _ in range(2):
            _,g=engine.gradients(backend.model,x,y,False,2)
            g,_=clip_gradients(g,1.);optimizer.update(backend.model,g);mx.eval(backend.model.parameters(),optimizer.state)
        path=save_bundle(root/'source',backend.model,optimizer,dict(step=2,arm='opensml'),None,best=True,reserve_gib=0)
        cfg=dict(source_bundle=path,source_step=2,additional_updates=4,learning_rate=3e-6,
                 batch=2,microbatch=2,betas=[.9,.95],weight_decay=.01,clip_norm=1.,
                 evaluation_updates=[0,2,4],checkpoint_every=1,max_new_tokens=8,legacy_max_new_tokens=8)
        evaluation=dict(unseen=metrics(),prose=3.)
        atomic_json(root/'original/opensml/evaluations/step_0000002.json',evaluation)
        return cfg,rows,optimizer

    def test_restores_all_optimizer_state_and_changes_only_lr(self):
        with tempfile.TemporaryDirectory() as d,patch.object(engine,'load',side_effect=toy_load):
            cfg,_,old=self.source(Path(d))
            b,new=launch.restore(Path(cfg['source_bundle'])/'model.safetensors',cfg)
            before=dict(tree_flatten(old.state));after=dict(tree_flatten(new.state))
            self.assertEqual(set(before),set(after))
            self.assertEqual(launch.optimizer_step(new),2)
            for key in before:
                if key=='learning_rate':continue
                self.assertEqual(mx.max(mx.abs(before[key].astype(mx.float32)-after[key].astype(mx.float32))).item(),0.)
            self.assertAlmostEqual(float(new.learning_rate.item()),3e-6,places=12)

    def test_interruption_exact_count_constant_lr_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as d,patch.object(engine,'load',side_effect=toy_load),contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);cfg,rows,_=self.source(root)
            def assess(backend,cfg,update,step,prose):
                return dict(new_tasks=metrics(.5+update*.01),legacy=metrics(.5+update*.01),prose=3.,prose_change_from_parent=0.)
            with patch.object(launch,'training_records',return_value=rows),patch.object(launch,'ORIGINAL',root/'original'), \
                 patch.object(launch,'assess',side_effect=assess),patch.object(launch,'evaluate_new',return_value=metrics()):
                stop=dict(requested=False);original=engine.gradients;calls=[]
                def interrupt(*args,**kwargs):
                    result=original(*args,**kwargs);calls.append(1)
                    if len(calls)==2:stop['requested']=True
                    return result
                with patch.object(launch,'OUTPUT',root/'resumed'),patch.object(engine,'gradients',side_effect=interrupt):
                    launch.run(cfg,{},stop)
                    report=read_json(root/'resumed/report.json')
                    self.assertEqual((report['status'],report['step'],report['additional_updates']),('stopped',4,2))
                with patch.object(launch,'OUTPUT',root/'resumed'):
                    launch.run(cfg,{},dict(requested=False))
                    first=read_json(root/'resumed/report.json')
                    launch.run(cfg,{},dict(requested=False))
                    self.assertEqual(first,read_json(root/'resumed/report.json'))
                with patch.object(launch,'OUTPUT',root/'continuous'):
                    launch.run(cfg,{},dict(requested=False))
                self.assertEqual(first['step'],6);self.assertEqual(first['additional_updates'],4)
                self.assertEqual([t['lr'] for t in first['training']],[3e-6]*4)
                self.assertEqual([t['step'] for t in first['training']],[3,4,5,6])
                second=read_json(root/'continuous/report.json')
                self.assertEqual(first['training'],second['training'])
                a=mx.load(resolve_bundle(root/'resumed/latest.json'));b=mx.load(resolve_bundle(root/'continuous/latest.json'))
                for k in a:self.assertEqual(mx.max(mx.abs(a[k]-b[k])).item(),0.)
                self.assertEqual([x['additional_updates'] for x in first['evaluations']],[0,2,4])

    def test_parent_remains_best_when_continuation_is_worse(self):
        with tempfile.TemporaryDirectory() as d,patch.object(engine,'load',side_effect=toy_load),contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);cfg,rows,_=self.source(root)
            def assess(backend,cfg,update,step,prose):
                return dict(new_tasks=metrics(.5 if update==0 else .25),legacy=metrics(.5 if update==0 else .25),prose=3.,prose_change_from_parent=0.)
            with patch.object(launch,'training_records',return_value=rows),patch.object(launch,'ORIGINAL',root/'original'), \
                 patch.object(launch,'OUTPUT',root/'out'),patch.object(launch,'assess',side_effect=assess), \
                 patch.object(launch,'evaluate_new',return_value=metrics()):
                launch.run(cfg,{},dict(requested=False))
            report=read_json(root/'out/report.json')
            self.assertEqual(report['best']['step'],2)
            self.assertFalse(report['improved_on_development'])
            source=mx.load(str(Path(cfg['source_bundle'])/'model.safetensors'))
            best=mx.load(resolve_bundle(root/'out/best.json'))
            for k in source:self.assertEqual(mx.max(mx.abs(source[k]-best[k])).item(),0.)


class DataTests(unittest.TestCase):
    def test_split_isolation_and_training_only_rehearsal(self):
        result=audit()
        self.assertEqual((result['train_examples'],result['new_examples'],result['rehearsal_examples']),(1024,768,256))
        self.assertEqual((result['development_examples'],result['test_examples']),(104,156))
        self.assertEqual(training_records(),training_records())
        self.assertEqual(len({r['id'] for r in training_records()}),1024)

    def test_all_authored_references_pass(self):
        for split in ('train','dev','test'):audit_references(new_records(split))

    def test_empty_echo_truncated_and_wrong_fact_fail(self):
        for row in new_records('dev'):
            for text,stop in [('', 'end'),(row['prompt'],'end'),(row['answer'],'length')]:
                self.assertFalse(grade(row,dict(text=text,stop=stop))['rule_pass'])
        row=next(r for r in new_records('dev') if r['family']=='owner')
        incorrect=row['answer'].replace(row['answer'].split()[-1].rstrip('.'),row['rubric']['forbidden'][0])
        self.assertFalse(grade(row,dict(text=incorrect,stop='end'))['rule_pass'])
        row=next(r for r in new_records('dev') if r['family']=='explanation')
        self.assertFalse(grade(row,dict(text='heat',stop='end'))['rule_pass'])


if __name__=='__main__':unittest.main()
