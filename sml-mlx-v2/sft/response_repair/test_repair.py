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
from sml_v2.checkpoint_bundle import resolve_bundle,save_bundle
from sml_v2.common import atomic_json,read_json
from sml_v2.precision import MasterAdamW
from sml_v2.pretrain import clip_gradients
from sft.transfer_control import engine
from sft.transfer_control.data import records
from sft.response_repair import launch
from sft.response_repair.data import audit,new_records,training_records
from sft.response_repair.rubrics import grade,audit_references


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(128,8);self.projection=nn.Linear(8,128)
    def __call__(self,x):return {'logits':self.projection(self.embedding(x))}


def toy_load(name):
    if name!='opensml':raise AssertionError('Reference model must never load')
    return SimpleNamespace(model=Tiny(),encode=lambda t:[ord(c) for c in t],eos=0,pad=0,name='current')


def metrics(score=.5):
    return dict(exact=score,both_correct=score,task_macro=score,fact_content=score,fact_format=score,
                fact_joint=score,echo_rate=0.,stopped=1.,assistant_nll=1.)


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
        self.assertEqual((result['training_examples'],result['new_factual_examples'],result['new_response_examples'],result['rehearsal_examples']),(1024,256,512,256))
        self.assertEqual((result['splits']['dev']['examples'],result['splits']['test']['examples']),(208,208))
        self.assertEqual(training_records(),training_records())
        self.assertEqual(len({r['id'] for r in training_records()}),1024)

    def test_all_authored_references_pass(self):
        for split in ('train','dev','test'):audit_references(new_records(split))

    def test_sentence_targets_obey_the_sentence_instruction(self):
        for split in ('train','dev','test'):
            for row in new_records(split):
                if 'one sentence' in row['prompt'].lower() or 'one friendly sentence' in row['prompt'].lower():
                    self.assertEqual(sum(row['answer'].count(p) for p in '.!?'),1,row['id'])

    def test_empty_echo_and_truncated_answers_fail(self):
        for row in new_records('dev'):
            for text,stop in [('', 'end'),(row['prompt'],'end'),(row['answer'],'length')]:
                self.assertFalse(grade(row,dict(text=text,stop=stop))['task_pass'])

    def test_content_and_requested_format_are_separate(self):
        rows=[r for r in new_records('dev') if r['family']=='location']
        brief,full=rows[:2]
        result=grade(brief,dict(text=full['answer'],stop='end'))
        self.assertTrue(result['content_match']);self.assertFalse(result['format_match'])
        wrong=rows[2]['answer']
        result=grade(brief,dict(text=wrong,stop='end'))
        self.assertFalse(result['content_match']);self.assertTrue(result['format_match'])

    def test_copying_only_passes_when_requested(self):
        rows=[r for r in new_records('dev') if r['family']=='copy_or_reply']
        reply,copy=rows[:2]
        self.assertTrue(grade(copy,dict(text=copy['answer'],stop='end'))['task_pass'])
        result=grade(reply,dict(text=copy['answer'],stop='end'))
        self.assertTrue(result['prompt_echo']);self.assertFalse(result['task_pass'])
        self.assertFalse(grade(copy,dict(text=copy['answer'].upper(),stop='end'))['task_pass'])

    def test_selection_cannot_trade_away_factual_or_format_accuracy(self):
        baseline=dict(new_tasks=metrics(.5),legacy=metrics(.6))
        floors=launch.parent_metrics(baseline)
        for key in ('fact_content','fact_format'):
            candidate=dict(new_tasks=metrics(.9),legacy=metrics(.7))
            candidate['new_tasks'][key]=.49
            self.assertGreater(launch.ranking(candidate),launch.ranking(baseline))
            self.assertFalse(launch.eligible(candidate,floors))
        candidate=dict(new_tasks=metrics(.9),legacy=metrics(.59))
        self.assertFalse(launch.eligible(candidate,floors))
        self.assertTrue(launch.eligible(baseline,floors))


if __name__=='__main__':unittest.main()
