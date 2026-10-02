"""Data separation, nontrivial grading, optimizer continuity, exact one-pass resume."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mlx.core as mx
import mlx.nn as nn
mx.set_default_device(mx.cpu)
from mlx.utils import tree_flatten
from sml_v2.checkpoint_bundle import save_bundle,resolve_bundle
from sml_v2.common import read_json
from sml_v2.precision import MasterAdamW
from sft.transfer_control import engine
from sft.transfer_control.data import records as legacy
from sft.response_repair.data import training_records as prior_training
from sft.reading_repair import launch
from sft.reading_repair.cards import records
from sft.reading_repair.source import split_for,rehearsal,candidate,ngrams,normalized
from sft.reading_repair.evaluate import grade,eligible,generate,verify_generation


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(128,8);self.projection=nn.Linear(8,128)
    def __call__(self,x):return {'logits':self.projection(self.embedding(x))}


def toy_load(name):
    if name!='opensml':raise AssertionError('Reference must never load')
    return SimpleNamespace(model=Tiny(),encode=lambda t:[ord(c) for c in t],eos=0,pad=0,name='current')


def metrics(update=0):
    return dict(new_tasks=dict(task_macro=.1+update*.1,natural_nll=2-update*.1,natural_f1=.2+update*.1,
                             natural_exact=.2,paired=.1,challenge_correct=update*2,repeat_rate=0.,cap_rate=0.),
                legacy=dict(both_correct=.7),repair_joint=.8,prose=3.,prose_change_from_parent=0.)


class DataTests(unittest.TestCase):
    def test_cached_generation_handles_prefill_3d_then_step_2d(self):
        class Fake:
            training=True
            def train(self,mode=True):self.training=mode
            def eval(self):self.training=False
            def logits(self,ids):
                result=mx.full((1,ids.shape[1],8),-10.)
                result[0,-1,4 if ids.shape[1]==1 else 5 if ids.shape[1]==2 else 6 if ids.shape[1]==3 else 1]=10.
                return result,ids.shape[1]
            def step(self,ids,caches):
                result=mx.full((1,8),-10.);result[0,5 if caches==1 else 6 if caches==2 else 1]=10.
                return result,caches+1
        b=SimpleNamespace(model=Fake(),encode=lambda s:[7],decode=lambda ids:' '.join(map(str,ids)),eos=1)
        result=generate(b,'fixture',8)
        self.assertEqual(result['token_ids'],[4,5,6]);self.assertEqual(result['stop'],'end')
        self.assertTrue(b.model.training)
        self.assertTrue(verify_generation(b,['fixture'],8)[0]['cached_uncached_equal'])

    def test_scenario_split_and_reference_rubrics(self):
        previous=set();styles=[]
        for split,count in [('train',2688),('dev',448),('test',448)]:
            rows=records(split);self.assertEqual(len(rows),count)
            prompts={r['prompt'] for r in rows};self.assertEqual(len(prompts),count)
            self.assertFalse(previous&prompts);previous|=prompts
            styles.append({r['style'] for r in rows})
            for r in rows:
                self.assertTrue(grade(r,r['answer'],'end')['passed'],r['id'])
                self.assertEqual(len([x for x in rows if x['group']==r['group']]),4)
        for i in range(3):
            for j in range(i):self.assertFalse(styles[i]&styles[j])

    def test_wrong_role_negation_empty_echo_truncation_do_not_pass(self):
        for r in records('dev'):
            for text,stop in [('', 'end'),(r['prompt'],'end'),(r['answer'],'length')]:
                self.assertFalse(grade(r,text,stop)['passed'],r['id'])
            for foil in r['foils']:self.assertFalse(grade(r,foil,'end')['passed'],r['id'])
        row=next(r for r in records('dev') if r['family']=='ownership')
        self.assertFalse(grade(row,' '.join([row['answer']]*8),'end')['passed'])
        self.assertTrue(grade(row,' '.join([row['answer']]*8),'end')['repeated'])
        rows=json.loads((launch.DIR/'challenge.json').read_text())
        self.assertEqual(len(rows),28)
        for row in rows:
            for ref in row['references']:self.assertTrue(grade(row,ref,'end')['passed'])

    def test_rehearsal_only_consumed_examples_and_no_copy_family(self):
        old={r['id']:r for r in prior_training()[:512]};rr=rehearsal();self.assertEqual(len(rr),256)
        for r in rr:
            source=old[r['id'].removeprefix('retained-')]
            self.assertEqual((r['prompt'],r['answer']),(source['prompt'],source['answer']))
            self.assertNotIn(r['family'],('copy_or_reply','replace','extract'))

    def test_article_split_is_deterministic_and_question_independent(self):
        for name in ('Ancient Rome','Music','England','Cats'):
            self.assertEqual(split_for(name),split_for(name.upper()))
        self.assertEqual({split_for(str(i)) for i in range(100)}, {'train','dev','test'})

    def test_human_source_filter_and_span_validation(self):
        tok=SimpleNamespace(encode=lambda s:list(s.encode()))
        context='The library was founded by Marianne and opened in the spring. It was built close to the river and the old town square.'
        raw=dict(id='fixture',title='Library',context=context,question='Who founded the local library?',answers=dict(text=['Marianne'],answer_start=[context.index('Marianne')]))
        self.assertIsNotNone(candidate(raw,tok,set(),set())[0])
        for question in ('Write python code for this library.','Calculate the percentage of visitors.','How many visitors came today?'):
            self.assertIsNone(candidate(dict(raw,question=question),tok,set(),set())[0])
        self.assertEqual(candidate(raw,tok,{normalized(raw['question'])},set())[1],'evaluation_overlap')
        self.assertEqual(candidate(raw,tok,set(),ngrams(context))[1],'evaluation_overlap')
        self.assertEqual(candidate(dict(raw,answers=dict(text=['Someone else'],answer_start=[0])),tok,set(),set())[1],'invalid_span')
        row,_=candidate(dict(raw,answers=dict(text=[],answer_start=[])),tok,set(),set())
        self.assertTrue(row['unknown']);self.assertEqual(row['answer'],'Not stated')

    def test_selection_requires_transfer_and_retention(self):
        cfg=read_json(launch.DIR/'config.json');parent=launch.parent_metrics(metrics())
        improved=metrics(1);self.assertTrue(eligible(improved,parent,cfg))
        self.assertFalse(eligible(dict(improved,prose_change_from_parent=.09),parent,cfg))
        bad=json.loads(json.dumps(improved));bad['legacy']['both_correct']=.5
        self.assertFalse(eligible(bad,parent,cfg))
        for key,value in [('natural_f1',.1),('challenge_correct',0),('task_macro',.1),('repeat_rate',.02),('cap_rate',.02)]:
            bad=json.loads(json.dumps(improved));bad['new_tasks'][key]=value
            self.assertFalse(eligible(bad,parent,cfg),key)

    def test_schedule_fixed_rate_and_end(self):
        cfg=read_json(launch.DIR/'config.json')
        for u in (1,32,128,512):self.assertEqual(launch.learning_rate(u,cfg),3e-6)
        with self.assertRaises(ValueError):launch.learning_rate(513,cfg)

    def test_prompt_mask_eos_and_padding(self):
        backend=toy_load('opensml');row=dict(prompt='Where?',answer='Here.')
        encoded=engine.encode(backend,row);head=backend.encode(engine.prefix(row['prompt']))
        self.assertTrue(all(x==-100 for x in encoded['y'][:len(head)-1]))
        self.assertEqual(encoded['y'][len(head)-1:],backend.encode(' '+row['answer'])+[backend.eos])
        short=engine.encode(backend,dict(prompt='A',answer='B'));x,y=engine.arrays([encoded,short])
        self.assertTrue(mx.all(y[1,len(short['y']):]==-100).item())


class ResumeTests(unittest.TestCase):
    def setup_source(self,root):
        mx.random.seed(313);b=toy_load('opensml');opt=MasterAdamW(learning_rate=3e-6,betas=(.9,.95),weight_decay=.01)
        train=legacy('train')[:8];x,y=engine.arrays([engine.encode(b,r) for r in train[:2]])
        for _ in range(2):
            _,g=engine.gradients(b.model,x,y,False,2);opt.update(b.model,g);mx.eval(b.model.parameters(),opt.state)
        path=save_bundle(root/'source',b.model,opt,dict(step=2),None,best=True,reserve_gib=0)
        cfg=read_json(launch.DIR/'config.json')
        cfg.update(source_bundle=path,source_step=2,additional_updates=4,batch=2,microbatch=2,warmup_updates=1,
                   checkpoint_every=1,evaluation_updates=[0,2,4])
        return cfg,dict(train=train),opt

    def test_optimizer_exact_restore(self):
        with tempfile.TemporaryDirectory() as d,patch.object(engine,'load',side_effect=toy_load):
            cfg,data,old=self.setup_source(Path(d));_,new=launch.restore_run(Path(cfg['source_bundle'])/'model.safetensors',cfg)
            a,b=dict(tree_flatten(old.state)),dict(tree_flatten(new.state));self.assertEqual(set(a),set(b))
            for key in a:
                if key!='learning_rate':self.assertTrue(mx.array_equal(a[key],b[key]).item(),key)
            self.assertEqual(launch.optimizer_step(new),2)

    def test_interrupted_equals_uninterrupted_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as d,patch.object(engine,'load',side_effect=toy_load),contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);cfg,data,_=self.setup_source(root)
            def assess(b,c,d,u,p):return metrics(u)
            with patch.object(launch,'assess',side_effect=assess),patch.object(launch,'assess_new',return_value={}):
                stop=dict(requested=False);original=engine.gradients;calls=[]
                def interrupt(*args,**kwargs):
                    out=original(*args,**kwargs);calls.append(1)
                    if len(calls)==2:stop['requested']=True
                    return out
                with patch.object(launch,'OUTPUT',root/'resumed'),patch.object(engine,'gradients',side_effect=interrupt):
                    launch.run(cfg,{},data,stop)
                    self.assertEqual(read_json(root/'resumed/report.json')['additional_updates'],2)
                with patch.object(launch,'OUTPUT',root/'resumed'):
                    launch.run(cfg,{},data,dict(requested=False));first=read_json(root/'resumed/report.json')
                    launch.run(cfg,{},data,dict(requested=False));self.assertEqual(first,read_json(root/'resumed/report.json'))
                with patch.object(launch,'OUTPUT',root/'continuous'):launch.run(cfg,{},data,dict(requested=False))
                second=read_json(root/'continuous/report.json');self.assertEqual(first['training'],second['training'])
                self.assertEqual(first['additional_updates'],4);self.assertEqual(first['best']['step'],6)
                for suffix in ('','.optimizer.safetensors'):
                    a=mx.load(resolve_bundle(root/'resumed/latest.json')+suffix);b=mx.load(resolve_bundle(root/'continuous/latest.json')+suffix)
                    for k in a:self.assertTrue(mx.array_equal(a[k],b[k]).item(),k)
                self.assertEqual([r['additional_updates'] for r in first['evaluations']],[0,2,4])

    def test_parent_remains_best_if_no_candidate_eligible(self):
        with tempfile.TemporaryDirectory() as d,patch.object(engine,'load',side_effect=toy_load),contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);cfg,data,_=self.setup_source(root)
            with patch.object(launch,'OUTPUT',root/'out'),patch.object(launch,'assess',return_value=metrics()),patch.object(launch,'assess_new',return_value={}):
                launch.run(cfg,{},data,dict(requested=False))
            report=read_json(root/'out/report.json');self.assertEqual(report['best']['step'],2)
            fresh=read_json(root/'out/fresh_test.json')
            self.assertEqual(fresh['candidates']['parent']['step'],2)
            self.assertEqual(fresh['candidates']['selected']['step'],2)
            self.assertEqual(fresh['candidates']['final']['step'],6)
            self.assertNotEqual(fresh['candidates']['parent']['model_sha256'],fresh['candidates']['final']['model_sha256'])
            self.assertFalse(report['improved_on_development'])
            source=mx.load(str(Path(cfg['source_bundle'])/'model.safetensors'));best=mx.load(resolve_bundle(root/'out/best.json'))
            for key in source:self.assertTrue(mx.array_equal(source[key],best[key]).item())


if __name__=='__main__':unittest.main()
