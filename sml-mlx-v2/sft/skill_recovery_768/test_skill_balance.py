"""Loss direction, frozen anchor, formatting, and exact interrupted resume."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from sml_v2.common import read_json
from sft.skill_recovery_768 import engine
from sft.skill_recovery_768.data import formats,unpack,partition,finite_sentence
from sft.skill_recovery_768.launch import run
from sft.skill_recovery_768.evaluate import eligible


class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(128,8); self.output=nn.Linear(8,128,bias=False)
    def __call__(self,x): return {'logits':self.output(self.embedding(x))}


def toy_load(cfg,weights=None):
    mx.random.seed(71); model=Tiny(); mx.eval(model.parameters())
    encode=lambda s:list(s.encode('ascii'))
    return SimpleNamespace(model=model,encode=encode,eos=1,pad=0,name='opensml',logits=lambda x:model(x)['logits'])


def row(i,answer='oak'):
    return dict(id=str(i),prompt='Which tree?',answer=answer,ranking_prompt='Tree:',choices=['oak','pine'],gold=0)


def cfg_small():
    cfg=read_json(Path(__file__).with_name('config.json'))
    cfg.update(updates=2,warmup_updates=1,evaluation_updates=[0,2],checkpoint_every=1,
               per_update=dict(commonsense=2,instruction=2,reading=1))
    return cfg


def tiny_data(cfg):
    return dict(train={f:[row(i,'oak' if i%2 else 'a tall oak') for i in range(n*cfg['updates'])]
                       for f,n in cfg['per_update'].items()},anchors=[[10,11,12,13]])


def snapshot(tree):
    mx.eval(tree)
    return {k:v.tolist() for k,v in tree_flatten(tree)}


class Tests(unittest.TestCase):
    def test_grounded_grade_requires_preserved_content_and_format(self):
        from .evaluate import grounded_grade
        r=dict(format='json',answer='{"answer": "Ada has the basket."}')
        self.assertTrue(grounded_grade(r,r['answer'],'end'))
        self.assertFalse(grounded_grade(r,'{"answer": "Ben has the basket."}','end'))
        self.assertFalse(grounded_grade(r,'Ada has the basket.','end'))
        self.assertFalse(grounded_grade(r,r['answer'],'length'))

    def test_qa_assistant_targets_are_not_used_and_every_qa_is_ranked(self):
        cfg=cfg_small();data=tiny_data(cfg);rows=engine.batch_at(data,cfg,0)
        b=toy_load(cfg);anchor=toy_load(cfg);opt=engine.optimizer(cfg)
        other=toy_load(cfg);otheropt=engine.optimizer(cfg)
        changed=copy.deepcopy(rows)
        for r in changed['commonsense']:r['answer']='THIS IS NOT A TRAINING TARGET'
        with patch.object(engine,'choice_arrays',wraps=engine.choice_arrays) as call:
            engine.update(b,anchor,opt,rows,data['anchors'][0],cfg,1)
            self.assertEqual(call.call_count,len(rows['commonsense']))
        engine.update(other,anchor,otheropt,changed,data['anchors'][0],cfg,1)
        for (_,a),(_,c) in zip(tree_flatten(b.model.parameters()),tree_flatten(other.model.parameters())):
            self.assertLess(float(mx.max(mx.abs(a-c)).item()),1e-7)

    def test_format_views_do_not_use_followup_or_other_split_rows(self):
        from .data import format_views
        rows=[dict(id='a',group='a',split='train',turn=0,answer='Ada has the basket.'),
              dict(id='b',group='a',split='train',turn=1,answer='Ben')]
        views=format_views(rows)
        self.assertEqual(len(views),3)
        self.assertTrue(all(r['group']=='a' and r['split']=='train' for r in views))
        self.assertTrue(all(r['base_answer']=='Ada has the basket.' for r in views))

    def test_format_roundtrips_and_rejects_invalid_outputs(self):
        for kind in ('plain','lower','upper','quote','json','bullet','lower_bullet','lower_json'):
            _,text=formats('A child holds a bowl.',kind)
            self.assertIsNotNone(unpack(text,kind))
        self.assertIsNone(unpack('{"answer":"oak","extra":1}','json'))
        self.assertIsNone(unpack('- oak\n- pine','bullet'))
        self.assertIsNone(unpack('OAK','lower'))
        self.assertIsNone(unpack('plain','quote'))

    def test_sentence_filter_rejects_caption_fragments(self):
        self.assertFalse(finite_sentence('A granite counter top in a hotel bathroom.'))
        self.assertFalse(finite_sentence('A silhouette of mother and daughter behind a bicycle.'))
        self.assertTrue(finite_sentence('A child uses a spatula to spread sauce on the dough.'))
        self.assertTrue(finite_sentence('A CHILD USES A SPATULA TO SPREAD SAUCE ON THE DOUGH.'))

    def test_group_split_is_stable(self):
        self.assertEqual(partition('same-concepts'),partition('same-concepts'))
        self.assertEqual({partition(str(i)) for i in range(200)},{'train','dev','test','excluded'})

    def test_historical_training_groups_never_enter_evaluation(self):
        from sft.skill_balance.data import partition as old
        from sft.base_curriculum_v2.data import partition as base
        for i in range(1000):
            group=str(i);split=partition(group)
            if split!='excluded':
                self.assertEqual(old(group),split)
                self.assertEqual(base(group),split)

    def test_prompt_eos_and_padding_masks(self):
        b=toy_load({}); enc=engine.encode(b,row(1),1024); head=b.encode(engine.prefix(row(1)['prompt']))
        self.assertEqual(enc['y'][:len(head)-1],[-100]*(len(head)-1))
        self.assertEqual(enc['y'][-1],b.eos)
        self.assertEqual([y for y in enc['y'] if y!=-100],b.encode(' oak')+[b.eos])
        e2=engine.encode(b,row(2,'a tall oak'),1024)
        x,y=engine.arrays([enc,e2],b.pad)
        self.assertTrue(bool(mx.all(y[0,len(enc['y']):]==-100).item()))
        batch=float(engine.objective(b.model,x,y,False).item()); singles=[]
        for e in (enc,e2):
            xx,yy=engine.arrays([e],b.pad); singles.append(float(engine.objective(b.model,xx,yy,False).item()))
        self.assertAlmostEqual(batch,sum(singles)/2,places=5)

    def test_choice_loss_pushes_correct_option_up(self):
        b=toy_load({}); x,y=engine.choice_arrays(b,row(1))
        temperature=cfg_small()['ranking_temperature']
        f=nn.value_and_grad(b.model,lambda a,c:engine.ranking_loss(b.model,a,c,0,temperature))
        before,g=f(x,y)
        import mlx.optimizers as optim
        opt=optim.SGD(learning_rate=.001); opt.update(b.model,g); mx.eval(b.model.parameters())
        after=engine.ranking_loss(b.model,x,y,0,temperature)
        self.assertLess(float(after.item()),float(before.item()))
        self.assertFalse(bool(mx.any(y==b.eos).item()))

    def test_zero_kl_at_identity_and_anchor_never_updated(self):
        cfg=cfg_small(); b=toy_load(cfg); anchor=toy_load(cfg); opt=engine.optimizer(cfg)
        before=snapshot(anchor.model.parameters()); x=mx.array([[10,11,12]])
        logits=anchor.model(x)['logits']; ref=logits-mx.logsumexp(logits,axis=-1,keepdims=True)
        self.assertAlmostEqual(float(engine.reference_kl(b.model,x,ref).item()),0.,places=6)
        data=tiny_data(cfg)
        metrics=engine.update(b,anchor,opt,engine.batch_at(data,cfg,0),data['anchors'][0],cfg,1)
        self.assertEqual(snapshot(anchor.model.parameters()),before)
        self.assertNotEqual(snapshot(b.model.parameters()),before)
        self.assertTrue(all(abs(v)<1e10 for v in metrics.values()))

    def test_interrupted_run_matches_uninterrupted_and_completed_does_not_repeat(self):
        cfg=cfg_small(); data=tiny_data(cfg); frozen={'test':True}; stop={'requested':False}
        m=dict(prose_nll=2.,reading_exact=.5,reading_f1=.6,stopped=1.,commonsense_macro=.5,
               commonsense_by_source=dict(a=.5,b=.5),instruction_proxy=.5)
        def assess(*args,**kwargs): return dict(metrics=copy.deepcopy(m))
        original=engine.update
        def interrupt(*args,**kwargs):
            result=original(*args,**kwargs); stop['requested']=True; return result
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a'; c=Path(folder)/'c'
            with patch('sft.skill_recovery_768.engine.load',toy_load),patch('sft.skill_recovery_768.evaluate.assess',assess),patch('sft.skill_recovery_768.launch.verify'):
                run(cfg,frozen,data,dict(requested=False),output=a)
                with patch('sft.skill_recovery_768.engine.update',interrupt): run(cfg,frozen,data,stop,output=c)
                (c/'STOP').unlink(missing_ok=True)
                run(cfg,frozen,data,dict(requested=False),output=c)
                def files(root):
                    p=root/read_json(root/'latest.json')['bundle']
                    return [mx.load(str(p/name)) for name in ('model.safetensors','model.safetensors.optimizer.safetensors')]
                for expected,actual in zip(files(a),files(c)):
                    self.assertEqual(expected.keys(),actual.keys())
                    gap=max(float(mx.max(mx.abs(expected[k]-actual[k])).item()) for k in expected)
                    self.assertLess(gap,1e-7)  # MLX accumulation can differ by FP32 roundoff.
                before=read_json(c/'latest.json')
                with patch('sft.skill_recovery_768.engine.update',side_effect=AssertionError('Repeated training')):
                    run(cfg,frozen,data,dict(requested=False),output=c)
                self.assertEqual(read_json(c/'latest.json'),before)

    def test_candidate_cannot_trade_large_reading_loss_for_proxy(self):
        cfg=cfg_small()
        base=dict(prose_nll=2.,reading_exact=.6,reading_f1=.7,stopped=1.,commonsense_by_source=dict(a=.5,b=.5))
        self.assertTrue(eligible(base,base,cfg))
        bad=dict(base,reading_exact=.5)
        self.assertFalse(eligible(bad,base,cfg))


if __name__=='__main__': unittest.main()
