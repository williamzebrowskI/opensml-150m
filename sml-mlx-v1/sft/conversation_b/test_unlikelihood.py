"""UL direction, negative labels, padding and no-repeat behavior."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from . import engine

class Toy(nn.Module):
    def __init__(self):super().__init__();self.scores=mx.array([0.,1.,2.,3.,4.,0.,0.,0.])
    def __call__(self,x):return {'logits':mx.broadcast_to(self.scores,(*x.shape,8))}

class UnlikelihoodTests(unittest.TestCase):
    def test_later_ngram_only(self):
        self.assertEqual(engine.repetition_mask([10,11,12,13]*2),[False]*4+[True]*4)
        self.assertEqual(engine.repetition_mask([10,11,12,13]),[False]*4)
        self.assertFalse(any(engine.repetition_mask([0,1,2,3]*3)))
        self.assertEqual(engine.repetition_mask([10]*7),[False]+[True]*6)

    def test_alignment_and_prompt_exclusion(self):
        b=SimpleNamespace(encode=lambda _: [10,11,12,13],pad=0,eos=1)
        r=engine.negative_row(b,'irrelevant',[10,11,12,13],4)
        self.assertEqual(r['targets'],0) # prompt repeats are not negatives
        r=engine.negative_row(b,'irrelevant',[10,11,12,13]*2,4)
        self.assertEqual(r['x'],[10,11,12,13,10,11,12,13,10,11,12])
        self.assertEqual(r['y'],[-100]*7+[10,11,12,13])
        self.assertEqual(r['targets'],4)

    def test_loss_and_gradient_direction(self):
        m=Toy();x=mx.array([[2,3]]);y=mx.array([[-100,4]])
        fn=nn.value_and_grad(m,lambda a,b:engine.ul_objective(m,a,b,1e-6))
        loss,g=fn(x,y)
        p=float(mx.softmax(m.scores)[4].item())
        import math
        self.assertAlmostEqual(float(loss.item()),-math.log(1-p),places=5)
        self.assertGreater(float(g['scores'][4].item()),0)
        m.update({'scores':m.scores-.1*g['scores']})
        self.assertLess(float(mx.softmax(m.scores)[4].item()),p)

    def test_padding_and_microbatch_normalization(self):
        m=Toy();x=mx.array([[2,3],[4,0]]);y=mx.array([[4,-100],[-100,-100]])
        v=float(engine.ul_objective(m,x,y,1e-6).item())
        single=float(engine.ul_objective(m,x[:1,:1],y[:1,:1],1e-6).item())
        self.assertAlmostEqual(v,single/2,places=6)
        rows=[dict(x=[2],y=[4],targets=1),dict(x=[4],y=[-100],targets=0)]
        cfg={'unlikelihood':{'epsilon':1e-6}}
        loss,g=engine.ul_gradients(m,rows,cfg)
        self.assertAlmostEqual(float(loss.item()),v,places=6)
        _,expected=nn.value_and_grad(m,lambda a,b:engine.ul_objective(m,a,b,1e-6))(x,y)
        self.assertLess(float(mx.max(mx.abs(g['scores']-expected['scores'])).item()),1e-6)
        zero,none=engine.ul_gradients(m,[rows[1]],cfg)
        self.assertEqual(float(zero.item()),0);self.assertIsNone(none)

    def test_extreme_probability_is_finite(self):
        m=Toy();m.scores=mx.array([0.,0.,0.,0.,100.,0.,0.,0.])
        v,g=nn.value_and_grad(m,lambda:engine.ul_objective(m,mx.array([[2]]),mx.array([[4]]),1e-6))()
        self.assertTrue(bool(mx.isfinite(v).item()))
        self.assertTrue(all(bool(mx.all(mx.isfinite(t)).item()) for _,t in tree_flatten(g)))

    def test_rehearsal_and_answers_not_used_for_negatives(self):
        b=SimpleNamespace(encode=lambda _: [5],pad=0,eos=1)
        cfg={'context':1024,'unlikelihood':{'max_new_tokens':96,'ngram':4}}
        rows=[dict(id='train',source='original',prompt='P',answer='unused reference'),dict(id='old',source='rehearsal',prompt='R',answer='also unused')]
        with patch.object(engine,'generate',return_value=dict(text='repeat',token_ids=[10,11,12,13]*2,tokens=8,stop='length')) as mocked:
            e,r=engine.collect(b,rows,cfg)
        mocked.assert_called_once_with(b,'P',96)
        self.assertEqual(len(e),1);self.assertEqual(e[0]['targets'],4)
        self.assertEqual(r[0]['id'],'train')

def run_unit_checks():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(UnlikelihoodTests))
    if not result.wasSuccessful():raise ValueError('Unlikelihood tests failed')
    return result.testsRun
if __name__=='__main__':run_unit_checks()
