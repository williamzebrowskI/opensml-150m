"""Explicit LR changes, immutable parent state, and bounded restart equivalence."""
import copy
import gzip
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from tests.test_five_mac_training import saved, contract
from tests import test_pipeline as fixtures
from sml_v2.common import atomic_json, fingerprint, read_json, file_sha256
from sml_v2.continuation import PRE_LR_TRIAL_CODE, check_resume
from sml_v2.data_review import reviewed_recipe
from sml_v2.lr_scheduler import DEFAULTS, adaptive_recipe, restore_scheduler
from sml_v2.lr_trial import KEY, end_step, trial_recipe
from sml_v2.recipe import learning_rate, tokens_per_update, validate
from sml_v2.topology_transition import migrate_recipe
import launch_five_mac as launch


def parent():
    m = saved()
    r = migrate_recipe(m, [4,3,2,2,2])
    r['plateau_scheduler']['min_lr'] = 5e-6
    state = dict(m['lr_scheduler'], cap=5e-6, last_loss=m['best_val_loss'],
                 config_fingerprint=fingerprint(r['plateau_scheduler']), cooldown_remaining=0)
    m.update(recipe=r, lr_scheduler=state, v2_contract=contract(m,r))
    r = reviewed_recipe(m)
    m.update(recipe=r, v2_contract=dict(contract(m,r),code=PRE_LR_TRIAL_CODE),
             validation_fingerprint='expanded-heldout')
    return m


class LRTrialTests(unittest.TestCase):
    def test_exact_ramp_and_control_preserve_all_other_state(self):
        m = parent(); before = copy.deepcopy(m)
        r = trial_recipe(m)
        self.assertEqual(m, before)
        self.assertEqual({k:v for k,v in r.items() if k != KEY}, m['recipe'])
        self.assertEqual(check_resume(m,contract(m,r),r), 'explicit-lr-trial-transition')
        s = restore_scheduler(r,m)
        self.assertEqual(s.state_dict(), dict(m['lr_scheduler'],cap=1e-5))
        u = tokens_per_update(r)
        for step, rate in [(0,5e-6),(1,5.05e-6),(50,7.5e-6),(100,1e-5),(999,1e-5)]:
            self.assertAlmostEqual(s.rate(learning_rate(m['tokens']+step*u,r)),rate,places=14)
        c = trial_recipe(m,'control')
        sc = restore_scheduler(c,m)
        self.assertEqual(sc.state_dict(),m['lr_scheduler'])
        for step in [1,100,1000]:
            self.assertEqual(sc.rate(learning_rate(m['tokens']+step*u,c)),5e-6)
        self.assertEqual(end_step(r),m['step']+1000)

    def test_mid_ramp_resume_and_later_reduction_are_not_replayed(self):
        m=parent();r=trial_recipe(m);s=restore_scheduler(r,m);u=tokens_per_update(r)
        current=dict(m,recipe=r,v2_contract=contract(m,r),step=m['step']+50,
                     tokens=m['tokens']+50*u,lr_scheduler=s.state_dict())
        self.assertEqual(trial_recipe(current),r)
        again=restore_scheduler(r,current)
        self.assertEqual(again.state_dict(),s.state_dict())
        self.assertAlmostEqual(again.rate(learning_rate(current['tokens']+u,r)),7.55e-6)
        for n in range(8):
            s.observe(3.2,step=m['step']+101+n,tokens=m['tokens']+(101+n)*u)
        current.update(step=m['step']+108,tokens=m['tokens']+108*u,lr_scheduler=s.state_dict())
        resumed=restore_scheduler(trial_recipe(current),current)
        self.assertEqual(resumed.state_dict(),s.state_dict())
        self.assertEqual(resumed.rate(learning_rate(current['tokens']+u,r)),5e-6)
        for changes in [dict(target_lr=2e-5),dict(trial_steps=2000),dict(warmup_steps=200),dict(arm='control')]:
            with self.assertRaises(ValueError): trial_recipe(current,**changes)

    def test_tampering_and_unevaluated_source_rejected(self):
        m=parent();r=trial_recipe(m)
        for kind in ['lr','batch','shuffle','parent','scheduler','validation','data','tokenizer','code']:
            bad=copy.deepcopy(r)
            if kind=='lr':bad['peak_lr']*=2
            if kind=='batch':bad['batches']=[3,4,2,2,2]
            if kind=='shuffle':bad['stream']['shuffle_buffer']=128
            if kind=='parent':bad[KEY]['parent_replica_hash']='wrong'
            if kind=='scheduler':bad[KEY]['parent_scheduler']['best']+=.1
            if kind=='validation':bad[KEY]['validation_fingerprint']='other'
            c=contract(m,bad)
            if kind in ['data','tokenizer','code']:c[kind]='other'
            with self.subTest(kind=kind),self.assertRaises(ValueError):check_resume(m,c,bad)
        with self.assertRaisesRegex(ValueError,'evaluated best'):
            trial_recipe(dict(m,step=m['step']+1,tokens=m['tokens']+tokens_per_update(r)))
        for value in [0,5e-6,float('nan'),float('inf'),True,.001]:
            with self.subTest(rate=value),self.assertRaises(ValueError):trial_recipe(m,target_lr=value)
        for updates in [0,-1,True]:
            with self.assertRaises(ValueError):trial_recipe(m,trial_steps=updates)
        bad=copy.deepcopy(r);bad['grad_accum']+=1
        with self.assertRaises(ValueError):validate(bad)
        self.assertEqual(launch.derive(m,[4,3,2,2,2])[3].state_dict(),m['lr_scheduler'])

    def test_plan_is_read_only_and_uses_fixed_parent(self):
        with patch('sys.argv',['launch','--lr-trial','rewarm']), \
                patch('pathlib.Path.exists',return_value=False), \
                patch.object(launch,'verified_metadata',return_value=parent()) as read, \
                patch('sys.stdout',new_callable=io.StringIO) as out:
            launch.main()
        read.assert_called_once_with(launch.TRIAL_PARENT)
        plan,_=json.JSONDecoder().raw_decode(out.getvalue())
        self.assertAlmostEqual(plan['next_learning_rate'],5.05e-6)
        self.assertIn('"remaining_updates": 1000',out.getvalue())
        self.assertIn('No model loaded',out.getvalue())
        with patch('sys.argv',['launch','--lr-trial','rewarm','--data-review']), \
                patch('sys.stderr',new_callable=io.StringIO),self.assertRaises(SystemExit):launch.main()
        with patch('sys.argv',['launch','--lr-trial','rewarm','--resume',str(launch.DATA_REVIEW_OUTPUT/'best.json'),
                               '--output',str(launch.DATA_REVIEW_OUTPUT)]), \
                patch.object(launch,'verified_metadata',return_value=parent()), \
                patch('sys.stderr',new_callable=io.StringIO),self.assertRaises(SystemExit):launch.main()
        m=parent();r=trial_recipe(m);s=restore_scheduler(r,m)
        m.update(recipe=r,v2_contract=contract(m,r),step=end_step(r),
                 tokens=m['tokens']+1000*tokens_per_update(r),lr_scheduler=s.state_dict())
        with patch('sys.argv',['launch','--lr-trial','rewarm']), \
                patch.object(launch,'verified_metadata',return_value=m), \
                patch('sys.stderr',new_callable=io.StringIO),self.assertRaises(SystemExit):launch.main()


class TinyLRTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.PipelineTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls):fixtures.PipelineTests.tearDownClass.__func__(cls)

    def test_worker_resume_is_exact_and_budget_forces_final_evaluation(self):
        import mlx.core as mx
        from sml_v2.pretrain import main
        from sml_v2.checkpoint_bundle import resolve_bundle
        mx.set_default_device(mx.cpu)
        cfg=read_json(ROOT/'configs/pilot.json')
        cfg['model'].update(vocab_size=300,d_model=16,n_heads=4,n_kv_heads=2,n_layers=1,
            mlp_multiple_of=8,max_seq_len=128,attention_impl='vanilla',ce_impl='reference',ffn_impl='reference')
        cfg.update(batches=[1],grad_accum=1,target_tokens=4096,reserve_gib=0,
            eval_batches_per_source=1,eval_batch_size=1,prompts=[],keep_checkpoints=10,
            eval_every_tokens=256,sample_every_tokens=1000000)
        corpus=self.root/'trial-corpus.json';atomic_json(corpus,self.config)
        def run(recipe,name,steps=0,resume=None,output=None):
            config=self.root/(name+'.json');atomic_json(config,recipe)
            output=output or self.root/name
            argv=['worker','--config',str(config),'--tokenizer',str(self.output),'--save-dir',str(output),
                  '--job-dir',str(self.root/(name+'-job')),'--corpus',str(corpus),'--stop-after-steps',str(steps)]
            if resume:argv+=['--resume',str(resume)]
            with patch.object(sys,'argv',argv),patch('sys.stdout',new_callable=io.StringIO) as log:main()
            return output,log.getvalue()
        def meta(path):return read_json(resolve_bundle(path)+'.json')
        base,_=run(cfg,'trial-base',2)
        adaptive=adaptive_recipe(meta(base/'latest.json'),dict(DEFAULTS,initial_lr=1e-5,min_lr=5e-6,patience=1,cooldown=0,min_delta=100.))
        old,_=run(adaptive,'trial-adaptive',2,base/'latest.json')
        review=reviewed_recipe(meta(old/'latest.json'),shuffle_buffer=8,eval_batches=4)
        reviewed,_=run(review,'trial-reviewed',1,old/'latest.json')
        source=reviewed/'best.json';m=meta(source)
        self.assertEqual(m['lr_scheduler']['cap'],5e-6)
        r=trial_recipe(m,warmup_steps=2,trial_steps=3)
        source_hash=file_sha256(resolve_bundle(source))
        full,log=run(r,'trial-full',resume=source)
        split,_=run(r,'trial-split',1,source)
        _,again=run(r,'trial-resumed',resume=split/'latest.json',output=split)
        a,b=resolve_bundle(full/'latest.json'),resolve_bundle(split/'latest.json')
        ma,mb=read_json(a+'.json'),read_json(b+'.json')
        self.assertEqual(ma,mb)
        self.assertEqual(ma['step'],m['step']+3)
        self.assertEqual(read_json(full/'last_eval.json')['step'],ma['step'])
        self.assertNotIn('[eval-baseline ',again)
        initial=next(p for p in full.glob('step_*') if read_json(p/'model.safetensors.json')['step']==m['step'])
        self.assertEqual(read_json(initial/'model.safetensors.json')['replica_hash'],m['replica_hash'])
        self.assertEqual(file_sha256(resolve_bundle(source)),source_hash)
        for suffix in ['', '.optimizer.safetensors']:
            left,right=mx.load(a+suffix),mx.load(b+suffix)
            for key in left:self.assertTrue(mx.array_equal(left[key],right[key]).item(),key)
        with gzip.open(a+'.rank0.data_state.json.gz','rt') as x,gzip.open(b+'.rank0.data_state.json.gz','rt') as y:
            self.assertEqual(json.load(x),json.load(y))


if __name__=='__main__':unittest.main()
