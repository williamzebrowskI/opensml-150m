import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sml_v2.common import ROOT, file_sha256, fingerprint, read_json
from sml_v2.continuation import PRE_PLATEAU_CODE, check_resume, extend_recipe, reviewed_upgrade
from sml_v2.lr_scheduler import DEFAULTS, PlateauScheduler, adaptive_recipe, restore_scheduler, validate_config
from sml_v2.recipe import learning_rate


class SchedulerTests(unittest.TestCase):
    def controller(self, **changes):
        return PlateauScheduler(dict(DEFAULTS, **changes), best=3.03, step=100, tokens=1000)

    def observe(self, scheduler, loss=3.04):
        return scheduler.observe(loss, step=scheduler.state['last_step'] + 1,
                                 tokens=scheduler.state['last_tokens'] + 100)

    def test_defaults_match_launch_configuration(self):
        self.assertEqual(read_json(ROOT / 'configs/plateau.json'), DEFAULTS)

    def test_exact_patience_after_initial_settling(self):
        c = self.controller()
        for _ in range(2):
            self.assertEqual(self.observe(c), 'cooldown')
        for _ in range(7):
            self.assertEqual(self.observe(c), 'monitoring')
            self.assertEqual(c.rate(.0003), .0002)
        self.assertEqual(self.observe(c), 'reduced')
        self.assertEqual(c.rate(.0003), .0001)
        self.assertEqual(c.state['reductions'], 1)
        self.assertEqual(c.state['cooldown_remaining'], 2)

    def test_significant_not_noisy_improvement_resets_patience(self):
        c = self.controller(cooldown=0, patience=3)
        self.assertEqual(self.observe(c, 3.029), 'monitoring')
        self.assertEqual(self.observe(c, 3.0285), 'monitoring')
        self.assertEqual(self.observe(c, 3.027), 'improved')
        self.assertEqual(c.state['bad_evals'], 0)
        self.assertEqual(c.state['best'], 3.027)

    def test_improvement_during_cooldown_updates_monitor_best(self):
        c = self.controller()
        self.assertEqual(self.observe(c, 3.02), 'cooldown')
        self.assertEqual(c.state['best'], 3.02)
        self.assertEqual(c.state['bad_evals'], 0)

    def test_floor_and_schedule_never_raise_lr(self):
        c = self.controller(patience=1, cooldown=0)
        rates = []
        for _ in range(10):
            self.observe(c)
            rates.append(c.rate(.0003))
        self.assertEqual(rates[:3], [.0001, .00005, .00003])
        self.assertEqual(c.state['reductions'], 3)
        self.assertEqual(c.rate(.00002), .00002)
        self.assertEqual(c.rate(0), 0)
        self.assertEqual(self.observe(c), 'at-floor')

    def test_json_resume_is_identical_to_uninterrupted(self):
        full = self.controller()
        resumed = self.controller()
        losses = [3.04] * 12 + [3.026, 3.03, 3.028] + [3.04] * 30
        for loss in losses:
            a, b = self.observe(full, loss), self.observe(resumed, loss)
            self.assertEqual(a, b)
            state = json.loads(json.dumps(resumed.state_dict()))
            resumed = PlateauScheduler(DEFAULTS, best=99, step=state['last_step'],
                                       tokens=state['last_tokens'], state=state)
            self.assertEqual(full.state_dict(), resumed.state_dict())
            self.assertEqual(full.rate(.0003), resumed.rate(.0003))

    def test_nonfinite_and_duplicate_observations_do_not_advance(self):
        c = self.controller()
        self.observe(c)
        before = c.state_dict()
        for loss in (float('nan'), float('inf'), -1, True):
            with self.assertRaises(ValueError):
                self.observe(c, loss)
            self.assertEqual(c.state_dict(), before)
        with self.assertRaises(ValueError):
            c.observe(3.03, step=before['last_step'], tokens=before['last_tokens'])
        self.assertEqual(c.state_dict(), before)

    def test_invalid_config_and_state_rejected(self):
        for key, value in [('patience', 0), ('patience', True), ('cooldown', -1),
                           ('factor', 1), ('initial_lr', float('nan')),
                           ('min_delta', -1), ('min_lr', .001)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config(dict(DEFAULTS, **{key: value}), .0003)
        c = self.controller()
        for key, value in [('cap', float('inf')), ('cap', 0), ('bad_evals', 8),
                           ('cooldown_remaining', 3), ('last_step', 1000),
                           ('last_tokens', 10000), ('config_fingerprint', 'wrong')]:
            state = dict(c.state_dict(), **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                PlateauScheduler(DEFAULTS, best=3.03, step=100, tokens=1000, state=state)


class TransitionTests(unittest.TestCase):
    def setUp(self):
        initial = read_json(ROOT / 'configs/pilot.json')
        base = dict(recipe=initial, step=4696, tokens=500105216, replica_hash='parent',
                    v2_contract=dict(recipe=fingerprint(initial), data='data', tokenizer='tokenizer', code=PRE_PLATEAU_CODE))
        self.recipe = extend_recipe(base, 15_000_000_000)
        self.metadata = dict(base, recipe=self.recipe, step=38056, tokens=4052811776,
                             best_val_loss=3.0306, ema=2.98,
                             v2_contract=dict(base['v2_contract'], recipe=fingerprint(self.recipe)))
        self.code = fingerprint({p.name: file_sha256(p) for p in (ROOT / 'sml_v2').glob('*.py')})

    def contract(self, recipe):
        return dict(self.metadata['v2_contract'], recipe=fingerprint(recipe), code=self.code)

    def test_explicit_transition_preserves_every_original_setting(self):
        new = adaptive_recipe(self.metadata, DEFAULTS)
        for key, value in self.recipe.items():
            self.assertEqual(new[key], value)
        self.assertEqual(new['scheduler_transition']['parent_contract'], self.metadata['v2_contract'])
        self.assertEqual(check_resume(self.metadata, self.contract(new), new), 'explicit-validation-scheduler-transition')
        c = restore_scheduler(new, self.metadata)
        self.assertEqual(c.rate(learning_rate(self.metadata['tokens'] + 106496, new)), .0002)
        self.assertEqual(c.state['best'], 3.0306)
        self.assertEqual(c.state['bad_evals'], 0)

    def test_reviewed_code_upgrade_without_opt_in_keeps_old_lr(self):
        self.assertTrue(reviewed_upgrade(PRE_PLATEAU_CODE, self.code))
        self.assertEqual(check_resume(self.metadata, self.contract(self.recipe), self.recipe),
                         'reviewed-scheduler-code-upgrade')
        self.assertIsNone(restore_scheduler(self.recipe, self.metadata))
        self.assertEqual(learning_rate(self.metadata['tokens'], self.recipe), .0003)

    def test_changed_data_model_and_transition_provenance_are_rejected(self):
        new = adaptive_recipe(self.metadata, DEFAULTS)
        for key in ('model', 'batches', 'target_tokens', 'scheduler_transition'):
            bad = copy.deepcopy(new)
            if key == 'model':
                bad[key]['d_model'] = 1024
            elif key == 'batches':
                bad[key][0] += 1
            elif key == 'target_tokens':
                bad[key] += 1
            else:
                bad[key]['parent_replica_hash'] = 'different'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'unrelated'):
                check_resume(self.metadata, self.contract(bad), bad)
        with self.assertRaisesRegex(ValueError, 'data or tokenizer'):
            check_resume(self.metadata, dict(self.contract(new), data='changed'), new)

    def test_resuming_does_not_reset_counters_or_rewarm(self):
        new = adaptive_recipe(self.metadata, DEFAULTS)
        c = restore_scheduler(new, self.metadata)
        for n in range(10):
            c.observe(3.04, step=38057+n, tokens=4052811776+(n+1)*106496)
        saved = dict(self.metadata, recipe=new, v2_contract=self.contract(new),
                     step=38066, tokens=4052811776+10*106496, lr_scheduler=c.state_dict())
        again = adaptive_recipe(saved, DEFAULTS)
        self.assertEqual(new, again)
        restored = restore_scheduler(again, saved)
        self.assertEqual(c.state_dict(), restored.state_dict())
        self.assertEqual(restored.rate(.0003), .0001)
        self.assertEqual(check_resume(saved, self.contract(again), again), 'unchanged')
        self.assertAlmostEqual(restored.rate(learning_rate(15_000_000_000, again)), .00003)
        with self.assertRaisesRegex(ValueError, 'refusing to reset'):
            adaptive_recipe(saved, dict(DEFAULTS, patience=7))
        with self.assertRaisesRegex(ValueError, 'missing required'):
            restore_scheduler(again, dict(saved, lr_scheduler=None))

    def test_no_fresh_training_or_automatic_increase(self):
        new = adaptive_recipe(self.metadata, DEFAULTS)
        with self.assertRaises(ValueError):
            restore_scheduler(new, None)
        late = dict(self.metadata, tokens=14_900_000_000)
        with self.assertRaisesRegex(ValueError, 'must not raise'):
            adaptive_recipe(late, DEFAULTS)


if __name__ == '__main__':
    unittest.main()
