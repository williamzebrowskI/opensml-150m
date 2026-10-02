"""Lower-floor continuation preserves training state and cannot replay the cut."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from tests.test_five_mac_training import saved, contract
from sml_v1.common import fingerprint
from sml_v1.continuation import PRE_LR_FLOOR_CODE, check_resume
from sml_v1.lr_scheduler import lower_floor_recipe, restore_scheduler
from sml_v1.recipe import learning_rate, validate
from sml_v1.topology_transition import migrate_recipe
import launch_five_mac as launch


class FloorTransitionTests(unittest.TestCase):
    def setUp(self):
        self.meta = saved()
        recipe = migrate_recipe(self.meta, [4, 3, 2, 2, 2])
        self.meta.update(recipe=recipe, v2_contract=contract(self.meta, recipe))
        scheduler = restore_scheduler(recipe, self.meta)
        for _ in range(32):
            scheduler.observe(3.1, step=scheduler.state['last_step'] + 1,
                              tokens=scheduler.state['last_tokens'] + 106496)
        self.meta.update(step=scheduler.state['last_step'], tokens=scheduler.state['last_tokens'],
                         lr_scheduler=scheduler.state_dict())
        self.meta['v2_contract']['code'] = PRE_LR_FLOOR_CODE
        self.assertEqual(scheduler.state['cap'], 3e-5)
        self.assertEqual(scheduler.state['cooldown_remaining'], 0)

    def test_one_explicit_cut_preserves_recipe_and_history(self):
        before = copy.deepcopy(self.meta)
        new = lower_floor_recipe(self.meta, 5e-6)
        self.assertEqual(self.meta, before)
        self.assertEqual(new['plateau_scheduler'], dict(before['recipe']['plateau_scheduler'], min_lr=5e-6))
        for key, value in before['recipe'].items():
            if key != 'plateau_scheduler':
                self.assertEqual(new[key], value, key)
        self.assertEqual(check_resume(self.meta, contract(self.meta, new), new), 'explicit-lr-floor-transition')
        scheduler = restore_scheduler(new, self.meta)
        self.assertEqual(scheduler.rate(learning_rate(self.meta['tokens'] + 106496, new)), 1.5e-5)
        for key in ('best', 'evaluations', 'last_step', 'last_tokens', 'last_loss'):
            self.assertEqual(scheduler.state[key], before['lr_scheduler'][key])
        self.assertEqual(scheduler.state['reductions'], before['lr_scheduler']['reductions'] + 1)
        self.assertEqual(scheduler.state['cooldown_remaining'], 2)

    def test_saved_resume_never_replays_reduction_or_cooldown(self):
        new = lower_floor_recipe(self.meta, 5e-6)
        scheduler = restore_scheduler(new, self.meta)
        rates = []
        for _ in range(30):
            # Simulate a save/restart after every evaluation, including at the floor.
            scheduler.observe(3.1, step=scheduler.state['last_step'] + 1,
                              tokens=scheduler.state['last_tokens'] + 106496)
            current = dict(self.meta, recipe=new, v2_contract=contract(self.meta, new),
                           step=scheduler.state['last_step'], tokens=scheduler.state['last_tokens'],
                           lr_scheduler=json.loads(json.dumps(scheduler.state_dict())))
            again = lower_floor_recipe(current, 5e-6)
            restored = restore_scheduler(again, current)
            self.assertEqual(restored.state_dict(), scheduler.state_dict())
            self.assertEqual(check_resume(current, current['v2_contract'], again), 'unchanged')
            rates.append(restored.rate(.0003))
            scheduler = restored
        self.assertIn(7.5e-6, rates)
        self.assertEqual(rates[-1], 5e-6)
        self.assertEqual(rates, sorted(rates, reverse=True))
        with self.assertRaisesRegex(ValueError, 'Saved LR floor differs'):
            lower_floor_recipe(current, 1e-6)

    def test_without_opt_in_keeps_old_rate(self):
        recipe, _, transition, scheduler = launch.derive(self.meta, [4, 3, 2, 2, 2])
        self.assertEqual(transition, 'reviewed-lr-floor-code-upgrade')
        self.assertEqual(recipe, self.meta['recipe'])
        self.assertEqual(scheduler.state_dict(), self.meta['lr_scheduler'])

    def test_invalid_floor_or_unsettled_source_rejected(self):
        for value in (0, -1, True, float('nan'), float('inf'), 3e-5, 4e-5):
            with self.subTest(floor=value), self.assertRaises(ValueError):
                lower_floor_recipe(self.meta, value)
        for changes in ({'cap': 5e-5}, {'cooldown_remaining': 1}):
            bad = copy.deepcopy(self.meta)
            bad['lr_scheduler'].update(changes)
            with self.assertRaisesRegex(ValueError, 'settling'):
                lower_floor_recipe(bad, 5e-6)
        with self.assertRaisesRegex(ValueError, 'pretraining'):
            lower_floor_recipe(dict(self.meta, sft=True), 5e-6)

    def test_unrelated_settings_and_provenance_cannot_change(self):
        new = lower_floor_recipe(self.meta, 5e-6)
        for kind in ('model', 'data', 'tokenizer', 'code', 'batches', 'patience', 'parent', 'state', 'cap'):
            bad = copy.deepcopy(new)
            if kind == 'model': bad['model']['d_model'] += 64
            if kind == 'batches': bad['batches'] = [3, 4, 2, 2, 2]
            if kind == 'patience': bad['plateau_scheduler']['patience'] += 1
            if kind == 'parent': bad['lr_floor_transition']['parent_replica_hash'] = 'wrong'
            if kind == 'state': bad['lr_floor_transition']['parent_scheduler']['best'] += 1
            if kind == 'cap': bad['lr_floor_transition']['resume_cap'] = 1e-5
            proposed = contract(self.meta, bad)
            if kind in ('data', 'tokenizer', 'code'): proposed[kind] = 'different'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                check_resume(self.meta, proposed, bad)
        with self.assertRaisesRegex(ValueError, 'incompatible scheduler'):
            restore_scheduler(dict(new, lr_floor_transition=new['lr_floor_transition']),
                              dict(self.meta, lr_scheduler=dict(self.meta['lr_scheduler'], config_fingerprint='wrong')))

    def test_floor_change_without_provenance_rejected(self):
        new = copy.deepcopy(self.meta['recipe'])
        new['plateau_scheduler']['min_lr'] = 5e-6
        with self.assertRaises(ValueError):
            check_resume(self.meta, contract(self.meta, new), new)
        with self.assertRaises(ValueError):
            restore_scheduler(new, self.meta)
        valid = lower_floor_recipe(self.meta, 5e-6)
        valid['lr_floor_transition']['parent_scheduler'] = None
        with self.assertRaisesRegex(ValueError, 'parent scheduler history'):
            validate(valid)

    def test_repeating_flag_selects_own_latest_and_plan_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, output = root / 'parent', root / 'output'
            parent.mkdir(); output.mkdir()
            (parent / 'latest.json').write_text('source')
            self.assertEqual(launch.select_checkpoint(output, parent=parent), parent / 'latest.json')
            (output / 'latest.json').write_text('resumed')
            self.assertEqual(launch.select_checkpoint(output, parent=parent), output / 'latest.json')
        with patch('sys.argv', ['launch', '--lr-floor', '5e-6']), \
                patch('sys.stdout', new_callable=io.StringIO) as out, \
                patch.object(launch, 'select_checkpoint', return_value=launch.DEFAULT_OUTPUT/'latest.json') as select, \
                patch.object(launch, 'verified_metadata', return_value=self.meta):
            launch.main()
        select.assert_called_once_with(launch.LOW_LR_OUTPUT.resolve(), None, launch.DEFAULT_OUTPUT)
        self.assertIn('No model loaded', out.getvalue())
        self.assertIn('"next_learning_rate": 1.5e-05', out.getvalue())
        self.assertIn('"lr_floor": 5e-06', out.getvalue())

    def test_new_floor_cannot_overwrite_parent_run(self):
        with patch('sys.argv', ['launch', '--lr-floor', '5e-6', '--output', str(launch.DEFAULT_OUTPUT)]), \
                patch('sys.stderr', new_callable=io.StringIO), \
                patch.object(launch, 'select_checkpoint', return_value=launch.DEFAULT_OUTPUT/'latest.json'), \
                patch.object(launch, 'verified_metadata', return_value=self.meta), self.assertRaises(SystemExit):
            launch.main()


if __name__ == '__main__':
    unittest.main()
