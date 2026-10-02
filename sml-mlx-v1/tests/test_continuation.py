import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sml_v1.common import ROOT, file_sha256, fingerprint, read_json
from sml_v1.continuation import PILOT_CODE, PRE_GATHER_CODE, check_resume, extend_recipe, reviewed_upgrade
from sml_v1.recipe import learning_rate, validate


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.recipe = read_json(ROOT / 'configs/pilot.json')
        self.code = fingerprint({p.name: file_sha256(p) for p in (ROOT / 'sml_v1').glob('*.py')})
        self.metadata = dict(recipe=self.recipe, step=4696, tokens=500_105_216,
            replica_hash='source-replica', v2_contract=dict(recipe=fingerprint(self.recipe),
            code=PILOT_CODE, data='unchanged-data', tokenizer='unchanged-tokenizer'))

    def contract(self, recipe):
        return dict(self.metadata['v2_contract'], recipe=fingerprint(recipe), code=self.code)

    def test_reviewed_upgrade_is_exactly_pinned(self):
        self.assertTrue(reviewed_upgrade(PILOT_CODE, self.code))
        self.assertFalse(reviewed_upgrade('unknown-source', self.code))
        self.assertFalse(reviewed_upgrade(PILOT_CODE, 'unknown-destination'))
        self.assertTrue(reviewed_upgrade(PRE_GATHER_CODE, self.code))

    def test_active_full_run_can_resume_control_upgrade_without_recipe_changes(self):
        extended = extend_recipe(self.metadata, 15_000_000_000)
        saved = dict(self.metadata, recipe=extended, tokens=900_000_000,
                     v2_contract=dict(self.contract(extended), code=PRE_GATHER_CODE))
        self.assertEqual(check_resume(saved, self.contract(extended), extended),
                         'reviewed-control-code-upgrade')
        changed = copy.deepcopy(extended)
        changed['peak_lr'] = .0001
        with self.assertRaisesRegex(ValueError, 'Resume rejected'):
            check_resume(saved, self.contract(changed), changed)

    def test_modified_upgrade_module_is_rejected(self):
        from unittest.mock import patch
        from sml_v1 import continuation
        original = continuation.file_sha256
        def changed(path):
            return 'changed' if path.name == 'jaccl_control.py' else original(path)
        with patch.object(continuation, 'file_sha256', side_effect=changed):
            hashes = {p.name: changed(p) for p in (ROOT / 'sml_v1').glob('*.py')}
            self.assertFalse(reviewed_upgrade(PRE_GATHER_CODE, fingerprint(hashes)))

    def test_only_explicit_settings_change(self):
        extended = extend_recipe(self.metadata, 15_000_000_000)
        self.assertEqual(self.metadata['recipe'], self.recipe)
        for key in self.recipe:
            if key not in ('name', 'target_tokens'):
                self.assertEqual(extended[key], self.recipe[key], key)
        self.assertEqual(extended['continuation']['parent_contract'], self.metadata['v2_contract'])
        self.assertEqual(check_resume(self.metadata, self.contract(extended), extended),
                         'explicit-token-budget-extension')

    def test_schedule_continuity_ramp_and_final_decay(self):
        extended = extend_recipe(self.metadata, 15_000_000_000)
        anchor = self.metadata['tokens']
        self.assertAlmostEqual(learning_rate(anchor, extended), learning_rate(anchor, self.recipe))
        self.assertAlmostEqual(learning_rate(anchor + 25_000_000, extended), .000165)
        self.assertAlmostEqual(learning_rate(anchor + 50_000_000, extended), .0003)
        self.assertAlmostEqual(learning_rate(1_000_000_000, extended), .0003)
        self.assertAlmostEqual(learning_rate(12_000_000_000, extended), .0003)
        self.assertAlmostEqual(learning_rate(13_500_000_000, extended), .000165)
        self.assertAlmostEqual(learning_rate(15_000_000_000, extended), .00003)
        self.assertAlmostEqual(learning_rate(15_000_100_000, extended), .00003)

    def test_partial_pilot_uses_its_actual_lr(self):
        self.metadata['tokens'] = 450_000_000
        extended = extend_recipe(self.metadata, 15_000_000_000)
        self.assertAlmostEqual(learning_rate(450_000_000, extended), .000165)

    def test_resuming_full_run_does_not_rewarm_again(self):
        extended = extend_recipe(self.metadata, 15_000_000_000)
        saved = dict(self.metadata, recipe=extended, tokens=900_000_000, step=8452,
                     v2_contract=self.contract(extended))
        resumed = extend_recipe(saved, 15_000_000_000, rewarm_tokens=1)
        self.assertEqual(resumed, extended)
        self.assertEqual(check_resume(saved, self.contract(resumed), resumed), 'unchanged')
        with self.assertRaisesRegex(ValueError, 'different target'):
            extend_recipe(saved, 20_000_000_000)

    def test_reject_silent_recipe_changes(self):
        extended = extend_recipe(self.metadata, 15_000_000_000)
        changes = [('peak_lr', .0001), ('batch', 1), ('optimizer', .8), ('source', 8),
                   ('anchor', 1), ('replica', 'other'), ('target', 14_000_000_000)]
        for kind, value in changes:
            with self.subTest(kind=kind):
                changed = copy.deepcopy(extended)
                if kind == 'batch':
                    changed['batches'][0] = value
                elif kind == 'optimizer':
                    changed['betas'][0] = value
                elif kind == 'source':
                    changed['stream']['dedup_window'] = value
                elif kind == 'anchor':
                    changed['continuation']['start_tokens'] = value
                elif kind == 'replica':
                    changed['continuation']['parent_replica_hash'] = value
                elif kind == 'target':
                    changed['target_tokens'] = value
                    # An existing full-run resume may not silently change budget.
                    saved = dict(self.metadata, recipe=extended, v2_contract=self.contract(extended))
                    with self.assertRaisesRegex(ValueError, 'Resume rejected'):
                        check_resume(saved, self.contract(changed), changed)
                    continue
                else:
                    changed[kind] = value
                with self.assertRaisesRegex(ValueError, 'Resume rejected'):
                    check_resume(self.metadata, self.contract(changed), changed)

    def test_reject_data_tokenizer_and_unknown_code_changes(self):
        extended = extend_recipe(self.metadata, 15_000_000_000)
        for key in ('data', 'tokenizer', 'code'):
            contract = self.contract(extended)
            contract[key] = 'changed'
            with self.assertRaisesRegex(ValueError, 'Resume rejected'):
                check_resume(self.metadata, contract, extended)

    def test_budget_and_ramp_validation(self):
        for target in (0, 500_000_000, 450_000_000, True):
            with self.assertRaises(ValueError):
                extend_recipe(self.metadata, target)
        for ramp in (0, -1, 20_000_000_000):
            with self.assertRaises(ValueError):
                extend_recipe(self.metadata, 15_000_000_000, ramp)
        recipe = extend_recipe(self.metadata, 15_000_000_000)
        recipe['continuation']['start_lr'] = float('nan')
        with self.assertRaises(ValueError):
            validate(recipe)

    def test_pilot_schedule_unchanged(self):
        self.assertAlmostEqual(learning_rate(0, self.recipe), 0)
        self.assertAlmostEqual(learning_rate(5_000_000, self.recipe), .0003)
        self.assertAlmostEqual(learning_rate(400_000_000, self.recipe), .0003)
        self.assertAlmostEqual(learning_rate(450_000_000, self.recipe), .000165)
        self.assertAlmostEqual(learning_rate(500_000_000, self.recipe), .00003)
        self.assertEqual(check_resume(self.metadata, self.contract(self.recipe), self.recipe),
                         'reviewed-pilot-code-upgrade')


if __name__ == '__main__':
    unittest.main()
