import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rename_five_macs import desired, main, rename_steps


class FiveMacNamesTests(unittest.TestCase):
    def test_names_follow_physical_labels(self):
        self.assertEqual(desired(4), dict(ComputerName='Mac-5', LocalHostName='mac-5', HostName='mac-5'))
        self.assertEqual(desired(0)['ComputerName'], 'Mac-1')

    def test_vacates_old_m3_collision_before_assigning_mac5(self):
        steps = rename_steps([{} for _ in range(5)])
        self.assertEqual(steps[0][0], 1)
        self.assertTrue(steps[0][1]['LocalHostName'].startswith('opensml-migration-'))
        self.assertEqual([s[0] for s in steps[1:]], [4,3,2,1,0])

    def test_repeating_completed_rename_is_noop(self):
        self.assertEqual(rename_steps([desired(i) for i in range(5)]), [])

    def test_default_plan_never_executes(self):
        with patch('sys.argv', ['rename']), patch('sys.stdout', new_callable=io.StringIO), \
                patch('rename_five_macs.run') as run, patch('subprocess.run') as sub:
            self.assertEqual(main(), 0)
            run.assert_not_called()
            sub.assert_not_called()


if __name__ == '__main__':
    unittest.main()
