from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


class AdaptiveLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sml-v2-adaptive-cli-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / 'runs/full_15b_v1'
        self.output = self.root / 'runs/full_15b_plateau_v1'
        self.parent.mkdir(parents=True)
        (self.parent / 'latest.json').write_text('{}')

    def invoke(self, args):
        calls = []
        def launch():
            calls.append(list(sys.argv))
            return 0
        with patch('launch_pretrain.cli', side_effect=launch), patch('sml_v2.common.ROOT', self.root), \
             patch.object(sys, 'argv', ['launch_adaptive', *args]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(ROOT / 'scripts/launch_adaptive.py'), run_name='__main__')
        return stopped.exception.code, calls

    def test_default_is_plan_only_from_latest_and_separate_output(self):
        status, calls = self.invoke([])
        self.assertEqual(status, 0)
        self.assertNotIn('--run', calls[0])
        self.assertIn(str(self.parent / 'latest.json'), calls[0])
        self.assertIn(str(self.output), calls[0])
        self.assertIn('--plateau-config', calls[0])
        self.assertNotIn('--extend-target-tokens', calls[0])
        self.assertEqual(calls[0][-2:], ['--ring-control', 'native'])

    def test_repeated_command_resumes_branch_not_original(self):
        self.output.mkdir(parents=True)
        (self.output / 'latest.json').write_text('{}')
        status, calls = self.invoke(['--run', '--clear-stop'])
        self.assertEqual(status, 0)
        self.assertIn(str(self.output / 'latest.json'), calls[0])
        self.assertNotIn(str(self.parent / 'latest.json'), calls[0])
        self.assertIn('--run', calls[0])
        self.assertIn('--clear-stop', calls[0])

    def test_no_checkpoint_does_not_start_fresh(self):
        (self.parent / 'latest.json').unlink()
        status, calls = self.invoke(['--run'])
        self.assertEqual(status, 2)
        self.assertFalse(calls)

    def test_transport_workaround_can_still_be_selected(self):
        status, calls = self.invoke(['--ring-control', 'all-sum'])
        self.assertEqual(status, 0)
        self.assertEqual(calls[0][-2:], ['--ring-control', 'all-sum'])


if __name__ == '__main__':
    unittest.main()
