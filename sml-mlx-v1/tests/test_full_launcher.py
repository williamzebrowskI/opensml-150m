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


class FullLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sml-v2-full-cli-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pilot = self.root / 'runs/pilot_v1'
        self.full = self.root / 'runs/full_15b_v1'
        self.pilot.mkdir(parents=True)
        (self.pilot / 'latest.json').write_text('{}')
        (self.pilot / 'best.json').write_text('{}')

    def invoke(self, args):
        calls = []
        def launch():
            calls.append(list(sys.argv))
            return 0
        with patch('launch_pretrain.cli', side_effect=launch), patch('sml_v1.common.ROOT', self.root), \
             patch.object(sys, 'argv', ['launch_full', *args]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(ROOT / 'scripts/launch_full.py'), run_name='__main__')
        return stopped.exception.code, calls

    def test_latest_pilot_by_default_is_plan_only(self):
        status, calls = self.invoke([])
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn(str(self.pilot / 'latest.json'), calls[0])
        self.assertIn('15000000000', calls[0])
        self.assertNotIn('--run', calls[0])
        self.assertEqual(calls[0][-2:], ['--ring-control', 'all-sum'])

    def test_native_control_forwarded_explicitly(self):
        status, calls = self.invoke(['--ring-control', 'native'])
        self.assertEqual(status, 0)
        self.assertEqual(calls[0][-2:], ['--ring-control', 'native'])
        self.assertNotIn('--run', calls[0])

    def test_best_pilot_explicit_run(self):
        status, calls = self.invoke(['--run', '--from', 'best', '--clear-stop'])
        self.assertEqual(status, 0)
        self.assertIn(str(self.pilot / 'best.json'), calls[0])
        self.assertIn('--run', calls[0])
        self.assertIn('--clear-stop', calls[0])

    def test_existing_full_run_resumes_itself(self):
        self.full.mkdir(parents=True)
        (self.full / 'latest.json').write_text('{}')
        status, calls = self.invoke(['--from', 'latest'])
        self.assertEqual(status, 0)
        self.assertIn(str(self.full / 'latest.json'), calls[0])
        self.assertNotIn(str(self.pilot / 'latest.json'), calls[0])

    def test_missing_full_best_never_falls_back_to_pilot(self):
        self.full.mkdir(parents=True)
        (self.full / 'latest.json').write_text('{}')
        status, calls = self.invoke(['--run', '--from', 'best'])
        self.assertEqual(status, 2)
        self.assertEqual(calls, [])

    def test_no_checkpoint_never_starts_fresh(self):
        (self.pilot / 'latest.json').unlink()
        status, calls = self.invoke(['--run'])
        self.assertEqual(status, 2)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
