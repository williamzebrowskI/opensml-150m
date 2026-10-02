from contextlib import redirect_stderr
import io
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LaunchCliTests(unittest.TestCase):
    def invoke(self, error=None):
        output = io.StringIO()
        with patch('sml_v1.launch.main', side_effect=error) as launch, redirect_stderr(output):
            with self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(ROOT / 'scripts/launch_pretrain.py'), run_name='__main__')
        launch.assert_called_once_with()
        return stopped.exception.code, output.getvalue()

    def test_success(self):
        self.assertEqual(self.invoke(), (0, ''))

    def test_busy_worker_is_shown_without_retrying(self):
        detail = 'RuntimeError: Other training/benchmark workers active: [1234 python -m sml_v1.pretrain]'
        status, output = self.invoke(subprocess.CalledProcessError(1, ['python', '-c', 'probe'], stderr=detail))
        self.assertEqual(status, 1)
        self.assertIn(detail, output)
        self.assertIn('idle safety check', output)
        self.assertIn('Wait for them to exit', output)
        self.assertIn('No automatic retry', output)

    def test_remote_failure_preserves_stdout_and_stderr(self):
        status, output = self.invoke(subprocess.CalledProcessError(255, ['ssh', 'mac-3'],
                                  output='Checking mac-3', stderr='ssh: Connection timed out'))
        self.assertEqual(status, 1)
        self.assertIn('status 255', output)
        self.assertIn('Checking mac-3', output)
        self.assertIn('ssh: Connection timed out', output)
        self.assertNotIn('idle safety check', output)

    def test_missing_diagnostics(self):
        status, output = self.invoke(subprocess.CalledProcessError(-15, ['python']))
        self.assertEqual(status, 1)
        self.assertIn('no diagnostic output', output)

    def test_unrelated_errors_are_not_suppressed(self):
        with patch('sml_v1.launch.main', side_effect=ValueError('Checkpoint mismatch')):
            with self.assertRaisesRegex(ValueError, 'Checkpoint mismatch'):
                runpy.run_path(str(ROOT / 'scripts/launch_pretrain.py'), run_name='__main__')


if __name__ == '__main__':
    unittest.main()
