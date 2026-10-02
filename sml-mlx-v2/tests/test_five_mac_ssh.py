import base64
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import inspect_five_mac_ssh as inspect
from setup_five_mac_ssh import INSTALL_KEY, main


class FiveMacSshTests(unittest.TestCase):
    def test_plan_never_runs_commands(self):
        with patch('sys.argv', ['setup']), patch('sys.stdout', new_callable=io.StringIO), \
                patch('setup_five_mac_ssh.run') as run, patch('subprocess.run') as sub:
            self.assertEqual(main(), 0)
            run.assert_not_called()
            sub.assert_not_called()

    def test_installer_preserves_existing_keys_and_is_idempotent(self):
        public = 'ssh-ed25519 AAAATEST opensml-five-mac-m5\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '.ssh').mkdir()
            target = root / '.ssh/authorized_keys'
            target.write_text('existing-key-without-final-newline')
            for _ in range(2):
                with patch('pathlib.Path.home', return_value=root), \
                        patch('sys.stdin', io.StringIO(public)), patch('sys.stdout', new_callable=io.StringIO):
                    exec(INSTALL_KEY, {})
            self.assertEqual(target.read_text(), 'existing-key-without-final-newline\n' + public)

    def test_installer_refuses_symlink_and_wrong_key_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'other').mkdir()
            (root / '.ssh').symlink_to(root / 'other', target_is_directory=True)
            for public in ['ssh-ed25519 AAAA opensml-five-mac-m5', 'ssh-rsa AAAA opensml-five-mac-m5']:
                with self.subTest(public=public), patch('pathlib.Path.home', return_value=root), \
                        patch('sys.stdin', io.StringIO(public)), self.assertRaises(SystemExit):
                    exec(INSTALL_KEY, {})

    def test_unpinned_host_key_is_never_written(self):
        route = dict(returncode=0, stdout='interface: en0', stderr='')
        scan = dict(returncode=0, stdout='192.0.2.1 ssh-ed25519 QUJD\n', stderr='')
        with tempfile.TemporaryDirectory() as tmp, patch.object(inspect, 'run', side_effect=[route, scan]):
            result, known = inspect.pinned_host((2, '192.0.2.1', 'SHA256:wrong'), Path(tmp))
            self.assertIsNone(known)
            self.assertIn('differs', result['error'])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_valid_pin_and_ethernet_route_required(self):
        ip = '192.0.2.1'
        key = base64.b64encode(b'test-public-key').decode()
        digest = 'SHA256:' + base64.b64encode(hashlib.sha256(b'test-public-key').digest()).decode().rstrip('=')
        for iface in ['en0', 'en1']:
            route = dict(returncode=0, stdout=f'interface: {iface}', stderr='')
            scan = dict(returncode=0, stdout=f'{ip} ssh-ed25519 {key}\n', stderr='')
            with tempfile.TemporaryDirectory() as tmp, patch.object(inspect, 'run', side_effect=[route, scan]):
                result, known = inspect.pinned_host((2, ip, digest), Path(tmp))
                self.assertEqual(known is not None, iface == 'en0')


if __name__ == '__main__':
    unittest.main()
