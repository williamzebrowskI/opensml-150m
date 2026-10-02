from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from five_mac_transport import command


class FiveMacTransportTests(unittest.TestCase):
    def setUp(self):
        self.env=dict(SML5_JOB='/tmp/five-job',SML5_SSH_CONFIG='/tmp/five-job/ssh_config',SML5_PORT='23456')

    def test_worker_tunnel_is_loopback_only(self):
        cmd=command(['-tt','opensml-five-mac-5','worker'],self.env)
        self.assertIn('127.0.0.1:23456:127.0.0.1:23456',cmd)
        self.assertIn('ExitOnForwardFailure=yes',cmd)

    def test_cleanup_does_not_create_coordinator_tunnel(self):
        cmd=command(['opensml-five-mac-4','cleanup'],self.env)
        self.assertNotIn('-R',cmd)

    def test_unknown_host_wrong_config_and_port_rejected(self):
        for args,env in [(['unverified-host'],self.env),
                (['opensml-five-mac-2'],dict(self.env,SML5_PORT='80')),
                (['opensml-five-mac-2'],dict(self.env,SML5_SSH_CONFIG='/tmp/unrelated'))]:
            with self.assertRaises(ValueError):command(args,env)


if __name__=='__main__':unittest.main()
