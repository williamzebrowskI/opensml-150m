import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from five_mac_topology import IDENTITIES, discover


def fixture(order=None):
    order = list(range(5)) if order is None else order
    reports = []
    for rank, (chip, mac, key) in enumerate(IDENTITIES):
        buses = []
        logical = order.index(rank)
        for side, peer_rank, peer_side in [('left', order[(logical - 1) % 5], 'right'),
                                           ('right', order[(logical + 1) % 5], 'left')]:
            buses.append(dict(domain_uuid_key=f'{rank}-{side}',
                receptacle_1_tag=dict(receptacle_id_key='1' if side == 'left' else '2'),
                _items=[dict(domain_uuid_key=f'{peer_rank}-{peer_side}')]))
        outputs = dict(chip=chip, hardware_ports=f'Hardware Port: Ethernet\nDevice: en0\nEthernet Address: {mac}\n'
            'Hardware Port: Thunderbolt 1\nDevice: en13\nHardware Port: Thunderbolt 2\nDevice: en2\n',
            ssh_host_fingerprint=f'256 {key} (ED25519)', rdma='enabled',
            ibv_devices='rdma_en13 guid\nrdma_en2 guid\n', thunderbolt=json.dumps(dict(SPThunderboltDataType=buses)))
        reports.append(dict(format='opensml-readonly-mac-inventory-v1', process_query_ok=True,
            training_processes=[], probes={k: dict(returncode=0, stdout=v) for k, v in outputs.items()}))
    return reports


class FiveMacTopologyTests(unittest.TestCase):
    def test_explicit_discovered_order_keeps_physical_identities(self):
        order = [0,1,2,4,3]
        reports = fixture(order)
        result = discover(reports, order)
        self.assertEqual(result['physical_order'], order)
        self.assertEqual(result['rdma'][0], [None,'rdma_en2',None,None,'rdma_en13'])
        self.assertIn((0,'en13',3,'en2'), result['edges'])
        with self.assertRaises(ValueError):
            discover(reports)

    def test_five_node_map_uses_discovered_interfaces(self):
        result = discover(fixture())
        self.assertEqual(len(result['edges']), 5)
        self.assertEqual(result['rdma'][0], [None, 'rdma_en2', None, None, 'rdma_en13'])
        self.assertFalse(result['readiness_verified'])

    def test_rejects_wrong_identity_idle_rdma_and_failed_probes(self):
        for probe, value in [('chip', 'Apple M4 Max'), ('ssh_host_fingerprint', '256 wrong'),
                             ('rdma', 'disabled'), ('ibv_devices', '')]:
            reports = fixture()
            reports[0]['probes'][probe]['stdout'] = value
            with self.subTest(probe=probe), self.assertRaises(ValueError):
                discover(reports)
        for key, value in [('process_query_ok', False), ('training_processes', ['worker'])]:
            reports = fixture()
            reports[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                discover(reports)
        reports = fixture()
        reports[3]['probes']['thunderbolt']['returncode'] = 1
        with self.assertRaises(ValueError):
            discover(reports)

    def test_missing_nonreciprocal_and_duplicate_domains_rejected(self):
        for change in ('missing', 'nonreciprocal', 'duplicate'):
            reports = fixture()
            value = json.loads(reports[0]['probes']['thunderbolt']['stdout'])
            bus = value['SPThunderboltDataType'][0]
            if change == 'missing':
                bus['_items'] = []
            elif change == 'nonreciprocal':
                bus['_items'][0]['domain_uuid_key'] = '2-right'
            else:
                bus['domain_uuid_key'] = '1-left'
            reports[0]['probes']['thunderbolt']['stdout'] = json.dumps(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                discover(reports)

    def test_four_reports_or_reordered_physical_machines_rejected(self):
        reports = fixture()
        with self.assertRaises(ValueError):
            discover(reports[:4])
        reports[3], reports[4] = reports[4], reports[3]
        with self.assertRaises(ValueError):
            discover(reports)


if __name__ == '__main__':
    unittest.main()
