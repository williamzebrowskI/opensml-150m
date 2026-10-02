import copy
import json
from pathlib import Path
import sys
import unittest
import tempfile
import hashlib
import shutil
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from five_mac_network import interfaces, plan, route_networks, verify_after
from test_five_mac_topology import fixture


def network_fixture():
    reports = fixture([0,1,2,4,3])
    for r in reports:
        r['thunderbolt_interface_labels'] = {'Thunderbolt 1':'en13','Thunderbolt 2':'en2'}
        r['probes']['interfaces'] = dict(returncode=0, stdout='''en0: flags=1<UP>
 inet 192.168.12.10 netmask 0xffffff00
en13: flags=1<UP>
 inet 10.77.0.1 netmask 0xfffffffc
en2: flags=1<UP>
bridge0: flags=1<UP>
 member: en13 flags=3
 member: en2 flags=3
''')
        r['probes']['routes'] = dict(returncode=0, stdout='default 192.168.12.1 UG en0\n192.168.12 link#1 UCS en0\n')
        r['probes']['default_route'] = dict(returncode=0, stdout='gateway: 192.168.12.1\ninterface: en0\n')
    return reports


class FiveMacNetworkTests(unittest.TestCase):
    def configured_fixture(self):
        reports = network_fixture()
        node = plan(reports,[0,1,2,4,3])['nodes'][0]
        report = copy.deepcopy(reports[0])
        text = 'en0: flags=1<UP>\n inet 192.168.12.10 netmask 0xffffff00\n'
        for e in node['endpoints']:
            text += f'{e["interface"]}: flags=1<UP>\n inet {e["ip"]} netmask 0xfffffffc\n'
        report['probes']['interfaces']['stdout'] = text + 'bridge0: flags=0<BROADCAST>\n'
        return report, node

    def test_runtime_allows_only_link_local_alias_and_detached_tb_bridge(self):
        report, node = self.configured_fixture()
        text = report['probes']['interfaces']['stdout']
        text = text.replace('bridge0: flags=0<BROADCAST>',
                            ' inet 169.254.128.119 netmask 0xffff0000\nbridge0: flags=1<UP>\n member: en8 flags=3')
        report['probes']['interfaces']['stdout'] = text
        report['thunderbolt_interface_labels']['Thunderbolt 3'] = 'en8'
        verify_after(report, node, runtime=True)
        # Applying a network plan retains the original exact-state requirement.
        with self.assertRaises(ValueError):
            verify_after(report, node)

    def test_runtime_still_rejects_wrong_address_mask_down_and_bridged_training_port(self):
        for kind in ('missing', 'mask', 'down', 'extra', 'link_local_mask', 'bridged', 'ethernet_bridge'):
            report, node = self.configured_fixture()
            e = node['endpoints'][0]
            text = report['probes']['interfaces']['stdout']
            if kind == 'missing': text = text.replace(f' inet {e["ip"]} netmask 0xfffffffc\n', '')
            if kind == 'mask': text = text.replace(f'inet {e["ip"]} netmask 0xfffffffc', f'inet {e["ip"]} netmask 0xffffff00')
            if kind == 'down': text = text.replace(f'{e["interface"]}: flags=1<UP>', f'{e["interface"]}: flags=0<BROADCAST>')
            if kind == 'extra': text = text.replace('bridge0:', ' inet 10.77.0.1 netmask 0xfffffffc\nbridge0:')
            if kind == 'link_local_mask': text = text.replace('bridge0:', ' inet 169.254.1.1 netmask 0xff000000\nbridge0:')
            if kind == 'bridged': text += f' member: {e["interface"]} flags=3\n'
            if kind == 'ethernet_bridge': text = text.replace('bridge0: flags=0<BROADCAST>', 'bridge0: flags=1<UP>\n member: en0 flags=3')
            report['probes']['interfaces']['stdout'] = text
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                verify_after(report, node, runtime=True)

    def test_worker_checks_before_mutating_and_verifies_after(self):
        import configure_five_mac_network as worker
        reports = network_fixture()
        value = plan(reports,[0,1,2,4,3])
        after = copy.deepcopy(reports[0])
        node = value['nodes'][0]
        text = 'en0: flags=1<UP>\n inet 192.168.12.10 netmask 0xffffff00\n'
        for e in node['endpoints']:
            text += f'{e["interface"]}: flags=1<UP>\n inet {e["ip"]} netmask 0xfffffffc\n'
        text += 'bridge0: flags=0<BROADCAST>\n'
        after['probes']['interfaces']['stdout'] = text
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'tools').mkdir()
            hashes={}
            for name in worker.MODULES:
                source=Path(worker.__file__).with_name(name)
                shutil.copyfile(source,p/'tools'/name)
                hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
            for name,obj in [('tools_sha256',hashes),('inventories',reports),('plan',value)]:
                (p/(name+'.json')).write_text(json.dumps(obj))
            with patch.object(worker,'collect',side_effect=[reports[0],after]), \
                    patch.object(worker.os,'geteuid',return_value=0), patch.object(worker.subprocess,'run') as run:
                worker.local_worker(p,0,'apply')
                self.assertEqual([c.args[0] for c in run.call_args_list],node['commands'])
                self.assertTrue((p/'applied_mac1.json').exists())
            bad=copy.deepcopy(reports[0]);bad['process_query_ok']=False
            with patch.object(worker,'collect',return_value=bad), patch.object(worker.subprocess,'run') as run:
                with self.assertRaises(ValueError):worker.local_worker(p,0,'apply')
                run.assert_not_called()

    def test_serialized_plan_roundtrip_and_tampering(self):
        from configure_five_mac_network import read_plan
        reports = network_fixture()
        value = plan(reports,[0,1,2,4,3])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'inventories.json').write_text(json.dumps(reports))
            (p/'plan.json').write_text(json.dumps(value))
            self.assertEqual(read_plan(p)[1]['format'],value['format'])
            value['nodes'][0]['commands'].append(['/sbin/ifconfig','en0','down'])
            (p/'plan.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                read_plan(p)

    def test_isolated_subnets_and_rollback_restore_old_addresses(self):
        p = plan(network_fixture(), [0,1,2,4,3])
        addresses = [e['ip'] for n in p['nodes'] for e in n['endpoints']]
        self.assertEqual(len(set(addresses)), 10)
        self.assertTrue(all(len(n['endpoints']) == 2 for n in p['nodes']))
        for n in p['nodes']:
            self.assertIn(['/sbin/ifconfig','bridge0','down'], n['commands'])
            self.assertIn(['/sbin/ifconfig','en13','inet','10.77.0.1','netmask','255.255.255.252','alias'], n['rollback'])
            self.assertFalse(any('en0' in c or 'en1' in c for c in n['commands']+n['rollback']))

    def test_non_thunderbolt_bridge_member_rejected(self):
        r = network_fixture()
        r[0]['probes']['interfaces']['stdout'] += ' member: en0 flags=3\n'
        with self.assertRaisesRegex(ValueError,'non-Thunderbolt'):
            plan(r,[0,1,2,4,3])

    def test_overlap_and_unknown_port_address_rejected(self):
        for mode in ('route','address','unknown'):
            r = network_fixture()
            if mode == 'route':
                r[0]['probes']['routes']['stdout'] += '10.85 link#1 UCS en0\n'
            elif mode == 'address':
                r[0]['probes']['interfaces']['stdout'] += 'en8: flags=1<UP>\n inet 10.85.1.2 netmask 0xfffffffc\n'
            else:
                r[0]['probes']['interfaces']['stdout'] = r[0]['probes']['interfaces']['stdout'].replace('10.77.0.1','172.16.20.1')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                plan(r,[0,1,2,4,3])

    def test_hex_masks_and_abbreviated_routes(self):
        r = network_fixture()[0]
        self.assertEqual(interfaces(r['probes']['interfaces']['stdout'])['en13']['ipv4'],[['10.77.0.1','255.255.255.252']])
        self.assertEqual(str(route_networks('10.85 link#3 UCS en4\n')[0]), '10.85.0.0/16')


if __name__ == '__main__':
    unittest.main()
