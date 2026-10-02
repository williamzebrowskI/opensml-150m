#!/usr/bin/env python3
"""Opt-in mesh/ring experiments. Never rewires, changes routes, or resumes production."""

import argparse
import datetime
import fcntl
import hashlib
import json
import ipaddress
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import uuid

import jaccl_benchmark as bench

mesh = bench.mesh
RING_EDGES = (mesh.EDGES[0], mesh.EDGES[3], mesh.EDGES[5], mesh.EDGES[2])
BATCHES = [32, 24, 24, 24]
RING_MAP = bench.ROOT / 'cluster/jaccl/ring_benchmark_map.json'


def port_inventory(state):
    mapping = dict(re.findall(r'Hardware Port: ([^\n]+)\nDevice: (\S+)', state['hardware_ports']))
    return {mapping.get('Thunderbolt ' + str(bus.get('receptacle_1_tag', {}).get('receptacle_id_key'))):
            (bus['domain_uuid_key'], {p['domain_uuid_key'] for p in bus.get('_items', []) if 'domain_uuid_key' in p})
            for bus in state['thunderbolt']['SPThunderboltDataType'] if 'domain_uuid_key' in bus}


def discover_ring(states):
    ports = [port_inventory(s) for s in states]
    owners = {}
    for rank, nodes in enumerate(ports):
        for iface, (uuid, _) in nodes.items():
            if iface is not None:
                mesh.require(uuid not in owners, 'Duplicate Thunderbolt domain identity')
                owners[uuid] = (rank, iface)
    cables = []
    for rank, nodes in enumerate(ports):
        for iface, (uuid, peers) in nodes.items():
            for peer in peers:
                if peer not in owners:
                    continue  # A display is not an inter-Mac cable.
                other, remote = owners[peer]
                mesh.require(other != rank and ports[other][remote][1] == {uuid} and peers == {peer},
                             'Ambiguous or nonreciprocal Thunderbolt connection')
                mesh.require(iface in ('en3', 'en4', 'en5') and remote in ('en3', 'en4', 'en5'),
                             'Unpinned Thunderbolt port; verify hardware before using it')
                if rank < other:
                    cables.append([rank, iface, other, remote])
    adjacency = {r: set() for r in range(4)}
    for a, _, b, _ in cables:
        adjacency[a].add(b)
        adjacency[b].add(a)
    mesh.require(len(cables) == 4 and all(len(v) == 2 for v in adjacency.values()),
                 'Need exactly four reciprocal cables and two different neighbors per Mac')
    order = [0, min(adjacency[0])]
    while len(order) < 4:
        following = adjacency[order[-1]] - {order[-2]}
        node = next(iter(following))
        mesh.require(node not in order, 'Ring does not connect all four Macs')
        order.append(node)
    mesh.require(0 in adjacency[order[-1]], 'Ring is not closed')
    return dict(edges=sorted(cables), physical_order=order)


def read_ring_map():
    if not RING_MAP.exists():
        return None
    value = json.loads(RING_MAP.read_text())
    order = value['physical_order']
    mesh.require(sorted(order) == list(range(4)) and order[0] == 0, 'Invalid ring rank order')
    mesh.require(len(value['edges']) == 4, 'Invalid ring cable map')
    return value


def edges(topology, mapping=None):
    if topology not in ('mesh', 'ring'):
        raise ValueError('Unknown physical topology')
    return mesh.EDGES if topology == 'mesh' else (mapping['edges'] if mapping else RING_EDGES)


def hostfile(topology, mapping=None):
    order = mapping['physical_order'] if mapping else list(range(4))
    matrix = [[None] * 4 for _ in range(4)]
    for a, ai, b, bi in edges(topology, mapping):
        a, b = order.index(a), order.index(b)
        matrix[a][b], matrix[b][a] = 'rdma_' + ai, 'rdma_' + bi
    return dict(backend='jaccl-ring' if topology == 'ring' else 'jaccl', hosts=[
        dict(ssh='127.0.0.1' if rank == 0 else mesh.ssh_alias(order[rank]),
             ips=[mesh.HOSTS[0][2]] if rank == 0 else [], rdma=matrix[rank])
        for rank in range(4)])


def inspect_host(rank, mac4_via_mac2=False, control_transport='wifi'):
    mesh.require(control_transport in ('wifi', 'ring-thunderbolt'), 'Unsupported ring inventory transport')
    mesh.require(not mac4_via_mac2 or control_transport == 'wifi', 'Setup relay is Wi-Fi-only')
    # Send a self-contained read-only collector: no dependency on remote project scripts.
    source = (bench.ROOT / 'scripts/configure_rdma_mesh.py').read_text()
    common = (bench.ROOT / 'sml_v2/cluster/jaccl_common.py').read_text()
    destinations = tuple(h[2] for i, h in enumerate(mesh.HOSTS) if i != rank) if control_transport == 'wifi' else ()
    code = f'''import json, os, importlib.metadata as metadata
from pathlib import Path
m = {{"__name__": "ring_read_only"}}
c = {{"__name__": "ring_common"}}
exec(compile({source!r}, "mesh_read_only.py", "exec"), m)
exec(compile({common!r}, "common_read_only.py", "exec"), c)
run = m["run"]
launcher = metadata.distribution("mlx").locate_file("mlx/_distributed_utils/launch.py").read_text()
state = dict(rank={rank}, interfaces=m["interfaces"](run(["/sbin/ifconfig"]).stdout),
    management={{ip: m["route"](ip) for ip in {destinations!r}}},
    ssh_connection=os.environ.get("SSH_CONNECTION", ""),
    rdma=run(["/usr/bin/rdma_ctl", "status"]).stdout.strip(),
    thunderbolt=json.loads(run(["/usr/sbin/system_profiler", "SPThunderboltDataType", "-json", "-timeout", "30"], timeout=45).stdout),
    hardware_ports=run(["/usr/sbin/networksetup", "-listallhardwareports"]).stdout,
    packages={{p: metadata.version(p) for p in c["PACKAGES"]}},
    ring_supported="jaccl-ring" in launcher and "MLX_JACCL_RING=1" in launcher,
    launcher_sha256=c["sha256"](metadata.distribution("mlx").locate_file("mlx/_distributed_utils/launch.py")),
    busy=c["training_processes"]())
print(json.dumps(state))
'''
    if mac4_via_mac2 and rank == 3:
        from repair_ring_network import transport
        command = transport(rank, [bench.PYTHON, '-c', code], relay=True)
    else:
        command = mesh.transport(rank, [bench.PYTHON, '-c', code], control_transport=control_transport)
    result = mesh.run(command, timeout=90,
                      label=f'mac-{rank + 1} read-only ring inventory ' +
                            ('via mac-2 setup relay' if mac4_via_mac2 and rank == 3 else f'over {control_transport}'))
    return json.loads(result.stdout)


def validate_states(states, topology, require_idle=True, mapping=None, allow_bridges=False, require_wifi=True):
    mesh.require(len(states) == 4, 'Need all four inventories')
    ports = []
    for rank, state in enumerate(states):
        devs = state['interfaces']
        mesh.require(state['rank'] == rank and devs['en0']['mac'] == mesh.HOSTS[rank][3],
                     f'mac-{rank + 1}: physical identity mismatch; do not move cables by name alone')
        if require_wifi:
            mesh.require([mesh.HOSTS[rank][2], '255.255.255.0'] in devs['en1']['ipv4'],
                         f'mac-{rank + 1}: pinned Wi-Fi address changed')
            wanted = {h[2] for i, h in enumerate(mesh.HOSTS) if i != rank}
            mesh.require(set(state['management']) == wanted and all(
                route.get('interface') == 'en1' for route in state['management'].values()),
                'This test requires Wi-Fi management; no automatic fallback')
        mesh.require(state['rdma'] == 'enabled' and state['ring_supported'],
                     f'mac-{rank + 1}: RDMA or installed jaccl-ring support missing')
        mesh.require(state['packages'] == states[0]['packages'] and
                     state['launcher_sha256'] == states[0]['launcher_sha256'], 'MLX/dependency versions differ')
        if require_idle:
            mesh.require(not state['busy'], f'mac-{rank + 1}: stop production yourself before testing: {state["busy"]}')
        needed = {ai if rank == a else bi for a, ai, b, bi in edges(topology, mapping) if rank in (a, b)}
        for i, iface in enumerate(('en3', 'en4', 'en5')):
            dev = devs[iface]
            mesh.require(dev['mac'] == mesh.PORT_MACS[rank][i], 'Thunderbolt port identity changed')
            if iface in needed:
                mesh.require(dev['up'] and dev['active'], f'mac-{rank + 1}: {iface} must be connected')
            else:
                mesh.require(not dev['active'], f'mac-{rank + 1}: remove the diagonal cable on {iface} for a physical ring')
        for name, dev in devs.items():
            if set(dev['members']) & {'en3', 'en4', 'en5'}:
                mesh.require(allow_bridges or not dev['up'], f'mac-{rank + 1}: active bridge {name}; repair before testing')
        hardware_mapping = dict(re.findall(r'Hardware Port: ([^\n]+)\nDevice: (\S+)', state['hardware_ports']))
        nodes = {}
        for item in state['thunderbolt']['SPThunderboltDataType']:
            iface = hardware_mapping.get('Thunderbolt ' + str(item.get('receptacle_1_tag', {}).get('receptacle_id_key')))
            if iface in needed:
                nodes[iface] = (item['domain_uuid_key'],
                    {p['domain_uuid_key'] for p in item.get('_items', []) if 'domain_uuid_key' in p})
        ports.append(nodes)
    for a, ai, b, bi in edges(topology, mapping):
        mesh.require(ai in ports[a] and bi in ports[b], 'Missing Thunderbolt port inventory')
        mesh.require(ports[a][ai][1] == {ports[b][bi][0]} and ports[b][bi][1] == {ports[a][ai][0]},
                     f'Wrong cable: mac-{a + 1} {ai} must connect to mac-{b + 1} {bi}')


def require_ipv4(states, topology, mapping=None):
    missing = [f'mac-{r + 1} {iface}' for r, state in enumerate(states)
               for iface in {ai if r == a else bi for a, ai, b, bi in edges(topology, mapping) if r in (a, b)}
               if not state['interfaces'][iface]['ipv4']]
    mesh.require(not missing, 'RDMA requires IPv4 on each selected port; missing: ' + ', '.join(missing)
                 + '. For ring wiring run scripts/configure_ring_ipv4.py --apply first.')
    for a, ai, b, bi in edges(topology, mapping):
        left, right = (states[r]['interfaces'][i]['ipv4'][0] for r, i in ((a, ai), (b, bi)))
        ln, rn = (ipaddress.IPv4Network(f'{ip}/{mask}', strict=False) for ip, mask in (left, right))
        mesh.require(ln == rn and left[0] != right[0],
                     f'Ring peer IPv4 mismatch: mac-{a + 1} {ai} {left[0]} <-> mac-{b + 1} {bi} {right[0]}. '
                     'Run configure_ring_ipv4.py --apply --align-peers; do not launch collectives yet.')


def preflight(topology, require_idle=True, mapping=None):
    states = [inspect_host(rank) for rank in range(4)]
    validate_states(states, topology, require_idle, mapping)
    if mapping:
        mesh.require(discover_ring(states) == mapping, 'Cables changed since remapping; run --remap-ring again')
    require_ipv4(states, topology, mapping)
    print(f'[preflight] all four physical identities, {topology} cables, RDMA and Wi-Fi management verified', flush=True)
    return states

