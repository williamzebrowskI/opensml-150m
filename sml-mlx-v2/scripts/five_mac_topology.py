"""Validate five fresh read-only inventories and derive the physical RDMA map.

This module does not configure interfaces, assign IPs, rename hosts, or launch
workers. Socket A/B labels must be confirmed by reciprocal peer evidence.
"""
import json
import re

IDENTITIES = (
    ('Apple M5 Ultra', '7c:d6:2c:00:09:df', 'SHA256:ixB3h7qjaTrCzEn9p0Va1m2tLFb4T37jLVYc9c6yCFk'),
    ('Apple M3 Ultra', '1c:1d:d3:dd:fa:84', 'SHA256:v45WFZM77qEGF9ZuVXV2Fj7llE/XdXUOqTwDCVDeAhE'),
    ('Apple M4 Max', '1c:1d:d3:db:7b:63', 'SHA256:oLLqAAJ679URqDBNzcrF7ebhPAduCHlIRa7lmlcOM2I'),
    ('Apple M4 Max', '1c:1d:d3:d8:19:b6', 'SHA256:8wdYLC64yOk+4wt9YQIf5U8eH+sUH2c0hKFBF4+1Z1s'),
    ('Apple M4 Max', '1c:1d:d3:d7:b4:da', 'SHA256:tty3pv5axM9NMBgu1Wz5BablZBwv/XUHffi+UqcdkPY'),
)
EXPECTED_EDGES = {frozenset((i, (i + 1) % 5)) for i in range(5)}


def probe(report, name):
    value = report['probes'][name]
    if value['returncode'] != 0:
        raise ValueError(f'{name}: inventory probe failed')
    return value['stdout']


def ports(report):
    labels = dict(re.findall(r'Hardware Port: (Thunderbolt \d+)\nDevice: (en\d+)',
                             probe(report, 'hardware_ports')))
    result = {}
    for bus in json.loads(probe(report, 'thunderbolt'))['SPThunderboltDataType']:
        tag = bus.get('receptacle_1_tag', {}).get('receptacle_id_key')
        domain = bus.get('domain_uuid_key')
        if not domain:
            continue
        iface = labels.get('Thunderbolt ' + str(tag))
        if iface is None or iface in result:
            raise ValueError('Unmapped or duplicate Thunderbolt controller')
        peers = {p['domain_uuid_key'] for p in bus.get('_items', []) if 'domain_uuid_key' in p}
        result[iface] = (domain, peers)
    if not result:
        raise ValueError('No Thunderbolt controller identities')
    return result


def validate_inventory(report, rank):
    chip, mac, host_key = IDENTITIES[rank]
    if report.get('format') != 'opensml-readonly-mac-inventory-v1':
        raise ValueError('Unknown inventory format')
    if probe(report, 'chip').strip() != chip:
        raise ValueError(f'Mac-{rank + 1}: chip identity differs')
    ethernet = re.search(r'Hardware Port: Ethernet\nDevice: en0\nEthernet Address: (\S+)',
                         probe(report, 'hardware_ports'))
    if not ethernet or ethernet[1].lower() != mac:
        raise ValueError(f'Mac-{rank + 1}: hardware Ethernet identity differs')
    fields = probe(report, 'ssh_host_fingerprint').split()
    if len(fields) < 2 or fields[1] != host_key:
        raise ValueError(f'Mac-{rank + 1}: SSH identity differs')
    if not report.get('process_query_ok') or report.get('training_processes') != []:
        raise ValueError(f'Mac-{rank + 1}: idle state is not verified')
    if probe(report, 'rdma').strip() != 'enabled':
        raise ValueError(f'Mac-{rank + 1}: RDMA is not enabled')
    devices = set(re.findall(r'^\s*(rdma_en\d+)\s', probe(report, 'ibv_devices'), re.M))
    return ports(report), devices


def discover(reports, physical_order=None):
    """Reports must be ordered by the NEW Mac labels, never old hostnames."""
    if len(reports) != 5:
        raise ValueError('Exactly five inventories required')
    order = list(range(5)) if physical_order is None else physical_order
    if (not isinstance(order, list) or any(type(i) is not int for i in order)
            or sorted(order) != list(range(5)) or order[:2] != [0, 1]):
        raise ValueError('Ring order must contain each physical Mac once, starting with M5/M3')
    expected_edges = {frozenset((order[i], order[(i + 1) % 5])) for i in range(5)}
    inventories = [validate_inventory(r, i) for i, r in enumerate(reports)]
    owners = {}
    for rank, (portmap, _) in enumerate(inventories):
        for iface, (domain, _) in portmap.items():
            if domain in owners:
                raise ValueError('Duplicate Thunderbolt domain identity')
            owners[domain] = (rank, iface)
    cables = []
    used = set()
    for rank, (portmap, devices) in enumerate(inventories):
        for iface, (domain, peers) in portmap.items():
            for peer in peers:
                if peer not in owners:
                    continue  # A peripheral is not evidence of a cluster edge.
                other, remote = owners[peer]
                if other == rank or peers != {peer} or inventories[other][0][remote][1] != {domain}:
                    raise ValueError('Ambiguous, self, or nonreciprocal cluster cable')
                if 'rdma_' + iface not in devices:
                    raise ValueError('Connected port lacks an enumerated RDMA device')
                if rank < other:
                    if (rank, iface) in used or (other, remote) in used:
                        raise ValueError('Port reused by multiple cluster cables')
                    used.update(((rank, iface), (other, remote)))
                    cables.append((rank, iface, other, remote))
    if len(cables) != 5 or {frozenset((a, b)) for a, _, b, _ in cables} != expected_edges:
        raise ValueError('Cables differ from the explicitly selected five-Mac ring order')
    matrix = [[None] * 5 for _ in range(5)]
    for a, ai, b, bi in cables:
        a, b = order.index(a), order.index(b)
        matrix[a][b], matrix[b][a] = 'rdma_' + ai, 'rdma_' + bi
    return dict(format='opensml-five-mac-cables-v1', physical_order=order,
                edges=sorted(cables), rdma=matrix, readiness_verified=False,
                note='Cable/identity evidence only; network configuration and live collectives remain required.')
