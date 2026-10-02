"""Pure, reversible runtime IPv4/bridge plans for the verified five-Mac ring."""
import ipaddress
import re

from five_mac_topology import discover, probe


def interfaces(text):
    result = {}
    current = None
    for line in text.splitlines():
        match = re.match(r'^(\w+): flags=[^<]+<([^>]+)>', line)
        if match:
            current = result[match[1]] = dict(up='UP' in match[2].split(','), ipv4=[], members=[])
        elif current is not None:
            match = re.search(r'\binet ([0-9.]+) netmask (\S+)', line)
            if match:
                mask = match[2]
                if mask.startswith('0x'):
                    mask = str(ipaddress.IPv4Address(int(mask, 16)))
                current['ipv4'].append([match[1], mask])
            match = re.search(r'\bmember: (\w+)', line)
            if match:
                current['members'].append(match[1])
    return result


def route_networks(text):
    result = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] == 'default':
            continue
        address, _, prefix = fields[0].partition('/')
        if not re.fullmatch(r'\d+(?:\.\d+){0,3}', address):
            continue
        parts = address.split('.')
        try:
            result.append(ipaddress.IPv4Network('.'.join(parts + ['0'] * (4 - len(parts)))
                          + '/' + (prefix or str(len(parts) * 8)), strict=False))
        except ValueError:
            raise ValueError('Unparseable numeric IPv4 route')
    return result


def endpoint_commands(iface, old, new):
    commands, undo = [], []
    for ip, mask in old['ipv4']:
        commands.append(['/sbin/ifconfig', iface, 'inet', ip, '-alias'])
        undo.append(['/sbin/ifconfig', iface, 'inet', ip, 'netmask', mask, 'alias'])
    commands.append(['/sbin/ifconfig', iface, 'inet', new, 'netmask', '255.255.255.252', 'alias'])
    commands.append(['/sbin/ifconfig', iface, 'up'])
    undo.insert(0, ['/sbin/ifconfig', iface, 'inet', new, '-alias'])
    undo.append(['/sbin/ifconfig', iface, 'up' if old['up'] else 'down'])
    return commands, undo


def plan(reports, order):
    topology = discover(reports, order)
    endpoints = [[] for _ in reports]
    for index, (a, ai, b, bi) in enumerate(topology['edges'], start=1):
        endpoints[a].append(dict(interface=ai, ip=f'10.85.{index}.1', peer=b,
                                 peer_interface=bi, peer_ip=f'10.85.{index}.2'))
        endpoints[b].append(dict(interface=bi, ip=f'10.85.{index}.2', peer=a,
                                 peer_interface=ai, peer_ip=f'10.85.{index}.1'))
    nodes = []
    for physical, report in enumerate(reports):
        devs = interfaces(probe(report, 'interfaces'))
        chosen = {e['interface'] for e in endpoints[physical]}
        tb = set(report['thunderbolt_interface_labels'].values())
        commands, rollback, bridges = [], [], {}
        for iface, dev in devs.items():
            if not iface.startswith('bridge') or not chosen.intersection(dev['members']):
                continue
            if not set(dev['members']) <= tb:
                raise ValueError('Bridge contains a non-Thunderbolt interface; manual review required')
            bridges[iface] = dev
            if dev['up']:
                commands.append(['/sbin/ifconfig', iface, 'down'])
            for member in sorted(chosen.intersection(dev['members'])):
                commands.append(['/sbin/ifconfig', iface, 'deletem', member])
                rollback.append(['/sbin/ifconfig', iface, 'addm', member])
            if dev['up']:
                rollback.append(['/sbin/ifconfig', iface, 'up'])
        old_ports = {iface: devs[iface] for iface in chosen}
        for endpoint in endpoints[physical]:
            subnet = ipaddress.IPv4Network(endpoint['ip'] + '/30', strict=False)
            existing = [ipaddress.IPv4Network(f'{ip}/{mask}', strict=False)
                        for dev in devs.values() for ip, mask in dev['ipv4']]
            existing += route_networks(probe(report, 'routes'))
            if any(subnet.overlaps(other) for other in existing):
                raise ValueError(f'Planned {subnet} overlaps existing address or route')
            old = old_ports[endpoint['interface']]
            for ip, mask in old['ipv4']:
                known = ipaddress.IPv4Address(ip)
                if not (known.is_link_local or known in ipaddress.IPv4Network('10.77.0.0/24')
                        or any(known in ipaddress.IPv4Network(f'10.78.{i}.0/24') for i in range(201,205))):
                    raise ValueError('Unexpected existing training-port IPv4 address; manual review required')
            add, undo = endpoint_commands(endpoint['interface'], old, endpoint['ip'])
            commands.extend(add)
            rollback = undo + rollback
        nodes.append(dict(physical=physical, endpoints=endpoints[physical], original_ports=old_ports,
                          original_bridges=bridges, commands=commands, rollback=rollback))
    return dict(format='opensml-five-mac-network-v1', topology=topology, nodes=nodes,
                control='ethernet-ssh-loopback-coordinator-tunnels', persistent=False,
                production_started=False)


def matches_before(report, node):
    devs = interfaces(probe(report, 'interfaces'))
    return all(devs.get(k) == v for k, v in {**node['original_ports'], **node['original_bridges']}.items())


def verify_after(report, node, *, runtime=False):
    """Check configured ports; live preflight also permits benign macOS extras.

    Applying a plan still requires its exact resulting state. At runtime macOS
    may add a link-local /16 alias or bring a detached Thunderbolt bridge up.
    Neither changes the required /30 address or reconnects a training port to
    a bridge. The cluster separately verifies routes and pings for every cable.
    """
    devs = interfaces(probe(report, 'interfaces'))
    for endpoint in node['endpoints']:
        iface = endpoint['interface']
        required = [endpoint['ip'], '255.255.255.252']
        addresses = devs[iface]['ipv4']
        matches = addresses == [required]
        if runtime:
            matches = addresses.count(required) == 1 and all(
                address == required or (ipaddress.IPv4Address(address[0]).is_link_local
                                         and address[1] == '255.255.0.0')
                for address in addresses)
        if not matches or not devs[iface]['up']:
            raise ValueError(f'{iface}: address/up state differs from plan')
        for bridge, state in devs.items():
            if bridge.startswith('bridge') and iface in state['members']:
                raise ValueError(f'{iface} remains in a bridge')
    for bridge in node['original_bridges']:
        if devs[bridge]['up']:
            thunderbolt = set(report['thunderbolt_interface_labels'].values())
            if not runtime or not set(devs[bridge]['members']) <= thunderbolt:
                raise ValueError('Training Thunderbolt bridge remains up')
