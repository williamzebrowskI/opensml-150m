#!/usr/bin/env python3
"""Configure the verified four-Studio mesh without changing Wi-Fi or training.

Runtime-only settings. Passwords are handled directly by sudo in Terminal.
Default is read-only; --apply configures and tests; --restore undoes an audit.
"""

import argparse
import datetime
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path('/Users/williamzebrowski/sml-mlx/sml-mlx-v1')
PYTHON = str(ROOT / '.venv/bin/python')
SELF = ROOT / 'scripts/configure_rdma_mesh.py'
POOL = ipaddress.IPv4Network('10.77.0.0/24')
HOSTS = [
    ('mac-1', None, '192.168.12.199', '1c:1d:d3:dd:fa:84'),
    ('mac-2', 'mac-4.local', '192.168.12.235', '1c:1d:d3:db:7b:63'),
    ('mac-3', 'mac-6.local', '192.168.12.119', '1c:1d:d3:d8:19:b6'),
    ('mac-4', 'mac-7.local', '192.168.12.101', '1c:1d:d3:d7:b4:da'),
]
# Rack wiring verified in diagnostics/rack_mesh_recheck_20260909_195002.
# Keep pair order unchanged so each Mac pair retains its original /30 subnet.
EDGES = [(0, 'en5', 1, 'en5'), (0, 'en3', 2, 'en5'), (0, 'en4', 3, 'en5'),
         (1, 'en3', 2, 'en4'), (1, 'en4', 3, 'en4'), (2, 'en3', 3, 'en3')]
PORT_MACS = [
    ('36:96:d6:2a:4d:04', '36:96:d6:2a:4d:08', '36:96:d6:2a:4d:0c'),
    ('36:68:88:7d:e9:04', '36:68:88:7d:e9:08', '36:68:88:7d:e9:0c'),
    ('36:5d:cd:c3:88:84', '36:5d:cd:c3:88:88', '36:5d:cd:c3:88:8c'),
    ('36:fa:27:d3:10:44', '36:fa:27:d3:10:48', '36:fa:27:d3:10:4c'),
]
BRIDGE_MACS = [None, '36:68:88:7d:e9:00', '36:5d:cd:c3:88:80', '36:fa:27:d3:10:40']
CONTROL_IPS = ['10.77.0.1', '10.77.0.2', '10.77.0.6', '10.77.0.10']
ROOT_LINKS = {1: 'en5', 2: 'en3', 3: 'en4'}


def control_settings(mode='wifi'):
    require(mode in ('wifi', 'thunderbolt', 'ring-thunderbolt'), 'Unknown control transport')
    if mode == 'ring-thunderbolt':
        return dict(coordinator='127.0.0.1', listen='127.0.0.1',
                    config=ROOT / 'cluster/jaccl/ssh_ring_thunderbolt_config',
                    hostfile=None)
    wired = mode == 'thunderbolt'
    return dict(coordinator=CONTROL_IPS[0] if wired else HOSTS[0][2],
                listen='0.0.0.0' if wired else HOSTS[0][2],
                config=ROOT / ('cluster/jaccl/ssh_thunderbolt_config' if wired else 'cluster/jaccl/ssh_config'),
                hostfile=ROOT / ('cluster/jaccl/hosts_thunderbolt.json' if wired else 'cluster/jaccl/hosts.json'))


def ssh_alias(rank, mode='wifi'):
    control_settings(mode)
    prefix = 'ring-tb-' if mode == 'ring-thunderbolt' else 'tb-' if mode == 'thunderbolt' else ''
    return f'sml-jaccl-{prefix}mac-{rank + 1}'


def root_link(rank):
    peer = plan(0)[ROOT_LINKS[rank]]
    interface = next(name for name, cfg in plan(rank).items() if cfg['peer'] == peer['ip'])
    return interface, peer['ip']


def worker_command(command, control_transport='wifi'):
    """Adapt mlx.launch's shared coordinator IP to this mesh's per-cable IPs."""
    network = control_settings(control_transport)
    if control_transport in ('wifi', 'ring-thunderbolt'):
        return list(command)
    endpoints = (network['listen'], *(root_link(rank)[1] for rank in range(1, 4)))
    source = f'''import os,sys
rank = int(os.environ["MLX_RANK"])
host, port = os.environ["MLX_JACCL_COORDINATOR"].rsplit(":", 1)
if rank not in range(4) or host != {network['coordinator']!r} or not port.isascii() or not port.isdecimal() or not 1024 <= int(port) <= 65535:
    raise RuntimeError("Unexpected rank or coordinator; refusing wired launch")
if os.environ.get("JACCL_RANK", str(rank)) != str(rank):
    raise RuntimeError("Conflicting JACCL_RANK override")
endpoint = {endpoints!r}[rank] + ":" + port
os.environ["MLX_JACCL_COORDINATOR"] = endpoint
os.environ["JACCL_COORDINATOR"] = endpoint
print(f"[coordinator] rank={{rank}} endpoint={{endpoint}} transport=thunderbolt", flush=True)
os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
'''
    return [PYTHON, '-c', source, *command]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(args, check=True, timeout=30, label=None):
    description = label or shlex.join(args)
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'{description}: timed out after {timeout}s') from exc
    if check and p.returncode:
        details = p.stderr.strip() or p.stdout.strip() or 'no command output'
        raise RuntimeError(f'{description} exited {p.returncode}: {details}')
    return p


def plan(rank):
    ports = {}
    for n, (a, ai, b, bi) in enumerate(EDGES):
        if rank in (a, b):
            local, peer = (1, 2) if rank == a else (2, 1)
            ports[ai if rank == a else bi] = {
                'ip': f'10.77.0.{n * 4 + local}',
                'peer': f'10.77.0.{n * 4 + peer}',
                'netmask': '255.255.255.252',
            }
    return ports


def interfaces(raw):
    result = {}
    for block in re.split(r'(?=^\w+: flags=)', raw, flags=re.M):
        name = re.match(r'(\w+): flags=\w+<([^>]*)>', block)
        if not name:
            continue
        mac = re.search(r'\bether (\S+)', block)
        ips = []
        for address, mask in re.findall(r'\binet (\S+) netmask (\S+)', block):
            mask = str(ipaddress.IPv4Address(int(mask, 16))) if mask.startswith('0x') else mask
            ips.append([str(ipaddress.IPv4Address(address)), mask])
        result[name[1]] = dict(mac=mac[1] if mac else None, ipv4=ips,
            up='UP' in name[2].split(','), active='status: active' in block,
            members=re.findall(r'\bmember: (\S+)', block))
    return result


def route(destination):
    raw = run(['/sbin/route', '-n', 'get', destination]).stdout
    return {key: value for key, value in re.findall(r'^\s*(interface|gateway):\s*(\S+)', raw, re.M)}


def snapshot(rank, topology=False, control_transport='wifi', allow_bridge_up=False):
    control_settings(control_transport)
    if control_transport == 'thunderbolt':
        destinations = CONTROL_IPS[1:] if rank == 0 else [root_link(rank)[1]]
        default_raw = run(['/sbin/route', '-n', 'get', 'default'], check=False)
        default = dict(re.findall(r'^\s*(interface|gateway):\s*(\S+)', default_raw.stdout, re.M))
    else:
        destinations = [h[2] for i, h in enumerate(HOSTS) if i != rank]
        default = route('default')
    state = dict(rank=rank, interfaces=interfaces(run(['/sbin/ifconfig']).stdout),
        default=default, management={ip: route(ip) for ip in destinations},
        routes=run(['/usr/sbin/netstat', '-rn', '-f', 'inet']).stdout,
        rdma=run(['/usr/bin/rdma_ctl', 'status']).stdout.strip(),
        control_transport=control_transport, ssh_connection=os.getenv('SSH_CONNECTION', ''))
    if topology:
        state['thunderbolt'] = json.loads(run(['/usr/sbin/system_profiler', 'SPThunderboltDataType',
            '-json', '-timeout', '30'], timeout=45).stdout)
        state['hardware_ports'] = run(['/usr/sbin/networksetup', '-listallhardwareports']).stdout
    guard(state, rank, allow_bridge_up=allow_bridge_up)
    return state


def guard(state, rank, allow_bridge_up=False):
    devs = state['interfaces']
    require(state['rank'] == rank and devs['en0']['mac'] == HOSTS[rank][3], 'Physical Mac identity changed')
    mode = state.get('control_transport', 'wifi')
    control_settings(mode)
    if mode == 'wifi':
        require([HOSTS[rank][2], '255.255.255.0'] in devs['en1']['ipv4'], 'Wi-Fi address changed')
        require(state['default'].get('interface') == 'en1', 'Default route must remain Wi-Fi')
        require(all(v.get('interface') == 'en1' for v in state['management'].values()), 'Management route is not Wi-Fi')
    else:
        expected = ({CONTROL_IPS[r]: ROOT_LINKS[r] for r in range(1, 4)} if rank == 0
                    else {root_link(rank)[1]: root_link(rank)[0]})
        require(set(state['management']) == set(expected), 'Missing Thunderbolt control routes')
        for ip, interface in expected.items():
            require(state['management'][ip].get('interface') == interface,
                    f'Thunderbolt control route to {ip} must use {interface}; run configure_thunderbolt_control.py --apply')
        if rank:
            connection = state.get('ssh_connection', '').split()
            require(len(connection) == 4 and connection[0] == root_link(rank)[1]
                    and connection[2] == CONTROL_IPS[rank], 'SSH did not arrive over the pinned Thunderbolt link')
    require(state['rdma'] == 'enabled', 'RDMA is not enabled')
    for i, name in enumerate(('en3', 'en4', 'en5')):
        require(devs[name]['mac'] == PORT_MACS[rank][i], 'Thunderbolt interface identity changed')
        require(devs[name]['up'] and devs[name]['active'], f'{HOSTS[rank][0]} {name} is not active')
    bridges = {n: d for n, d in devs.items() if n.startswith('bridge') and d['members']}
    for name, bridge in bridges.items():
        if set(bridge['members']) & {'en3', 'en4', 'en5'}:
            require(name == 'bridge0', 'Unexpected bridge contains mesh ports')
    if 'bridge0' in devs:
        bridge = devs['bridge0']
        require(bridge['mac'] == BRIDGE_MACS[rank] and (allow_bridge_up or not bridge['up']),
                'Bridge identity/state changed; stop and investigate')
        require(set(bridge['members']) <= {'en2', 'en3', 'en4', 'en5'}, 'Bridge contains a non-Thunderbolt port')
    else:
        require(rank == 0, 'Expected remote bridge is missing')


def topology_check(states):
    ports = []
    for state in states:
        mapping = dict(re.findall(r'Hardware Port: ([^\n]+)\nDevice: (\S+)', state['hardware_ports']))
        nodes = {}
        for item in state['thunderbolt']['SPThunderboltDataType']:
            label = 'Thunderbolt ' + str(item.get('receptacle_1_tag', {}).get('receptacle_id_key'))
            iface = mapping.get(label)
            if iface in ('en3', 'en4', 'en5'):
                nodes[iface] = (item['domain_uuid_key'],
                    {p['domain_uuid_key'] for p in item.get('_items', []) if 'domain_uuid_key' in p})
        ports.append(nodes)
    for a, ai, b, bi in EDGES:
        require(ai in ports[a] and bi in ports[b], 'Missing Thunderbolt port')
        require(ports[b][bi][0] in ports[a][ai][1] and ports[a][ai][0] in ports[b][bi][1],
            f'Cabling changed: {HOSTS[a][0]} {ai} <-> {HOSTS[b][0]} {bi}')


def unused_pool(state):
    for dev in state['interfaces'].values():
        for ip, mask in dev['ipv4']:
            require(not ipaddress.IPv4Network(f'{ip}/{mask}', strict=False).overlaps(POOL), '10.77.0.0/24 is already in use')
    for line in state['routes'].splitlines():
        token = line.split()[0] if line.split() else ''
        if not re.match(r'^\d', token):
            continue
        address, _, prefix = token.partition('/')
        parts = address.split('.')
        network = '.'.join(parts + ['0'] * (4 - len(parts))) + '/' + (prefix or str(len(parts) * 8))
        require(not ipaddress.IPv4Network(network, strict=False).overlaps(POOL), 'Existing route overlaps the proposed RDMA subnets')


def original_check(state, rank):
    guard(state, rank)
    unused_pool(state)
    for name in plan(rank):
        require(all(ipaddress.IPv4Address(ip).is_link_local and mask == '255.255.0.0'
                    for ip, mask in state['interfaces'][name]['ipv4']),
                f'{name} has a non-link-local address; refusing to replace it')


def transport(rank, args, tty=False, control_transport='wifi'):
    control_settings(control_transport)
    if rank == 0:
        return args
    if control_transport == 'ring-thunderbolt':
        return ['/usr/bin/ssh', '-F', str(control_settings(control_transport)['config']),
                '-tt' if tty else '-T', ssh_alias(rank, control_transport), shlex.join(args)]
    _, alias, ip, _ = HOSTS[rank]
    bind = []
    if control_transport == 'thunderbolt':
        ip = CONTROL_IPS[rank]
        bind = ['-o', 'BindAddress=' + root_link(rank)[1]]
    return ['/usr/bin/ssh', '-tt' if tty else '-T', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=8', '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no',
        '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3',
        *bind, '-o', 'HostName=' + ip, '-o', 'HostKeyAlias=' + alias,
        'williamzebrowski@' + alias, shlex.join(args)]


def invoke(rank, action, before=None, interactive=False, control_transport='wifi'):
    # Send the reviewed helper as code; no remote project file or password is copied.
    bootstrap = f'exec(compile({SELF.read_text()!r}, {str(SELF)!r}, "exec"))'
    args = [PYTHON, '-c', bootstrap, '--worker', json.dumps(dict(rank=rank, action=action, before=before,
                                                             control_transport=control_transport))]
    if action in ('apply', 'restore'):
        args = ['/usr/bin/sudo'] + ([] if interactive else ['-n']) + args
    cmd = transport(rank, args, tty=interactive, control_transport=control_transport)
    if interactive:
        result = subprocess.run(cmd)
        require(result.returncode == 0, f'{HOSTS[rank][0]} {action} exited {result.returncode}; see output above')
        return None
    return json.loads(run(cmd, timeout=100, label=f'{HOSTS[rank][0]} {action}').stdout)


def unchanged(before, current, rank):
    for name in ('en0', 'en1', 'en3', 'en4', 'en5', 'bridge0'):
        require(before['interfaces'].get(name) == current['interfaces'].get(name), f'{name} changed since preflight')
    require(before['default'] == current['default'], 'Default route changed since preflight')
    require(before['management'] == current['management'], 'Management routing changed since preflight')
    original_check(current, rank)


def restore(rank, before):
    original_check(before, rank)
    current = snapshot(rank)
    # Never erase unrelated addresses added after this helper ran.
    for name, cfg in plan(rank).items():
        allowed = before['interfaces'][name]['ipv4'] + [[cfg['ip'], cfg['netmask']]]
        require(all(pair in allowed for pair in current['interfaces'][name]['ipv4']), 'Unexpected address; inspect before restoring')
    errors = []
    for name, cfg in plan(rank).items():
        try:
            # Only the peer /32 routes created by apply are candidates for deletion.
            raw = run(['/usr/sbin/netstat', '-rn', '-f', 'inet']).stdout
            for line in raw.splitlines():
                fields = line.split()
                if fields and fields[0] in (cfg['peer'], cfg['peer'] + '/32'):
                    require(name in fields and 'H' in fields[2], 'Peer route unexpectedly changed')
                    run(['/sbin/route', '-n', 'delete', '-host', cfg['peer']])
            dev = interfaces(run(['/sbin/ifconfig', name]).stdout)[name]
            if [cfg['ip'], cfg['netmask']] in dev['ipv4']:
                run(['/sbin/ifconfig', name, 'inet', cfg['ip'], '-alias'])
            for ip, mask in before['interfaces'][name]['ipv4']:
                if [ip, mask] not in interfaces(run(['/sbin/ifconfig', name]).stdout)[name]['ipv4']:
                    run(['/sbin/ifconfig', name, 'inet', ip, 'netmask', mask, 'alias'])
            old_members = before['interfaces'].get('bridge0', {}).get('members', [])
            if name in old_members:
                bridge = interfaces(run(['/sbin/ifconfig', 'bridge0']).stdout)['bridge0']
                if name not in bridge['members']:
                    run(['/sbin/ifconfig', 'bridge0', 'addm', name])
        except Exception as exc:
            errors.append(str(exc))
    require(not errors, 'Rollback incomplete: ' + '; '.join(errors))
    restored = snapshot(rank)
    for name in ('en3', 'en4', 'en5', 'bridge0'):
        old = before['interfaces'].get(name)
        new = restored['interfaces'].get(name)
        require(old is None and new is None or old is not None and new is not None
            and old['ipv4'] == new['ipv4'] and set(old['members']) == set(new['members'])
            and old['up'] == new['up'], f'Rollback state mismatch: {name}')
    return restored


def apply(rank, before):
    unchanged(before, snapshot(rank), rank)
    try:
        members = before['interfaces'].get('bridge0', {}).get('members', [])
        for name, cfg in plan(rank).items():
            if name in members:
                run(['/sbin/ifconfig', 'bridge0', 'deletem', name])
            for ip, _ in before['interfaces'][name]['ipv4']:
                run(['/sbin/ifconfig', name, 'inet', ip, '-alias'])
            run(['/sbin/ifconfig', name, 'inet', cfg['ip'], 'netmask', cfg['netmask']])
            # The /30 connected route handles ARP. A host route using an Ethernet
            # interface name creates a bogus peer entry with our own MAC on macOS.
        after = snapshot(rank)
        configured_check(after, rank)
        return after
    except BaseException:
        print('Apply interrupted; attempting local rollback.', file=sys.stderr, flush=True)
        restore(rank, before)
        raise


def configured_check(state, rank):
    guard(state, rank)
    for name, cfg in plan(rank).items():
        addresses = state['interfaces'][name]['ipv4']
        expected = [cfg['ip'], cfg['netmask']]
        require(expected in addresses, f'{name} does not have its planned /30 address')
        # macOS may add a self-assigned alias without removing our mesh IP.
        require(all(pair == expected or (ipaddress.IPv4Address(pair[0]).is_link_local
                    and pair[1] == '255.255.0.0') for pair in addresses),
                f'{name} has an unexpected non-mesh IPv4 address')
        require(name not in state['interfaces'].get('bridge0', {}).get('members', []), f'{name} is still bridged')


def verify_worker(rank):
    state = snapshot(rank)
    configured_check(state, rank)
    state['link_checks'] = {}
    for name, cfg in plan(rank).items():
        require(route(cfg['peer']).get('interface') == name, f'{name} peer route is wrong')
        info = run(['ibv_devinfo', '-d', 'rdma_' + name, '-v']).stdout
        gids = re.findall(r'::ffff:[0-9.]+', info)
        require('::ffff:' + cfg['ip'] in gids and 'PORT_ACTIVE' in info, f'{name} RDMA address is missing')
        # ICMP is a separate IP diagnostic, not proof of RDMA success or failure.
        ping_args = ['/sbin/ping', '-n', '-S', cfg['ip'], '-c', '2', '-W', '1000', cfg['peer']]
        try:
            ping = run(ping_args, check=False, timeout=8)
            ping_result = dict(returncode=ping.returncode, stdout=ping.stdout, stderr=ping.stderr)
        except RuntimeError as exc:
            ping_result = dict(returncode=None, stdout='', stderr=str(exc))
        state['link_checks'][name] = dict(peer=cfg['peer'], rdma=info, ping=ping_result)
    return state


def collectives(audit):
    # Use the live topology and current hostfile, never a historical audit's map.
    result = run([PYTHON, str(ROOT / 'scripts/check_jaccl.py'), '--control-transport', 'wifi'],
                 check=False, timeout=600, label='current-mesh RDMA collective test')
    output = result.stdout + result.stderr
    (audit / 'collectives.log').write_text(output)
    print(output, end='', flush=True)
    ranks = set(re.findall(r'\[ok\] rank=(\d+)\b', output))
    require(result.returncode == 0 and ranks == {'0', '1', '2', '3'},
            'RDMA collective failed; keep these logs for diagnosis')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true')
    group.add_argument('--apply', action='store_true')
    group.add_argument('--verify', action='store_true')
    group.add_argument('--restore', type=Path, metavar='AUDIT_DIRECTORY')
    group.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        req = json.loads(args.worker)
        rank = req['rank']
        require(type(rank) is int and 0 <= rank < 4, 'Invalid rank')
        action = req['action']
        if action == 'inspect':
            result = snapshot(rank, topology=True, control_transport=req.get('control_transport', 'wifi'))
        elif action == 'inspect-setup':
            # Only this read-only setup inventory accepts a rebooted bridge.
            # Production preflight and all mutation paths keep strict guards.
            result = snapshot(rank, topology=True, allow_bridge_up=True)
        elif action == 'verify':
            result = verify_worker(rank)
        elif action in ('apply', 'restore'):
            result = globals()[action](rank, req['before'])
            print(HOSTS[rank][0] + ': ' + action + ' verified', flush=True)
            return
        else:
            raise RuntimeError('Unknown action')
        print(json.dumps(result))
        return

    require(interfaces(run(['/sbin/ifconfig', 'en0']).stdout)['en0']['mac'] == HOSTS[0][3], 'Run this on mac-1')
    if args.apply or args.restore:
        require(sys.stdin.isatty(), 'Run --apply/--restore in your own interactive Terminal for administrator passwords. No changes made.')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    audit = ROOT / 'diagnostics' / ('rdma_mesh_' + stamp)
    audit.mkdir()
    states = []
    for rank, host in enumerate(HOSTS):
        print('Checking ' + host[0] + ' through ' + host[2], flush=True)
        state = invoke(rank, 'inspect')
        (audit / (host[0] + '.json')).write_text(json.dumps(state, indent=2))
        states.append(state)
    topology_check(states)
    print('All six cable pairs and Wi-Fi management routes verified.', flush=True)

    if args.restore:
        old = json.loads((args.restore / 'before.json').read_text())
        require(len(old) == 4, 'Invalid rollback audit')
        topology_check(old)
        for rank in range(4):
            original_check(old[rank], rank)
        for rank in reversed(range(4)):
            print('Restoring ' + HOSTS[rank][0] + '. Enter that Mac administrator password if prompted.', flush=True)
            invoke(rank, 'restore', old[rank], interactive=True)
        print('Original mesh IPv4 addresses and membership restored; bridges remain down.', flush=True)
        return

    if not args.verify:
        for rank, state in enumerate(states):
            original_check(state, rank)
        (audit / 'before.json').write_text(json.dumps(states, indent=2))
        (audit / 'plan.json').write_text(json.dumps([plan(r) for r in range(4)], indent=2))
        for n, (a, ai, b, bi) in enumerate(EDGES):
            print(f'{HOSTS[a][0]} {ai} 10.77.0.{n*4+1}/30 <-> {HOSTS[b][0]} {bi} 10.77.0.{n*4+2}/30')
        if not args.apply:
            print('Read-only checks passed. No changes made. Audit: ' + str(audit))
            return
        print('Undo: ' + shlex.join([PYTHON, str(SELF), '--restore', str(audit)]), flush=True)
        try:
            for rank in range(4):
                print('\nConfiguring ' + HOSTS[rank][0] + '. Enter that Mac administrator password if prompted.', flush=True)
                invoke(rank, 'apply', states[rank], interactive=True)
        except BaseException:
            print('Some Macs may have been configured. Use the undo command above. Audit: ' + str(audit), file=sys.stderr)
            raise

    ping_warnings = []
    for rank, host in enumerate(HOSTS):
        print('Verifying all three links on ' + host[0], flush=True)
        result = invoke(rank, 'verify')
        (audit / (host[0] + '-verified.json')).write_text(json.dumps(result, indent=2))
        for interface, link in result['link_checks'].items():
            if link['ping']['returncode'] != 0:
                warning = f"{host[0]} {interface} -> {link['peer']}: ICMP ping did not pass"
                ping_warnings.append(warning)
                print('WARNING: ' + warning, flush=True)
    (audit / 'ip_diagnostics.json').write_text(json.dumps(dict(ping_warnings=ping_warnings), indent=2))
    print('All twelve RDMA port addresses and routes verified. Testing strict four-rank RDMA...', flush=True)
    collectives(audit)
    print('PASS: four-Mac CPU RDMA collectives verified. No training job was launched or stopped.')
    if ping_warnings:
        print(f'NOTE: {len(ping_warnings)} ICMP checks did not pass; ordinary IP connectivity is not fully verified.')
    print('Runtime-only configuration: recheck after reboot. GPU training is not yet validated.')
    print('Audit: ' + str(audit))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(1)
