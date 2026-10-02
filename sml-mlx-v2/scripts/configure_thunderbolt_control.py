#!/usr/bin/env python3
"""Repair only known mesh routes, then verify Thunderbolt SSH and CPU JACCL.

Read-only unless --apply or --restore is supplied. Administrator passwords are
entered in the user's Terminal. Never starts training or changes Wi-Fi settings.
"""

import argparse
import datetime
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import threading
import time

import configure_rdma_mesh as mesh

ROOT = mesh.ROOT
SELF = ROOT / 'scripts/configure_thunderbolt_control.py'
FORMAT = 'sml-thunderbolt-control-routes-v1'


def setup_transport(rank, args, tty=False, mac4_via_mac2=False):
    cmd = mesh.transport(rank, args, tty=tty)
    if mac4_via_mac2 and rank == 3:
        # Both endpoints retain pinned host keys. No keys or agent go to the relay.
        cmd[1:1] = ['-F', str(mesh.control_settings('wifi')['config']),
                    '-o', 'ForwardAgent=no', '-J', mesh.ssh_alias(1)]
    return cmd


def inspect_host(rank, mac4_via_mac2=False):
    if not mac4_via_mac2 or rank != 3:
        return mesh.invoke(rank, 'inspect')
    bootstrap = f'exec(compile({mesh.SELF.read_text()!r}, {str(mesh.SELF)!r}, "exec"))'
    request = json.dumps(dict(rank=rank, action='inspect', before=None, control_transport='wifi'))
    cmd = setup_transport(rank, [mesh.PYTHON, '-c', bootstrap, '--worker', request],
                          mac4_via_mac2=True)
    state = json.loads(mesh.run(cmd, timeout=100, label='mac-4 inspect through mac-2').stdout)
    mesh.configured_check(state, rank)
    connection = state.get('ssh_connection', '').split()
    mesh.require(len(connection) == 4 and connection[0] == mesh.HOSTS[1][2]
                 and connection[2] == mesh.HOSTS[3][2], 'mac-4 setup SSH did not arrive through mac-2')
    return state


def retry_read_check(operation, label):
    """Retry transient SSH failures only, never an administrator route change."""
    for attempt in range(3):
        try:
            return operation()
        except RuntimeError as exc:
            detail = str(exc).lower()
            transient = 'exited 255: ssh:' in detail and any(message in detail for message in (
                'operation timed out', 'connection timed out', 'connection reset by peer',
                'connection closed by remote host', 'no route to host'))
            if not transient or attempt == 2:
                raise
            delay = 2 ** (attempt + 1)
            print(f'[retry] {label}: temporary SSH connection failure; retry {attempt + 2}/3 in {delay}s', flush=True)
            time.sleep(delay)


def entries(raw, destination):
    found = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in (destination, destination + '/32'):
            found.append(dict(destination=destination, gateway=parts[1], flags=parts[2], interface=parts[3]))
    return found


def mac_address(value):
    parts = value.lower().split(':')
    if len(parts) != 6 or any(not re.fullmatch('[0-9a-f]{1,2}', part) for part in parts):
        return None
    return ':'.join(part.zfill(2) for part in parts)


def connected(raw, ip, interface):
    expected = ipaddress.IPv4Network(ip + '/30', strict=False)
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[1].startswith('link#') or parts[3] != interface:
            continue
        address, sep, prefix = parts[0].partition('/')
        if not sep or prefix != '30':
            continue
        octets = address.split('.')
        try:
            network = ipaddress.IPv4Network('.'.join(octets + ['0'] * (4 - len(octets))) + '/30')
        except ValueError:
            continue
        if network == expected and 'U' in parts[2] and 'H' not in parts[2]:
            return True
    return False


def repair_plan(state, rank):
    mesh.configured_check(state, rank)
    actions = []
    for interface, cfg in mesh.plan(rank).items():
        mesh.require(connected(state['routes'], cfg['ip'], interface), f'Missing connected /30 route on {interface}')
        current = entries(state['routes'], cfg['peer'])
        mesh.require(len(current) <= 1, f'Ambiguous routes to {cfg["peer"]}')
        if not current:
            continue
        item = current[0]
        mesh.require(item['interface'] == interface, f'Unexpected peer route interface: {item}')
        if 'S' not in item['flags']:
            mesh.require('H' in item['flags'] and 'L' in item['flags'] and 'G' not in item['flags'],
                         f'Unexpected dynamic peer route: {item}')
            continue
        # Only remove the exact broken shape created by our earlier helper.
        mesh.require(set('UHLS') <= set(item['flags']) and not set('GIRB') & set(item['flags'])
                     and mac_address(item['gateway']) == state['interfaces'][interface]['mac'],
                     f'Unrecognized static peer route; refusing to change it: {item}')
        actions.append(dict(kind='remove-self-peer', peer=cfg['peer'], interface=interface))
    if rank in (2, 3):
        interface, gateway = mesh.root_link(rank)
        current = entries(state['routes'], mesh.CONTROL_IPS[0])
        mesh.require(len(current) <= 1, 'Ambiguous coordinator routes')
        if current:
            item = current[0]
            mesh.require(item['gateway'] == gateway and item['interface'] == interface
                         and set('UGHS') <= set(item['flags']) and not set('ILRB') & set(item['flags']),
                         f'Unexpected coordinator route; refusing to replace it: {item}')
        else:
            actions.append(dict(kind='add-coordinator', peer=mesh.CONTROL_IPS[0], interface=interface, gateway=gateway))
    return actions


def route_command(action, undo=False):
    if action['kind'] == 'remove-self-peer':
        return (['/sbin/route', '-n', 'add', '-host', action['peer'], '-interface', action['interface']]
                if undo else ['/sbin/route', '-n', 'delete', '-host', action['peer']])
    mesh.require(action['kind'] == 'add-coordinator', 'Unknown route action')
    return (['/sbin/route', '-n', 'delete', '-host', action['peer']] if undo else
            ['/sbin/route', '-n', 'add', '-host', action['peer'], action['gateway']])


def assert_idle():
    # Reuse the reviewed worker detector without importing MLX.
    common = sys.modules.get('sml_jaccl_common')
    if common is None:
        scope = {}
        exec(compile((ROOT / 'sml_v2/cluster/jaccl_common.py').read_text(), 'jaccl_common.py', 'exec'), scope)
    else:
        scope = vars(common)
    busy = scope['training_processes']()
    mesh.require(not busy, f'Stop existing GPU workers before route changes: {busy}')


def stable(before, after):
    mesh.require(before['interfaces'] == after['interfaces'], 'Interfaces changed since the audit; rerun --check')
    mesh.require(before['default'] == after['default'], 'Internet/default route changed since the audit')
    mesh.require(before['management'] == after['management'], 'Wi-Fi management routing changed since the audit')


def undo_routes(rank, before, actions):
    current = mesh.snapshot(rank)
    stable(before, current)
    for action in reversed(actions):
        raw = mesh.run(['/usr/sbin/netstat', '-rn', '-f', 'inet']).stdout
        found = entries(raw, action['peer'])
        mesh.require(len(found) <= 1, 'Ambiguous route during rollback')
        if action['kind'] == 'add-coordinator':
            if not found:
                continue
            item = found[0]
            mesh.require(item['gateway'] == action['gateway'] and item['interface'] == action['interface']
                         and set('UGHS') <= set(item['flags']) and not set('ILRB') & set(item['flags']),
                         'Coordinator route changed; refusing to delete it during rollback')
        else:
            if found and 'S' in found[0]['flags']:
                item = found[0]
                mesh.require(item['interface'] == action['interface']
                             and mac_address(item['gateway']) == before['interfaces'][action['interface']]['mac']
                             and set('UHLS') <= set(item['flags']) and not set('GIRB') & set(item['flags']),
                             'Static peer route changed during rollback')
                continue
            if found:
                item = found[0]
                mesh.require(item['interface'] == action['interface'] and 'L' in item['flags']
                             and not set('SGRB') & set(item['flags']), 'Peer route changed during rollback')
                cmd = ['/sbin/route', '-n', 'delete', '-host', action['peer']]
                if 'I' in item['flags']:
                    cmd += ['-ifscope', action['interface']]
                mesh.run(cmd)
        mesh.run(route_command(action, undo=True))
    stable(before, mesh.snapshot(rank))


def change_routes(rank, before, restore=False):
    mesh.require(os.geteuid() == 0, 'Route changes require sudo in your Terminal')
    assert_idle()
    actions = repair_plan(before, rank)
    if restore:
        undo_routes(rank, before, actions)
        return
    current = mesh.snapshot(rank)
    stable(before, current)
    mesh.require(repair_plan(current, rank) == actions, 'Routes changed since the audit; rerun --check')
    completed = []
    try:
        for action in actions:
            mesh.run(route_command(action))
            completed.append(action)
        after = mesh.snapshot(rank)
        stable(before, after)
        mesh.require(not repair_plan(after, rank), 'Route repair is incomplete')
    except BaseException:
        undo_routes(rank, before, completed)
        raise


def worker_source():
    # Ship every project dependency, including the last-minute GPU-worker guard.
    source = ['import sys,types']
    for name, path in (('configure_rdma_mesh', mesh.SELF),
                       ('sml_jaccl_common', ROOT / 'sml_v2/cluster/jaccl_common.py')):
        source += [f'module=types.ModuleType({name!r})',
                   f'sys.modules[{name!r}]=module',
                   f'exec(compile({path.read_text()!r}, {str(path)!r}, "exec"), module.__dict__)']
    source.append(f'exec(compile({SELF.read_text()!r}, {str(SELF)!r}, "exec"))')
    return '; '.join(source)


def invoke_change(rank, before, restore=False, check_only=False, mac4_via_mac2=False):
    req = json.dumps(dict(rank=rank, before=before, restore=restore, check_only=check_only))
    args = [mesh.PYTHON, '-c', worker_source(), '--worker', req]
    if check_only:
        result = mesh.run(setup_transport(rank, args, mac4_via_mac2=mac4_via_mac2),
                          timeout=60, label=f'mac-{rank + 1} repair-helper check')
        print(result.stdout, end='', flush=True)
        return
    cmd = setup_transport(rank, ['/usr/bin/sudo', *args], tty=True, mac4_via_mac2=mac4_via_mac2)
    result = subprocess.run(cmd)
    mesh.require(result.returncode == 0, f'mac-{rank + 1}: route change failed; use the printed rollback command')


def verify_coordinator():
    """Prove each rank reaches the coordinator through its direct cable address."""
    for rank in range(1, 4):
        target = mesh.root_link(rank)[1]
        payload = os.urandom(65536)
        expected = hashlib.sha256(payload).hexdigest().encode()
        failures = []
        with socket.socket() as listener:
            listener.settimeout(12)
            listener.bind((mesh.control_settings('thunderbolt')['listen'], 0))
            listener.listen(1)
            port = listener.getsockname()[1]

            def serve():
                try:
                    with listener.accept()[0] as connection:
                        connection.settimeout(8)
                        mesh.require(connection.getpeername()[0] == mesh.CONTROL_IPS[rank], 'Wrong coordinator client address')
                        connection.sendall(payload)
                        digest = b''
                        while len(digest) < len(expected):
                            chunk = connection.recv(len(expected) - len(digest))
                            mesh.require(bool(chunk), 'Short coordinator response')
                            digest += chunk
                        mesh.require(digest == expected, 'Coordinator TCP payload checksum mismatch')
                except Exception as exc:
                    failures.append(str(exc))

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            code = f'''import socket,hashlib
s=socket.socket(); s.settimeout(8); s.bind(({mesh.CONTROL_IPS[rank]!r},0))
s.connect(({target!r},{port})); data=b""
while len(data)<65536:
 part=s.recv(65536-len(data))
 if not part: raise RuntimeError("Short coordinator payload")
 data+=part
s.sendall(hashlib.sha256(data).hexdigest().encode()); s.close()
print("coordinator TCP verified")'''
            try:
                mesh.run(mesh.transport(rank, [mesh.PYTHON, '-c', code], control_transport='thunderbolt'),
                         timeout=20, label=f'mac-{rank + 1} Thunderbolt coordinator TCP')
            finally:
                thread.join(timeout=13)
            mesh.require(not thread.is_alive() and not failures, 'Coordinator test failed: ' + '; '.join(failures))
        print(f'[control-ok] mac-{rank + 1}: pinned Thunderbolt SSH + coordinator {target} TCP checksum verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--apply', action='store_true')
    group.add_argument('--check', action='store_true')
    group.add_argument('--verify', action='store_true')
    group.add_argument('--restore', type=Path)
    group.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--mac4-via-mac2', action='store_true',
                        help='Use pinned SSH through mac-2 for mac-4 setup/rollback only; final verification stays Thunderbolt-only')
    args = parser.parse_args()
    if args.worker:
        req = json.loads(args.worker)
        rank = req['rank']
        mesh.require(type(rank) is int and 0 <= rank < 4, 'Invalid rank')
        if req.get('check_only', False):
            assert_idle()
            print(f'[helper-ok] mac-{rank + 1}: embedded dependencies loaded; no GPU workers; no changes made')
            return
        change_routes(rank, req['before'], restore=req['restore'])
        print(f'mac-{rank + 1}: routes {"restored" if req["restore"] else "repaired"}; Wi-Fi and RDMA addresses unchanged')
        return
    if args.apply or args.restore:
        mesh.require(sys.stdin.isatty(), 'Run this command in your own Terminal for sudo passwords; no changes made')
    if args.verify:
        # check_jaccl verifies identities, cabling, routes, pinned SSH and coordinator TCP.
        subprocess.run([mesh.PYTHON, str(ROOT / 'scripts/check_jaccl.py'), '--control-transport', 'thunderbolt'], check=True)
        print('PASS: Thunderbolt-only control and four-rank CPU RDMA verified. No training started.')
        return
    record = None
    relay = args.mac4_via_mac2
    if args.restore:
        record = json.loads((args.restore / 'before.json').read_text())
        mesh.require(record['format'] == FORMAT and len(record['states']) == 4, 'Invalid rollback audit')
        saved_relay = record.get('mac4_via_mac2', False)
        mesh.require(type(saved_relay) is bool, 'Invalid rollback relay setting')
        relay = relay or saved_relay
    if relay:
        print('[setup] mac-4 via mac-2; setup relay only, not used by wired verification or training', flush=True)
    states = []
    for rank in range(4):
        path = 'mac-2 (Wi-Fi setup relay)' if relay and rank == 3 else 'Wi-Fi'
        print(f'[inspect] mac-{rank + 1}: auditing through {path}; no changes yet', flush=True)
        state = retry_read_check(lambda: inspect_host(rank, relay), f'mac-{rank + 1} inspection')
        mesh.configured_check(state, rank)
        states.append(state)
    mesh.topology_check(states)
    for rank in range(4):
        retry_read_check(lambda: invoke_change(rank, states[rank], check_only=True, mac4_via_mac2=relay),
                         f'mac-{rank + 1} repair-helper check')
    if args.restore:
        for rank, old in enumerate(record['states']):
            repair_plan(old, rank)
            stable(old, states[rank])
        for rank in reversed(range(4)):
            print(f'Restoring mac-{rank + 1}; enter its administrator password if prompted.', flush=True)
            invoke_change(rank, record['states'][rank], restore=True, mac4_via_mac2=relay)
        print('Previous routes restored. Use explicit --control-transport wifi until wired control is repaired.')
        return
    plans = [repair_plan(state, rank) for rank, state in enumerate(states)]
    audit = ROOT / 'diagnostics' / ('thunderbolt_control_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    audit.mkdir()
    (audit / 'before.json').write_text(json.dumps(dict(format=FORMAT, states=states, mac4_via_mac2=relay), indent=2))
    (audit / 'plan.json').write_text(json.dumps(plans, indent=2))
    for rank, actions in enumerate(plans):
        print(f'mac-{rank + 1}:', flush=True)
        for action in actions:
            print('  ' + shlex.join(route_command(action)), flush=True)
        if not actions:
            print('  Routes already repaired.', flush=True)
    undo = shlex.join([mesh.PYTHON, str(SELF), '--restore', str(audit)] + (['--mac4-via-mac2'] if relay else []))
    print(f'Audit: {audit}\nRollback (restores the old IP limitation):\n{undo}', flush=True)
    if not args.apply:
        print('Read-only check finished. Run again with --apply to repair routes and test CPU RDMA.')
        return
    for rank in range(4):
        if plans[rank]:
            print(f'\nRepairing mac-{rank + 1}; enter its administrator password if prompted.', flush=True)
            invoke_change(rank, states[rank], mac4_via_mac2=relay)
    subprocess.run([mesh.PYTHON, str(SELF), '--verify'], check=True)
    (audit / 'verified.json').write_text(json.dumps(dict(control_transport='thunderbolt', cpu_rdma=True)))
    print('Ready. Resume yourself with:\nbash ' + str(ROOT / 'stage_b/launch.py'))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
