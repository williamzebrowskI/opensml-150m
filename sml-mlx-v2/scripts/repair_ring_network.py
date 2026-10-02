#!/usr/bin/env python3
"""Guarded runtime bridge isolation and neighbor refresh; never launches training."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import configure_rdma_mesh as mesh

ROOT = mesh.ROOT
SELF = ROOT / 'scripts/repair_ring_network.py'
FORMAT = 'sml-ring-network-repair-v1'


def inspect(rank, common):
    devs = mesh.interfaces(mesh.run(['/sbin/ifconfig']).stdout)
    mesh.require(devs['en0']['mac'] == mesh.HOSTS[rank][3], 'Physical Mac identity mismatch')
    mesh.require([mesh.HOSTS[rank][2], '255.255.255.0'] in devs['en1']['ipv4'], 'Wi-Fi IP changed')
    destinations = [h[2] for i, h in enumerate(mesh.HOSTS) if i != rank]
    routes = {ip: mesh.route(ip) for ip in destinations}
    mesh.require(all(r.get('interface') == 'en1' for r in routes.values()), 'Management must use Wi-Fi')
    default = mesh.route('default')
    mesh.require(default.get('interface') == 'en1', 'Default route must use Wi-Fi')
    mesh.require(not common['training_processes'](), 'GPU workers are active; stop training yourself first')
    bridges = {}
    for name, dev in devs.items():
        if not set(dev['members']) & {'en3', 'en4', 'en5'}:
            continue
        mesh.require(rank != 0 and name == 'bridge0' and dev['mac'] == mesh.BRIDGE_MACS[rank],
                     'Unknown bridge; refusing changes')
        mesh.require(set(dev['members']) <= {'en2', 'en3', 'en4', 'en5'}, 'Unexpected bridge members')
        mesh.require(not dev['ipv4'], 'Bridge has an IPv4 address; refusing to disrupt it')
        bridges[name] = dict(mac=dev['mac'], members=dev['members'], up=dev['up'])
    return dict(rank=rank, identity=devs['en0']['mac'], wifi=devs['en1']['ipv4'],
                default=default, management=routes, bridges=bridges,
                rdma=mesh.run(['/usr/bin/rdma_ctl', 'status']).stdout.strip())


def verify_stable(before, current):
    for key in ('rank', 'identity', 'wifi', 'default', 'management', 'rdma'):
        mesh.require(before[key] == current[key], f'{key} changed since audit; refusing changes')
    mesh.require(set(before['bridges']) == set(current['bridges']), 'Bridge inventory changed')
    for name, dev in before['bridges'].items():
        for key in ('mac', 'members'):
            mesh.require(dev[key] == current['bridges'][name][key], 'Bridge identity/members changed')


def worker(request, common):
    rank, action = request['rank'], request['action']
    current = inspect(rank, common)
    if action == 'inspect':
        print(json.dumps(current))
        return
    mesh.require(action in ('apply', 'restore'), 'Unknown action')
    mesh.require(os.geteuid() == 0, 'Administrator privileges required')
    before = request['before']
    verify_stable(before, current)
    for name, dev in before['bridges'].items():
        target = dev['up'] if action == 'restore' else False
        if current['bridges'][name]['up'] != target:
            mesh.run(['/sbin/ifconfig', name, 'up' if target else 'down'])
            print(f'mac-{rank + 1}: {name} -> {"up" if target else "down"}', flush=True)
    # Refresh only the problematic pair's disposable ARP entries, never static entries.
    if action == 'apply' and rank in (0, 3):
        peer = mesh.HOSTS[3 if rank == 0 else 0][2]
        arp = mesh.run(['/usr/sbin/arp', '-n', peer], check=False)
        if arp.returncode == 0 and ' on en1 ' in arp.stdout and 'permanent' not in arp.stdout.lower():
            result = mesh.run(['/usr/sbin/arp', '-d', peer, 'ifscope', 'en1'], check=False)
            print(f'mac-{rank + 1}: neighbor refresh {peer}: {result.stdout.strip()} {result.stderr.strip()}', flush=True)
            mesh.require(result.returncode == 0 or 'cannot locate' in result.stderr.lower(), 'ARP refresh failed')
        else:
            print(f'mac-{rank + 1}: no disposable Wi-Fi neighbor entry to clear', flush=True)
    after = inspect(rank, common)
    verify_stable(before, after)
    for name, dev in before['bridges'].items():
        mesh.require(after['bridges'][name]['up'] == (dev['up'] if action == 'restore' else False),
                     'Bridge state did not persist')
    print(f'mac-{rank + 1}: verified; Wi-Fi/default route and RDMA enablement unchanged', flush=True)


def transport(rank, argv, relay=False, tty=False):
    cmd = mesh.transport(rank, argv, tty=tty)
    if relay and rank == 3:
        proxy = shlex.join(['/usr/bin/ssh', '-F', str(ROOT / 'cluster/jaccl/ssh_config'),
                            '-o', 'ForwardAgent=no', '-W', '%h:%p', mesh.ssh_alias(1)])
        cmd[1:1] = ['-o', 'ForwardAgent=no', '-o', 'ProxyCommand=' + proxy]
    return cmd


def invoke(rank, action='inspect', before=None, relay=False):
    # Embed dependencies so remote project copies need not be up to date.
    source = SELF.read_text().replace('import configure_rdma_mesh as mesh', '')
    mesh_source = (ROOT / 'scripts/configure_rdma_mesh.py').read_text()
    common_source = (ROOT / 'sml_v2/cluster/jaccl_common.py').read_text()
    code = f'''import types
m = types.ModuleType("ring_mesh_helper")
exec(compile({mesh_source!r}, "mesh_helper.py", "exec"), m.__dict__)
c = {{"__name__": "ring_common_helper"}}
exec(compile({common_source!r}, "common_helper.py", "exec"), c)
s = {{"__name__": "ring_network_worker", "mesh": m}}
exec(compile({source!r}, "ring_network_worker.py", "exec"), s)
s["worker"]({dict(rank=rank, action=action, before=before)!r}, c)
'''
    command = [mesh.PYTHON, '-c', code]
    if action != 'inspect':
        command = ['/usr/bin/sudo', '--', *command]
        subprocess.run(transport(rank, command, relay, tty=True), check=True)
        return None
    result = mesh.run(transport(rank, command, relay), timeout=45, label=f'mac-{rank + 1} network audit')
    return json.loads(result.stdout)


def verify_direct():
    failures = []
    for rank in range(4):
        try:
            invoke(rank)
            print(f'[direct-ok] mac-{rank + 1}: identity, SSH and Wi-Fi routes verified', flush=True)
        except RuntimeError as exc:
            failures.append(str(exc))
    mesh.require(not failures, 'Direct access still blocked:\n' + '\n'.join(failures)
                 + '\nNo benchmark started. Investigate Wi-Fi/router connectivity; do not rewire based on this alone.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='Read-only audit (default)')
    modes.add_argument('--apply', action='store_true')
    modes.add_argument('--restore', type=Path, metavar='AUDIT_DIRECTORY')
    parser.add_argument('--mac4-via-mac2', action='store_true', help='Setup relay only; final checks remain direct')
    args = parser.parse_args()
    changing = bool(args.apply or args.restore)
    mesh.require(not changing or sys.stdin.isatty(), 'Run --apply/--restore in your interactive Terminal for sudo')
    common = {'__name__': 'ring_common_local'}
    exec(compile((ROOT / 'sml_v2/cluster/jaccl_common.py').read_text(), 'ring_common_local.py', 'exec'), common)
    inspect(0, common)
    lockdir = ROOT / 'runs/cluster_setup_locks'
    lockdir.mkdir(parents=True, exist_ok=True)
    with (lockdir / '.launcher.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        states = []
        for rank in range(4):
            print(f'[inspect] mac-{rank + 1}' + (' via mac-2' if rank == 3 and args.mac4_via_mac2 else ''), flush=True)
            states.append(invoke(rank, relay=args.mac4_via_mac2))
        for state in states:
            print(f'mac-{state["rank"] + 1}: bridges={state["bridges"]}')
        if not changing:
            print('Read-only audit complete. No changes or jobs started.')
            return
        if args.restore:
            previous = json.loads((args.restore / 'before.json').read_text())
            mesh.require(previous['format'] == FORMAT and len(previous['states']) == 4, 'Invalid rollback audit')
            targets = previous['states']
            for before, state in zip(targets, states):
                verify_stable(before, state)
        else:
            targets = states
        audit = ROOT / 'diagnostics' / ('ring_network_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        audit.mkdir(parents=True)
        common['atomic_json'](audit / 'before.json', dict(format=FORMAT, states=states))
        suffix = ' --mac4-via-mac2' if args.mac4_via_mac2 else ''
        print(f'Audit: {audit}\nRollback (may restore unwanted bridges):\n{mesh.PYTHON} {SELF} --restore {audit}{suffix}', flush=True)
        print('Neighbor-cache refresh is temporary; macOS relearns entries automatically.', flush=True)
        try:
            for rank in (1, 2, 3, 0):
                common['atomic_json'](audit / f'rank{rank}-requested.json', dict(action='restore' if args.restore else 'apply'))
                invoke(rank, 'restore' if args.restore else 'apply', targets[rank], args.mac4_via_mac2)
                common['atomic_json'](audit / f'rank{rank}-completed.json', dict(completed=True))
            time.sleep(5)
            verify_direct()
        except BaseException as exc:
            common['atomic_json'](audit / 'failure.json', dict(error=str(exc)))
            print(f'INCOMPLETE: some changes may have applied. Audit and rollback: {audit}', flush=True)
            raise
        print('Direct SSH verified. No training, RDMA collective test, or remapping was started.')
        print(f'Next: {mesh.PYTHON} {ROOT / "scripts/benchmark_jaccl_ring.py"} --remap-ring')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
