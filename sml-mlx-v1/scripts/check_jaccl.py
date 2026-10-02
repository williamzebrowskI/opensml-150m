#!/usr/bin/env python3
"""Read-only mesh preflight followed by an official mlx.launch CPU RDMA test."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys

import configure_rdma_mesh as mesh

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'cluster/jaccl'
PYTHON = str(ROOT / '.venv/bin/python')
PROBE = ROOT / 'scripts/mlx_dist_check.py'


def check_hostfile(control_transport='wifi'):
    network = mesh.control_settings(control_transport)
    data = json.loads(network['hostfile'].read_text())
    matrix = [[None] * 4 for _ in range(4)]
    for a, ai, b, bi in mesh.EDGES:
        matrix[a][b], matrix[b][a] = 'rdma_' + ai, 'rdma_' + bi
    expected = [dict(ssh='127.0.0.1' if rank == 0 else mesh.ssh_alias(rank, control_transport),
                     ips=[network['coordinator']] if rank == 0 else [], rdma=matrix[rank])
                for rank in range(4)]
    mesh.require(data == dict(backend='jaccl', hosts=expected),
                 'Hostfile differs from the verified mesh; inspect before launching')
    if control_transport == 'thunderbolt':
        for rank in range(1, 4):
            raw = mesh.run(['/usr/bin/ssh', '-G', '-F', str(network['config']), mesh.ssh_alias(rank, control_transport)]).stdout
            options = dict(line.split(None, 1) for line in raw.splitlines() if ' ' in line)
            required = dict(hostname=mesh.CONTROL_IPS[rank], bindaddress=mesh.root_link(rank)[1],
                            hostkeyalias=mesh.HOSTS[rank][1], user='williamzebrowski', port='22',
                            addressfamily='inet', batchmode='yes', stricthostkeychecking='true')
            mesh.require(all(options.get(k) == v for k, v in required.items())
                         and options.get('proxycommand', 'none') == 'none'
                         and options.get('proxyjump', 'none') == 'none',
                         f'mac-{rank + 1}: SSH configuration does not match pinned direct Thunderbolt transport')


def command(port, control_transport='wifi'):
    worker = ['/usr/bin/env', '-u', 'MLX_METAL_FAST_SYNCH', '-u', 'MLX_JACCL_RING',
              PYTHON, '-u', str(PROBE), '--backend', 'jaccl', '--expected-world', '4',
              '--stream', 'cpu', '--timeout', '60', '--no-p2p', '--hostnames',
              '--tensor-mb', '1', '--iters', '5', '--verbose']
    return [str(ROOT / '.venv/bin/mlx.launch'), '--backend', 'jaccl',
            '--hostfile', str(mesh.control_settings(control_transport)['hostfile']), '--cwd', str(ROOT),
            '--starting-port', str(port), '--', *mesh.worker_command(worker, control_transport)]


def passed(returncode, output):
    ranks = set(re.findall(r'\[ok\] rank=(\d+)\b', output))
    return returncode == 0 and ranks == {'0', '1', '2', '3'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-transport', choices=('wifi', 'thunderbolt'), default='wifi')
    args = parser.parse_args()
    mode = args.control_transport
    network = mesh.control_settings(mode)
    check_hostfile(mode)
    mesh.require(os.access(CONFIG / 'bin/ssh', os.X_OK), 'Project SSH shim is not executable')
    audit = ROOT / 'diagnostics' / ('jaccl_launch_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    audit.mkdir(parents=True)
    print(f'CPU-only JACCL check; no training or network changes. Audit: {audit}', flush=True)
    states, versions = [], []
    probe_hash = hashlib.sha256(PROBE.read_bytes()).hexdigest()
    inventory_code = (
        'import hashlib, importlib.metadata as m, json; from pathlib import Path; '
        f'print(json.dumps(dict(probe=hashlib.sha256(Path({str(PROBE)!r}).read_bytes()).hexdigest(), '
        'mlx=m.version("mlx"), metal=m.version("mlx-metal"))))'
    )
    for rank in range(4):
        state = mesh.invoke(rank, 'inspect', control_transport=mode)
        mesh.configured_check(state, rank)
        states.append(state)
        inventory = json.loads(mesh.run(mesh.transport(rank, [PYTHON, '-c', inventory_code], control_transport=mode),
                                        label=f'mac-{rank + 1} probe/version check').stdout)
        mesh.require(inventory['probe'] == probe_hash, f'mac-{rank + 1}: probe differs; sync it first')
        versions.append(inventory)
        print(f'mac-{rank + 1}: identity and ports verified, MLX {inventory["mlx"]}', flush=True)
    mesh.topology_check(states)
    if mode == 'thunderbolt':
        from configure_thunderbolt_control import verify_coordinator
        verify_coordinator()
    mesh.require(all(v == versions[0] for v in versions), 'MLX or mlx-metal versions differ')
    (audit / 'preflight.json').write_text(json.dumps(dict(states=states, versions=versions), indent=2))
    with socket.socket() as sock:
        sock.bind((network['listen'], 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, PATH=str(CONFIG / 'bin') + os.pathsep + os.environ.get('PATH', ''))
    env['SML_JACCL_CONTROL_TRANSPORT'] = mode
    env.pop('MLX_METAL_FAST_SYNCH', None)
    env.pop('MLX_JACCL_RING', None)
    print(f'Launching strict four-rank collectives on coordinator port {port}...', flush=True)
    log = audit / 'collectives.log'
    # Each rank has a 60-second watchdog; the outer deadline also bounds SSH/launcher failures.
    with log.open('w') as output:
        process = subprocess.Popen(command(port, mode), env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = process.wait(timeout=120)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    text = log.read_text()
    print(text, end='')
    success = passed(result, text)
    (audit / 'result.json').write_text(json.dumps(dict(passed=success, returncode=result,
                                                      backend='jaccl', stream='cpu', world=4), indent=2))
    mesh.require(success, f'JACCL test did not pass on all four ranks; see {log}')
    print(f'PASS: all four ranks completed strict CPU JACCL collectives. Log: {log}')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
