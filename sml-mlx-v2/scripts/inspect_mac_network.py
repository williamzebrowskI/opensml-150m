#!/usr/bin/env python3
"""Collect local, read-only migration evidence; never configure or train."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def probe(argv, timeout=15):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return dict(command=argv, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(command=argv, returncode=None, stdout='', stderr=str(exc))


def collect(label):
    commands = {
        'computer_name': ['/usr/sbin/scutil', '--get', 'ComputerName'],
        'local_hostname': ['/usr/sbin/scutil', '--get', 'LocalHostName'],
        'macos': ['/usr/bin/sw_vers'],
        'chip': ['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'],
        'memory_bytes': ['/usr/sbin/sysctl', '-n', 'hw.memsize'],
        'ethernet_ip': ['/usr/sbin/ipconfig', 'getifaddr', 'en0'],
        'ethernet_identity': ['/sbin/ifconfig', 'en0'],
        'interfaces': ['/sbin/ifconfig'],
        'hardware_ports': ['/usr/sbin/networksetup', '-listallhardwareports'],
        'routes': ['/usr/sbin/netstat', '-rn', '-f', 'inet'],
        'default_route': ['/sbin/route', '-n', 'get', 'default'],
        'rdma': ['/usr/bin/rdma_ctl', 'status'],
        'power': ['/usr/bin/pmset', '-g', 'custom'],
        'ssh_host_fingerprint': ['/usr/bin/ssh-keygen', '-lf', '/etc/ssh/ssh_host_ed25519_key.pub'],
    }
    result = dict(format='opensml-readonly-mac-inventory-v1', label=label,
                  collected_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), architecture=platform.machine(),
                  network_changed=False, training_started=False, readiness_verified=False,
                  probes={key: probe(argv) for key, argv in commands.items()})
    result['probes']['thunderbolt'] = probe([
        '/usr/sbin/system_profiler', 'SPThunderboltDataType', '-json', '-timeout', '30'], timeout=45)
    result['probes']['ibv_devices'] = probe([shutil.which('ibv_devices') or '/usr/bin/ibv_devices'])
    result['packages'] = {}
    for name in ('mlx', 'mlx-metal', 'numpy', 'datasets', 'huggingface-hub', 'tokenizers'):
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
    processes = probe(['/bin/ps', '-axo', 'pid=,ppid=,command='])
    result['process_query_ok'] = processes['returncode'] == 0
    result['training_processes'] = []
    for row in processes['stdout'].splitlines():
        fields = row.strip().split(None, 2)
        if len(fields) != 3 or int(fields[0]) in (os.getpid(), os.getppid()):
            continue
        command = fields[2]
        if ' -c ' in command:
            continue
        if any(term in command for term in ('-m sml_v2.pretrain', '-m sft.train', '-m sft.probe', '-m train.pretrain_jaccl',
                   '-m train.jaccl_worker', 'mlx.launch', '/launch_adaptive.py',
                   '/launch_full.py', '/launch_pretrain.py')):
            result['training_processes'].append(row.strip())
    result['thunderbolt_interface_labels'] = dict(re.findall(
        r'Hardware Port: (Thunderbolt \d+)\nDevice: (\S+)', result['probes']['hardware_ports']['stdout']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True, help='Inventory label only; does NOT rename this Mac')
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--stdout', action='store_true', help='Print JSON, writing no files')
    target.add_argument('--output', type=Path, help='Write a new JSON report; refuse overwriting')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new report filename')
    result = collect(args.label)
    payload = json.dumps(result, indent=2) + '\n'
    if args.stdout:
        print(payload, end='')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(payload)
        print(f'Inventory: {args.output.resolve()}')
        print('Read-only evidence only. Not a cable/RDMA readiness test. No settings changed.')
        print(f"RDMA: {result['probes']['rdma']['stdout'].strip() or 'unknown'}")
        print(f"Training process query succeeded: {result['process_query_ok']}; matches: {len(result['training_processes'])}")


if __name__ == '__main__':
    main()
