#!/usr/bin/env python3
"""Authorize this M5's dedicated SSH public key on four pinned Ethernet peers.

Plan-only by default. Run --run in a normal Terminal; SSH asks for peer login
passwords directly at the terminal. Never pass passwords through this script.
No network settings, hostnames, training state, or existing keys are replaced.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

from five_mac_topology import IDENTITIES
from inspect_five_mac_ssh import IDENTITY, PEERS, inspect, pinned_host, run, ssh_options

ROOT = Path(__file__).resolve().parents[1]
REMOTE_PYTHON = str(Path(__file__).resolve().parents[2] / '.venv/bin/python')
INSTALL_KEY = '''import os, pathlib, sys
key = sys.stdin.read().strip()
parts = key.split()
if len(parts) != 3 or parts[0] != 'ssh-ed25519' or parts[2] != 'opensml-five-mac-m5':
    raise SystemExit('Unexpected public key format')
root = pathlib.Path.home() / '.ssh'
if root.is_symlink(): raise SystemExit('Refusing symlinked SSH directory')
root.mkdir(mode=0o700, exist_ok=True)
path = root / 'authorized_keys'
if path.is_symlink(): raise SystemExit('Refusing symlinked authorized_keys')
existing = path.read_text() if path.exists() else ''
if not any(line.split()[0:2] == parts[0:2] for line in existing.splitlines()):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, 'w') as stream:
        if existing and not existing.endswith('\\n'): stream.write('\\n')
        stream.write(key + '\\n')
print('Dedicated M5 public key authorized; existing keys preserved')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    print(f'Dedicated local key: {IDENTITY}')
    print('Creates an Ed25519 key without a passphrase for unattended cluster SSH, if absent.')
    print('Only the public key is sent to each peer; private key stays on this M5.')
    for label, ip, _ in PEERS:
        print(f'Mac-{label}: williamzebrowski@{ip} over Ethernet; pinned host key required')
    if not args.run:
        print('Plan only. Run with --run in Terminal to authorize peers interactively.')
        return 0
    if not sys.stdin.isatty():
        parser.error('--run requires an interactive Terminal for password prompts')
    local = run(['/usr/bin/ssh-keygen', '-lf', '/etc/ssh/ssh_host_ed25519_key.pub'])
    if local['returncode'] != 0 or local['stdout'].split()[1] != IDENTITIES[0][2]:
        parser.error('This is not the handoff-pinned M5; refusing credential setup')
    output = ROOT / 'diagnostics/m5_migration_20260922' / datetime.now().strftime('ssh_setup_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    verified = []
    for peer in PEERS:
        result, known = pinned_host(peer, output)
        if known is None:
            parser.error(f'Mac-{peer[0]} identity/route verification failed: {result.get("error")}')
        verified.append((peer, known))
    if IDENTITY.is_symlink() or IDENTITY.with_suffix('.pub').is_symlink():
        parser.error('Refusing symlinked dedicated key')
    IDENTITY.parent.mkdir(mode=0o700, exist_ok=True)
    if not IDENTITY.exists():
        if IDENTITY.with_suffix('.pub').exists():
            parser.error('Public key exists without private key; refusing replacement')
        subprocess.run(['/usr/bin/ssh-keygen', '-q', '-t', 'ed25519', '-N', '',
                        '-C', 'opensml-five-mac-m5', '-f', str(IDENTITY)], check=True)
    public = subprocess.check_output(['/usr/bin/ssh-keygen', '-y', '-P', '', '-f', str(IDENTITY)], text=True)
    # ssh-keygen versions may include the comment with -y; normalize it.
    public = ' '.join(public.split()[:2]) + ' opensml-five-mac-m5\n'
    for peer, known in verified:
        label, ip, _ = peer
        print(f'\nAuthorizing Mac-{label} ({ip}). Enter that Mac\'s login password if prompted.', flush=True)
        subprocess.run(ssh_options(known, batch=False) + [f'williamzebrowski@{ip}',
                       shlex.join([REMOTE_PYTHON, '-c', INSTALL_KEY])],
                       input=public, text=True, check=True)
    results = [inspect(peer, output) for peer in PEERS]
    (output / 'results.json').write_text(json.dumps(dict(peers=results), indent=2) + '\n')
    for result in results:
        status = result.get('ssh', {}).get('returncode')
        print(f'{result["label"]}: authenticated inventory {"passed" if status == 0 else "FAILED"}')
        if status == 0:
            report = json.loads(result['ssh']['stdout'])
            (output / (result['label'] + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(f'Audit: {output}')
    return 0 if all(r.get('ssh', {}).get('returncode') == 0 for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
