#!/usr/bin/env python3
"""Read-only Ethernet bootstrap checks against handoff-pinned host keys.

Writes public host keys and audit results locally; never accepts an unpinned
key, installs credentials, modifies a peer, or starts training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import base64
import json
from pathlib import Path
import subprocess

PEERS = (
    (2, '192.168.12.117', 'SHA256:v45WFZM77qEGF9ZuVXV1Fj7llE/XdXUOqTwDCVDeAhE'),
    (3, '192.168.12.102', 'SHA256:oLLqAAJ679URqDBNzcrF7ebhPAduCHlIRa7lmlcOM2I'),
    (4, '192.168.12.167', 'SHA256:8wdYLC64yOk+4wt9YQIf5U8eH+sUH2c0hKFBF4+1Z1s'),
    (5, '192.168.12.193', 'SHA256:tty3pv5axM9NMBgu1Wz5BablZBwv/XUHffi+UqcdkPY'),
)
IDENTITY = Path.home() / '.ssh/id_ed25519_opensml_five_mac'


def run(argv, **kwargs):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=60, **kwargs)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(returncode=None, stdout='', stderr=str(exc))


def pinned_host(peer, output):
    label, ip, expected = peer
    result = dict(label=f'new-mac-{label}', ip=ip, expected_fingerprint=expected)
    route = run(['/sbin/route', '-n', 'get', ip])
    result['route'] = route
    if route['returncode'] != 0 or 'interface: en0' not in route['stdout']:
        result['error'] = 'Peer route is not verified Ethernet en0'
        return result, None
    scan = run(['/usr/bin/ssh-keyscan', '-T', '5', '-t', 'ed25519', ip])
    keys = []
    for line in scan['stdout'].splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[0] != ip or parts[1] != 'ssh-ed25519':
            continue
        digest = 'SHA256:' + base64.b64encode(hashlib.sha256(base64.b64decode(parts[2])).digest()).decode().rstrip('=')
        if digest != expected:
            result['error'] = 'Host key differs from trusted handoff fingerprint'
            return result, None
        keys.append(line)
    if not keys:
        result['error'] = 'No pinned host key received'
        result['scan'] = scan
        return result, None
    known = output / f'mac{label}_known_hosts'
    known.write_text('\n'.join(sorted(set(keys))) + '\n')
    result['host_key_verified'] = True
    return result, known


def ssh_options(known, *, batch=True):
    local = run(['/usr/sbin/ipconfig', 'getifaddr', 'en0'])
    if local['returncode'] != 0 or not local['stdout'].strip():
        raise RuntimeError('Local Ethernet address unavailable')
    options = [
        '/usr/bin/ssh', '-F', '/dev/null', '-o', f'BatchMode={"yes" if batch else "no"}',
        '-o', 'ConnectTimeout=8', '-o', 'StrictHostKeyChecking=yes',
        '-o', f'UserKnownHostsFile={known}', '-o', 'GlobalKnownHostsFile=/dev/null',
        '-o', 'HostKeyAlgorithms=ssh-ed25519', '-o', 'ForwardAgent=no',
        '-b', local['stdout'].strip()]
    if IDENTITY.exists():
        options += ['-i', str(IDENTITY), '-o', 'IdentitiesOnly=yes']
    return options


def inspect(peer, output):
    label, ip, _ = peer
    result, known = pinned_host(peer, output)
    if known is None:
        return result
    collector = Path(__file__).with_name('inspect_mac_network.py').read_text()
    result['ssh'] = run(ssh_options(known) + [f'williamzebrowski@{ip}',
        str(Path(__file__).resolve().parents[2] / '.venv/bin/python'), '-',
        '--label', f'new-mac-{label}', '--stdout'], input=collector)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda peer: inspect(peer, output), PEERS))
    (output / 'results.json').write_text(json.dumps(dict(
        collected_utc=datetime.now(timezone.utc).isoformat(), peers=results), indent=2) + '\n')
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k not in ('route', 'scan', 'ssh')}))
        if 'ssh' in result:
            print('SSH exit:', result['ssh']['returncode'])
            if result['ssh']['returncode']:
                print(result['ssh']['stderr'].strip())
    return 0 if all(r.get('ssh', {}).get('returncode') == 0 for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
