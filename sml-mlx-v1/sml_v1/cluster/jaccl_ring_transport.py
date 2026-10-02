"""Job-scoped coordinator forwarding over pinned Thunderbolt SSH sessions."""

import os
import json
from pathlib import Path
import re
import subprocess
import sys
import time

MODE = 'ring-thunderbolt'
PORT_ENV = 'SML_JACCL_RING_COORDINATOR_PORT'
JOB_ENV = 'SML_JACCL_RING_JOB_DIR'
ALIASES = tuple(f'sml-jaccl-ring-tb-mac-{i}' for i in (2, 3, 4))


def coordinator_port(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4,5}', value) or not 1024 <= int(value) <= 65535:
        raise ValueError('Missing or invalid job-scoped ring coordinator port')
    return int(value)


def ssh_command(arguments, environ, root):
    """mlx.launch uses -tt for workers, but not for its later cleanup SSH."""
    hosts = [arg for arg in arguments if arg in ALIASES]
    if len(hosts) != 1:
        raise ValueError('Wired ring SSH requires exactly one pinned physical Mac alias')
    forward = []
    if '-tt' in arguments:
        port = coordinator_port(environ.get(PORT_ENV))
        # The remote loopback listener dies with this worker's SSH session.
        # Mac-4's SSH session traverses Mac-2 using the pinned wired jump host.
        forward = ['-o', 'ExitOnForwardFailure=yes', '-R', f'127.0.0.1:{port}:127.0.0.1:{port}']
    return ['/usr/bin/ssh', '-F', str(Path(root) / 'cluster/jaccl/ssh_ring_thunderbolt_config'),
            '-o', 'LogLevel=ERROR', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=12',
            *forward, *arguments]


def check_coordinator(environ):
    endpoint = environ.get('MLX_JACCL_COORDINATOR', '')
    host, _, value = endpoint.rpartition(':')
    port = coordinator_port(value)
    if host != '127.0.0.1':
        raise ValueError('Wired ring coordinator must use the job-scoped loopback tunnel')
    if environ.get('JACCL_COORDINATOR', endpoint) != endpoint:
        raise ValueError('Conflicting legacy coordinator override')
    return port


def wait_for_coordinator(port, job, timeout=120):
    """Observe the root socket without consuming a native JACCL handshake."""
    if not isinstance(job, str) or not Path(job).is_absolute():
        raise ValueError('Missing absolute wired ring job directory')
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = Path(job) / 'worker_rank0.json'
        if record.exists():
            value = json.loads(record.read_text())
            pid = value.get('pid')
            if type(pid) is not int or pid <= 1 or value.get('output') != job:
                raise ValueError('Invalid root worker identity for coordinator readiness')
            result = subprocess.run(['/usr/sbin/lsof', '-nP', '-a', '-p', str(pid),
                                     '-iTCP:' + str(port), '-sTCP:LISTEN', '-Fn'],
                                    capture_output=True, text=True, timeout=5)
            listeners = [line[1:] for line in result.stdout.splitlines() if line.startswith('n')]
            if listeners:
                if listeners != [f'127.0.0.1:{port}']:
                    raise ValueError(f'Unexpected coordinator listener: {listeners}')
                return
        time.sleep(.2)
    raise RuntimeError('Root coordinator did not become ready; no remote workers launched')


if __name__ == '__main__':
    try:
        command = ssh_command(sys.argv[1:], os.environ, Path(__file__).resolve().parents[1])
        if '-tt' in sys.argv[1:]:
            wait_for_coordinator(coordinator_port(os.environ.get(PORT_ENV)), os.environ.get(JOB_ENV))
        os.execv(command[0], command)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
