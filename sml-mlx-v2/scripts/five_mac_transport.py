"""Job-scoped Ethernet SSH with loopback-only reverse coordinator tunnels."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ALIASES = {f'opensml-five-mac-{i}' for i in range(2,6)}


def command(arguments, environ):
    if len([a for a in arguments if a in ALIASES]) != 1:
        raise ValueError('Exactly one pinned five-Mac SSH alias required')
    config = Path(environ['SML5_SSH_CONFIG'])
    job = Path(environ['SML5_JOB'])
    port = environ['SML5_PORT']
    if not (config.is_absolute() and job.is_absolute() and config.parent == job
            and port.isdigit() and 1024 <= int(port) <= 65535):
        raise ValueError('Invalid job-scoped SSH configuration')
    forward = []
    if '-tt' in arguments:
        forward = ['-o','ExitOnForwardFailure=yes','-R',f'127.0.0.1:{port}:127.0.0.1:{port}']
    return ['/usr/bin/ssh','-F',str(config),'-o','LogLevel=ERROR',*forward,*arguments]


def wait_root(job, port):
    deadline = time.monotonic()+120
    while time.monotonic()<deadline:
        record = job/'worker_rank0.json'
        if record.exists():
            value = json.loads(record.read_text())
            if value.get('output') != str(job) or type(value.get('pid')) is not int or value['pid'] <= 1:
                raise ValueError('Invalid coordinator PID record')
            p = subprocess.run(['/usr/sbin/lsof','-nP','-a','-p',str(value['pid']),
                                '-iTCP:'+str(port),'-sTCP:LISTEN','-Fn'],capture_output=True,text=True,timeout=5)
            listeners = [l[1:] for l in p.stdout.splitlines() if l.startswith('n')]
            if listeners:
                if listeners != [f'127.0.0.1:{port}']:
                    raise ValueError('Coordinator listener is not loopback-only')
                return
        time.sleep(.2)
    raise RuntimeError('Coordinator did not listen within 120 seconds')


if __name__ == '__main__':
    try:
        cmd = command(sys.argv[1:], os.environ)
        if '-tt' in sys.argv[1:]:
            wait_root(Path(os.environ['SML5_JOB']),int(os.environ['SML5_PORT']))
        os.execv(cmd[0],cmd)
    except Exception as exc:
        print(f'Five-Mac SSH failed: {exc}',file=sys.stderr)
        raise SystemExit(2)
