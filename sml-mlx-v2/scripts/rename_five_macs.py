#!/usr/bin/env python3
"""Rename the five pinned Macs. Run --run in Terminal for administrator prompts.

The physical labels follow the original handoff, not the cable sequence.
Uses Ethernet IPs throughout, saves old names, and preserves SSH identities.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

from five_mac_topology import IDENTITIES
from inspect_five_mac_ssh import PEERS, pinned_host, run, ssh_options

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT.parent / '.venv/bin/python')
FIELDS = ('ComputerName', 'LocalHostName', 'HostName')
READ_NAMES = '''import json,subprocess
out={}
for n in ('ComputerName','LocalHostName','HostName'):
 p=subprocess.run(['/usr/sbin/scutil','--get',n],text=True,capture_output=True)
 if p.returncode and not (n=='HostName' and 'not set' in p.stderr): raise SystemExit(p.stderr)
 out[n]=p.stdout.strip() if p.returncode==0 else None
p=subprocess.check_output(['/usr/bin/ssh-keygen','-lf','/etc/ssh/ssh_host_ed25519_key.pub'],text=True)
out['fingerprint']=p.split()[1]
print(json.dumps(out))
'''
WRITE_NAMES = '''import json,subprocess,sys
values=json.loads(sys.argv[1]); expected=sys.argv[2]
actual=subprocess.check_output(['/usr/bin/ssh-keygen','-lf','/etc/ssh/ssh_host_ed25519_key.pub'],text=True).split()[1]
if actual!=expected: raise SystemExit('Wrong physical Mac; refusing rename')
for field,value in values.items():
 if field not in ('ComputerName','LocalHostName','HostName'): raise SystemExit('Invalid name field')
 subprocess.run(['/usr/sbin/scutil','--set',field,value or ''],check=True)
print('Names updated:',json.dumps(values))
'''


def desired(physical):
    return dict(ComputerName=f'Mac-{physical+1}', LocalHostName=f'mac-{physical+1}', HostName=f'mac-{physical+1}')


def rename_steps(before):
    if all(all(before[i].get(k) == v for k, v in desired(i).items()) for i in range(5)):
        return []
    # The old M3 LocalHostName occupies mac-5 while old peer HostNames occupy
    # mac-2/3/4. Vacate that cycle first, then work back toward the new M5.
    temporary = dict.fromkeys(FIELDS, 'opensml-migration-m3-117')
    return [(1, temporary)] + [(i, desired(i)) for i in (4, 3, 2, 1, 0)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--rollback', type=Path, help='Explicitly restore names_before.json from this audit directory')
    args = parser.parse_args()
    for i, identity in enumerate(IDENTITIES):
        print(f'Mac-{i+1}: {identity[0]}; Ethernet hardware ID {identity[1]}')
    if not args.run:
        print('Plan only. Sets ComputerName, LocalHostName and HostName. No training or networking changes.')
        print('Run --run in normal Terminal. Administrator passwords are entered there, never logged.')
        return 0
    if not sys.stdin.isatty():
        parser.error('An interactive Terminal is required for administrator authentication')
    output = ROOT / 'diagnostics/m5_migration_20260922' / datetime.now().strftime('rename_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    known = {}
    for peer in PEERS:
        result, host = pinned_host(peer, output)
        if host is None:
            parser.error(str(result.get('error')))
        known[peer[0]-1] = host

    def transport(i, command, interactive=False):
        if not i:
            return command
        options = ssh_options(known[i])
        if interactive:
            options += ['-t']
        return options + ['williamzebrowski@'+PEERS[i-1][1], shlex.join(command)]

    def read_all():
        result = []
        for i in range(5):
            response = run(transport(i, [PYTHON, '-c', READ_NAMES]))
            if response['returncode'] != 0:
                raise RuntimeError(response['stderr'])
            value = json.loads(response['stdout'])
            if value['fingerprint'] != IDENTITIES[i][2]:
                raise RuntimeError(f'Mac-{i+1} identity differs; no rename permitted')
            result.append(value)
        return result

    before = read_all()
    (output / 'names_before.json').write_text(json.dumps(before, indent=2)+'\n')
    if args.rollback:
        restore = json.loads((args.rollback / 'names_before.json').read_text())
        if len(restore) != 5 or any(v['fingerprint'] != IDENTITIES[i][2] for i,v in enumerate(restore)):
            parser.error('Rollback identities differ')
        steps = [(i, {k: restore[i][k] for k in FIELDS}) for i in (0,1,2,3,4)]
    else:
        steps = rename_steps(before)
    (output / 'steps.json').write_text(json.dumps(steps, indent=2)+'\n')
    print(f'Original names saved in {output}', flush=True)
    for i, names in steps:
        print(f'\nMac-{i+1} ({IDENTITIES[i][0]}) → {names["ComputerName"]}. Enter its administrator password if prompted.', flush=True)
        subprocess.run(transport(i, ['/usr/bin/sudo', PYTHON, '-c', WRITE_NAMES,
                       json.dumps(names), IDENTITIES[i][2]], interactive=True), check=True)
    after = read_all()
    (output / 'names_after.json').write_text(json.dumps(after, indent=2)+'\n')
    expected = restore if args.rollback else [desired(i) for i in range(5)]
    if any(any((after[i].get(k) or '') != (expected[i].get(k) or '') for k in FIELDS) for i in range(5)):
        raise RuntimeError('A name did not persist exactly; inspect the audit before retrying')
    print('Verified all five Mac names. Cable order and training state unchanged.')
    print(f'Rollback command: {shlex.join([PYTHON,str(Path(__file__).resolve()),"--run","--rollback",str(output)])}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
