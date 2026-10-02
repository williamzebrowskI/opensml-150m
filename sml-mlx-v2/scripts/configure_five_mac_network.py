#!/usr/bin/env python3
"""Prepare/review/apply a reversible five-Mac runtime network plan.

--prepare collects fresh evidence and writes a plan. --plan DIR prints it.
--plan DIR --apply requires an interactive Terminal for sudo authentication.
Only verified Thunderbolt bridge membership and IPv4 aliases are changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from five_mac_network import interfaces, matches_before, plan, verify_after
from five_mac_topology import discover, ports, probe, validate_inventory
from inspect_five_mac_ssh import PEERS, inspect, pinned_host, run, ssh_options
from inspect_mac_network import collect

ORDER = [0,1,2,4,3]  # User accepted the discovered cable order on 2026-09-22.
ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT.parent / '.venv/bin/python')
MODULES = ('configure_five_mac_network.py','five_mac_network.py','five_mac_topology.py',
           'inspect_five_mac_ssh.py','inspect_mac_network.py')


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def read_plan(directory):
    reports = json.loads((directory/'inventories.json').read_text())
    value = json.loads((directory/'plan.json').read_text())
    if value != json.loads(json.dumps(plan(reports, ORDER))):
        raise ValueError('Plan differs from independently reconstructed, identity-pinned actions')
    return reports, value


def protected(report):
    devs = interfaces(probe(report,'interfaces'))
    default = probe(report,'default_route')
    return dict(ethernet=devs['en0'], wifi=devs.get('en1'),
                default=[line.strip() for line in default.splitlines() if 'gateway:' in line or 'interface:' in line])


def local_worker(directory, physical, action):
    hashes=json.loads((directory/'tools_sha256.json').read_text())
    if set(hashes) != set(MODULES) or any(hashlib.sha256((directory/'tools'/n).read_bytes()).hexdigest()!=v for n,v in hashes.items()):
        raise ValueError('Staged worker code checksum mismatch')
    reports, value = read_plan(directory)
    node = value['nodes'][physical]
    before = collect(f'network-worker-mac-{physical+1}')
    validate_inventory(before, physical)
    if ports(before) != ports(reports[physical]):
        raise ValueError('Thunderbolt peer/controller identity changed since the plan')
    if protected(before) != protected(reports[physical]):
        raise ValueError('Ethernet, Wi-Fi or default route changed since the plan')
    if action == 'rollback':
        if os.geteuid() != 0:
            raise ValueError('Administrator privileges required')
        receipt = directory/f'applied_mac{physical+1}.json'
        if not receipt.exists():
            raise ValueError('No successful local apply receipt; inspect partial logs before rollback')
        verify_after(before, node)
        for command in node['rollback']:
            subprocess.run(command, check=True)
        after = collect(f'rolled-back-mac-{physical+1}')
        if not matches_before(after, node) or protected(after) != protected(before):
            raise ValueError('Rollback did not reproduce the recorded network state')
        save(directory/f'rollback_mac{physical+1}.json',after)
        print(f'Mac-{physical+1}: rollback verified')
        return
    try:
        verify_after(before, node)
        print(f'Mac-{physical+1}: configuration already matches plan')
        return
    except ValueError:
        pass
    if not matches_before(before,node):
        raise ValueError('Port/bridge state changed since the plan; regenerate the plan without applying it')
    if action == 'check':
        print(f'Mac-{physical+1}: live precondition checks passed')
        return
    if os.geteuid() != 0:
        raise ValueError('Administrator privileges required')
    executed = []
    try:
        for command in node['commands']:
            subprocess.run(command, check=True)
            executed.append(command)
        after = collect(f'configured-mac-{physical+1}')
        verify_after(after,node)
        if protected(after) != protected(before):
            raise ValueError('Ethernet, Wi-Fi or default route changed during application')
        save(directory/f'applied_mac{physical+1}.json',dict(before=before,after=after,executed=executed))
        print(f'Mac-{physical+1}: training IPv4 and bridge settings verified; Ethernet/Wi-Fi/default route unchanged')
    except BaseException:
        recovery = []
        for command in node['rollback']:
            p=subprocess.run(command,capture_output=True,text=True)
            recovery.append(dict(command=command,returncode=p.returncode,stderr=p.stderr))
        save(directory/f'failed_mac{physical+1}.json',dict(before=before,executed=executed,recovery=recovery))
        raise


def collect_all(directory):
    directory.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        local = pool.submit(collect,'new-mac-1-m5')
        remote = list(pool.map(lambda peer:inspect(peer,directory),PEERS))
        reports=[local.result()]
    for i,result in enumerate(remote,start=2):
        if result.get('ssh',{}).get('returncode') != 0:
            raise RuntimeError(f'Mac-{i}: inventory failed: {result}')
        reports.append(json.loads(result['ssh']['stdout']))
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--rollback',action='store_true')
    parser.add_argument('--worker',type=int,choices=range(5),help=argparse.SUPPRESS)
    parser.add_argument('--action',choices=('check','apply','rollback'),default='check',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker is not None:
        return local_worker(args.plan.resolve(),args.worker,args.action)
    if args.apply and args.rollback:
        parser.error('Choose apply or rollback')
    if args.prepare:
        if args.plan or args.apply or args.rollback:
            parser.error('--prepare only collects evidence and constructs a new plan')
        directory=ROOT/'diagnostics/m5_migration_20260922'/datetime.now().strftime('network_%Y%m%d_%H%M%S')
        directory.mkdir(parents=True,exist_ok=False)
        reports=collect_all(directory)
        value=plan(reports,ORDER)
        save(directory/'inventories.json',reports)
        save(directory/'plan.json',value)
        code=directory/'tools';code.mkdir()
        for name in MODULES:
            shutil.copyfile(Path(__file__).with_name(name),code/name)
        save(directory/'tools_sha256.json',{n:hashlib.sha256((code/n).read_bytes()).hexdigest() for n in MODULES})
    elif args.plan:
        directory=args.plan.resolve()
        reports,value=read_plan(directory)
    else:
        parser.error('Use --prepare or --plan DIR')
    print(f'Plan directory: {directory}')
    for node in value['nodes']:
        print(f'\nMac-{node["physical"]+1}:')
        for c in node['commands']:
            print('  '+shlex.join(c))
    print('\nEthernet SSH and loopback-only coordinator tunnels; no forwarding or default-route changes.')
    print('Runtime settings only. A reboot requires checking/reapplying the network; production stays stopped.')
    if not (args.apply or args.rollback):
        print('Review only. No network settings changed.')
        return
    if not sys.stdin.isatty():
        parser.error('Run in normal Terminal for administrator password prompts')
    hashes=json.loads((directory/'tools_sha256.json').read_text())
    if set(hashes) != set(MODULES) or any(hashlib.sha256((directory/'tools'/n).read_bytes()).hexdigest()!=v for n,v in hashes.items()):
        raise ValueError('Staged worker code changed; prepare a new plan')
    known={}
    for peer in PEERS:
        result,path=pinned_host(peer,directory)
        if path is None: raise RuntimeError(result['error'])
        known[peer[0]-1]=path
    def transport(i,cmd,tty=False):
        if not i: return cmd
        return ssh_options(known[i])+(['-t'] if tty else [])+['williamzebrowski@'+PEERS[i-1][1],shlex.join(cmd)]
    # Copy only this new plan and its public worker scripts, never training state.
    for i in range(1,5):
        subprocess.run(transport(i,['/bin/mkdir','-p',str(directory/'tools')]),check=True)
        ssh=shlex.join(ssh_options(known[i]))
        for source in ('plan.json','inventories.json','tools_sha256.json','tools'):
            subprocess.run(['/usr/bin/rsync','-a','-e',ssh,str(directory/source),
                           'williamzebrowski@'+PEERS[i-1][1]+':'+str(directory)+'/'],check=True)
    action='rollback' if args.rollback else 'apply'
    for i in range(5):
        cmd=[PYTHON,str(directory/'tools'/Path(__file__).name),'--plan',str(directory),'--worker',str(i)]
        if not args.rollback:
            subprocess.run(transport(i,cmd+['--action','check']),check=True)
    for i in range(5):
        print(f'\nMac-{i+1}: {action}. Enter its administrator password if prompted.',flush=True)
        cmd=['/usr/bin/sudo',PYTHON,str(directory/'tools'/Path(__file__).name),
             '--plan',str(directory),'--worker',str(i),'--action',action]
        subprocess.run(transport(i,cmd,tty=True),check=True)
    after=collect_all(directory/'after')
    discover(after,ORDER)
    for i,report in enumerate(after):
        if args.rollback:
            if not matches_before(report,value['nodes'][i]): raise ValueError('Rollback mismatch')
        else:
            verify_after(report,value['nodes'][i])
        if protected(report) != protected(reports[i]): raise ValueError('Protected network state changed')
    save(directory/f'{action}_verified.json',dict(reports=after,production_started=False))
    print(f'All five network configurations verified. RDMA collective tests still required. Audit: {directory}')


if __name__=='__main__':
    main()
