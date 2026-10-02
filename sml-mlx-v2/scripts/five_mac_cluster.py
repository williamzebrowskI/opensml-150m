"""Verified five-Mac staging and bounded process orchestration over Ethernet."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading

from configure_five_mac_network import ORDER, ROOT, PYTHON, collect_all, protected, read_plan
from five_mac_network import verify_after
from five_mac_topology import discover, probe
from inspect_five_mac_ssh import IDENTITY, PEERS, ssh_options


class Cluster:
    def __init__(self, network):
        self.network=Path(network).resolve()
        self.original,self.plan=read_plan(self.network)
        if not (self.network/'apply_verified.json').is_file():
            raise ValueError('Network plan has no successful five-Mac verification')

    def remote(self, physical, argv, *, timeout=120):
        if physical:
            argv=ssh_options(self.network/f'mac{physical+1}_known_hosts')+[
                'williamzebrowski@'+PEERS[physical-1][1],shlex.join(argv)]
        return subprocess.run(argv,capture_output=True,text=True,check=True,timeout=timeout).stdout

    def preflight(self, job):
        reports=collect_all(job/'preflight')
        topology=discover(reports,ORDER)
        for physical,r in enumerate(reports):
            verify_after(r,self.plan['nodes'][physical],runtime=True)
            if protected(r)!=protected(self.original[physical]):
                raise ValueError('Protected Ethernet/Wi-Fi settings changed since the network plan')
            if r['packages']!=reports[0]['packages']:
                raise ValueError('Runtime package versions differ across Macs')
            for e in self.plan['nodes'][physical]['endpoints']:
                code="import subprocess,json; ip="+repr(e['peer_ip'])+"; print(subprocess.check_output(['/sbin/route','-n','get',ip],text=True)); subprocess.run(['/sbin/ping','-c','2','-W','1000','-S',"+repr(e['ip'])+",ip],check=True)"
                answer=self.remote(physical,[PYTHON,'-c',code],timeout=15)
                if 'interface: '+e['interface'] not in answer:
                    raise ValueError('Training link is routed over the wrong interface')
            # Verify coordinator control routing on the peer itself, not just the M5.
            if physical:
                answer=self.remote(physical,['/sbin/route','-n','get',probe(reports[0],'ethernet_ip').strip()])
                if 'interface: en0' not in answer:
                    raise ValueError('Peer coordinator route is not Ethernet')
        (job/'preflight.json').write_text(json.dumps(reports,indent=2)+'\n')
        print('[preflight] five identities, RDMA, packages, idle workers and all ten directed links verified',flush=True)
        return topology

    def stage(self, source):
        files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in source.rglob('*') if p.is_file()}
        (source/'stage_hashes.json').write_text(json.dumps(files,sort_keys=True)+'\n')
        code="from pathlib import Path; import hashlib,json; p=Path("+repr(str(source))+"); m=json.loads((p/'stage_hashes.json').read_text()); assert all((p/n).is_file() and not (p/n).is_symlink() and hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in m.items()); print('Staged inputs verified')"
        def copy(physical):
            if physical:
                self.remote(physical,['/bin/mkdir','-p',str(source)])
                ssh=shlex.join(ssh_options(self.network/f'mac{physical+1}_known_hosts'))
                subprocess.run(['/usr/bin/rsync','-a','--partial','-e',ssh,str(source)+'/',
                    'williamzebrowski@'+PEERS[physical-1][1]+':'+str(source)+'/'],check=True,timeout=900)
            self.remote(physical,[PYTHON,'-c',code],timeout=180)
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(copy,range(5)))
        print('[stage] all input hashes verified on five Macs',flush=True)

    def launch(self, job, source, worker, *, timeout, stop_dir=None):
        topology=discover(json.loads((job/'preflight.json').read_text()),ORDER)
        local_ip=probe(self.original[0],'ethernet_ip').strip()
        cfg=[]
        for physical in range(1,5):
            cfg += [f'Host opensml-five-mac-{physical+1}',
                f'  HostName {PEERS[physical-1][1]}','  User williamzebrowski',
                f'  IdentityFile {IDENTITY}','  IdentitiesOnly yes','  BatchMode yes',
                '  StrictHostKeyChecking yes',f'  UserKnownHostsFile {self.network/f"mac{physical+1}_known_hosts"}',
                '  GlobalKnownHostsFile /dev/null','  HostKeyAlgorithms ssh-ed25519',
                f'  BindAddress {local_ip}','  ForwardAgent no','  ServerAliveInterval 15','  ServerAliveCountMax 12']
        (job/'ssh_config').write_text('\n'.join(cfg)+'\n')
        hosts=dict(backend='jaccl-ring',hosts=[dict(ssh='127.0.0.1' if not physical else f'opensml-five-mac-{physical+1}',
                ips=['127.0.0.1'] if not physical else [],rdma=topology['rdma'][logical])
                for logical,physical in enumerate(ORDER)])
        (job/'hosts.json').write_text(json.dumps(hosts,indent=2)+'\n')
        bindir=job/'bin';bindir.mkdir()
        wrapper=bindir/'ssh'
        shutil.copyfile(Path(__file__).with_name('five_mac_transport.py'),job/'five_mac_transport.py')
        wrapper.write_text('#!/bin/sh\nexec '+shlex.join([PYTHON,str(job/'five_mac_transport.py')])+' "$@"\n')
        wrapper.chmod(0o755)
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        env=dict(os.environ)
        for k in list(env):
            if k.startswith(('MLX_','JACCL_','SML_JACCL_','SML5_')) or k=='PYTHONPATH':env.pop(k)
        env.update(SML5_JOB=str(job),SML5_PORT=str(port),SML5_SSH_CONFIG=str(job/'ssh_config'),PATH=str(bindir)+os.pathsep+env['PATH'])
        command=[str(ROOT.parent/'.venv/bin/mlx.launch'),'--backend','jaccl-ring','--hostfile',str(job/'hosts.json'),
                 '--cwd',str(source),'--starting-port',str(port),'--',*worker]
        (job/'launch.json').write_text(json.dumps(dict(command=command,physical_order=ORDER,timeout=timeout),indent=2)+'\n')
        process=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        timed_out=threading.Event()
        def kill():
            if process.poll() is None:
                timed_out.set()
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
        timer=threading.Timer(timeout,kill) if timeout else None
        if timer:timer.daemon=True;timer.start()
        def stop(sig,frame):
            if stop_dir:
                (stop_dir/'STOP').touch()
                print('[stop] Save/stop requested; waiting for all five workers',flush=True)
            else:kill()
        handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
        def cleanup(physical):
            logical=ORDER.index(physical)
            code=f'''import json,os,signal,subprocess
from pathlib import Path
p=Path({str(job/f'worker_rank{logical}.json')!r})
if p.exists():
 d=json.loads(p.read_text()); pid=d.get('pid',0)
 if type(pid)==int and pid>1:
  r=subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
  if r.returncode==0 and {str(job)!r} in r.stdout and any(s in r.stdout for s in ('-m sml_v2.pretrain','probe_five_mac.py','-m sft.train','-m sft.probe')): os.kill(pid,signal.SIGKILL)
'''
            return self.remote(physical,[PYTHON,'-c',code],timeout=30)
        try:
            failed_rank=False
            with (job/'train.log').open('w') as log:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                    if re.search(r'Node with rank \d+ exited with code (?!0\b)',line):failed_rank=True
            status=process.wait()
            if timed_out.is_set() or status or failed_rank:
                raise RuntimeError(f'Five-Mac job failed: exit={status}, timed_out={timed_out.is_set()}')
        finally:
            if process.poll() is None:kill()
            if timer:timer.cancel()
            with ThreadPoolExecutor(max_workers=5) as pool:
                for f in [pool.submit(cleanup,i) for i in range(5)]:
                    try:f.result()
                    except Exception as e:print(f'[cleanup-unconfirmed] {e}',flush=True)
            for sig,handler in handlers.items():signal.signal(sig,handler)
