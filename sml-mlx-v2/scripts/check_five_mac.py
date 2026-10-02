#!/usr/bin/env python3
"""Run bounded native five-Mac correctness tests; never touch production state."""
import argparse
import hashlib
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

from five_mac_cluster import Cluster, ORDER, ROOT, PYTHON


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--network',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    args=p.parse_args()
    if not args.run:
        print('Plan only. --run executes five-rank RDMA checks with a five-minute timeout and tiny disposable checkpoints.')
        return
    cluster=Cluster(args.network)
    job=ROOT/'diagnostics/m5_migration_20260922'/datetime.now().strftime('collectives_%Y%m%d_%H%M%S')
    job.mkdir(parents=True,exist_ok=False)
    cluster.preflight(job)
    source=job/'input';source.mkdir()
    shutil.copytree(ROOT/'sml_v2',source/'sml_v2',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(Path(__file__).with_name('probe_five_mac.py'),source/'probe_five_mac.py')
    cluster.stage(source)
    cluster.launch(job,source,[PYTHON,'-u',str(source/'probe_five_mac.py'),'--job-dir',str(job)],timeout=360)
    receipts=[]
    for rank,physical in enumerate(ORDER):
        receipt=json.loads(cluster.remote(physical,['/bin/cat',str(job/f'probe_rank{rank}.json')]))
        if receipt.get('rank')!=rank or receipt.get('status')!='passed':raise RuntimeError('Missing rank success receipt')
        receipts.append(receipt)
    if len({r['checkpoint_digest'] for r in receipts})!=1:raise RuntimeError('Saved replicas differ')
    proof=dict(network=str(args.network.resolve()),receipts=receipts,job=str(job),
        modules={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'sml_v2').glob('*.py')})
    (job/'verified.json').write_text(json.dumps(proof,indent=2)+'\n')
    (args.network/'five_rank_correctness.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(f'Five-rank correctness PASSED: {job}',flush=True)


if __name__=='__main__':main()
