"""Verify a downloaded training record archive without extracting or executing it."""
import argparse,hashlib,json,tarfile
from pathlib import Path

def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();m=json.loads(a.manifest.read_text())
 if sha256_file(a.archive)!=m['archive_sha256']:raise SystemExit('Archive checksum mismatch')
 expected={x['path']:x for x in m['files']}
 with tarfile.open(a.archive,'r:gz') as tf:
  members=tf.getmembers()
  if len(members)!=len(expected) or {x.name for x in members}!=set(expected):raise SystemExit('Archive member list mismatch')
  for entry in members:
   name=Path(entry.name)
   if name.is_absolute() or '..' in name.parts or not entry.isfile():raise SystemExit('Unsafe archive member: '+entry.name)
   r=expected[entry.name];h=hashlib.sha256();size=0
   with tf.extractfile(entry) as src:
    for b in iter(lambda:src.read(4*1024**2),b''):h.update(b);size+=len(b)
   if size!=r['bytes'] or h.hexdigest()!=r['sha256']:raise SystemExit('Member checksum mismatch: '+entry.name)
 print('Verified archive and',len(expected),'files; nothing was extracted.')
if __name__=='__main__':main()
