import hashlib,json
from pathlib import Path
def read_json(p):return json.loads(Path(p).read_text())
def file_sha256(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
