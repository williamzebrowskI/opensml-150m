"""Check the packaged source snapshot without training or inference."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'provenance/SNAPSHOT_MANIFEST.json').read_text())
for rel,expected in manifest['packaged_files'].items():
 p=ROOT/rel
 if not p.is_file() or p.is_symlink():raise SystemExit('Missing/unsafe file: '+rel)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise SystemExit('Changed file: '+rel)
print(f"Verified {len(manifest['packaged_files'])} packaged files.")
