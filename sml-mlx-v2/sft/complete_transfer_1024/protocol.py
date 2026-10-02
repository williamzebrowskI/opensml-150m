"""Freeze source, replay, curriculum and runtime before production."""
from pathlib import Path
from sml_v2.common import read_json,file_sha256,fingerprint
from sft.protected_update_1024.protocol import contract as old_contract,code_files as old_code
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent;OUTPUT=ROOT/'runs/sft_complete_transfer_1024_v1'
def code_files():return sorted(set(old_code()+list(DIR.glob('*.py'))))
def contract(cfg):
 oldcfg=read_json(ROOT/'sft/protected_update_1024/config.json');oldcfg['arm']='control';f=old_contract(oldcfg)
 if cfg['source_bundle']!=oldcfg['source_bundle'] or cfg['source_model_sha256']!=oldcfg['source_model_sha256']:raise ValueError('Wrong parent')
 if (cfg['source_step'],cfg['updates'])!=(1024,192):raise ValueError('Unexpected run length')
 if cfg['per_update']!={'commonsense':2,'instruction':4,'reading':2}:raise ValueError('Unexpected curriculum')
 sel=read_json(DIR/'selection.json')
 if sel['config']!=fingerprint(cfg):raise ValueError('Selection config changed')
 review=read_json(DIR/'review.json')
 if review['selection']!=fingerprint(sel) or review['status']!='sample-reviewed':raise ValueError('Sample review is stale')
 f['config']=fingerprint(cfg);f['selection']=fingerprint(sel)
 for name in ('config.json','selection.json','rejections.json','review.json'):
  p=DIR/name;f['protected'][str(p)]=file_sha256(p)
 f['code'].update({str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()})
 return f
from sft.protected_update_1024.protocol import verify
