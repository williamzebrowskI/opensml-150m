"""Download official evaluation splits, validate every row, freeze cached inputs."""
import io,json,re,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from evaluation.full_benchmarks.spec import *
from sml_v1.common import file_sha256,atomic_json
from sml_v1.tokenization import Tokenizer

def download(url,target,sha=None):
 import requests
 if not target.exists():
  response=requests.get(url,timeout=(20,180));response.raise_for_status()
  target.parent.mkdir(parents=True,exist_ok=True)
  temp=target.with_suffix(target.suffix+'.download');temp.write_bytes(response.content)
  if sha and file_sha256(temp)!=sha:raise ValueError('Source checksum mismatch: '+url)
  temp.replace(target)
 if sha and file_sha256(target)!=sha:raise ValueError('Cached source checksum mismatch: '+str(target))
 return target

def preprocess(text):
 return re.sub(r'\[.*?\]','',text.strip().replace(' [title]','. ')).replace('  ',' ')

def convert(name,raw):
 rows=[]
 for index,r in enumerate(raw):
  if name.startswith('arc_'):
   choices=r['choices']['text'];gold=r['choices']['label'].index(r['answerKey'])
   prompt='Question: '+r['question']+'\nAnswer:'
  elif name=='piqa':
   choices=[r['sol1'],r['sol2']];gold=int(r['label']);prompt='Question: '+r['goal']+'\nAnswer:'
  elif name=='hellaswag':
   choices=[preprocess(x) for x in r['endings']];gold=int(r['label'])
   prompt=preprocess(r['activity_label']+': '+r['ctx_a']+' '+r['ctx_b'].capitalize())
  else:
   rows.append(dict(id=str(r['key']),**r));continue
  if not choices or not 0<=gold<len(choices) or any(not x for x in choices):raise ValueError('Malformed choice row')
  rows.append(dict(id=f'{name}:{index}',task=name,source_id=r.get('id',r.get('ind',index)),prompt=prompt,choices=choices,gold=gold))
 if len(rows)!=DATA[name]['rows'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Missing/duplicate source row')
 return rows

def prepare():
 import pyarrow.parquet as pq
 CACHE.mkdir(exist_ok=True)
 if (CACHE/'prepared.json').exists():
  saved=json.loads((CACHE/'prepared.json').read_text())
  for path,sha in saved['files'].items():
   if file_sha256(DIR/path)!=sha:raise ValueError('Prepared file changed: '+path)
  print('[prepared] Existing dataset cache verified.',flush=True);return saved
 tokenizer=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
 if tokenizer.fingerprint!=TOKENIZER:raise ValueError('Wrong tokenizer')
 files={};counts={};lengths={}
 for name,spec in DATA.items():
  url=f'https://huggingface.co/datasets/{spec["repo"]}/resolve/{spec["revision"]}/{spec["file"]}'
  path=download(url,CACHE/'raw'/(name+Path(spec['file']).suffix),spec['sha256'])
  raw=[json.loads(x) for x in path.read_text().splitlines()] if name=='ifeval' else pq.read_table(path).to_pylist()
  rows=convert(name,raw);maximum=0;limited=0
  for row in rows:
   if name=='ifeval':
    n=len(tokenizer.encode('User: '+row['prompt']+'\nAssistant:'))
    if n>=MAX_CONTEXT:raise ValueError('Official prompt cannot fit; no truncation: '+row['id'])
    maximum=max(maximum,n);limited+=n+GENERATION_LIMIT>MAX_CONTEXT
   else:
    head=tokenizer.encode(row['prompt'])
    for choice in row['choices']:
     ids=tokenizer.encode(row['prompt']+' '+choice)
     if ids[:len(head)]!=head or not len(head)<len(ids)<=MAX_CONTEXT:raise ValueError('Boundary/context violation: '+row['id'])
     maximum=max(maximum,len(ids))
  if name=='ifeval':
   if sum(len(r['instruction_id_list']) for r in rows)!=834:raise ValueError('IFEval instruction count mismatch')
  target=CACHE/(name+'.json');atomic_json(target,rows)
  for p in (path,target):files[str(p.relative_to(DIR))]=file_sha256(p)
  counts[name]=len(rows);lengths[name]=dict(maximum_tokens=maximum,generation_context_limited_prompts=limited)
  print('[dataset]',name,len(rows),'maximum tokens',maximum,'context-limited',limited,flush=True)
 punkt=download(f'https://raw.githubusercontent.com/nltk/nltk_data/{PUNKT_REVISION}/packages/tokenizers/punkt_tab.zip',CACHE/'punkt_tab.zip')
 with zipfile.ZipFile(punkt) as z:
  # Only text tables; no pickle deserialization and no archive path traversal.
  for item in z.infolist():
   if item.is_dir():continue
   rel=Path(item.filename)
   if rel.is_absolute() or '..' in rel.parts or rel.parts[0]!='punkt_tab':raise ValueError('Unsafe archive path')
   p=DIR/'_nltk_data/tokenizers'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(item))
   files[str(p.relative_to(DIR))]=file_sha256(p)
 files[str(punkt.relative_to(DIR))]=file_sha256(punkt)
 result=dict(datasets=DATA,harness=HARNESS,tokenizer=TOKENIZER,files=files,counts=counts,lengths=lengths,
             training=False,ifeval_split_note='Official evaluation split is named train; all rows remain evaluation-only.')
 atomic_json(CACHE/'prepared.json',result);return result
if __name__=='__main__':prepare()
