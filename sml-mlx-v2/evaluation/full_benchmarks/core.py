"""Local inference, official IFEval scoring, and resumable evaluation records."""
import dataclasses,hashlib,importlib.metadata,json,math,os,random,sys
from pathlib import Path
from evaluation.full_benchmarks.spec import *
from sml_v2.common import atomic_json,file_sha256,fingerprint

def verify(hashes):
 for path,sha in hashes.items():
  if file_sha256(path)!=sha:raise ValueError('Frozen input changed: '+str(path))

def verify_architecture(contract,base):
 # Older runs pin this file in protected; newer runs pin it as architecture.
 # Require a saved hash, and enforce both pins when both formats are present.
 pins=[]
 if 'protected' in contract:
  if not isinstance(contract['protected'],dict):raise ValueError('Invalid protected input hashes')
  if str(base) in contract['protected']:pins.append(contract['protected'][str(base)])
 if 'architecture' in contract:pins.append(contract['architecture'])
 if not pins:raise ValueError('Run contract has no pinned base architecture metadata hash')
 actual=file_sha256(base)
 if any(pin!=actual for pin in pins):raise ValueError('Base architecture metadata changed')
 return actual

def runtime():
 sys.path.insert(0,str(DIR/'_runtime'));sys.path.insert(0,str(DIR/'vendor'))
 os.environ['NLTK_DATA']=str(DIR/'_nltk_data')
 # Do not permit the vendored verifier to trigger an implicit download.
 if not (DIR/'_nltk_data/tokenizers/punkt_tab/english').is_dir():raise ValueError('Missing tokenizer tables; run prepare.py')
 import nltk
 nltk.data.path[:]=[str(DIR/'_nltk_data')]
 from langdetect import DetectorFactory
 DetectorFactory.seed=0
 from lm_eval.tasks.ifeval import utils
 return utils

def get_rows(suite):
 names=MC if suite=='multiple-choice' else ('ifeval',)
 return [r for name in names for r in json.loads((CACHE/(name+'.json')).read_text())]

def frozen(model,suite):
 if suite=='ifeval' and model=='pretrained':raise ValueError('IFEval requires an SFT model with the trained chat format')
 prepared=json.loads((CACHE/'prepared.json').read_text())
 if prepared['datasets']!=DATA or prepared['tokenizer']!=TOKENIZER:raise ValueError('Prepared specification changed')
 verify({str(DIR/p):h for p,h in prepared['files'].items()})
 spec=MODELS[model];bundle=ROOT/spec['bundle'];weights=bundle/'model.safetensors'
 if file_sha256(weights)!=spec['sha256']:raise ValueError('Unexpected checkpoint weights')
 # Check the committed bundle, metadata, and optimizer bytes without loading an optimizer.
 manifest=json.loads((bundle/'manifest.json').read_text())
 protected={str(bundle/p):v['sha256'] for p,v in manifest['files'].items()}
 protected[str(bundle/'manifest.json')]=file_sha256(bundle/'manifest.json')
 protected.update({str(DIR/p):h for p,h in prepared['files'].items()})
 protected[str(CACHE/'prepared.json')]=file_sha256(CACHE/'prepared.json')
 verify(protected)
 meta=json.loads((bundle/'model.safetensors.json').read_text())
 if meta['step']!=spec['step']:raise ValueError('Wrong source step')
 if model!='pretrained' and meta['training_format']!='plain-user-assistant-eos-v1':raise ValueError('Wrong SFT format')
 base=ROOT/BASE/'model.safetensors.json'
 if model!='pretrained':
  contract=json.loads((bundle.parent/'contract.json').read_text())
  protected[str(base)]=verify_architecture(contract,base)
 else:protected[str(base)]=file_sha256(base)
 from sml_v2.tokenization import Tokenizer
 tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
 if tok.fingerprint!=TOKENIZER:raise ValueError('Tokenizer mismatch')
 for p in (ROOT/'tokenizer/bytebpe32k_v1').iterdir():
  if p.is_file():protected[str(p)]=file_sha256(p)
 code=list(DIR.glob('*.py'))+[ROOT/'sml_v2'/n for n in ('model.py','tokenization.py','common.py')]
 versions={p:importlib.metadata.version(p) for p in ('mlx','numpy','tokenizers','pyarrow')}
 if suite=='ifeval':
  vendor=json.loads((DIR/'vendor_sources.json').read_text())
  verify({str(DIR/p):v['sha256'] for p,v in vendor.items()})
  protected.update({str(DIR/p):v['sha256'] for p,v in vendor.items()})
  for p in (DIR/'vendor_sources.json',DIR/'requirements-ifeval.txt'):
   protected[str(p)]=file_sha256(p)
  runtime()
  for line in (DIR/'requirements-ifeval.txt').read_text().splitlines():
   package,wanted=line.split('==');actual=importlib.metadata.version(package)
   if actual!=wanted:raise ValueError('IFEval dependency changed: '+package)
   versions[package]=actual
  code+=list((DIR/'vendor').rglob('*.py'))
  # Freeze installed dependency sources and language profiles as well as versions.
  for p in (DIR/'_runtime').rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':protected[str(p)]=file_sha256(p)
 return dict(version=1,model=model,suite=suite,step=spec['step'],weights=str(weights),
             prepared_sha256=file_sha256(CACHE/'prepared.json'),protected=protected,
             code={str(p):file_sha256(p) for p in code},versions=versions,context=MAX_CONTEXT,
             fp32=True,attention='MLX explicit vanilla attention; PyTorch equivalence not claimed',ffn='reference',
             batch_size=1,zero_shot=True,training=False,
             protocol=dict(harness=HARNESS,mc_template='plain official question/continuation; no chat/BOS/EOS',
                           mc_metrics=['acc','acc_norm'],ifeval_template='User: {unchanged prompt}\nAssistant:',
                           max_new_tokens=GENERATION_LIMIT,greedy=True,repetition_penalty=1,
                           decode_mode=spec.get('decode_mode','cached'),
                           eos=tok.eos,extra_stop_strings=[],seed=SEED))

class Backend:
 def __init__(self,model):
  import mlx.core as mx
  from mlx.utils import tree_map
  from sml_v2.model import TransformerConfig,TransformerLM
  from sml_v2.tokenization import Tokenizer
  cfg=json.loads((ROOT/BASE/'model.safetensors.json').read_text())['recipe']['model']
  cfg=dict(cfg,attention_impl='vanilla',ffn_impl='reference',ce_impl='reference',loss_dtype='float32')
  self.model=TransformerLM(TransformerConfig(**cfg))
  self.model.load_weights(str(ROOT/MODELS[model]['bundle']/'model.safetensors'),strict=True)
  self.model.update(tree_map(lambda x:x.astype(mx.float32),self.model.parameters()))
  self.model.eval();mx.eval(self.model.parameters());mx.set_cache_limit(256*1024**2)
  self.tokenizer=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
  self.encode=self.tokenizer.encode;self.eos=self.tokenizer.eos
  self.decode_mode=MODELS[model].get('decode_mode','cached')
 def logits(self,ids):return self.model.logits(ids)[0]

def candidate_tokens(backend,prompt,choice):
 # Same joint-tokenization boundary policy as TemplateLM._encode_pair; this
 # tokenizer additionally verifies that the joint prompt prefix is unchanged.
 nspaces=len(prompt)-len(prompt.rstrip())
 continuation=' '+choice
 if nspaces:continuation=prompt[-nspaces:]+continuation;prompt=prompt[:-nspaces]
 head=backend.encode(prompt);ids=backend.encode(prompt+continuation)
 if not head or ids[:len(head)]!=head:raise ValueError('Tokenization boundary changed')
 if not len(head)<len(ids)<=MAX_CONTEXT:raise ValueError('Empty answer or context overflow; no truncation')
 if not choice:raise ValueError('Empty candidate')
 return ids,len(head)

def candidate_score(backend,prompt,choice):
 import mlx.core as mx
 ids,n=candidate_tokens(backend,prompt,choice)
 logits=backend.logits(mx.array([ids[:-1]],dtype=mx.int32))[0,n-1:].astype(mx.float32)
 labels=mx.array(ids[n:],dtype=mx.int32)
 logp=mx.take_along_axis(logits,labels[:,None],axis=-1).squeeze(-1)-mx.logsumexp(logits,axis=-1)
 total=float(logp.sum().item())
 if not math.isfinite(total) or total>0:raise ValueError('Invalid likelihood')
 return dict(log_likelihood=total,normalized=total/len(choice),answer_tokens=len(labels),characters=len(choice))

def score_mc(backend,row):
 scores=[candidate_score(backend,row['prompt'],c) for c in row['choices']]
 raw=max(range(len(scores)),key=lambda i:scores[i]['log_likelihood'])
 norm=max(range(len(scores)),key=lambda i:scores[i]['normalized'])
 return dict(id=row['id'],task=row['task'],gold=row['gold'],scores=scores,
             prediction=raw,prediction_norm=norm,correct=raw==row['gold'],correct_norm=norm==row['gold'])

def generate(backend,prompt,limit=GENERATION_LIMIT,stop_requested=lambda:False):
 import mlx.core as mx
 ids=backend.encode('User: '+prompt+'\nAssistant:')
 if not ids or len(ids)>=MAX_CONTEXT:raise ValueError('Prompt cannot fit; no truncation')
 budget=min(limit,MAX_CONTEXT-len(ids));outputs=[];reason='length';cache=None
 scores,cache=backend.model.logits(mx.array([ids],dtype=mx.int32));scores=scores[:,-1,:];mx.eval(scores,cache)
 try:
  for _ in range(budget):
   if stop_requested():raise InterruptedError('Interrupted during response; this prompt will resume from its start')
   token=int(mx.argmax(scores[0]).item())
   if token==backend.eos:reason='eos';break
   outputs.append(token)
   if len(outputs)<budget:
    if getattr(backend,'decode_mode','cached')=='full_context':
     cache=None
     scores=backend.logits(mx.array([ids+outputs],dtype=mx.int32))[:,-1,:];mx.eval(scores)
    else:
     scores,cache=backend.model.step(mx.array([[token]],dtype=mx.int32),caches=cache);mx.eval(scores,cache)
  if reason=='length' and budget<limit:reason='context_limit'
  return dict(text=backend.tokenizer.decode(outputs),tokens=outputs,generated_tokens=len(outputs),
              prompt_tokens=len(ids),effective_limit=budget,stop_reason=reason)
 finally:cache=None

def grade_ifeval(row,text):
 utils=runtime()
 inp=utils.InputExample(key=row['key'],instruction_id_list=row['instruction_id_list'],prompt=row['prompt'],kwargs=row['kwargs'])
 # Make a resumed evaluation identical even if a checker draws a default value.
 seed=SEED+int(row['key']);random.seed(seed)
 strict=utils.test_instruction_following_strict(inp,text)
 random.seed(seed);loose=utils.test_instruction_following_loose(inp,text)
 return dict(strict=dataclasses.asdict(strict),loose=dataclasses.asdict(loose))

class Journal:
 """Append-only item commits. Incomplete trailing writes are not treated as results."""
 def __init__(self,path,expected_ids):
  self.path=Path(path);self.expected_ids=list(expected_ids);self.records=[]
  if self.path.exists():
   raw=self.path.read_bytes();end=raw.rfind(b'\n')+1
   if end!=len(raw):
    # Preserve interrupted bytes for inspection before restoring the last commit.
    self.path.with_suffix('.incomplete-tail').write_bytes(raw[end:])
    with self.path.open('r+b') as f:f.truncate(end)
    raw=raw[:end]
   self.records=[json.loads(x) for x in raw.splitlines()]
  ids=[r['id'] for r in self.records]
  if ids!=self.expected_ids[:len(ids)] or len(ids)>len(self.expected_ids):raise ValueError('Result order/identity mismatch')
 def append(self,row):
  n=len(self.records)
  if n>=len(self.expected_ids) or row['id']!=self.expected_ids[n]:raise ValueError('Duplicate or out-of-order result')
  self.path.parent.mkdir(parents=True,exist_ok=True)
  with self.path.open('ab') as f:
   f.write((json.dumps(row,ensure_ascii=False)+'\n').encode());f.flush();os.fsync(f.fileno())
  self.records.append(row)

def summary(records,suite,expected):
 result=dict(status='complete' if len(records)==expected else 'partial',completed=len(records),expected=expected,training=False)
 if suite=='multiple-choice':
  result['tasks']={}
  for task in MC:
   rr=[r for r in records if r['task']==task]
   if rr:result['tasks'][task]=dict(completed=len(rr),expected=DATA[task]['rows'],acc=sum(r['correct'] for r in rr)/len(rr),acc_norm=sum(r['correct_norm'] for r in rr)/len(rr))
 else:
  for mode in ('strict','loose'):
   if records:
    prompt=sum(r[mode]['follow_all_instructions'] for r in records)
    total=sum(len(r[mode]['follow_instruction_list']) for r in records)
    correct=sum(sum(r[mode]['follow_instruction_list']) for r in records)
    result[mode]=dict(prompt_correct=prompt,prompt_total=len(records),prompt_accuracy=prompt/len(records),instruction_correct=correct,instruction_total=total,instruction_accuracy=correct/total)
  result['stop_counts']={k:sum(r['generation']['stop_reason']==k for r in records) for k in ('eos','length','context_limit')}
 return result
