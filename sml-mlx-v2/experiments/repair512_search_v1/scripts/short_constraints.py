"""Short complete public SmolTalk targets with parent replay and family coverage."""
import random,re,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];PUBLIC=ROOT/'experiments/public_repair_384_v1'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PUBLIC/'scripts'))
import data as rules
from data_focused import parse,verify
from sml_v2.common import atomic_json,read_json,fingerprint,file_sha256
from sml_v2.tokenization import Tokenizer

def select(rows,n):
 counts=Counter();chosen=[];left=list(rows)
 for _ in range(n):
  assert left,'Insufficient short instruction records'
  index=min(range(len(left)),key=lambda i:sum(counts[v['family']] for v in left[i]['named_checks'])/len(left[i]['named_checks']))
  r=left.pop(index);chosen.append(r);counts.update(v['family'] for v in r['named_checks'])
 return chosen

def prepare(c,cfg):
 import pyarrow.parquet as pq
 tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');raw=read_json(PUBLIC/'data/prepared.json')['data'];pins=read_json(c/'source_pins.json');pin=pins['constraints'];path=c/'raw'/pin['local_file'];assert file_sha256(path)==pin['sha256']
 denied,dgrams=rules.exclusions();held=set();prior=set();seen=set();rejected=Counter();buckets=defaultdict(list)
 for path in [PUBLIC/'data/prepared.json',ROOT/'sft/conversation_foundation_v1/prepared.json',ROOT/'sft/text_followup_512_v1/prepared.json',ROOT/'experiments/unified_text_v2_sft_v1/data/prepared.json']:
  d=read_json(path)['data']
  for split in ('train','dev','test'):
   for r in d[split]:
    prior.add(rules.group_id(r['messages']))
    if split!='train':held.update(g for m in r['messages'] if m['role']=='user' for g in rules.grams(m['content']))
 for i,r in enumerate(pq.read_table(c/'raw'/pin['local_file']).to_pylist()):
  messages=r['messages'];prompt=messages[0]['content'];answer=messages[-1]['content'];checks=parse(prompt)
  if not checks or len(messages)!=2 or not re.search(r'\?|\b(?:explain|describe|discuss|draft|summarize|rewrite|write (?:a|an|the)|create (?:a|an|the))\b',prompt,re.I):rejected['unsupported_or_no_task']+=1;continue
  if re.search(r'crafted to|specified constraints|given instructions|here is my response|meet (?:all )?(?:the )?constraints|ensure that the word|this (?:part of the )?response',answer,re.I):rejected['meta_answer']+=1;continue
  if re.search(r'\b(?:diagnosis|dosage|medical|arthritis|cancer|curcumin|treatment|investment|lawyer)\b',prompt,re.I):rejected['specialist_scope']+=1;continue
  group=rules.group_id(messages)
  if group in seen or group in prior:rejected['duplicate']+=1;continue
  if rules.basic_quality(messages):rejected['quality']+=1;continue
  if any(rules.norm(m['content']) in denied or rules.grams(m['content'])&dgrams for m in messages):rejected['benchmark_overlap']+=1;continue
  if rules.grams(prompt)&held:rejected['heldout_overlap']+=1;continue
  if not all(verify(answer,checks)):rejected['named_constraint_failure']+=1;continue
  row=dict(id=fingerprint(messages),group=group,source='instructions',dataset_category='constraints',messages=messages,named_checks=checks,origin=dict(dataset=pin['repo'],file=pin['file'],row_index=i,split='train'))
  try:enc=rules.turns(tok,row,2048)
  except OverflowError:rejected['overflow']+=1;continue
  if max(e['targets'] for e in enc)>128:rejected['long_target']+=1;continue
  seen.add(group);bucket=int(group[:8],16)%20;split='dev' if bucket==0 else 'test' if bucket==1 else 'train';buckets[split].append(row)
 selected={};n=cfg['updates']*cfg['fresh_per_batch']
 for split in ('train','dev','test'):
  rows=buckets[split];random.Random(2026100123).shuffle(rows);selected[split]=select(rows,n if split=='train' else 32)
 rng=random.Random(cfg['seed']);replay={s:[r for r in raw['train'][:2048] if r['source']==s] for s in ('chat','human','grounded','instructions')}
 for rr in replay.values():rng.shuffle(rr)
 count=(16-cfg['fresh_per_batch'])//4;train=[]
 for u in range(cfg['updates']):
  batch=selected['train'][u*cfg['fresh_per_batch']:(u+1)*cfg['fresh_per_batch']]
  for rr in replay.values():batch += [rr[(u*count+j)%len(rr)] for j in range(count)]
  rng.shuffle(batch);assert len(batch)==16;train+=batch
 assert not {r['group'] for r in train}&{r['group'] for split in ('dev','test') for r in raw[split]+selected[split]}
 encoded=[e for r in train for e in rules.turns(tok,r,cfg['context'])]
 assert all(e['y'][-1]==tok.eos and len(e['x'])==len(e['y'])<=cfg['context'] for e in encoded)
 cfg.update(training_counts=dict(Counter(r['source'] for r in train)),method='Ordinary assistant-only conversation-balanced CE + EOS; short verified public instruction targets with parent replay')
 atomic_json(c/'config.json',cfg);atomic_json(c/'data.json',dict(train=train,dev=raw['dev'],test=raw['test'],targeted_dev=selected['dev'],targeted_test=selected['test']))
 receipt=dict(training_conversations=len(train),unique_conversations=len({r['group'] for r in train}),fresh_public_examples=n,parent_replay_fraction=(16-cfg['fresh_per_batch'])/16,dataset_mix=dict(smol_constraints=n,parent_replay=len(train)-n),assistant_turns=len(encoded),fresh_dev=32,fresh_reserved_test=32,eligible={k:len(v) for k,v in buckets.items()},rejections=dict(rejected),constraint_families=dict(Counter(v['family'] for r in selected['train'] for v in r['named_checks'])),maximum_fresh_answer_tokens=128,whole_answers=True,targets_truncated=False,benchmark_examples_used_for_training=False,limitations='Only parsed named controls mechanically verified. Public answers are not guaranteed factually correct. Fresh constraint development differs from trial21; use paired parent/candidate comparison.')
 atomic_json(c/'selection.json',receipt);print('[prepared-short-constraints]',receipt,flush=True)

def assess(c,b,dataset,u):
 from sft.conversation_foundation_v1.engine import generate,repeated
 p=c/'runs/sft/evaluations'/f'targeted_{u:05d}.json'
 if p.exists():return
 answers=[];families=Counter();passes=Counter()
 for row in dataset['targeted_dev']:
  g=generate(b,row['messages'][:-1],384);ok=all(verify(g['text'],row['named_checks']))
  for check in row['named_checks']:families[check['family']]+=1;passes[check['family']]+=all(verify(g['text'],[check]))
  answers.append(dict(id=row['id'],reference=row['messages'][-1]['content'],prompt=row['messages'][0]['content'],correct=ok,repeated=repeated(g['tokens']),**g))
 metrics=dict(correct=sum(a['correct'] for a in answers),total=len(answers),accuracy=sum(a['correct'] for a in answers)/len(answers),stopped=sum(a['stop']=='eos' for a in answers)/len(answers),repeated=sum(a['repeated'] for a in answers)/len(answers),by_family={f:dict(passed=passes[f],total=n) for f,n in families.items()})
 atomic_json(p,dict(metrics=metrics,answers=answers,reserved_test_used=False));print('[fresh-constraint-eval]',u,metrics,flush=True)
