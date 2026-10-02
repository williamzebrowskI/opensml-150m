"""Tree-disjoint reviewed English human conversations from OASST1 TRAIN."""
import random,re,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];PUBLIC=ROOT/'experiments/public_repair_384_v1'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PUBLIC/'scripts'))
import data as rules
from sml_v1.common import atomic_json,read_json,fingerprint,file_sha256
from sml_v1.tokenization import Tokenizer

def prepare(c,cfg):
 import pyarrow.parquet as pq
 pin=read_json(c/'source_pins.json')['oasst'];p=c/'raw'/pin['local_file'];assert file_sha256(p)==pin['sha256']
 tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');raw=read_json(PUBLIC/'data/prepared.json')['data'];nodes=pq.read_table(p).to_pylist();by_id={r['message_id']:r for r in nodes};denied,dgrams=rules.exclusions();held=set();prior=set();reject=Counter();trees=defaultdict(list)
 for path in [PUBLIC/'data/prepared.json',ROOT/'sft/conversation_foundation_original/prepared.json',ROOT/'sft/text_followup_512_v1/prepared.json',ROOT/'experiments/unified_text_v1_sft_v1/data/prepared.json']:
  d=read_json(path)['data']
  for split in ('train','dev','test'):
   for r in d[split]:
    prior.add(rules.group_id(r['messages']))
    if split!='train':held.update(g for m in r['messages'] if m['role']=='user' for g in rules.grams(m['content']))
 for leaf in nodes:
  if leaf['role']!='assistant' or leaf['lang']!='en' or leaf['rank']!=0:continue
  chain=[];node=leaf;valid=True
  while node is not None:
   if node['deleted'] or node['synthetic'] or node['lang']!='en' or node['review_result'] is not True:valid=False;break
   if node['role']=='assistant':
    labels=dict(zip((node.get('labels') or {}).get('name',[]),(node.get('labels') or {}).get('value',[])))
    if node['rank']!=0 or labels.get('quality',0)<.7 or labels.get('fails_task',0)>.1:valid=False;break
   chain.append(node)
   if node['parent_id'] is None:break
   node=by_id.get(node['parent_id'])
   if node is None:valid=False
  if not valid:reject['unreviewed_or_lower_rank_ancestor']+=1;continue
  chain.reverse();messages=[dict(role='user' if n['role']=='prompter' else 'assistant',content=n['text'].strip()) for n in chain]
  if not 2<=len(messages)<=6 or rules.basic_quality(messages):reject['structure']+=1;continue
  if rules.SCOPE.search(' '.join(m['content'] for m in messages)) or any(rules.BOILERPLATE.search(m['content']) for m in messages if m['role']=='assistant'):reject['scope_or_boilerplate']+=1;continue
  if any(rules.norm(m['content']) in denied or rules.grams(m['content'])&dgrams for m in messages) or any(rules.grams(m['content'])&held for m in messages if m['role']=='user'):reject['evaluation_overlap']+=1;continue
  if rules.group_id(messages) in prior:reject['prior_data']+=1;continue
  repeated=False
  for m in messages:
   if m['role']=='assistant':
    words=rules.norm(m['content']).split();counts=Counter(tuple(words[i:i+8]) for i in range(len(words)-7));repeated |= max(counts.values(),default=0)>=2
  if repeated:reject['repeated_reference_span']+=1;continue
  group='oasst-tree-'+leaf['message_tree_id'];row=dict(id=fingerprint(messages),group=group,source='chat',dataset_category='oasst',messages=messages,origin=dict(dataset=pin['repo'],split='train',tree=leaf['message_tree_id'],message_ids=[n['message_id'] for n in chain]),review_quality_floor=.7)
  try:enc=rules.turns(tok,row,2048)
  except OverflowError:reject['overflow']+=1;continue
  if max(e['targets'] for e in enc)>192:reject['long_complete_answer']+=1;continue
  row['assistant_turns']=len(enc);trees[group].append(row)
 buckets=defaultdict(list)
 for group,rows in trees.items():
  # Select one complete highest-depth chain per tree, preserving its real history.
  row=sorted(rows,key=lambda r:(-r['assistant_turns'],r['id']))[0]
  bucket=int(fingerprint(group)[:8],16)%20;split='dev' if bucket==0 else 'test' if bucket==1 else 'train';buckets[split].append(row)
 n=cfg['updates']*cfg['fresh_per_batch'];selected={}
 for split in ('train','dev','test'):
  rows=buckets[split];random.Random(2026100124).shuffle(rows);k=n if split=='train' else 32;assert len(rows)>=k,(split,len(rows),k)
  # Round robin by number of assistant turns avoids only single-turn QA.
  strata=defaultdict(list)
  for row in rows:strata[row['assistant_turns']].append(row)
  out=[]
  while len(out)<k:
   for key in sorted(strata,reverse=True):
    if strata[key] and len(out)<k:out.append(strata[key].pop())
  selected[split]=out
 rng=random.Random(cfg['seed']);replay={s:[r for r in raw['train'][:2048] if r['source']==s] for s in ('chat','human','grounded','instructions')}
 for rr in replay.values():rng.shuffle(rr)
 count=(16-cfg['fresh_per_batch'])//4;train=[]
 for u in range(cfg['updates']):
  batch=selected['train'][u*cfg['fresh_per_batch']:(u+1)*cfg['fresh_per_batch']]
  for rr in replay.values():batch += [rr[(u*count+j)%len(rr)] for j in range(count)]
  rng.shuffle(batch);assert len(batch)==16;train+=batch
 assert not {r['group'] for r in train}&{r['group'] for split in ('dev','test') for r in raw[split]+selected[split]}
 encoded=[e for r in train for e in rules.turns(tok,r,cfg['context'])];assert all(e['y'][-1]==tok.eos and len(e['x'])==len(e['y'])<=cfg['context'] for e in encoded)
 cfg.update(training_counts=dict(Counter(r['source'] for r in train)),method='Ordinary assistant-only conversation-balanced CE + EOS; reviewed short OASST1 conversations with parent replay')
 atomic_json(c/'config.json',cfg);atomic_json(c/'data.json',dict(train=train,dev=raw['dev'],test=raw['test'],targeted_dev=selected['dev'],targeted_test=selected['test']))
 receipt=dict(training_conversations=len(train),unique_conversations=len({r['group'] for r in train}),fresh_public_examples=n,parent_replay_fraction=(16-cfg['fresh_per_batch'])/16,dataset_mix=dict(oasst=n,parent_replay=len(train)-n),assistant_turns=len(encoded),fresh_assistant_turn_counts=dict(Counter(r['assistant_turns'] for r in selected['train'])),fresh_dev=32,fresh_reserved_test=32,eligible={k:len(v) for k,v in buckets.items()},rejections=dict(reject),tree_disjoint=True,review_quality_floor=.7,assistant_rank=0,maximum_fresh_answer_tokens=192,whole_answers=True,targets_truncated=False,benchmark_examples_used_for_training=False,limitations='Human review/ranking selects better source answers but does not guarantee factual correctness. Fresh dialogue NLL and repetition are development checks, not helpfulness scores.')
 atomic_json(c/'selection.json',receipt);print('[prepared-human-dialogue]',receipt,flush=True)

def assess(c,b,dataset,u):
 from sft.conversation_foundation_original.engine import generate,repeated
 p=c/'runs/sft/evaluations'/f'targeted_{u:05d}.json'
 if p.exists():return
 answers=[]
 for row in dataset['targeted_dev']:
  messages=[]
  for m in row['messages']:
   if m['role']=='user':messages.append(m)
   else:
    g=generate(b,messages,384);answers.append(dict(id=row['id'],prompt=messages,reference=m['content'],repeated=repeated(g['tokens']),**g));messages.append(dict(role='assistant',content=g['text']))
 metrics=dict(generated_turns=len(answers),stopped=sum(a['stop']=='eos' for a in answers)/len(answers),repeated=sum(a['repeated'] for a in answers)/len(answers),empty=sum(not a['text'].strip() for a in answers)/len(answers))
 atomic_json(p,dict(metrics=metrics,answers=answers,reserved_test_used=False));print('[fresh-human-eval]',u,metrics,flush=True)
