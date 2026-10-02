"""Unused short conversations from the original verified public source pool."""
import random,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];PUBLIC=ROOT/'experiments/public_repair_384_v1'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PUBLIC/'scripts'))
import data as rules
from sml_v2.common import atomic_json,read_json,fingerprint,file_sha256
from sml_v2.tokenization import Tokenizer

def prepare(c,cfg):
 pin=read_json(c/'source_pins.json')['foundation'];path=c/'raw'/pin['local_file'];assert file_sha256(path)==pin['sha256'];foundation=read_json(path)['data'];raw=read_json(PUBLIC/'data/prepared.json')['data'];tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
 denied,dgrams=rules.exclusions();held=set();prior=set();reject=Counter();buckets=defaultdict(list)
 for dataset in [raw,read_json(ROOT/'sft/text_followup_512_v1/prepared.json')['data'],read_json(ROOT/'experiments/unified_text_v2_sft_v1/data/prepared.json')['data']]:
  for split in ('train','dev','test'):
   for row in dataset[split]:
    prior.add(rules.group_id(row['messages']))
    if split!='train':held.update(g for m in row['messages'] if m['role']=='user' for g in rules.grams(m['content']))
 for row in foundation['train'][:8192]:prior.add(rules.group_id(row['messages']))
 for split in ('dev','test'):
  for row in foundation[split]:
   prior.add(rules.group_id(row['messages']));held.update(g for m in row['messages'] if m['role']=='user' for g in rules.grams(m['content']))
 seen=set()
 for r in foundation['train'][8192:]:
  if r['source'] not in ('smol-magpie-ultra-short','everyday-conversations'):continue
  # Retain a complete conversational prefix, ending after a whole assistant
  # answer. Later turns may be omitted; no answer text is shortened.
  messages=[];pending=[];a=0
  for m in r['messages']:
   pending.append(m)
   if m['role']=='assistant':
    if r['answer_lengths'][a]>192:break
    messages+=pending;pending=[];a+=1
  if not messages:reject['long_first_answer']+=1;continue
  group=rules.group_id(messages)
  if group in prior or group in seen:reject['prior_or_duplicate']+=1;continue
  if rules.basic_quality(messages) or rules.SCOPE.search(' '.join(m['content'] for m in messages)):reject['scope_quality']+=1;continue
  if any(rules.BOILERPLATE.search(m['content']) for m in messages if m['role']=='assistant'):reject['boilerplate']+=1;continue
  if any(rules.norm(m['content']) in denied or rules.grams(m['content'])&dgrams for m in messages) or any(rules.grams(m['content'])&held for m in messages if m['role']=='user'):reject['evaluation_overlap']+=1;continue
  row=dict(id=r['id'],group=group,source='chat',dataset_category='familiar',messages=messages,origin=dict(dataset=r['repo'],file=r['file'],row_index=r['row_index'],source=r['source'],split='train'))
  try:enc=rules.turns(tok,row,2048)
  except OverflowError:reject['overflow']+=1;continue
  if max(e['targets'] for e in enc)>192:reject['long_complete_answer']+=1;continue
  row['assistant_turns']=len(enc);seen.add(group);bucket=int(fingerprint(['familiar-isolate',group])[:8],16)%20;split='dev' if bucket==0 else 'test' if bucket==1 else 'train';buckets[split].append(row)
 selected={};n=min(512,cfg['updates']*cfg['fresh_per_batch'])
 print('[familiar-pool]',dict(reject),{k:len(v) for k,v in buckets.items()},flush=True)
 for split in ('train','dev','test'):
  rows=buckets[split];random.Random(2026100125).shuffle(rows);k=n if split=='train' else 32;assert len(rows)>=k,(split,len(rows),k)
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
  batch=[selected['train'][(u*cfg['fresh_per_batch']+j)%n] for j in range(cfg['fresh_per_batch'])]
  for rr in replay.values():batch += [rr[(u*count+j)%len(rr)] for j in range(count)]
  rng.shuffle(batch);assert len(batch)==16;train+=batch
 assert not {r['group'] for r in train}&{r['group'] for split in ('dev','test') for r in raw[split]+selected[split]}
 encoded=[e for r in train for e in rules.turns(tok,r,cfg['context'])];assert all(e['y'][-1]==tok.eos and len(e['x'])==len(e['y'])<=cfg['context'] for e in encoded)
 cfg.update(training_counts=dict(Counter(r['source'] for r in train)),evaluation_updates=[0,128,256] if cfg['updates']==256 else [0,128],method='Ordinary assistant-only conversation-balanced CE + EOS; unused original-source short conversations with replay')
 atomic_json(c/'config.json',cfg);atomic_json(c/'data.json',dict(train=train,dev=raw['dev'],test=raw['test'],targeted_dev=selected['dev'],targeted_test=selected['test']))
 receipt=dict(training_conversations=len(train),unique_conversations=len({r['group'] for r in train}),fresh_public_examples=n,fresh_exposures=cfg['updates']*cfg['fresh_per_batch'],fresh_passes=cfg['updates']*cfg['fresh_per_batch']/n,parent_replay_fraction=(16-cfg['fresh_per_batch'])/16,dataset_mix=dict(unused_short_chat=cfg['updates']*cfg['fresh_per_batch'],parent_replay=len(train)-cfg['updates']*cfg['fresh_per_batch']),assistant_turns=len(encoded),fresh_assistant_turn_counts=dict(Counter(r['assistant_turns'] for r in selected['train'])),fresh_dev=32,fresh_reserved_test=32,eligible={k:len(v) for k,v in buckets.items()},rejections=dict(reject),maximum_fresh_answer_tokens=192,whole_answers=True,targets_truncated=False,benchmark_examples_used_for_training=False,limitations='Same source families as earlier SFT, but groups used by retained runs and all holdout groups/phrases excluded. Complete prefixes selected; later turns may be omitted, answer text never truncated. Public synthetic target facts are not guaranteed correct. Replay and 512 fresh conversations repeat explicitly over the longer run.')
 atomic_json(c/'selection.json',receipt);print('[prepared-familiar-dialogue]',receipt,flush=True)
