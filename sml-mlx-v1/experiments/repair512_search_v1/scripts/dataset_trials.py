"""Public TRAIN data isolation, fixed parent/replay and independent held-out QA."""
import random,re,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
PUBLIC=ROOT/'experiments/public_repair_384_v1'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PUBLIC/'scripts'))
import data as rules
from sml_v1.common import atomic_json,read_json,fingerprint,file_sha256
from sml_v1.tokenization import Tokenizer

def prepare(c,cfg):
 if cfg['dataset_plan']=='breadth-stable':
  from breadth_stable import prepare
  return prepare(c,cfg)
 if cfg['dataset_plan']=='mixed-public':
  from mixed_public import prepare
  return prepare(c,cfg)
 if cfg['dataset_plan']=='science-replay':
  from science_dialogue import prepare
  return prepare(c,cfg)
 if cfg['dataset_plan']=='familiar-replay':
  from familiar_dialogue import prepare
  return prepare(c,cfg)
 if cfg['dataset_plan']=='human-replay':
  from human_dialogue import prepare
  return prepare(c,cfg)
 if cfg['dataset_plan']=='constraints-replay':
  from short_constraints import prepare
  return prepare(c,cfg)
 import pyarrow.parquet as pq
 from sft.reading_repair.source import split_for,REVIEW_EXCLUSIONS
 from sft.transfer_control.data import digest,normalized
 pins=read_json(c/'source_pins.json')
 for key,pin in pins.items():assert file_sha256(c/'raw'/pin['local_file'])==pin['sha256']
 tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');raw=read_json(PUBLIC/'data/prepared.json')['data']
 denied,dgrams=rules.exclusions();held=set();prior=set();used_context=set();used_articles=set()
 for path in [PUBLIC/'data/prepared.json',ROOT/'sft/conversation_foundation_original/prepared.json',ROOT/'sft/text_followup_512_v1/prepared.json',ROOT/'experiments/unified_text_v1_sft_v1/data/prepared.json']:
  d=read_json(path)['data']
  for split in ('train','dev','test'):
   for row in d[split]:
    prior.add(rules.group_id(row['messages']))
    if split!='train':held.update(g for m in row['messages'] if m['role']=='user' for g in rules.grams(m['content']))
 ground=read_json(ROOT/'sft/grounded_rank_384/prepared.json')
 for split in ('train','dev','test'):
  for row in ground[split]['grounded']:
   if row.get('context'):used_context.add(rules.norm(row['context']))
   if row.get('article_group'):used_articles.add(row['article_group'])
 rejected=Counter();seen=set();buckets=defaultdict(list)
 assert cfg['dataset_plan']=='squad-replay'
 pin=pins['squad']
 for r in pq.read_table(c/'raw'/pin['local_file']).to_pylist():
  context=r['context'];article='article-'+digest(normalized(r['title']))
  if split_for(r['title'])!='train' or article in used_articles or rules.norm(context) in used_context:rejected['prior_article_or_historical_holdout']+=1;continue
  if not 35<=len(context.split())<=200 or r['id'] in REVIEW_EXCLUSIONS:rejected['context_or_review_filter']+=1;continue
  if r['answers']['text']:
   refs=r['answers']['text'];answer=refs[0];unknown=False
   if not 1<=len(answer.split())<=12 or any(context[start:start+len(a)]!=a for a,start in zip(refs,r['answers']['answer_start'])):rejected['span_filter']+=1;continue
  else:answer='Not stated';refs=[answer];unknown=True
  prompt='Answer briefly using only the passage. If the answer is absent, say "Not stated".\n\nPassage: '+context+'\n\nQuestion: '+r['question']
  messages=[dict(role='user',content=prompt),dict(role='assistant',content=answer)]
  group=fingerprint(['squad-context',rules.norm(context),unknown])
  if group in seen or rules.group_id(messages) in prior:rejected['duplicate']+=1;continue
  if rules.basic_quality(messages):rejected['quality']+=1;continue
  if any(rules.norm(m['content']) in denied or rules.grams(m['content'])&dgrams for m in messages) or rules.norm(r['question']) in denied or rules.grams(r['question'])&dgrams:rejected['benchmark_overlap']+=1;continue
  if rules.grams(prompt)&held:rejected['heldout_overlap']+=1;continue
  row=dict(id=fingerprint(messages),group=group,source='grounded',dataset_category='squad',messages=messages,references=refs,unknown=unknown,article=article,origin=dict(dataset=pin['repo'],id=r['id'],title=r['title'],split='train'))
  try:enc=rules.turns(tok,row,2048)
  except OverflowError:rejected['overflow']+=1;continue
  if max(e['targets'] for e in enc)>64:rejected['target_length']+=1;continue
  seen.add(group)
  # Fixed category seed and article partition across all future QA isolates.
  split=['dev','test','train'][min(int(fingerprint(['trial21',article])[:8],16)%20,2)]
  buckets[split].append(row)
 n=cfg['updates']*cfg['fresh_per_batch'];selected={}
 for split in ('train','dev','test'):
  rows=buckets[split];random.Random(2026100122).shuffle(rows);k=n if split=='train' else 32
  yes=[r for r in rows if not r['unknown']];no=[r for r in rows if r['unknown']]
  selected[split]=yes[:k*7//8]+no[:k//8]
  assert len(selected[split])==k,(split,len(selected[split]),k)
  random.Random(2026100123).shuffle(selected[split])
 for a in selected:
  for b in selected:
   if a!=b:assert not {r['article'] for r in selected[a]}&{r['article'] for r in selected[b]}
 rng=random.Random(cfg['seed']);replay={s:[r for r in raw['train'][:2048] if r['source']==s] for s in ('chat','human','grounded','instructions')}
 for rr in replay.values():rng.shuffle(rr)
 count=(16-cfg['fresh_per_batch'])//4;train=[]
 for u in range(cfg['updates']):
  batch=selected['train'][u*cfg['fresh_per_batch']:(u+1)*cfg['fresh_per_batch']]
  for s,rr in replay.items():batch += [rr[(u*count+j)%len(rr)] for j in range(count)]
  rng.shuffle(batch);assert len(batch)==16;train+=batch
 assert not {r['group'] for r in train}&{r['group'] for split in ('dev','test') for r in raw[split]+selected[split]}
 encoded=[e for r in train for e in rules.turns(tok,r,cfg['context'])]
 assert all(e['y'][-1]==tok.eos and len(e['x'])==len(e['y'])<=cfg['context'] for e in encoded)
 cfg.update(evaluation_updates=[0,cfg['updates']],training_counts=dict(Counter(r['source'] for r in train)),method='Ordinary conversation-balanced assistant-only CE + EOS; isolated fresh SQuAD TRAIN with parent replay')
 atomic_json(c/'config.json',cfg);atomic_json(c/'data.json',dict(train=train,dev=raw['dev'],test=raw['test'],targeted_dev=selected['dev'],targeted_test=selected['test']))
 receipt=dict(training_conversations=len(train),unique_conversations=len({r['group'] for r in train}),fresh_public_examples=n,parent_replay_fraction=(16-cfg['fresh_per_batch'])/16,dataset_mix=dict(squad=n,parent_replay=len(train)-n),assistant_turns=len(encoded),fresh_dev=32,fresh_reserved_test=32,eligible={k:len(v) for k,v in buckets.items()},rejections=dict(rejected),article_disjoint=True,whole_answers=True,targets_truncated=False,benchmark_examples_used_for_training=False,limitations='Annotated spans and absent-answer labels retained; exact match can undercount valid paraphrases. Fresh QA development differs from trial21; compare paired parent/candidate within this trial.')
 atomic_json(c/'selection.json',receipt);print('[prepared-dataset]',receipt,flush=True)

def normal(text):return re.sub(r'\b(a|an|the)\b',' ',re.sub(r'[^\w\s]',' ',text.lower())).split()
def f1(text,ref):
 a=normal(text);b=normal(ref);common=sum((Counter(a)&Counter(b)).values())
 return 2*common/(len(a)+len(b)) if a or b else 1.
def assess(c,b,dataset,u):
 if dataset.get('dataset_plan')=='breadth-stable':
  from data_focused import assess
  return assess(c,b,dataset,u)
 if dataset.get('dataset_plan')=='mixed-public':
  from mixed_public import assess
  return assess(c,b,dataset,u)
 if dataset['targeted_dev'][0]['dataset_category']=='openbook':
  from science_dialogue import assess
  return assess(c,b,dataset,u)
 if dataset['targeted_dev'][0]['dataset_category'] in ('oasst','familiar'):
  from human_dialogue import assess
  return assess(c,b,dataset,u)
 if b is not None and dataset['targeted_dev'][0]['dataset_category']=='constraints':
  from short_constraints import assess
  return assess(c,b,dataset,u)
 from sft.conversation_foundation_original.engine import generate,repeated
 p=c/'runs/sft/evaluations'/f'targeted_{u:05d}.json'
 if p.exists():return
 answers=[]
 for row in dataset['targeted_dev'][:32]:
  g=generate(b,row['messages'][:-1],384)
  answers.append(dict(id=row['id'],reference=row['messages'][-1]['content'],references=row['references'],prompt=row['messages'][0]['content'],unknown=row['unknown'],correct=any(normal(g['text'])==normal(a) for a in row['references']),token_f1=max(f1(g['text'],a) for a in row['references']),repeated=repeated(g['tokens']),**g))
 metrics=dict(correct=sum(a['correct'] for a in answers),total=len(answers),accuracy=sum(a['correct'] for a in answers)/len(answers),token_f1=sum(a['token_f1'] for a in answers)/len(answers),stopped=sum(a['stop']=='eos' for a in answers)/len(answers),repeated=sum(a['repeated'] for a in answers)/len(answers))
 atomic_json(p,dict(metrics=metrics,answers=answers,reserved_test_used=False));print('[fresh-qa-eval]',u,metrics,flush=True)
