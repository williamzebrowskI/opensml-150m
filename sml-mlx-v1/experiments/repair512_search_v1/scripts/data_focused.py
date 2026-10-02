"""Trial 21: fresh public TRAIN sources; annotated QA and verified named formats."""
import json,random,re,sys
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[3]
PUBLIC=ROOT/'experiments/public_repair_384_v1'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PUBLIC/'scripts'))
import data as rules
from sml_v1.common import read_json,atomic_json,fingerprint,file_sha256
from sml_v1.tokenization import Tokenizer

# Conservatively parse the canonical SmolTalk constraints. Reject unsupported or
# unresolved controls rather than claiming to verify arbitrary natural language.
def parse(prompt):
 p=prompt.lower();checks=[]
 if re.search(r'\[(?:relation|num_\w+|keywords|ender|forbidden_words)\]|json|quotation|repeat (?:the |your )?prompt|two responses|\bletter\s+[a-z]\b|\beach (?:paragraph|sentence)|\bper (?:paragraph|sentence)|\b(?:french|spanish|german|chinese)\b',p):return None
 for f,unit in [('words','words?'),('sentences','sentences?'),('bullets','bullet points?'),('paragraphs','paragraphs?'),('placeholders','placeholders?'),('highlights','sections?')]:
  pattern=r'(exactly|at least|at most|less than|fewer than|more than|no more than|no fewer than)?\s*(\d+)\s+'+unit
  ms=list(re.finditer(pattern,p))
  if len(ms)>1:return None
  if not ms:continue
  m=ms[0]
  if f=='highlights' and 'highlight' not in p:continue
  n=int(m[2]);rel=m[1] or 'exactly'
  if n<1 or n>(180 if f=='words' else 6):return None
  c=dict(family=f,number=n,relation=rel)
  if f=='paragraphs':c['separator']='divider' if '***' in p else 'blank_line'
  checks.append(c)
 for f,pattern in [('lowercase',r'all lowercase'),('uppercase',r'all (?:capital letters|uppercase)'),('title',r'title.{0,65}double (?:angular|angle) brackets'),('no_comma',r'no commas?|without (?:using )?commas?')]:
  if re.search(pattern,p):checks.append(dict(family=f))
 if 'postscript' in p:
  if 'p.p.s' in p:return None
  if 'p.s' not in p:return None
  checks.append(dict(family='postscript'))
 if 'finish your response with this exact phrase' in p:
  ms=re.findall(r'finish your response with this exact phrase\s*["\']([^"\'\n]+)["\']',prompt,re.I)
  if len(ms)!=1:return None
  checks.append(dict(family='ending',phrase=ms[0]))
 elif re.search(r'finish your response|end your response|exact phrase',p):return None
 if 'keyword' in p:
  if 'do not include' in p or 'incorporate' in p:return None
  ms=re.findall(r'include keywords\s+\[([^\]\n]+)\]',prompt,re.I)
  if len(ms)!=1:return None
  words=[v.strip() for v in ms[0].split(',')]
  if not words or any(not re.fullmatch(r'[A-Za-z][A-Za-z -]{0,30}',w) for w in words):return None
  checks.append(dict(family='keywords',words=words))
 if 'should appear' in p:
  ms=re.findall(r"word\s+[\"']([^\"'\n]+)[\"']\s+should appear\s+(at least|exactly)\s+(\d+)\s+times",prompt,re.I)
  if len(ms)!=1:return None
  word,relation,n=ms[0]
  if not re.fullmatch(r'[A-Za-z]+',word) or not 1<=int(n)<=6:return None
  checks.append(dict(family='frequency',word=word,number=int(n),relation=relation))
 if not 1<=len(checks)<=4:return None
 if any(x in p for x in ['my answer is','[num','[relation','[language']):return None
 return checks

def verify(text,checks):
 base=[c for c in checks if c['family'] not in ('placeholders','highlights','postscript','keywords','frequency')]
 out=rules.verify(text,base)
 for c in checks:
  f=c['family'];t=text.strip()
  if f=='keywords':out.append(all(re.search(r'(?<!\w)'+re.escape(w)+r'(?!\w)',t,re.I) for w in c['words']))
  elif f=='frequency':
   n=len(re.findall(r'(?<!\w)'+re.escape(c['word'])+r'(?!\w)',t,re.I));out.append(n>=c['number'] if c['relation']=='at least' else n==c['number'])
  elif f=='postscript':out.append(bool(re.search(r'(?im)^\s*p\.s\.',t)))
  elif f in ('placeholders','highlights'):
   n=len(re.findall(r'\[[^\[\]\n]+\]',t)) if f=='placeholders' else len(re.findall(r'(?<!\*)\*[^*\n]+\*(?!\*)',t))
   v=c['number'];r=c['relation'];out.append({'exactly':n==v,'at least':n>=v,'at most':n<=v,'less than':n<v,'fewer than':n<v,'more than':n>v,'no more than':n<=v,'no fewer than':n>=v}[r])
 return out

def prepare(c,cfg):
 import pyarrow.parquet as pq
 tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');rng=random.Random(2026100121)
 pins=read_json(c/'source_pins.json')
 for name,pin in pins.items():assert file_sha256(c/'raw'/f'{name}.parquet')==pin['sha256']
 raw=read_json(PUBLIC/'data/prepared.json')['data'];denied,dgrams=rules.exclusions()
 seen=set();held=set();used_context=set();used_articles=set()
 for path in [PUBLIC/'data/prepared.json',ROOT/'sft/conversation_foundation_original/prepared.json',ROOT/'sft/text_followup_512_v1/prepared.json',ROOT/'experiments/unified_text_v1_sft_v1/data/prepared.json']:
  dataset=read_json(path)['data']
  for split in ('train','dev','test'):
   for row in dataset[split]:
    seen.add(rules.group_id(row['messages']))
    if split!='train':held.update(g for m in row['messages'] if m['role']=='user' for g in rules.grams(m['content']))
 ground=read_json(ROOT/'sft/grounded_rank_384/prepared.json')
 for split in ('train','dev','test'):
  for row in ground[split]['grounded']:
   if row.get('context'):used_context.add(rules.norm(row['context']))
   if row.get('article_group'):used_articles.add(row['article_group'])
 # Preserve historical article partitions as well as selected held-out articles.
 from sft.reading_repair.source import split_for
 from sft.transfer_control.data import digest,normalized
 rejects=Counter();pools=defaultdict(list);consumed=set()
 def add(messages,category,origin,group=None,**extras):
  group=group or rules.group_id(messages)
  if group in seen or group in consumed:rejects['duplicate_or_prior_data']+=1;return
  if rules.basic_quality(messages):rejects['structure_quality']+=1;return
  if any(rules.norm(m['content']) in denied for m in messages) or any(rules.grams(m['content'])&dgrams for m in messages):rejects['benchmark_overlap']+=1;return
  if any(rules.grams(m['content'])&held for m in messages if m['role']=='user'):rejects['holdout_overlap']+=1;return
  row=dict(id=fingerprint(messages),group=group,source={'constraints':'instructions','squad':'grounded','sciq':'human'}[category],dataset_category=category,messages=messages,origin=origin,**extras)
  try:encoded=rules.turns(tok,row,2048)
  except OverflowError:rejects['context_overflow']+=1;return
  if max(e['targets'] for e in encoded)>(384 if category=='constraints' else 64):rejects['long_target']+=1;return
  consumed.add(group);pools[category].append(row)
 for i,r in enumerate(pq.read_table(c/'raw/constraints.parquet').to_pylist()):
  messages=r['messages'];checks=parse(messages[0]['content'])
  # Prefer actual tasks over demonstrations that only talk about obeying rules.
  if not re.search(r'\?|\b(?:explain|describe|discuss|draft|summarize|rewrite|write (?:a|an|the)|create (?:a|an|the))\b',messages[0]['content'],re.I):
   rejects['format_only_without_task']+=1;continue
  if re.search(r'crafted to|specified constraints|given instructions|here is my response|meet (?:all )?(?:the )?constraints|ensure that the word|this (?:part of the )?response',messages[-1]['content'],re.I):
   rejects['meta_answer']+=1;continue
  if not checks:rejects['unsupported_constraints']+=1;continue
  if re.search(r'\b(?:diagnosis|dosage|medical|arthritis|cancer|curcumin|treatment|investment|lawyer)\b',messages[0]['content'],re.I):rejects['specialist_scope']+=1;continue
  if not all(verify(messages[-1]['content'],checks)):rejects['target_constraint_failure']+=1;continue
  add(messages,'constraints',dict(dataset=pins['constraints']['repo'],file=pins['constraints']['file'],row_index=i,split='train'),named_checks=checks)
 print('[fresh-pool] constraints',len(pools['constraints']),flush=True)
 for i,r in enumerate(pq.read_table(c/'raw/squad.parquet').to_pylist()):
  context=r['context'];article='article-'+digest(normalized(r['title']))
  if split_for(r['title'])!='train' or article in used_articles or rules.norm(context) in used_context:continue
  if not 35<=len(context.split())<=200:continue
  if r['id'] in __import__('sft.reading_repair.source',fromlist=['REVIEW_EXCLUSIONS']).REVIEW_EXCLUSIONS:continue
  if r['answers']['text']:
   refs=r['answers']['text']
   if any(context[start:start+len(a)]!=a for a,start in zip(refs,r['answers']['answer_start'])):continue
   answer=refs[0];unknown=False
   if not 1<=len(answer.split())<=12:continue
  else:answer='Not stated';refs=[answer];unknown=True
  prompt='Answer briefly using only the passage. If the answer is absent, say "Not stated".\n\nPassage: '+context+'\n\nQuestion: '+r['question']
  add([dict(role='user',content=prompt),dict(role='assistant',content=answer)],'squad',dict(dataset=pins['squad']['repo'],id=r['id'],title=r['title'],split='train'),group=fingerprint(['squad-context',rules.norm(context),unknown]),references=refs,unknown=unknown,article=article)
 print('[fresh-pool] squad',len(pools['squad']),flush=True)
 for i,r in enumerate(pq.read_table(c/'raw/sciq.parquet').to_pylist()):
  if not r['support'] or not r['correct_answer'].strip():continue
  if not 4<=len(r['question'].split())<=45 or len(tok.encode(r['correct_answer']))>24:continue
  # Preserve the official answer label without invented rationales.
  prompt=r['question']+'\n\nGive the answer briefly.'
  add([dict(role='user',content=prompt),dict(role='assistant',content=r['correct_answer'])],'sciq',dict(dataset=pins['sciq']['repo'],row_index=i,split='train'),group=fingerprint(['sciq-support',rules.norm(r['support'])]),references=[r['correct_answer']])
 print('[fresh-pool] sciq',len(pools['sciq']),flush=True)
 # Context/support groups define splits. Holdout content never enters training.
 selected={s:{} for s in ('train','dev','test')}
 counts=dict(constraints=1536,squad=1024,sciq=512)
 for category,rows in pools.items():
  rng.shuffle(rows);groups={};buckets=defaultdict(list)
  for row in rows:
   split='dev' if int(row['group'][:8],16)%20==0 else 'test' if int(row['group'][:8],16)%20==1 else 'train'
   # SQuAD article-level splitting prevents sibling paragraphs across new splits.
   if category=='squad':split=['dev','test','train'][min(int(fingerprint(['trial21',row['article']])[:8],16)%20,2)]
   buckets[split].append(row)
  for split in ('dev','test','train'):
   n=counts[category] if split=='train' else 32
   if category=='constraints':
    # Greedy rare-family coverage, unique examples, no oversampling.
    families=sorted({v['family'] for r in buckets[split] for v in r['named_checks']});chosen=[];ids=set();family_counts=Counter()
    while len(chosen)<n:
     eligible=[r for r in buckets[split] if r['id'] not in ids]
     if not eligible:raise ValueError('Insufficient fresh constraint data')
     r=min(eligible,key=lambda r:sum(family_counts[x['family']] for x in r['named_checks'])/len(r['named_checks']))
     chosen.append(r);ids.add(r['id']);family_counts.update(v['family'] for v in r['named_checks'])
   elif category=='squad':
    yes=[r for r in buckets[split] if not r['unknown']];no=[r for r in buckets[split] if r['unknown']]
    chosen=yes[:n*7//8]+no[:n//8]
   else:chosen=buckets[split][:n]
   assert len(chosen)==n,(category,split,len(chosen),n)
   selected[split][category]=chosen
 replay=[]
 for source in ('chat','human','grounded','instructions'):
  rr=[r for r in raw['train'][:2048] if r['source']==source];rng.shuffle(rr);replay+=rr[:256]
 rng.shuffle(replay);train=[]
 for u in range(256):
  batch=selected['train']['constraints'][u*6:u*6+6]+selected['train']['squad'][u*4:u*4+4]+selected['train']['sciq'][u*2:u*2+2]+replay[u*4:u*4+4];rng.shuffle(batch);train+=batch
 assert len(train)==4096 and len({r['group'] for r in train})==4096
 assert not {r['group'] for r in train}&{r['group'] for split in ('dev','test') for cat in selected[split].values() for r in cat}
 cfg.update(data_focused=True,updates=256,evaluation_updates=[0,256],training_counts=dict(Counter(r['source'] for r in train)),source_caps=dict(chat=192,human=160,grounded=160,instructions=384),method='Ordinary assistant-only conversation-balanced CE + EOS; fresh public source mixture')
 atomic_json(c/'config.json',cfg)
 atomic_json(c/'data.json',dict(train=train,dev=raw['dev'],test=raw['test'],targeted_dev=[r for cat in selected['dev'].values() for r in cat],targeted_test=[r for cat in selected['test'].values() for r in cat]))
 receipt=dict(training_conversations=4096,unique_conversations=4096,parent_replay_fraction=.25,fresh_public_examples=3072,dataset_mix=dict(smol_constraints=1536,squad=1024,sciq=512,parent_replay=1024),fresh_dev=96,fresh_reserved_test=96,rejections=dict(rejects),eligible=dict((k,len(v)) for k,v in pools.items()),constraint_families=dict(Counter(v['family'] for r in selected['train']['constraints'] for v in r['named_checks'])),whole_answers=True,targets_truncated=False,benchmark_examples_used_for_training=False,limitations='Named format checks and SQuAD spans checked mechanically. Public generated answer facts are not independently guaranteed correct. SciQ labels preserved; no invented explanations. SQuAD is a previously used source, with unused contexts and TRAIN articles selected.')
 atomic_json(c/'selection.json',receipt);print('[prepared-data-focused]',json.dumps(receipt),flush=True)

def assess(c,b,dataset,u):
 from sft.conversation_foundation_original.engine import generate,repeated
 p=c/'runs/sft/evaluations'/f'targeted_{u:05d}.json'
 if p.exists():return
 answers=[];stats={}
 for category in ('constraints','squad','sciq'):
  selected=[r for r in dataset['targeted_dev'] if r['dataset_category']==category][:16]
  correct=0;family=Counter();total=Counter();eos=0;rep=0
  for row in selected:
   g=generate(b,row['messages'][:-1],384);eos+=g['stop']=='eos';rep+=repeated(g['tokens'])
   if category=='constraints':
    checks=verify(g['text'],row['named_checks']);ok=all(checks)
    for v in row['named_checks']:total[v['family']]+=1
    # Check each separately to preserve family ordering for extension checks.
    for v in row['named_checks']:family[v['family']]+=all(verify(g['text'],[v]))
   else:
    def normal(t):return re.sub(r'\b(a|an|the)\b',' ',re.sub(r'[^\w\s]',' ',t.lower())).split()
    ok=any(normal(g['text'])==normal(a) for a in row['references'])
   correct+=ok;answers.append(dict(id=row['id'],category=category,reference=row['messages'][-1]['content'],prompt=row['messages'][0]['content'],correct=ok,**g))
  stats[category]=dict(correct=correct,total=len(selected),accuracy=correct/len(selected),stopped=eos/len(selected),repeated=rep/len(selected),by_family={f:dict(passed=family[f],total=n) for f,n in total.items()})
 atomic_json(p,dict(metrics=stats,answers=answers,reserved_test_used=False));print('[fresh-data-eval]',u,json.dumps(stats),flush=True)
