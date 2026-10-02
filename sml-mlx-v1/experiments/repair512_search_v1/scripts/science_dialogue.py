"""Annotated elementary science QA, with earlier conversation/follow-up replay."""
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
 pin=read_json(c/'source_pins.json')['openbook'];path=c/'raw'/pin['local_file'];assert file_sha256(path)==pin['sha256'];raw=read_json(PUBLIC/'data/prepared.json')['data'];unified=read_json(ROOT/'experiments/unified_text_v1_sft_v1/data/prepared.json')['data'];tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');denied,dgrams=rules.exclusions();prior=set();held=set();reject=Counter();seen=set();buckets=defaultdict(list)
 for dataset in (raw,unified,read_json(ROOT/'sft/conversation_foundation_original/prepared.json')['data'],read_json(ROOT/'sft/text_followup_512_v1/prepared.json')['data']):
  for split in ('train','dev','test'):
   for row in dataset[split]:
    prior.add(rules.group_id(row['messages']))
    if split!='train':held.update(g for m in row['messages'] if m['role']=='user' for g in rules.grams(m['content']))
 for r in pq.read_table(path).to_pylist():
  if r['humanScore']<.999 or r['clarity']<1.6:reject['human_agreement_clarity']+=1;continue
  stem=r['question_stem'].strip();fact=r['fact1'].strip();choices=r['choices'];gold=choices['text'][choices['label'].index(r['answerKey'])].strip();group=fingerprint(['openbook-fact',rules.norm(fact)])
  if group in seen:reject['shared_fact']+=1;continue
  if not 3<=len(stem.split())<=60 or not gold or not fact:reject['structure']+=1;continue
  prompt=stem+'\n\nChoices:\n'+'\n'.join(label+'. '+text for label,text in zip(choices['label'],choices['text']))+'\n\nGive the correct answer and a brief explanation.'
  answer=gold.rstrip('.')+'. '+fact[0].upper()+fact[1:].rstrip('.')+'.'
  messages=[dict(role='user',content=prompt),dict(role='assistant',content=answer)]
  if rules.basic_quality(messages) or rules.group_id(messages) in prior:reject['quality_prior']+=1;continue
  if rules.norm(stem) in denied or rules.grams(stem)&dgrams or any(rules.norm(m['content']) in denied or rules.grams(m['content'])&dgrams for m in messages) or rules.grams(prompt)&held:reject['evaluation_overlap']+=1;continue
  row=dict(id=fingerprint(messages),group=group,source='grounded',dataset_category='openbook',messages=messages,gold_answer=gold,fact=fact,origin=dict(dataset=pin['repo'],id=r['id'],split='train',human_score=r['humanScore'],clarity=r['clarity']))
  try:enc=rules.turns(tok,row,2048)
  except OverflowError:reject['overflow']+=1;continue
  if max(e['targets'] for e in enc)>96:reject['long_answer']+=1;continue
  seen.add(group);bucket=int(group[:8],16)%20;split='dev' if bucket==0 else 'test' if bucket==1 else 'train';buckets[split].append(row)
 selected={};n=min(512,cfg['updates']*4)
 for split in ('train','dev','test'):
  rows=buckets[split];random.Random(2026100126).shuffle(rows);k=n if split=='train' else 32;assert len(rows)>=k,(split,len(rows),k);selected[split]=rows[:k]
 replay={}
 replay['public_instruction']=[r for r in raw['train'][:2048] if r['source']=='instructions']
 for source,mapped in [('conversation','chat'),('writing','human'),('format_followup','instructions'),('recovery','chat')]:
  rr=[]
  for r in unified['train'][:6144]:
   if r['source']!=source:continue
   row=dict(r,source=mapped,group=rules.group_id(r['messages']))
   if max(e['targets'] for e in rules.turns(tok,row,2048))<=192:rr.append(row)
  replay[source]=rr
 rng=random.Random(cfg['seed'])
 for rr in replay.values():rng.shuffle(rr)
 quotas=dict(public_instruction=4,conversation=2,writing=2,format_followup=2,recovery=2)
 assert all(replay[k] for k in quotas)
 train=[]
 for u in range(cfg['updates']):
  batch=[selected['train'][(u*4+j)%n] for j in range(4)]
  for key,count in quotas.items():
   rr=replay[key];batch += [rr[(u*count+j)%len(rr)] for j in range(count)]
  rng.shuffle(batch);assert len(batch)==16;train+=batch
 assert not {r['group'] for r in train}&{rules.group_id(r['messages']) for dataset in (raw,unified) for split in ('dev','test') for r in dataset[split]}
 assert not {r['group'] for r in train}&{r['group'] for split in ('dev','test') for r in selected[split]}
 encoded=[e for r in train for e in rules.turns(tok,r,2048)];assert all(e['y'][-1]==tok.eos and len(e['x'])==len(e['y'])<=2048 for e in encoded)
 cfg.update(training_counts=dict(Counter(r['source'] for r in train)),evaluation_updates=[0,128,256] if cfg['updates']==256 else [0,128],method='Ordinary assistant-only conversation-balanced CE + EOS; annotated OpenBookQA plus earlier conversation and follow-up replay')
 atomic_json(c/'config.json',cfg);atomic_json(c/'data.json',dict(train=train,dev=raw['dev'],test=raw['test'],targeted_dev=selected['dev'],targeted_test=selected['test']))
 receipt=dict(training_conversations=len(train),unique_conversations=len({r['group'] for r in train}),fresh_public_examples=n,fresh_exposures=cfg['updates']*4,fresh_passes=cfg['updates']*4/n,parent_replay_fraction=.75,dataset_mix=dict(openbook=cfg['updates']*4,**{k:v*cfg['updates'] for k,v in quotas.items()}),replay_eligible={k:len(v) for k,v in replay.items()},replay_origin='repair512 instruction rows plus consumed unified384 conversation/writing/format-followup/recovery rows',assistant_turns=len(encoded),fresh_dev=32,fresh_reserved_test=32,eligible={k:len(v) for k,v in buckets.items()},rejections=dict(reject),fact_group_disjoint=True,whole_answers=True,targets_truncated=False,benchmark_examples_used_for_training=False,limitations='Correct choice and supporting fact copied from public annotation. Supporting fact is not a complete independently verified rationale. Holdout gold-answer mention is a content proxy, not complete semantic correctness.')
 atomic_json(c/'selection.json',receipt);print('[prepared-science]',receipt,flush=True)

def assess(c,b,dataset,u):
 from sft.conversation_foundation_original.engine import generate,repeated
 from dataset_trials import normal
 p=c/'runs/sft/evaluations'/f'targeted_{u:05d}.json'
 if p.exists():return
 answers=[]
 for row in dataset['targeted_dev']:
  g=generate(b,row['messages'][:-1],384);gold=' '.join(normal(row['gold_answer']));ok=gold in ' '.join(normal(g['text']))
  answers.append(dict(id=row['id'],prompt=row['messages'][0]['content'],reference=row['messages'][-1]['content'],gold_answer=row['gold_answer'],gold_mention_proxy=ok,repeated=repeated(g['tokens']),**g))
 metrics=dict(generated_turns=len(answers),gold_mention_proxy=sum(a['gold_mention_proxy'] for a in answers)/len(answers),stopped=sum(a['stop']=='eos' for a in answers)/len(answers),repeated=sum(a['repeated'] for a in answers)/len(answers))
 atomic_json(p,dict(metrics=metrics,answers=answers,reserved_test_used=False));print('[fresh-science-eval]',u,metrics,flush=True)
