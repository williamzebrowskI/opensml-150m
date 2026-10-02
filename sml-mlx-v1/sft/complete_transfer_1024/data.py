"""Pinned UltraChat first replies plus context-disjoint complete transformations.
Raw Parquet is streamed into one bounded RAM buffer, verified and discarded.
"""
import hashlib,io,random,re
from collections import Counter
from pathlib import Path
from sml_v1.common import read_json,fingerprint
from sft.natural_control.data import BLOCK,QUANT,MATH,MEDICAL,LIVE,BAD_RESPONSE,SPECIAL,norm,grams,exclusions
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
REPO='HuggingFaceH4/ultrachat_200k';REV='8049631c405ae6576f93f445c6b8166f76f5505a'
FILES=[('data/train_sft-00000-of-00003-a3ecf92756993583.parquet',243999189,'afa8fa7426081b2a0e732fb50dbb5cd402a28ad5f0dbe66c0d996d63e7220727'),('data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet',243897199,'c8230190bc8b29084c974ff32f05589b458fad1908fc4b2fb2b9e1e9f7921f03'),('data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet',244131731,'6fe7d2a5e95cf660f972ecaf304aa5632d7f53384e4b0e0d0f44d9c96733c03e')]
EXTRA=re.compile(r'\b(poison|weapons?|suicide|sexual|porn\w*|politic\w*|elections?|religion|God|astrolog\w*|horoscope|pharmac\w*|laws?|legal|tax|stock market|cryptocurrency|copyright|patent|latest|current president|as an? (?:ai|language)|I cannot|I can.t|I don.t have|I do not have|sorry|unable|lyrics|citation|references|studies show|research shows|according to|Nobel|World War|king of|queen of)\b|\d{4}|```|https?://|\$|\\(?:frac|sum)',re.I)
SCOPE=re.compile(r"\b(tours?|tourist|hotels?|museums?|restaurants?|visitors?|travel|transportation|bus|buses|trains?|taxis?|airlines?|flights?|tickets?|booking|phone|email address|contact information|mounting|chassis|hardware|plugin|plugins|install|download|podcast|documentary|film|movies?|novel|author|author's|report|article|passage|text below|website|webpage|web page|link|recipe|recipes|oven|bake|boil|cook|cooking|pregnan\w*|diabet\w*|nutrition|supplement|diet|healthy|weight loss|disorder|symptom|psycholog\w*|biolog\w*|genetic|neurolog\w*|gorillas?|elephants?|species|juicer|girlfriend|boyfriend|politics|history|historical|ancient|pyramids?|Egypt|origin|etymology|telephone|streaming|mobile operators|TV|WMC|DVBlink|religious|bible|Hussain|Allah|gods?|guns?|explosives|ammunition|celebrit\w*|scientific|quantum|chemistry|physics|thermodynamic|nuclear|electrons?|molecules?|chemical|astronomy|planet|climate|geology|geological|scientists?|studies|evidence suggests|evidence to suggest)\b|@|\.com\b|\.org\b",re.I)
EXTRA_SCOPE=re.compile(r'\b(programs?|swift|rust|code|functions?|print|console|readline|computer|software|apps?|interface|websites?|poems?|poetry|sonnets?|rondeau|rhyme|rhyming|iambic|haiku|meme|poster|sticker|playlist|mixtape|songs?|historic|landmarks?|visiting|visit|cuisine|dish|dishes|ingredients?|salad|grill|grilled|blood|arteries|circulatory|cardiovascular|lymphatic|immune|nervous system|mental health|medical|therapy|counseling|diagnosis|symptoms?|physiolog\w*|research|experiment|study|economic|economics|finance|financial)\b|[{}]|\w+\([^)]*\)',re.I)
START=re.compile(r"^(?:how (?:can|do|does|would)|what (?:are (?:some|the benefits)|is the (?:difference|importance))|(?:can|could) you (?:suggest|give|explain|describe)|(?:write|create|describe|explain|suggest|give|provide|list|draft|compose|discuss|design)\b)",re.I)
def stream(cancelled):
 import requests,pyarrow.parquet as pq
 for file,size,sha in FILES:
  buf=io.BytesIO();h=hashlib.sha256();n=0
  with requests.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{file}',stream=True,timeout=(30,120)) as res:
   res.raise_for_status()
   for part in res.iter_content(1024*1024):
    if cancelled():raise InterruptedError('Stopped during download')
    n+=len(part)
    if n>size:raise ValueError('Source exceeds pinned size')
    h.update(part);buf.write(part)
  if n!=size or h.hexdigest()!=sha:raise ValueError('Pinned source integrity mismatch')
  buf.seek(0);print('[source-verified]',file,flush=True)
  for batch in pq.ParquetFile(buf).iter_batches(batch_size=128):
   if cancelled():raise InterruptedError('Stopped during selection')
   yield from batch.to_pylist()
  buf.close()

def sentences(text):
 return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])',text) if s.strip()]

def transform(row,index):
 ss=sentences(' '.join(row['answer'].split()))
 if re.search(r'(?m)^\s*(?:[-*]\s|\d+\.)',row['answer']) or any(c in row['answer'] for c in ('"','“','”')) or not 2<=len(ss)<=6 or any(not 6<=len(s.split())<=45 for s in ss):return None
 count=min(2+index%3,len(ss)); selected=ss[:count]
 mode=('bullets','paragraphs','numbered')[index%3]
 directive={'bullets':f'Present all {count} supplied sentences as {count} bullet points, starting each with "- ".', 'paragraphs':f'Present all {count} supplied sentences as {count} separate paragraphs, with a blank line between paragraphs.', 'numbered':f'Present all {count} supplied sentences as a numbered list from 1 to {count}.'}[mode]
 answer='\n'.join('- '+s for s in selected) if mode=='bullets' else '\n\n'.join(selected) if mode=='paragraphs' else '\n'.join(f'{i+1}. {s}' for i,s in enumerate(selected))
 return dict(id=row['id']+':complete',group=row['group'],source='ultrachat-grounded',format=mode,sentences=selected,prompt='Prepare a complete response using the supplied material. '+directive+' Preserve every sentence exactly, in order. Add no introduction or extra claims.\n\nMaterial:\n'+' '.join(selected),answer=answer)

def build(cfg,cancelled=lambda:False):
 from sml_v1.tokenization import Tokenizer
 from sft.transfer_control.engine import encode
 from types import SimpleNamespace
 tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');backend=SimpleNamespace(encode=tok.encode,eos=tok.eos)
 deny,dg=exclusions();pool=[];seen=set();seen_answers=set();rejected=Counter()
 rejects=read_json(DIR/'rejections.json') if (DIR/'rejections.json').exists() else {}
 for raw in stream(cancelled):
  msg=raw['messages']
  if len(msg)<2 or [x['role'] for x in msg[:2]]!=['user','assistant']:continue
  p,a=(x['content'].strip() for x in msg[:2]);key=fingerprint(norm(p));ak=fingerprint(norm(a))
  if key in seen or ak in seen_answers or raw['prompt_id'] in rejects:continue
  if not START.search(p) or re.search(r'\b(?:[6-9]|\d{2,})\b',p):continue
  if SCOPE.search(p+' '+a) or EXTRA_SCOPE.search(p+' '+a):continue
  if not 5<=len(p.split())<=80 or not 40<=len(a.split())<=240:continue
  if any(rx.search(p+'\n'+a) for rx in (BLOCK,QUANT,MATH,MEDICAL,LIVE,BAD_RESPONSE,SPECIAL,EXTRA)):continue
  if a[-1] not in '.!?"”' or len(re.findall(r'[.!?](?:\s|$)',a))<2:continue
  if sum(c.isascii() for c in p+a)/len(p+a)<.98:continue
  if re.search(r'\b(continue|previous|above|earlier|again|this text|this passage|this article)\b',p,re.I):continue
  if norm(p) in deny or grams(p)&dg or grams(a)&dg:continue
  words=norm(a).split();ng=Counter(tuple(words[i:i+5]) for i in range(len(words)-4))
  if ng and max(ng.values())>1:continue
  row=dict(id=raw['prompt_id'],group=key,source='ultrachat-first-reply',prompt=p,answer=a)
  try:enc=encode(backend,row,cfg['context'])
  except ValueError:continue
  if enc['targets']>400:continue
  seen.add(key);seen_answers.add(ak);pool.append(row)
 print('[eligible]',len(pool),'independent first replies',flush=True)
 # Split on normalized root before choosing task view, keep all heldout groups out of train.
 splits={s:[] for s in ('train','dev','test')}
 for row in sorted(pool,key=lambda r:fingerprint([cfg['seed'],r['group']])):
  n=int(row['group'][:8],16)%10;s='train' if n<8 else 'dev' if n==8 else 'test';splits[s].append(row)
 new={};stats={}
 for s,rows in splits.items():
  need_n=cfg['updates']*3 if s=='train' else 32;need_g=cfg['updates'] if s=='train' else 16
  ground=[];used=set()
  for row in rows:
   g=transform(row,len(ground))
   if g:
    encode(backend,g,cfg['context']);ground.append(g);used.add(row['group'])
   if len(ground)==need_g:break
  natural=[r for r in rows if r['group'] not in used][:need_n]
  if len(natural)!=need_n or len(ground)!=need_g:raise ValueError(f'Not enough independent {s} rows: {len(natural)}/{need_n}, {len(ground)}/{need_g}')
  new[s]=dict(natural=natural,complete=ground)
  stats[s]={f:dict(count=len(rr),groups=len({r['group'] for r in rr}),hash=fingerprint(rr),answer_tokens=sum(encode(backend,r,cfg['context'])['targets'] for r in rr)) for f,rr in new[s].items()}
 from sft.protected_update_1024.data import build as old_build
 oldcfg=read_json(ROOT/'sft/protected_update_1024/config.json');oldcfg['arm']='control'
 old,sel=old_build(oldcfg,cancelled)
 if sel!=read_json(ROOT/'sft/protected_update_1024/selection_control.json'):raise ValueError('Historical replay recipe changed')
 train=[]
 for i in range(cfg['updates']):train+=new['train']['natural'][3*i:3*i+3]+new['train']['complete'][i:i+1]
 replay=list({r['id']:r for r in old['train']['reading']}.values());random.Random(cfg['seed']).shuffle(replay)
 if len(replay)<cfg['updates']*2:raise ValueError('Not enough unique replay')
 qa=old['train']['commonsense']
 old['train']=dict(instruction=train,commonsense=[r for i in range(cfg['updates']) for r in (qa[4*i],qa[4*i+2])],reading=replay[:cfg['updates']*2])
 old['new']=new
 manifest=dict(data_hash=fingerprint(old),config=fingerprint(cfg),source=dict(repo=REPO,revision=REV,files=[list(x) for x in FILES],license='MIT',synthetic=True),tokenizer=tok.fingerprint,new=stats,train_hashes={f:fingerprint(rr) for f,rr in old['train'].items()},replay_source=fingerprint(sel),eligible=len(pool))
 return old,manifest
