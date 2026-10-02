"""Ranged-HTTP Parquet streaming; bounded selections in RAM, no raw cache."""
import hashlib,heapq,json,random,re
from collections import Counter,defaultdict
from pathlib import Path
from sml_v2.common import fingerprint
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
REPO='HuggingFaceTB/smol-smoltalk'
REV='f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc'
FILES=[
 ('data/train-00000-of-00004.parquet',229710350,'498cd4580014f42c40cfde066573564ed28e3e1548a5566715b615f5ea932856'),
 ('data/train-00001-of-00004.parquet',230298184,'5412c03df579fc7dd7d06c7f5249627d99d0621882fb8f2cd5c13582eb2a80f6'),
 ('data/train-00002-of-00004.parquet',230587255,'aefcac6571abf275b4e1629baa01f326c0fe279d35cd0349b7439319923815a7'),
 ('data/train-00003-of-00004.parquet',231879938,'c15edfca888c9924a626dad7b7f4e9f08d8f406f997ac13be46bc9be0a6192db')]
SOURCES={'smol-magpie-ultra-short':'conversation','smollm-rewrite-30k':'rewrite','smol-summarize-20k':'summary','everyday-conversations':'everyday'}
BLOCK=re.compile(r'```|https?://|<\|[^>]+\|>|\b(python|javascript|typescript|programming|coding|source code|function calling|sql|html|css|java|algorithm|api|debug|code snippet|github|linux|terminal command|calculus|algebra|equation|arithmetic|calculate|calculation|mathematics|maths?|fractions?|trigonometry|ANOVA|multiply|multiplication|divide|division|derivative|integral|probability|percent|percentage|interest rate|investment|diagnosis|dosage|prescription|lawsuit)\b|\d\s*[+*/=]\s*\d|\\(?:frac|sum|int)\b',re.I)
QUANT=re.compile(r'\b(how many|how much|compute|solve|mean, median|statistics|statistical|geometry|square root|mathematical|count the|prime number|theorem)\b',re.I)
ROLEPLAY=re.compile(r'\b(pretend|role.?play|you are (?:a|an|the) (?!AI|assistant)|time travel|imagine (?:you|that you)|act as (?:a|an)|playing the (?:part|role))\b',re.I)
BAD_RESPONSE=re.compile(r"\b(as (?:an? )?(?:AI|language model)|I (?:grew up|was born|visited|live in|worked as)|today.s (?:weather|news)|current (?:price|president)|as of 20\d\d|latest news)\b",re.I)
SETUP=re.compile(r'^(?:sure|of course|certainly|absolutely)[!.,:]?\s*(?:please|could you|can you|what (?:would|do)|provide|share|tell me)',re.I)
SPECIAL=re.compile(r'(^|\n)\s*(?:User|Assistant|System|Human|AI):',re.I)
PERSONA=re.compile(r"\b(?:(?:play|playing|assume|adopt|take on) (?:the |a |an )?(?:role|part|persona)|you(?: are|'re| will be| should be) (?!free\b|able\b|welcome\b)|want you to be|your (?:role|character|background) (?:is|will)|in character)\b",re.I)
MEDICAL=re.compile(r'\b(patient|patients|cancer|tumou?r|syndrome|diagnos\w*|therap\w*|treatment|medication|antibiotic|diarrhea)\b',re.I)
LIVE=re.compile(r'\b(forecast|our area|tonight.s|current news|recent news|today.s news|weather today|weather tomorrow|stock price)\b|[$€£]\s*\d',re.I)
LOCAL_LIVE=re.compile(r'\b(current|latest|weather|traffic|forecast|president|prime minister|mayor|prices?|costs?|news|nearby)\b|\bnear me\b',re.I)
MATH=re.compile(r'\b(exponent\w*|factorials?|logarithm\w*|polynomials?|quadratic|trigonomet\w*|integers?|decimals?|divisib\w*|numerator|denominator|subtraction|addition|matrices|matrix|permutation\w*|probabilit\w*|correlation\w*|regression|quantitative|dataset|data set|mean|median|standard deviation|variance|average|ratios?|fractions?|total|inequality|inequalities|geometric|geometry|algebra\w*|calculus|numerical|statistic\w*)\b|\d\s*[\^×÷+*/=]\s*[\d(]|[=<>]\s*[-+]?\d',re.I)
REVIEW_REJECTED={
 'c54431477d6cf923a1961ee58da74bbe3db190f2d4e39975246ac62f16f134c2':'Wrongly says photosynthesis releases carbon dioxide and broadly treats weathering as a source.',
}
STOPWORDS=set('a an the and or but if as in on at of to for by with from into onto up out about over under through after before while during that this these those it its itself he his him himself she her hers herself they their them themselves you your we our us i my me is are was were be been being am do does did done have has had having will would can could may might must should not no yes also very more most some any all both each such than then there here who what when where why how which whose whom so just now still only new other one two three'.split())

def word_forms(w):
 forms={w}
 for suffix in ('ing','ed','es','s'):
  if w.endswith(suffix) and len(w)>len(suffix)+2:
   base=w[:-len(suffix)];forms.update((base,base+'e'))
   if len(base)>2 and base[-1]==base[-2]:forms.add(base[:-1])
 if w.endswith('ies'):forms.add(w[:-3]+'y')
 return forms

def grounded_target(content,answer):
 # Conservative lexical screen with simple inflection variants. This does not
 # prove entailment or detect every unsupported relationship in a summary.
 source=set().union(*(word_forms(w) for w in norm(content).split()))
 target={w for w in norm(answer).split() if w not in STOPWORDS}
 unsupported=sum(not bool(word_forms(w)&source) for w in target)
 return unsupported<=max(1,int(len(target)*.06))

def norm(s):return ' '.join(re.findall(r"[a-z0-9]+",s.lower()))
def grams(s,n=13):
 w=norm(s).split();return {hashlib.blake2b(' '.join(w[i:i+n]).encode(),digest_size=8).digest() for i in range(len(w)-n+1)}

def tokens():
 from sml_v2.tokenization import Tokenizer
 from transformers import AutoTokenizer
 a=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
 b=AutoTokenizer.from_pretrained(ROOT/'diagnostics/base_capability_20260924/reference',local_files_only=True,trust_remote_code=False)
 return a,lambda s:b.encode(s,add_special_tokens=False)

def exclusions():
 prompts=[]
 for f in (ROOT/'evaluation/full_benchmarks/_cache').glob('*.json'):
  if f.stem in ('arc_easy','arc_challenge','piqa','hellaswag','ifeval'):
   prompts.extend(r['prompt'] for r in json.loads(f.read_text()))
 p=ROOT/'evaluation/reading_transfer/protocol.json';prompts.extend(r['prompt'] for r in json.loads(p.read_text())['rows'])
 p=DIR/'probes.json'
 if p.exists():prompts.extend(r['prompt'] for rr in json.loads(p.read_text()).values() for r in rr)
 return {norm(p) for p in prompts},set().union(*(grams(p) for p in prompts))

def stream(cancelled=lambda:False):
 import fsspec,pyarrow.parquet as pq
 fs=fsspec.filesystem('http',block_size=4*1024**2)
 for file,size,sha in FILES:
  url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{file}'
  with fs.open(url,cache_type='readahead') as f:
   if f.size!=size:raise ValueError('Pinned source size changed: '+file)
   pf=pq.ParquetFile(f);index=0
   for batch in pf.iter_batches(batch_size=128,columns=['messages','source']):
    if cancelled():raise InterruptedError('Stop requested during stream preparation')
    for row in batch.to_pylist():
     yield file,index,row;index+=1
   print('[streamed]',file,index,'rows; raw blocks discarded',flush=True)

def candidate(raw,cfg,a,b):
 family=SOURCES.get(raw['source'])
 if family is None:return None,'source'
 msgs=raw['messages'];system='';j=0
 if msgs and msgs[0]['role']=='system':system=msgs[0]['content'].strip();j=1
 if len(msgs)<j+2 or msgs[j]['role']!='user' or msgs[j+1]['role']!='assistant':return None,'roles'
 # Everyday's repeated greeting is not the substantive task; keep its next pair.
 if family=='everyday' and len(msgs)>=j+4 and len(msgs[j]['content'].split())<4:j+=2
 content=msgs[j]['content'].strip();p=content;answer=msgs[j+1]['content'].strip()
 # Keep actual task instructions from the system field; remove only generic persona.
 if system and system not in ('You are a helpful AI assistant.','You are a helpful assistant.'):p=system+'\n\n'+p
 if not 1<=len(p.split())<=400:return None,'prompt_length'
 if not 8<=len(answer.split())<=cfg.get('max_answer_words',256):return None,'answer_length'
 if BLOCK.search(p+' '+answer) or QUANT.search(p):return None,'code_math_scope'
 if MATH.search(p+' '+answer):return None,'math_scope_extended'
 if family=='conversation' and len(re.findall(r'\b\d+(?:\.\d+)?\b',p))>3:return None,'numeric_table_or_problem'
 if ROLEPLAY.search(p) or BAD_RESPONSE.search(answer) or SETUP.search(answer):return None,'roleplay_or_setup'
 if PERSONA.search(content.replace('’',"'")) or (system and family not in ('rewrite','summary') and PERSONA.search(system.replace('’',"'"))):return None,'persona'
 if MEDICAL.search(p+' '+answer) or LIVE.search(p+' '+answer):return None,'specialist_or_live'
 if re.search(r'\b(drugs?|pharmac\w*|metabolites?|midazolam|CYP\w*)\b',p+' '+answer,re.I):return None,'specialist_drug'
 if family=='everyday' and LOCAL_LIVE.search(p+' '+answer):return None,'unverifiable_live_context'
 if family=='conversation' and re.search(r'\b(edit\w*|rewrit\w*|rephras\w*|gramma\w*|punctua\w*|refin\w*|polish\w*)\b',content,re.I):return None,'dedicated_rewrite_source'
 if family in ('summary','rewrite') and not set(re.findall(r'\d+',answer))<=set(re.findall(r'\d+',content)):return None,'new_numbers'
 if family=='summary':
  if not grounded_target(content,answer):return None,'summary_unsupported_words'
  source_words=set(norm(content).split())
  capitals=re.findall(r'\b[A-Z][A-Za-z]+\b',answer)
  allowed=STOPWORDS|{'summary','according','following','despite','meanwhile','however'}
  if any(w.lower() not in source_words|allowed for w in capitals):return None,'summary_new_name'
  if any(w in answer.lower().split() and w not in content.lower().split() for w in ('agrees','agreed','confirmed','scheduled')):return None,'summary_overstated_commitment'
  if 'without using second or third person pronouns' in system and re.search(r'\b(you|your|he|his|him|she|her|they|their|them|it|its)\b',answer,re.I):return None,'summary_instruction'
  sentences=len(re.split(r'(?<=[.!?])\s+(?=[A-Z])',answer.strip()))
  limit=1 if 'one very short sentence' in system else 3
  if sentences>limit:return None,'summary_length_instruction'
 if SPECIAL.search(p+'\n'+answer):return None,'embedded_roles'
 # A complete rewritten email can end with a name rather than punctuation.
 signed=family=='rewrite' and bool(re.search(r'[.!?][\s\S]*\n(?:Best(?: regards)?|Kind regards|Regards|Sincerely|Thanks),?\s*\n[\w .,-]{2,100}$',answer))
 if answer[-1] not in '.!?"”\'’):' and not signed:return None,'incomplete_ending'
 if len(re.findall('[^\x00-\x7f]',p+answer))/len(p+answer)>.08:return None,'language'
 aw=norm(answer).split()
 if len(aw)>=12 and max(Counter(tuple(aw[i:i+4]) for i in range(len(aw)-3)).values())>=3:return None,'repetition'
 if norm(p)==norm(answer):return None,'echo'
 ids=[]
 for encode in (a.encode,b):
  head=encode('User: '+p+'\nAssistant:');joined=encode('User: '+p+'\nAssistant: '+answer)
  if joined[:len(head)]!=head:raise ValueError('Tokenizer boundary changed')
  targets=len(joined)-len(head)+1
  if not cfg['min_answer_tokens']<=targets<=cfg['max_answer_tokens']:return None,'answer_tokens'
  if len(joined)>cfg['context']:return None,'context'
  ids.append(targets)
 # First user text (including input passage), not response, determines split.
 key=fingerprint(norm(content))
 bucket=int(key[:8],16)%100
 split='dev' if bucket<cfg.get('dev_percent',10) else 'test' if bucket<cfg.get('dev_percent',10)+cfg.get('test_percent',10) else 'train'
 identity=fingerprint([raw['source'],p,answer])
 if identity in REVIEW_REJECTED:return None,'manual_review_rejection'
 return dict(id=identity,family=family,prompt=p,content=content,answer=answer,
    group=key,split=split,answer_targets=ids),None

def build(cfg,show_samples=False,cancelled=lambda:False):
 a,b=tokens();blocked,bgrams=exclusions();heaps=defaultdict(list);eligible=Counter();reject=Counter();source_counts=Counter()
 for file,index,raw in stream(cancelled):
  source_counts[raw['source']]+=1
  row,reason=candidate(raw,cfg,a,b)
  if row is None:reject[reason]+=1;continue
  if norm(row['prompt']) in blocked or grams(row['prompt'])&bgrams:reject['evaluation_overlap']+=1;continue
  row.update(file=file,row_index=index)
  key=(row['split'],row['family']);eligible[key]+=1
  n=cfg['training_counts'][row['family']] if row['split']=='train' else cfg['validation_per_source']
  cap=n*3;rank=int(fingerprint([cfg['seed'],row['id']]),16);entry=(-rank,row['id'],index,row)
  if len(heaps[key])<cap:heapq.heappush(heaps[key],entry)
  elif entry>heaps[key][0]:heapq.heapreplace(heaps[key],entry)
  if sum(eligible.values())%20000==0:print('[eligible]',sum(eligible.values()),flush=True)
 pools={s:[] for s in ('train','dev','test')};seen_p=set();seen_groups=set();seen_grams=set()
 # Reserve eval first; remove near-duplicate prompts before accepting training.
 for split in ('test','dev','train'):
  split_grams=set();split_groups=set()
  for family,ntrain in cfg['training_counts'].items():
   n=ntrain if split=='train' else cfg['validation_per_source'];added=0
   for _,_,_,row in sorted(heaps[(split,family)],reverse=True):
    p=norm(row['prompt']);g=grams(row['content'])
    if p in seen_p or row['group'] in seen_groups or g&seen_grams:continue
    pools[split].append(row);seen_p.add(p);split_grams|=g;split_groups.add(row['group']);added+=1
    if added==n:break
   if added!=n:raise ValueError(f'Insufficient {split}/{family}: {added}/{n}; eligible={dict(eligible)}; sources={dict(source_counts)}')
  seen_grams|=split_grams;seen_groups|=split_groups
 random.Random(cfg['seed']).shuffle(pools['train'])
 for s in ('dev','test'):random.Random(cfg['seed']+len(s)).shuffle(pools[s])
 train_tokens=[sum(r['answer_targets'][i] for r in pools['train']) for i in (0,1)]
 if min(train_tokens)<cfg['minimum_answer_targets']:raise ValueError('Insufficient supervised answer tokens')
 stats={s:dict(count=len(rows),families=dict(Counter(r['family'] for r in rows)),answer_targets=[sum(r['answer_targets'][i] for r in rows) for i in (0,1)],hash=fingerprint(rows)) for s,rows in pools.items()}
 manifest=dict(repo=REPO,revision=REV,files=[list(f) for f in FILES],declared_license='Apache-2.0',source_integrity='Revision, file size, selected row content hashes; full-file SHA values are upstream metadata, not rehashed range reads.',
   config=fingerprint(cfg),eligible={str(k):v for k,v in eligible.items()},rejections=dict(reject),source_counts=dict(source_counts),
   stats=stats,selection={s:[dict(id=r['id'],file=r['file'],row=r['row_index']) for r in rows] for s,rows in pools.items()},
   limitations='Conversation-group and 13-word overlap separation, not guaranteed semantic topic isolation. Synthetic data; sampled review is not full factual verification.')
 if show_samples:
  for family in cfg['training_counts']:
   rr=[r for r in pools['train'] if r['family']==family]
   for r in rr[::max(1,len(rr)//20)][:20]:print('[review]',json.dumps({k:r[k] for k in ('id','family','prompt','answer')},ensure_ascii=False),flush=True)
 return pools,manifest

if __name__=='__main__':
 cfg=json.loads((DIR/'config.json').read_text());data,manifest=build(cfg,show_samples=True)
 from sml_v2.common import atomic_json
 atomic_json(DIR/'selection.json',manifest)
 print('[prepared]',json.dumps(manifest['stats']),flush=True)
