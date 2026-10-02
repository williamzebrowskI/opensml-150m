"""Pinned-source screens, also allowing complete replies shorter than eight words."""
from sft.natural_control.data import *

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
 if not 1<=len(answer.split())<=cfg.get('max_answer_words',256):return None,'answer_length'
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
