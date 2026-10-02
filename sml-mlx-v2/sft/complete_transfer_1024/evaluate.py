"""New complete-content tasks and saved natural answers; reused skill retention."""
import re
from collections import Counter
from sft.skill_recovery_768.evaluate import assess as legacy_assess,eligible as legacy_eligible
from sft.reading_repair.generation import generate
from sft.transfer_control.engine import encode,arrays,objective

def grade(row,text,stop):
 if row['format']=='paragraphs':parts=[p.strip() for p in text.strip().split('\n\n') if p.strip()];formatted=all('\n' not in p for p in parts)
 else:
  lines=text.strip().splitlines();parts=[];formatted=True
  for i,line in enumerate(lines):
   prefix='- ' if row['format']=='bullets' else f'{i+1}. '
   if not line.startswith(prefix):formatted=False
   parts.append(line[len(prefix):].strip() if line.startswith(prefix) else line.strip())
 content=parts==row['sentences'];complete=len(parts)==len(row['sentences']) and content
 return dict(content=content,format_ok=formatted and len(parts)==len(row['sentences']),complete=complete,passed=content and complete and formatted and stop=='end')

def repetition(text):
 words=text.lower().split();counts=Counter(tuple(words[i:i+5]) for i in range(len(words)-4));return bool(counts and max(counts.values())>=3)

def assess(b,data,split,cfg,cancelled=lambda:False):
 result=legacy_assess(b,data,split,dict(cfg,max_new_tokens=128),cancelled)
 natural=[];complete=[]
 for family,dest in [('natural',natural),('complete',complete)]:
  for row in data['new'][split][family]:
   if cancelled():raise InterruptedError('Evaluation stopped')
   g=generate(b,row['prompt'],cfg['max_new_tokens']);x,y=arrays([encode(b,row,cfg['context'])],b.pad)
   loss=float(objective(b.model,x,y).item())
   dest.append(dict(**row,generation=g,assistant_nll=loss,repeated=repetition(g['text']),**(grade(row,g['text'],g['stop']) if family=='complete' else {})))
 result['complete_answers']=complete;result['natural_review_answers']=natural
 m=result['metrics'];m.update(complete_pass=sum(r['passed'] for r in complete)/len(complete),complete_content=sum(r['content'] for r in complete)/len(complete),complete_format=sum(r['format_ok'] for r in complete)/len(complete),natural_nll=sum(r['assistant_nll'] for r in natural)/len(natural),natural_stopped=sum(r['generation']['stop']=='end' for r in natural)/len(natural),natural_repeated=sum(r['repeated'] for r in natural)/len(natural),natural_mean_tokens=sum(r['generation']['tokens'] for r in natural)/len(natural))
 result['limitation']='Complete tasks check exact preservation of source content and structure, not open-ended truth. Natural replies require semantic review. Historical skills diagnostics reused. No automatic best or promotion.'
 return result

def eligible(m,b,cfg):return legacy_eligible(m,b,cfg) and m['natural_repeated']<=b['natural_repeated']+.05 and m['natural_stopped']>=b['natural_stopped']-.05
