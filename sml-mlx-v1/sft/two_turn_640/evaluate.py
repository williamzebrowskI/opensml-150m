"""Paired follow-ups and own-history generations; actual answers always saved."""
from collections import Counter,defaultdict
from sft.answer_completion.evaluate import teacher_stats,summary,normalized
from sft.reading_repair.generation import generate
from sft.transfer_control.engine import prose_loss
from sft.transfer_control.launch import prose_texts
from .data import history

def assess(b,data,cfg,cancelled=lambda:False):
 b.model.eval();result={}
 for split,rows in [('development',data['dev']),('retention',data['retention'])]:
  records=[]
  for row in rows:
   if cancelled():raise InterruptedError('Evaluation stopped')
   t=teacher_stats(b,row,cfg);g=generate(b,row['prompt'],cfg['max_new_tokens'])
   words=g['text'].lower().split();grams=Counter(tuple(words[i:i+5]) for i in range(len(words)-4))
   records.append(dict(**row,teacher=t,generation=g,exact_reference=normalized(g['text'])==normalized(row['answer']) and g['stop']=='end',repeated=bool(grams and max(grams.values())>=2)))
  result[split]=dict(metrics=summary(records),answers=records)
 dev=result['development']['answers'];first={r['group']:r['generation']['text'] for r in dev if r['turn']==0};pairs=defaultdict(list);own=[]
 for row in dev:
  if row['turn']==0:continue
  if cancelled():raise InterruptedError('Evaluation stopped')
  pairs[row['group']].append(row)
  p=history(row['first_prompt'],first[row['group']],row['current']);g=generate(b,p,cfg['max_new_tokens'])
  own.append(dict(id=row['id'],family=row['family'],first_answer=first[row['group']],prompt=p,reference=row['answer'],generation=g,exact_reference=normalized(g['text'])==normalized(row['answer']) and g['stop']=='end'))
 result['own_history']=own
 result['metrics']=dict(development=result['development']['metrics'],retention=result['retention']['metrics'],paired_both_exact=sum(all(r['exact_reference'] for r in rr) for rr in pairs.values())/len(pairs),paired_groups=len(pairs),followup_exact=sum(r['exact_reference'] for r in dev if r['turn']==1)/(2*len(pairs)),own_history_exact=sum(r['exact_reference'] for r in own)/len(own),prose_nll=prose_loss(b,prose_texts()))
 result['families']={f:dict(followups=len(rr),exact=sum(r['exact_reference'] for r in rr)/len(rr)) for f in {r['family'] for r in dev} for rr in [[r for r in dev if r['family']==f and r['turn']==1]]}
 result['note']='Exact-reference scores are strict diagnostics, not semantic chat grades. Alternative correct wording can fail. Own-history exact includes first-reply errors. Review actual complete answers; no automatic selection. Retention/prose checks are reused; test stays unevaluated.'
 return result
