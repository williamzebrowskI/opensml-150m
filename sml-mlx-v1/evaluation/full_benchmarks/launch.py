"""Full evaluation on one Mac; immutable models, resumable results, no training."""
import argparse,fcntl,gc,json,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation.full_benchmarks.spec import *
from evaluation.full_benchmarks.core import Backend,Journal,candidate_score,candidate_tokens,frozen,generate,get_rows,grade_ifeval,summary,verify
from sml_v1.common import atomic_json,file_sha256,fingerprint

def check(model,suite,manifest):
 import mlx.core as mx
 rows=get_rows(suite);backend=Backend(model)
 lengths=[]
 if suite=='multiple-choice':
  for row in rows:
   for choice in row['choices']:lengths.append(len(candidate_tokens(backend,row['prompt'],choice)[0]))
  fixture=dict(id='synthetic-check',task='synthetic',prompt='Question: A green scarf is on the bench. What is on the bench?\nAnswer:',choices=['the scarf','the oven'],gold=0)
  from evaluation.full_benchmarks.core import score_mc
  probe=score_mc(backend,fixture)
  # Independent scalar token-by-token score checks the answer mask and shift.
  p='A quiet garden contains';a=' flowers.';ids,n=candidate_tokens(backend,p,a)
  expected=0.
  for i in range(n,len(ids)):
   logits=backend.logits(mx.array([ids[:i]],dtype=mx.int32))[0,-1].astype(mx.float32)
   expected+=float((logits[ids[i]]-mx.logsumexp(logits)).item())
  error=abs(candidate_score(backend,p,a)['log_likelihood']-expected)
  if error>.01:raise ValueError('Likelihood numerical check failed')
  checks=dict(synthetic_choice_probe=probe,tokenwise_score_abs_error=error,candidates=len(lengths),maximum_tokens=max(lengths))
 else:
  for row in rows:
   n=len(backend.encode('User: '+row['prompt']+'\nAssistant:'));lengths.append(n)
   if n+GENERATION_LIMIT>MAX_CONTEXT:raise ValueError('Full IFEval response budget cannot fit; protocol review required')
   # Exercise every official constraint with dummy text, without generating official answers.
   grade_ifeval(row,'This is a setup check. It is not a benchmark answer.')
  fixture=dict(key=-1,prompt='Write the word violet without a comma.',instruction_id_list=['keywords:existence','punctuation:no_comma'],kwargs=[dict(keywords=['violet']),{}])
  good=grade_ifeval(fixture,'violet');bad=grade_ifeval(fixture,'blue,')
  if not good['strict']['follow_all_instructions'] or any(bad['strict']['follow_instruction_list']):raise ValueError('Official verifier fixture failed')
  probe=generate(backend,'Hello! Can I ask you something?',limit=32)
  # Verify the configured decoding path against independent full-prefix scoring.
  ids=backend.encode('User: Hello! Can I ask you something?\nAssistant:');uncached=[]
  for _ in range(32):
   t=int(mx.argmax(backend.logits(mx.array([ids],dtype=mx.int32))[0,-1]).item())
   if t==backend.eos:break
   ids.append(t);uncached.append(t)
  if uncached!=probe['tokens']:raise ValueError('Configured/full-context generation mismatch')
  checks=dict(verifier_rows_checked=len(rows),instruction_count=sum(len(r['instruction_id_list']) for r in rows),maximum_prompt_tokens=max(lengths),context_limited_prompts=0,synthetic_probe=probe,decode_mode=backend.decode_mode,full_context_equal=True)
 del backend;gc.collect();mx.clear_cache();verify(manifest['protected']);verify(manifest['code'])
 receipt=dict(status='passed',contract=manifest,checks=checks,full_evaluation_started=False,training=False)
 atomic_json(DIR/f'readiness_{model}_{suite}.json',receipt)
 print('[check-passed]',json.dumps(checks),flush=True)

def run(model,suite,manifest):
 import mlx.core as mx
 out=RESULTS/model/suite;out.mkdir(parents=True,exist_ok=True)
 mp=out/'manifest.json'
 if (out/'records.jsonl').exists() and not mp.exists():raise ValueError('Results exist without an identity manifest; refusing to adopt them')
 if mp.exists():
  if json.loads(mp.read_text())!=manifest:raise ValueError('Run inputs/code/settings changed; use a new versioned output directory')
 else:atomic_json(mp,manifest)
 rows=get_rows(suite);journal=Journal(out/'records.jsonl',[r['id'] for r in rows])
 if len(journal.records)==len(rows):
  atomic_json(out/'summary.json',summary(journal.records,suite,len(rows)))
  print('[already-complete]',out/'summary.json');return
 stopped=[False];signals=[0]
 def interrupt(*_):
  signals[0]+=1;stopped[0]=True
  print('[stop] Saving completed evaluation items; no model weights are modified.',flush=True)
  if signals[0]>1:raise KeyboardInterrupt
 previous=signal.signal(signal.SIGINT,interrupt);start=time.monotonic();initial=len(journal.records)
 backend=Backend(model)
 print(f'[ready] {model} {suite}: {initial}/{len(rows)}; inference only',flush=True)
 try:
  for row in rows[initial:]:
   if stopped[0]:break
   if suite=='multiple-choice':
    from evaluation.full_benchmarks.core import score_mc
    result=score_mc(backend,row)
   else:
    gen=generate(backend,row['prompt'],stop_requested=lambda:stopped[0])
    result=dict(id=row['id'],generation=gen,**grade_ifeval(row,gen['text']))
   journal.append(result);n=len(journal.records)
   if n%25==0 or n==len(rows):
    print(f'[progress] {n}/{len(rows)}; elapsed {time.monotonic()-start:.1f}s',flush=True)
    atomic_json(out/'summary.json',summary(journal.records,suite,len(rows)));mx.clear_cache()
 except (KeyboardInterrupt,InterruptedError):stopped[0]=True
 finally:
  signal.signal(signal.SIGINT,previous);del backend;gc.collect();mx.clear_cache()
  atomic_json(out/'summary.json',summary(journal.records,suite,len(rows)))
  verify(manifest['protected']);verify(manifest['code'])
  atomic_json(out/'integrity.json',dict(inputs_unchanged=True,training=False,completed=len(journal.records),manifest_sha256=file_sha256(mp)))
 status='finished' if len(journal.records)==len(rows) else 'paused'
 print(f'[{status}] {out / "summary.json"}; rerun the same command to resume.',flush=True)
 print(json.dumps(summary(journal.records,suite,len(rows)),indent=2),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--suite',choices=['multiple-choice','ifeval'],required=True)
 p.add_argument('--model',choices=list(MODELS),default='sft-448')
 action=p.add_mutually_exclusive_group();action.add_argument('--run',action='store_true');action.add_argument('--check',action='store_true')
 args=p.parse_args()
 print(json.dumps(dict(model=args.model,suite=args.suite,mode='run' if args.run else 'check' if args.check else 'plan',source=MODELS[args.model]['bundle'],counts={k:v['rows'] for k,v in DATA.items() if (k=='ifeval')==(args.suite=='ifeval')},training=False,output=str(RESULTS/args.model/args.suite)),indent=2),flush=True)
 if not args.run and not args.check:return
 # Share the existing single-Mac experiment lock with the training launchers.
 with (ROOT/'sft/.experiment.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit('Another experiment is running; finish it before benchmarking.')
  manifest=frozen(args.model,args.suite)
  if args.check:check(args.model,args.suite,manifest);return
  receipt=DIR/f'readiness_{args.model}_{args.suite}.json'
  if not receipt.exists() or json.loads(receipt.read_text())['contract']!=manifest:
   raise ValueError('Readiness check missing or stale; run this command with --check first')
  run(args.model,args.suite,manifest)
if __name__=='__main__':main()
