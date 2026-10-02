"""Numerical, official-verifier and interrupted-result correctness tests."""
import json,math,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mlx.core as mx
from evaluation.full_benchmarks.core import Journal,candidate_score,candidate_tokens,generate,grade_ifeval,score_mc,summary,verify_architecture
from sml_v2.common import file_sha256
from evaluation.full_benchmarks.prepare import preprocess,convert

class Fake:
 @staticmethod
 def encode(text):return [{'A':1,'B':2,'C':3,' ':4,'D':5}[x] for x in text]
 @staticmethod
 def logits(ids):return mx.stack([ids.astype(mx.float32)*(.1*i) for i in range(6)],axis=-1)

class Tests(unittest.TestCase):
 def test_architecture_pins_across_contract_versions(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d)/'model.safetensors.json';base.write_text('{"model":"original"}')
   sha=file_sha256(base)
   contracts=[dict(protected={str(base):sha}),dict(architecture=sha),
              dict(protected={str(base):sha},architecture=sha)]
   for contract in contracts:
    with self.subTest(contract=contract):self.assertEqual(verify_architecture(contract,base),sha)
   for contract in [{},dict(protected={}),dict(protected=None),dict(architecture=None),
                    dict(architecture='0'*64),dict(protected={str(base):'0'*64}),
                    dict(protected={str(base):sha},architecture='0'*64),
                    dict(protected={str(base):'0'*64},architecture=sha)]:
    with self.subTest(contract=contract),self.assertRaises(ValueError):verify_architecture(contract,base)
   base.write_text('{"model":"changed"}')
   for contract in contracts:
    with self.subTest(contract=contract),self.assertRaisesRegex(ValueError,'metadata changed'):
     verify_architecture(contract,base)
 def test_full_context_generation_eos_limit_and_interrupt(self):
  from types import SimpleNamespace
  class Model:
   def __init__(self):self.prefixes=[]
   def logits(self,ids):
    self.prefixes.append(ids.tolist()[0])
    token=1 if ids.shape[1]<4 else 0
    scores=mx.array([10. if i==token else -10. for i in range(3)])
    return mx.broadcast_to(scores,(1,ids.shape[1],3)),None
   def step(self,*a,**kw):raise AssertionError('Full-context generation used KV step')
  model=Model()
  b=SimpleNamespace(model=model,decode_mode='full_context',encode=lambda _: [2,2],eos=0,
                    tokenizer=SimpleNamespace(decode=lambda ids: 'x'*len(ids)),logits=lambda ids:model.logits(ids)[0])
  result=generate(b,'fixture',limit=5)
  self.assertEqual(result['tokens'],[1,1]);self.assertEqual(result['stop_reason'],'eos')
  self.assertEqual(model.prefixes,[[2,2],[2,2,1],[2,2,1,1]])
  result=generate(b,'fixture',limit=1)
  self.assertEqual(result['tokens'],[1]);self.assertEqual(result['stop_reason'],'length')
  with self.assertRaises(InterruptedError):generate(b,'fixture',stop_requested=lambda:True)
 def test_exact_likelihood_shift_mask_and_normalization(self):
  b=Fake();r=candidate_score(b,'AB','C')
  # Joint tokens A B space C; supervised targets are space and C only.
  expected=sum(target*(.1*previous)-math.log(sum(math.exp(k*(.1*previous)) for k in range(6))) for previous,target in [(2,4),(4,3)])
  self.assertAlmostEqual(r['log_likelihood'],expected,places=5)
  self.assertEqual(r['answer_tokens'],2)
  self.assertEqual(r['normalized'],r['log_likelihood'])
 def test_trailing_whitespace_boundary_and_overflow(self):
  ids,n=candidate_tokens(Fake(),'AB ','C')
  self.assertEqual((ids,n),([1,2,4,4,3],2))
  with self.assertRaises(ValueError):candidate_tokens(Fake(),'A'*2048,'C')
  with self.assertRaises(ValueError):candidate_tokens(Fake(),'A','')
 def test_first_max_tie_break(self):
  row=dict(id='x',task='test',prompt='A',choices=['C','C'],gold=0)
  r=score_mc(Fake(),row);self.assertEqual((r['prediction'],r['prediction_norm']),(0,0))
 def test_interrupted_journal_resumes_once(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'records.jsonl';j=Journal(p,['a','b','c']);j.append(dict(id='a',value=1))
   with p.open('ab') as f:f.write(b'{"id":"b",')
   resumed=Journal(p,['a','b','c']);self.assertEqual(len(resumed.records),1)
   resumed.append(dict(id='b',value=2));resumed.append(dict(id='c',value=3))
   self.assertEqual([r['value'] for r in Journal(p,['a','b','c']).records],[1,2,3])
   with self.assertRaises(ValueError):resumed.append(dict(id='c'))
   with self.assertRaises(ValueError):Journal(p,['different'])
 def test_ifeval_official_positive_and_negative(self):
  row=dict(key=-7,prompt='Write violet without a comma.',instruction_id_list=['keywords:existence','punctuation:no_comma'],kwargs=[dict(keywords=['violet']),{}])
  self.assertTrue(grade_ifeval(row,'violet')['strict']['follow_all_instructions'])
  self.assertEqual(grade_ifeval(row,'blue,')['strict']['follow_instruction_list'],[False,False])
 def test_ifeval_strict_loose_and_denominators(self):
  row=dict(key=-9,prompt='Use JSON.',instruction_id_list=['detectable_format:json_format'],kwargs=[{}])
  result=grade_ifeval(row,'Here is the result.\n{"ok": true}')
  self.assertFalse(result['strict']['follow_all_instructions']);self.assertTrue(result['loose']['follow_all_instructions'])
  records=[dict(id='one',generation=dict(stop_reason='eos'),**result)]
  report=summary(records,'ifeval',541)
  self.assertEqual(report['status'],'partial');self.assertEqual(report['strict']['instruction_total'],1)
  self.assertEqual(report['loose']['prompt_accuracy'],1)
 def test_hellaswag_preprocessing(self):
  self.assertEqual(preprocess('  Sailing [title] [noise] A  boat.  '),'Sailing.  A boat.')
 def test_partial_mc_is_not_full_score(self):
  r=summary([dict(id='arc_easy:0',task='arc_easy',correct=True,correct_norm=False)],'multiple-choice',15428)
  self.assertEqual(r['status'],'partial');self.assertEqual(r['tasks']['arc_easy']['expected'],2376)
  self.assertEqual(r['tasks']['arc_easy']['acc'],1)
 def test_runner_interrupt_resume_and_changed_manifest(self):
  from evaluation.full_benchmarks import launch
  rows=[dict(id=f'arc_easy:{i}',task='arc_easy',prompt='A',choices=['C','D'],gold=0) for i in range(3)]
  def result(row):return dict(id=row['id'],task=row['task'],correct=True,correct_norm=True)
  manifest=dict(protected={},code={},training=False)
  with tempfile.TemporaryDirectory() as d, patch.object(launch,'RESULTS',Path(d)), patch.object(launch,'Backend',return_value=Fake()), patch.object(launch,'get_rows',return_value=rows):
   with patch('evaluation.full_benchmarks.core.score_mc',side_effect=[result(rows[0]),KeyboardInterrupt()]):
    launch.run('test','multiple-choice',manifest)
   output=Path(d)/'test/multiple-choice'
   self.assertEqual(json.loads((output/'summary.json').read_text())['status'],'partial')
   with patch('evaluation.full_benchmarks.core.score_mc',side_effect=lambda _,r:result(r)) as mocked:
    launch.run('test','multiple-choice',manifest)
    self.assertEqual(mocked.call_count,2)
   self.assertEqual(len((output/'records.jsonl').read_text().splitlines()),3)
   with self.assertRaisesRegex(ValueError,'changed'):
    launch.run('test','multiple-choice',dict(manifest,changed=True))

if __name__=='__main__':unittest.main()
