"""Protocol tests; no benchmark generation or judge/model loading."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('mt_runner',Path(__file__).with_name('run.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class ProtocolTests(unittest.TestCase):
    def test_official_questions_and_references(self):
        qq=r.questions();refs={x['question_id'] for x in r.rows(r.DIR/'upstream/reference.jsonl')}
        self.assertTrue(all(q['question_id'] in refs for q in qq if q['category'] in ('math','coding','reasoning')))

    def test_own_history_and_reference_history_are_distinct(self):
        q=dict(turns=['Remember Maya.','What name?'])
        answer=dict(generations=[dict(text='I remember Maya.'),dict(text='Maya')])
        ref=dict(choices=[dict(turns=['Hello Maya','Your name is Maya'])])
        _,prompt=r.make_judge_prompt(q,answer,1,ref)
        self.assertIn('Assistant: I remember Maya.',prompt)
        self.assertIn('Latest user request: What name?',prompt)
        self.assertIn('Reference first answer (context):\nHello Maya',prompt)

    def test_no_reference_and_malformed_scores(self):
        _,prompt=r.make_judge_prompt(dict(turns=['Hello']),dict(generations=[dict(text='Hi')]),0)
        self.assertNotIn('Reference Answer (Score 5)',prompt)
        self.assertEqual(r.parse_score('Feedback [RESULT] 5\n'),5)
        for text in ['5','[RESULT] 10','[RESULT] 0','[RESULT] 3 or 5','No score']:
            self.assertIsNone(r.parse_score(text))

    def test_resume_tail_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'answers.jsonl';j=r.Journal(p,[1,2]);j.append(dict(id=1))
            with p.open('ab') as f:f.write(b'{"id":')
            j=r.Journal(p,[1,2]);self.assertEqual(len(j.records),1)
            self.assertTrue(p.with_suffix('.incomplete-tail').exists())
            with self.assertRaises(ValueError):j.append(dict(id=1))
            j.append(dict(id=2));self.assertEqual(len(r.Journal(p,[1,2]).records),2)
            with self.assertRaises(ValueError):r.Journal(p,[2,1])

    def test_standard_answer_export_and_partial_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);q=r.questions()[0]
            row=dict(id=q['question_id'],answer_id='abc',tstamp=0,generations=[dict(text='First',stop_reason='eos'),dict(text='Second',stop_reason='eos')])
            r.Journal(out/'answers.jsonl',[q['question_id']]).append(row)
            r.export_answers(out,[row],'test')
            saved=r.rows(out/'model_answer/test.jsonl')[0]
            self.assertEqual(saved['choices'],[dict(index=0,turns=['First','Second'])])
            summary=r.summarize(out,r.questions());self.assertIsNone(summary['overall_score'])
            self.assertEqual(summary['status'],'partial')

if __name__=='__main__':unittest.main()
