"""CPU checks for the training boundary and selection safeguards."""
import unittest
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v2.tokenization import Tokenizer
from sft.conversation_foundation_v1.data import turns,visible_prefix,basic_quality,group_id,batch_at

class TrainingBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')

    def test_history_is_conditioning_not_target(self):
        messages=[dict(role='system',content='Answer briefly.'),dict(role='user',content='My name is Lena.'),
                  dict(role='assistant',content='Hello, Lena!'),dict(role='user',content='What is my name?'),
                  dict(role='assistant',content='Your name is Lena.')]
        encoded=turns(self.tok,dict(messages=messages),2048)
        self.assertEqual(len(encoded),2)
        for e,i in zip(encoded,(2,4)):
            prefix=visible_prefix(messages[:i]);head=self.tok.encode(prefix)
            joined=self.tok.encode(prefix+' '+messages[i]['content'])+[self.tok.eos]
            self.assertEqual(e['x'],joined[:-1])
            self.assertEqual(e['y'][:len(head)-1],[-100]*(len(head)-1))
            self.assertEqual([t for t in e['y'] if t!=-100],joined[len(head):])
            self.assertEqual(e['y'][-1],self.tok.eos)

    def test_overflow_rejected_not_silently_cut(self):
        row=dict(messages=[dict(role='user',content='Tell a story.'),dict(role='assistant',content='a long story '*200)])
        with self.assertRaises(OverflowError):turns(self.tok,row,32)

    def test_group_invariant_to_answer(self):
        a=[dict(role='user',content='Please explain how a bicycle works.'),dict(role='assistant',content='First answer.')]
        b=[a[0],dict(role='assistant',content='Another answer.')]
        self.assertEqual(group_id(a),group_id(b))

    def test_bad_target_filters(self):
        prompt=dict(role='user',content='Explain an idea.')
        self.assertEqual(basic_quality([prompt,dict(role='assistant',content='```python\nx=1')]),'unclosed_code_fence')
        self.assertEqual(basic_quality([prompt,dict(role='assistant',content=('one two three four five six seven eight nine '*8))]),'repetitive_target')
        self.assertIsNone(basic_quality([prompt,dict(role='assistant',content='Here is a complete explanation.')]))

    def test_resume_cursor_covers_each_conversation_once(self):
        data=dict(train=[dict(id=i) for i in range(32)]);cfg=dict(updates=2,batch_conversations=16)
        self.assertEqual([r['id'] for u in range(2) for r in batch_at(data,cfg,u)],list(range(32)))
        self.assertEqual(batch_at(data,cfg,1)[0]['id'],16)

if __name__=='__main__':unittest.main()
