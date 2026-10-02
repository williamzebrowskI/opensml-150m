import unittest
from collections import Counter
from sft.teacher_chat_768_v1.data import batch_at,decode_object,validate_review,turns,visible_prefix,annotate
from sft.teacher_chat_768_v1.launch import arm_config
from sml_v1.tokenization import Tokenizer
from sft.teacher_chat_768_v1.data import ROOT

class PipelineTests(unittest.TestCase):
    def test_balanced_exposures_and_resume_cursor(self):
        data={'train':[dict(id=f'{s}{i}',source=s) for s in ('teacher','replay') for i in range(2048)]}
        cfg=dict(seed=4,updates=512,batch_conversations=16)
        batches=[batch_at(data,cfg,u) for u in range(512)]
        self.assertTrue(all(Counter(r['source'] for r in b)=={'teacher':8,'replay':8} for b in batches))
        self.assertEqual(set(Counter(r['id'] for b in batches for r in b).values()),{2})
        self.assertEqual(batches[64],batch_at(data,cfg,64))
        self.assertNotEqual(batches[0],batches[256])

    def test_student_error_history_never_target(self):
        tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
        messages=[dict(role='user',content='My scarf is green.'),dict(role='assistant',content='Your scarf is blue.'),
                  dict(role='user',content='What color did I say?'),dict(role='assistant',content='Green.')]
        enc=turns(tok,dict(messages=messages,last_only=True),2048)
        self.assertEqual(len(enc),1);self.assertEqual(enc[0]['message_index'],3)
        head=tok.encode(visible_prefix(messages[:-1]));e=enc[0]
        self.assertEqual(e['y'][:len(head)-1],[-100]*(len(head)-1))
        self.assertEqual(e['y'][-1],tok.eos)
        self.assertNotIn('blue',tok.decode([t for t in e['y'] if t>=0]).lower())

    def test_rejects_ambiguous_teacher_review(self):
        fields=['scope','first_complete','final_complete','history_correct','supported','instruction_following']
        good={k:True for k in fields};self.assertTrue(validate_review(good))
        self.assertFalse(validate_review(dict(good,supported=False)))
        with self.assertRaises(ValueError):validate_review(dict(good,supported='true'))
        with self.assertRaises(ValueError):decode_object('[1,2]')
        with self.assertRaises(ValueError):decode_object('prefix {"answer":"ok"}')

    def test_repetitive_student_context_is_allowed_but_target_is_not(self):
        tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');cfg=dict(context=2048,max_assistant_tokens=512)
        repeated='I would be happy to help you with that. '*8
        messages=[dict(role='user',content='Write a birthday greeting.'),dict(role='assistant',content=repeated),
                  dict(role='user',content='Please write the greeting itself.'),dict(role='assistant',content='Happy birthday! Have a wonderful day.')]
        row=annotate(tok,dict(messages=messages,last_only=True),cfg)
        self.assertEqual(row['assistant_turns'],1)
        messages[-1]=dict(role='assistant',content=repeated)
        with self.assertRaises(ValueError):annotate(tok,dict(messages=messages,last_only=True),cfg)

    def test_rates_do_not_change_data_or_full_schedule(self):
        cfg=dict(updates=512,seed=4,rates={'a':3e-7,'b':1e-6})
        a=arm_config(cfg,'a');b=arm_config(cfg,'b')
        self.assertEqual(a['updates'],512);self.assertEqual(a['seed'],b['seed'])
        self.assertAlmostEqual(a['final_lr'],3e-8)

if __name__=='__main__':unittest.main()
