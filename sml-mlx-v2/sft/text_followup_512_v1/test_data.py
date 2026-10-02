"""CPU checks for plain-text constraints and conversation preservation."""
import unittest
from sft.text_followup_512_v1.data import constraints, verify_text, eligible, turns, visible_prefix, ROOT
from sml_v2.tokenization import Tokenizer

class DataChecks(unittest.TestCase):
    def test_constraint_oracle_rejects_missing_content_and_wrong_format(self):
        for row in constraints('train',48):
            answer=row['messages'][-1]['content'];checks=row['checks']
            self.assertTrue(verify_text(answer,checks)['joint_proxy'])
            self.assertFalse(verify_text('I would be happy to help.',checks)['joint_proxy'])
            changed=answer.replace(checks['required'][0], 'Someone').replace(checks['required'][0].lower(),'someone').replace(checks['required'][0].upper(),'SOMEONE')
            self.assertFalse(verify_text(changed,checks)['content_proxy'])
        r=next(r for r in constraints('train',8) if r['checks']['family']=='bullets')
        self.assertFalse(verify_text(r['messages'][-1]['content']+'\n- Extra item.',r['checks'])['format'])

    def test_text_scope(self):
        for prompt in ('Return JSON with a name.', 'Write Python to sort a list.', 'Calculate the average.', 'Solve for x: 4 + 5 = x'):
            self.assertFalse(eligible([dict(role='user',content=prompt),dict(role='assistant',content='Example')]))
        self.assertTrue(eligible([dict(role='user',content='Write two bullet points about a picnic.'),dict(role='assistant',content='- Pack a blanket.\n- Bring water.')]))

    def test_followup_masks_history_and_keeps_complete_eos(self):
        tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');r=constraints('train',1)[0]
        encoded=turns(tok,r,2048);self.assertEqual(len(encoded),2)
        e=encoded[1];head=tok.encode(visible_prefix(r['messages'][:3]))
        self.assertEqual(e['y'][:len(head)-1],[-100]*(len(head)-1))
        self.assertEqual(e['y'][-1],tok.eos)
        self.assertIn(r['messages'][1]['content'],visible_prefix(r['messages'][:3]))
        self.assertEqual(e['targets'],sum(x!=-100 for x in e['y']))

if __name__=='__main__':unittest.main()
