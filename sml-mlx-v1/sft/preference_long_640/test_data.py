import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from sml_v1.common import read_json
from sml_v1.tokenization import Tokenizer
from sft.preference_long_640.data import DIR,ROOT,encode_reply
from sft.preference_long_640.verified import partition as preference_split,passes

class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=read_json(DIR/'config.json');cls.data=read_json(DIR/'prepared.json');cls.tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    def test_pairs_have_curriculum_then_shuffled_pass(self):
        rows=self.data['train']['preference'];n=self.cfg['preference_train_pairs']
        self.assertEqual(len(rows),n*2)
        self.assertTrue(all(c==2 for c in Counter(r['id'] for r in rows).values()))
        self.assertEqual({r['id'] for r in rows[:n]},{r['id'] for r in rows[n:]})
        self.assertEqual([r['difficulty'] for r in rows[:n]],sorted(r['difficulty'] for r in rows[:n]))
        self.assertNotEqual([r['id'] for r in rows[:n]],[r['id'] for r in rows[n:]])
    def test_split_and_benchmark_exclusion(self):
        from sft.grounded_rank_384.data import exclusions,overlaps
        denied,grams,_,_=exclusions();groups={}
        for split in ('train','dev','test'):
            rr=self.data[split]['preference'];groups[split]={r['group'] for r in rr}
            for r in {r['id']:r for r in rr}.values():
                self.assertEqual(preference_split(r['group']),split)
                self.assertTrue(passes(r,r['answer']))
                self.assertFalse(passes(r,r['rejected']))
                self.assertFalse(overlaps([r['prompt'],r['answer'],r['rejected']],denied,grams))
        for a,b in (('train','dev'),('train','test'),('dev','test')):self.assertFalse(groups[a]&groups[b])
        from sft.grounded_rank_384.data import norm
        texts={s:{norm(r['prompt']) for r in self.data[s]['preference']} for s in groups}
        for a,b in (('train','dev'),('train','test'),('dev','test')):self.assertFalse(texts[a]&texts[b])
    def test_complete_answer_masks_and_no_truncation(self):
        prompt='Return the word green.';answer='green'
        e=encode_reply(self.tok,prompt,answer,1024);h=self.tok.encode('User: '+prompt+'\nAssistant:')
        self.assertEqual(e['y'][:len(h)-1],[-100]*(len(h)-1));self.assertEqual(e['y'][-1],self.tok.eos)
        self.assertEqual(e['targets'],len(e['y'])-len(h)+1)
        with self.assertRaises(ValueError):encode_reply(self.tok,prompt,'green '*2000,32)
    def test_replay_full_history_and_retention_family_counts(self):
        from sft.intact_smoltalk_base_pilot.data import turns
        for family,n in self.cfg['per_update'].items():self.assertEqual(len(self.data['train'][family]),self.cfg['updates']*n)
        for row in {r['id']:r for r in self.data['train']['replay']}.values():
            parts=turns(self.tok,row,self.cfg['context'])
            self.assertEqual(len(parts),sum(m['role']=='assistant' for m in row['messages']))
            self.assertTrue(all(p['y'][-1]==self.tok.eos for p in parts))

if __name__=='__main__':unittest.main()
