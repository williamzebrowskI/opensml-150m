"""CPU-only checks of data boundaries, run lifecycle and candidate retention."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from sml_v2.common import atomic_json,fingerprint,read_json
from sft.natural_control import data
from sft.natural_control.protocol import validate,DIR,arms

class Encoding:
    def encode(self,s):return list(s.encode())

class DataTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read_json(DIR/'config.json');self.cfg.update(min_answer_tokens=1,max_answer_tokens=1000,context=2000)
    def row(self,messages,source='smol-magpie-ultra-short'):
        return data.candidate(dict(messages=messages,source=source),self.cfg,Encoding(),Encoding().encode)
    def test_mask_source_filters_and_complete_signed_email(self):
        good=[dict(role='system',content='Rewrite the following politely.'),dict(role='user',content='Please send the meeting notes when you have a chance.'),dict(role='assistant',content='Could you please share the meeting notes when you have a moment?\n\nBest,\nErin')]
        row,why=self.row(good,'smollm-rewrite-30k');self.assertIsNone(why);self.assertIn('Rewrite',row['prompt'])
        self.assertEqual(row['content'],good[1]['content'])
        for text in ('Write a Python program to sort these values.','Explain some maths and fractions to me.'):
            bad=[dict(role='user',content=text),dict(role='assistant',content='Here is a long enough explanation of the requested topic.')]
            self.assertEqual(self.row(bad)[1],'code_math_scope')
    def test_everyday_group_uses_substantive_turn(self):
        rows=[]
        for topic in ('Tell me about the garden beside your school.','Tell me about the birds beside the river.'):
            msgs=[dict(role='user',content='Hello'),dict(role='assistant',content='Hi!'),dict(role='user',content=topic),dict(role='assistant',content='A peaceful outdoor place can be a pleasant spot to relax.')]
            row,why=self.row(msgs,'everyday-conversations');self.assertIsNone(why);rows.append(row)
        self.assertNotEqual(rows[0]['group'],rows[1]['group'])
    def test_reviewed_failures_are_rejected(self):
        for prompt in ("Play the role of a gruff retired officer.","You're a detective working for the FBI.","You are Arthur and won a prize."):
            row,reason=self.row([dict(role='user',content=prompt),dict(role='assistant',content='I have worked in this field for many years and welcome you.')])
            self.assertIsNone(row);self.assertIn(reason,('persona','roleplay_or_setup'))
        raw=[dict(role='system',content='Extract the main point in one very short sentence.'),dict(role='user',content='The garden gate will stay closed until Friday for repairs.'),dict(role='assistant',content='A golden monument was discovered at a virtual meeting on Monday.')]
        self.assertEqual(self.row(raw,'smol-summarize-20k')[1],'summary_unsupported_words')
        row,reason=self.row([dict(role='user',content='Is a storm coming soon?'),dict(role='assistant',content='A storm is forecast to arrive in our area tonight.')],'everyday-conversations')
        self.assertIsNone(row);self.assertEqual(reason,'specialist_or_live')
        row,reason=self.row([dict(role='user',content='What is the current traffic situation in the city?'),dict(role='assistant',content='The city has moderate traffic with roadwork on Main Street.')],'everyday-conversations')
        self.assertIsNone(row);self.assertEqual(reason,'unverifiable_live_context')
        row,reason=self.row([dict(role='system',content='Extract the main point in one very short sentence.'),dict(role='user',content='Lucas wants to improve the app and meet on Wednesday.'),dict(role='assistant',content='Lucas wants to improve the AirGuard app and meet on Wednesday.')],'smol-summarize-20k')
        self.assertIsNone(row);self.assertEqual(reason,'summary_new_name')
        row,reason=self.row([dict(role='system',content='Extract the main point in one very short sentence.'),dict(role='user',content='Olivia is free to meet at the library on Saturday. Please let Olivia know if this works.'),dict(role='assistant',content='Olivia agrees to meet at the library on Saturday.')],'smol-summarize-20k')
        self.assertIsNone(row);self.assertEqual(reason,'summary_overstated_commitment')
        row,reason=self.row([dict(role='user',content='What is the value of the expression 2^(3^2)?'),dict(role='assistant',content='The value is five hundred twelve when powers are evaluated inside first.')])
        self.assertIsNone(row);self.assertEqual(reason,'math_scope_extended')
        row,reason=self.row([dict(role='user',content='Consider this dataset of plastics usage over several years.'),dict(role='assistant',content='I am ready to analyze this table and answer your questions.')])
        self.assertIsNone(row);self.assertEqual(reason,'math_scope_extended')
    def test_configuration_exposure(self):
        cfg=read_json(DIR/'config.json');validate(cfg);self.assertEqual(len(arms(cfg)),4)
        self.assertEqual(cfg['updates']*cfg['batch'],20480)
    def test_semantic_probe_split_not_shared(self):
        probes=read_json(DIR/'probes.json')
        self.assertFalse({data.norm(r['prompt']) for r in probes['dev']}&{data.norm(r['prompt']) for r in probes['test']})
        for rows in probes.values():
            self.assertEqual(len(rows),40)
            self.assertTrue(all(r['rubric'] and r['reference'] for r in rows))
    def test_manifest_survives_json_roundtrip_and_shared_directive(self):
        cfg=dict(training_counts={'conversation':2},validation_per_source=1,seed=3,minimum_answer_targets=0)
        shared='Please read the entire passage carefully and then provide a complete response using only the supplied information. '
        raw=[]
        for i,split in enumerate(('train','train','dev','test')):
            content=f'Unique input {i}'
            row=dict(id=str(i),family='conversation',prompt=shared+content,content=content,answer='A complete answer.',group=str(i),split=split,answer_targets=[4,5])
            raw.append(('fixture.parquet',i,dict(source='fixture',candidate=row)))
        with patch.object(data,'stream',return_value=iter(raw)),patch.object(data,'tokens',return_value=(None,None)),patch.object(data,'exclusions',return_value=(set(),set())),patch.object(data,'candidate',side_effect=lambda r,*args:(r['candidate'],None)):
            pools,manifest=data.build(cfg)
        self.assertEqual(manifest,json.loads(json.dumps(manifest)))
        self.assertEqual([len(pools[s]) for s in ('train','dev','test')],[2,1,1])

class LifecycleTests(unittest.TestCase):
    def test_pruning_keeps_all_evaluation_candidates_and_latest(self):
        from sft.natural_control.launch import prune_checkpoints
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for i,step in enumerate([0,160,320,480,640,800,960,1120,1280,1280]):
                p=root/f'step_{step:07d}_{i:04d}';p.mkdir()
                atomic_json(p/'manifest.json',dict(format='sml-pretrain-bundle-v1',created_ns=i,step=step))
            foreign=root/'step_ignored';foreign.mkdir();(foreign/'unrelated').touch()
            prune_checkpoints(root,dict(evaluation_updates=[0,320,640,960,1280]))
            steps=[read_json(p/'manifest.json')['step'] for p in root.glob('step_*') if (p/'manifest.json').exists()]
            self.assertTrue({0,320,640,960,1280}<=set(steps));self.assertTrue(foreign.exists());self.assertNotIn(480,steps)
    def test_default_launcher_only_prints_plan(self):
        from sft.natural_control.launch import main
        with patch('sys.argv',['launch.py']),patch('sft.natural_control.launch.preflight',side_effect=AssertionError('should not run')):
            with patch('builtins.print'):main()

class TrainingLifecycleTests(unittest.TestCase):
    def test_stop_resume_and_completed_fit_do_not_repeat_examples(self):
        # Fake updates/evaluation, real launcher control flow. Real tensor save /
        # next-update parity is tested separately by check.py on both full models.
        import contextlib
        from unittest.mock import MagicMock
        from sft.natural_control.launch import run_arm
        seen=[];stop={'requested':False};evaluated=[];counter=[0]
        cfg=dict(updates=2,batch=1,microbatch=1,seed=7,checkpoint_every=1,evaluation_updates=[0,1,2])
        arm=dict(name='fixture',model='opensml',peak_lr=0.1);frozen={'fixture':True}
        def update(b,opt,rows,cfg,step,peak):
            seen.extend(r['id'] for r in rows)
            if step==1:stop['requested']=True
            return dict(step=step,loss=1.,lr=peak)
        def save(root,model,opt,meta,stream,**kw):
            counter[0]+=1;p=root/f"step_{meta['step']:07d}_{counter[0]:04d}";p.mkdir()
            atomic_json(p/'manifest.json',dict(format='sml-pretrain-bundle-v1',created_ns=counter[0],step=meta['step']))
            atomic_json(p/'meta.json',meta);atomic_json(root/'latest.json',dict(path=str(p)));return str(p)
        def restore(b,opt,path,*args):
            meta=read_json(Path(read_json(path)['path'])/'meta.json');return meta['step'],meta['training']
        def assess(*args):
            evaluated.append(len(seen))
            return dict(natural_likelihood={},prose_nll=0,structure={},reading_exact_diagnostic={})
        with tempfile.TemporaryDirectory() as temp,contextlib.ExitStack() as stack:
            out=Path(temp)
            for name,value in [('sft.natural_control.engine.load',MagicMock()),('sft.natural_control.engine.optimizer',MagicMock()),
                ('sft.natural_control.engine.update',update),('sft.natural_control.evaluate.assess',assess),
                ('sml_v2.checkpoint_bundle.save_bundle',save),('sft.natural_control.launch.restore',restore),
                ('builtins.print',MagicMock())]:stack.enter_context(patch(name,value))
            data={'train':[dict(id='first'),dict(id='second')]}
            run_arm(arm,cfg,frozen,data,stop,out)
            self.assertEqual(seen,['first']);self.assertEqual(read_json(out/'fixture/report.json')['status'],'stopped')
            stop['requested']=False;run_arm(arm,cfg,frozen,data,stop,out)
            self.assertEqual(seen,['first','second']);self.assertEqual(evaluated,[0,1,2])
            self.assertEqual(read_json(out/'fixture/report.json')['status'],'complete')
            run_arm(arm,cfg,frozen,data,stop,out);self.assertEqual(seen,['first','second'])

if __name__=='__main__':unittest.main()
