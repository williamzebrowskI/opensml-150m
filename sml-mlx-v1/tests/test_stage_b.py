import copy
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from sml_v1.common import atomic_json, fingerprint, read_json, file_sha256
from sml_v1.corpus import content_key, partition
from sml_v1.stage_b_data import select_document, document_keys, StageBStreamingTokens, POLICY
from sml_v1.stage_b import KEY, stage_recipe, training_manifest, migrate_cursor, end_step
from sml_v1.stream import stream_manifest, StreamingTokens, PrefetchedStream, validation_batches
from sml_v1.recipe import tokens_per_update, learning_rate, validate
from sml_v1.continuation import check_resume
from sml_v1.lr_scheduler import DEFAULTS, adaptive_recipe, restore_scheduler
from sml_v1.data_review import reviewed_recipe
from tests.test_five_mac_training import contract
from tests.test_lr_trial import parent
from tests import test_pipeline as fixtures

PROSE = ('A visitor walks through a quiet garden and watches birds settling among the branches. '
         'The gardener explains how fallen leaves protect the soil during winter. Nearby trees provide '
         'shelter for small animals, while flowers attract insects throughout the warmer seasons. '
         'Careful observation helps the visitor understand how these living things depend on one another.')


def train_text(text):
    for i in range(1000):
        candidate = f'{text}\nThis passage belongs to collection {i}.'
        if partition(content_key(candidate)) == 'train':
            return candidate
    raise AssertionError('No training fixture')


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(min_chars=200, max_bytes=200000)
        self.source = dict(label='tutorial', text_field='rollout_results[0].text', config='tutorial')
        self.original = train_text(PROSE)

    def row(self, text, finish='stop'):
        return dict(text=self.original, id='same-origin', url='https://example.org/garden',
                    rollout_results=[dict(text=train_text(text), finish_reason=finish)])

    def test_reads_generated_text_and_rejects_bad_topics_and_outputs(self):
        text='1. Read the following explanation.\n'+PROSE+'\n2. Observe the garden carefully.\n'+PROSE.replace('visitor','reader')
        # Avoid repeated long spans in the successful fixture.
        text='1. Walk slowly through the garden.\n'+PROSE+'\n2. Write down what you notice about the changing weather.'
        row=self.row(text)
        got,keys,reason=select_document(row,self.source,self.cfg)
        self.assertIsNone(reason)
        self.assertEqual(got,row['rollout_results'][0]['text'])
        self.assertNotEqual(got,row['text'])
        self.assertIn('content:'+content_key(row['text']),keys)
        for bad,expected in [
            (dict(row,rollout_results=[dict(text='Short fragment.',finish_reason='stop')]),'length'),
            (self.row(text,'length'),'incomplete_generation'),
            (dict(row,prompt='An algebra lesson'),'math_topic'),
            (dict(row,text=train_text(PROSE+' A Python programming guide.')),'code_topic'),
            (self.row('To create:\n'+text),'boilerplate_or_encoding'),
            (self.row(PROSE+' The visitor returns another day to listen to the birds and enjoy the shade.'),'unstructured_tutorial')]:
            with self.subTest(expected=expected):self.assertEqual(select_document(bad,self.source,self.cfg)[2],expected)
        faq=dict(self.source,config='faq')
        text=train_text(PROSE+'\nWhat should the reader do next?')
        # Make the output end with a dangling question, while keeping train split.
        for i in range(1000):
            text=f'{PROSE}\nQuestion {i}: What should the reader do next?'
            if partition(content_key(text))=='train':break
        row=dict(self.row(PROSE),rollout_results=[dict(text=text,finish_reason='stop')])
        self.assertEqual(select_document(row,faq,self.cfg)[2],'unanswered_faq')
        for phrase in ['Jump to navigation','Subscribe to updates','( Full Answer )','Clients are delighted']:
            self.assertEqual(select_document(self.row(text+' '+phrase),self.source,self.cfg)[2], 'boilerplate_or_encoding')

    def test_original_split_and_shared_identity_cannot_be_hidden_by_rewrite(self):
        for i in range(10000):
            original=f'{PROSE} Item {i}.'
            if partition(content_key(original))=='validation':break
        row=self.row('1. Read the passage.\n'+PROSE+'\n2. Observe the weather carefully.')
        row['text']=original
        self.assertEqual(select_document(row,self.source,self.cfg)[2],'reserved_partition')
        a=document_keys(dict(id='document',url='https://www.example.org/garden/'),PROSE)
        b=document_keys(dict(metadata=dict(id='document',url='http://example.org/garden')),PROSE+' Different.')
        self.assertEqual(len(a&b),2)


class TransitionTests(unittest.TestCase):
    def test_exact_parent_schedule_and_budget_and_tamper_guards(self):
        m=parent()
        reference=dict(fake='reference',settings=m['recipe']['stream'])
        m['v2_contract']['data']=fingerprint(reference)
        manifest=dict(format='sml-v2-stage-b-v1',reference=reference,settings=m['recipe']['stream'],tokenizer=m['v2_contract']['tokenizer'])
        before=copy.deepcopy(m)
        r=stage_recipe(m,manifest)
        self.assertEqual(m,before)
        self.assertEqual({k:v for k,v in r.items() if k!=KEY},m['recipe'])
        c=dict(contract(m,r),data=fingerprint(manifest))
        self.assertEqual(check_resume(m,c,r,manifest),'explicit-stage-b-transition')
        self.assertEqual(restore_scheduler(r,m).state_dict(),m['lr_scheduler'])
        self.assertEqual(learning_rate(m['tokens']+tokens_per_update(r),r),5e-6)
        self.assertEqual(end_step(r),m['step']+2348)
        for key in ['batches','grad_accum','model','peak_lr']:
            bad=copy.deepcopy(r)
            if key=='batches':bad[key]=[3,4,2,2,2]
            elif key=='model':bad[key]['d_model']+=1
            else:bad[key]*=2
            with self.subTest(key=key),self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(manifest);bad['reference']['fake']='other'
        with self.assertRaises(ValueError):check_resume(m,c,r,bad)
        with self.assertRaises(ValueError):stage_recipe(dict(m,step=m['step']+1),manifest)
        saved=dict(m,recipe=r,v2_contract=c)
        self.assertEqual(stage_recipe(saved,manifest),r)
        with self.assertRaises(ValueError):stage_recipe(saved,manifest,500_000_000)


class LauncherTests(unittest.TestCase):
    def test_stale_data_and_worker_evidence_are_rejected(self):
        from stage_b import launch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); corpus=root/'corpus.json'; receipt=root/'web.json'; audit=root/'audit.json'
            atomic_json(corpus,dict(sources=['web']));atomic_json(receipt,dict(status='passed'))
            modules={'stage_b_data.py':'current'}
            source=dict(accepted=20,scanned=300,hf_cursor_restart=True,
                        receipt='web.json',receipt_sha256=file_sha256(receipt))
            record=dict(corpus_sha256=file_sha256(corpus),selection_sha256='current',
                        review_status='passed',sources=dict(web=source))
            atomic_json(audit,record)
            atomic_json(root/'five_rank_correctness.json',dict(modules=modules,
                        receipts=[dict(status='passed') for _ in range(5)]))
            with patch.multiple(launch,ROOT=root,CORPUS=corpus,AUDIT=audit,NETWORK=root):
                launch.verify_readiness(modules,{'web':100})
                for key,value in [('review_status','failed'),('selection_sha256','old'),('sources',{})]:
                    atomic_json(audit,dict(record,**{key:value}))
                    with self.subTest(key=key),self.assertRaises(ValueError):
                        launch.verify_readiness(modules,{'web':100})
                atomic_json(audit,record)
                with self.assertRaises(ValueError):launch.verify_readiness(dict(modules,other='changed'),{'web':100})
                atomic_json(receipt,dict(status='changed'))
                with self.assertRaises(ValueError):launch.verify_readiness(modules,{'web':100})

    def test_plan_mode_does_not_launch_and_finished_pilot_cannot_restart(self):
        from stage_b import launch
        m=parent(); reference=dict(fake='reference',settings=m['recipe']['stream'])
        m['v2_contract']['data']=fingerprint(reference)
        manifest=dict(format='sml-v2-stage-b-v1',reference=reference,settings=m['recipe']['stream'],
                      tokenizer=m['v2_contract']['tokenizer'])
        r=stage_recipe(m,manifest)
        m.update(recipe=r,step=end_step(r),v2_contract=contract(m,r))
        with patch.object(launch,'verify_bundle_metadata',return_value=m), \
                patch.object(launch,'load_corpus'),patch.object(launch,'Tokenizer'), \
                patch.object(launch,'training_manifest',return_value=manifest),patch.object(launch,'check_resume'):
            with self.assertRaisesRegex(ValueError,'pilot is complete'):launch.prepare()
        with patch.object(launch,'prepare',return_value=(None,None,None,None,None,{})), \
                patch.object(launch,'verify_readiness') as verify,patch.object(sys,'argv',['launch']), \
                patch('sys.stdout',new_callable=io.StringIO):
            launch.main()
            verify.assert_not_called()


class TinyStageBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.PipelineTests.setUpClass.__func__(cls)
        # All model/tokenizer fixtures remain tiny; make prose long enough for
        # the real production selection rules before any training starts.
        file=Path(cls.config['sources'][0]['local_jsonl'])
        with file.open('w') as f:
            for i in range(6500):f.write(json.dumps(dict(text=f'Garden passage {i}. '+PROSE,id=str(i)))+'\n')
        extra=cls.root/'extra.jsonl'
        with extra.open('w') as f:
            for i in range(2000):f.write(json.dumps(dict(text=f'Another account {i}. '+PROSE,id='extra-'+str(i)))+'\n')
        cls.stage_corpus=copy.deepcopy(cls.config)
        cls.stage_corpus['reference_corpus']=copy.deepcopy(cls.config)
        cls.stage_corpus['stage_b_selection']=POLICY
        cls.stage_corpus['sources'][0]['weight']=45
        cls.stage_corpus['sources'][1]['weight']=20
        cls.stage_corpus['sources'].append(dict(label='c',weight=35,text_field='text',local_jsonl=str(extra)))

    @classmethod
    def tearDownClass(cls):fixtures.PipelineTests.tearDownClass.__func__(cls)

    def test_stream_mixture_exclusions_and_exact_prefetch_resume(self):
        settings=dict(prefetch_batches=2,read_attempts=1,dedup_window=100,shuffle_buffer=8)
        old=stream_manifest(self.config,self.tokenizer,settings)
        new=training_manifest(self.stage_corpus,self.tokenizer,settings)
        old_stream=StreamingTokens(old,self.tokenizer,'train',128)
        old_stream.batch(11)
        state=old_stream.state_dict();old_stream.close()
        meta=dict(tokens=11*128,v2_contract=dict(data=fingerprint(old)))
        converted=migrate_cursor(state,meta,new)
        for label in ['a','b']:
            self.assertEqual(converted['sources'][label]['dataset'],state['sources'][label]['dataset'])
            self.assertEqual(converted['sources'][label]['buffer'],[])
        exclusions=set()
        validation_batches(old,self.tokenizer,128,2,1,exclusion_keys=exclusions)
        stream=StageBStreamingTokens(new,self.tokenizer,'train',128,converted,exclusions=exclusions)
        data=PrefetchedStream(stream,2)
        try:
            data.global_batch(1,[2])
            consumed=data.state_dict()
            expected=data.global_batch(1,[2])
        finally:data.close()
        restored=StageBStreamingTokens(new,self.tokenizer,'train',128,consumed,exclusions=exclusions)
        import numpy as np
        np.testing.assert_array_equal(restored.global_batch(1,[2]),expected)
        for i in range(200):restored.row()
        counters=restored.state_dict()['stage_b']['offsets']
        total=sum(counters.values())
        for label,w in new['weights'].items():self.assertLess(abs(counters[label]/total-w/100),.02)
        self.assertEqual(sum(restored.offsets.values()),meta['tokens']+total)
        self.assertFalse(set(restored.state_dict()['stage_b']['seen_keys'])&exclusions)
        restored.close()
        with self.assertRaises(ValueError):StageBStreamingTokens(new,self.tokenizer,'train',128,consumed,exclusions={'changed'})

    def test_tiny_worker_keeps_reference_and_exact_optimizer_resume(self):
        import mlx.core as mx
        from sml_v1.pretrain import main
        from sml_v1.checkpoint_bundle import resolve_bundle
        mx.set_default_device(mx.cpu)
        cfg=read_json(ROOT/'configs/pilot.json')
        cfg['model'].update(vocab_size=300,d_model=16,n_heads=4,n_kv_heads=2,n_layers=1,
            mlp_multiple_of=8,max_seq_len=128,attention_impl='vanilla',ce_impl='reference',ffn_impl='reference')
        cfg.update(batches=[1],grad_accum=1,target_tokens=4096,reserve_gib=0,
            eval_batches_per_source=1,eval_batch_size=1,prompts=[],keep_checkpoints=10,
            eval_every_tokens=256,sample_every_tokens=1000000)
        def run(recipe,name,steps=0,resume=None,output=None,stage=False):
            config=self.root/(name+'.json');atomic_json(config,recipe)
            corpus=self.root/(name+'-corpus.json');atomic_json(corpus,self.stage_corpus if stage else self.config)
            output=output or self.root/name
            argv=['worker','--config',str(config),'--tokenizer',str(self.output),'--save-dir',str(output),
                  '--job-dir',str(self.root/(name+'-job')),'--corpus',str(corpus),'--stop-after-steps',str(steps)]
            if resume:argv+=['--resume',str(resume)]
            with patch.object(sys,'argv',argv),patch('sys.stdout',new_callable=io.StringIO) as log, \
                    patch('sml_v1.stage_b_eval.CASES',(('fixture','A bird rests in a tree.','Where is the bird?',('in a tree',)),)):
                main()
            return output,log.getvalue()
        def meta(path):return read_json(resolve_bundle(path)+'.json')
        base,_=run(cfg,'stage-base',2)
        adaptive=adaptive_recipe(meta(base/'latest.json'),dict(DEFAULTS,initial_lr=1e-5,min_lr=5e-6,patience=1,cooldown=0,min_delta=100.))
        old,_=run(adaptive,'stage-adaptive',2,base/'latest.json')
        review=reviewed_recipe(meta(old/'latest.json'),shuffle_buffer=8,eval_batches=4)
        reviewed,_=run(review,'stage-reviewed',1,old/'latest.json')
        source=reviewed/'best.json';m=meta(source)
        manifest=training_manifest(self.stage_corpus,self.tokenizer,m['recipe']['stream'])
        r=stage_recipe(m,manifest,384)
        source_hash=file_sha256(resolve_bundle(source))
        full,log=run(r,'stage-full',resume=source,stage=True)
        split,_=run(r,'stage-split',1,source,stage=True)
        _,again=run(r,'stage-resumed',resume=split/'latest.json',output=split,stage=True)
        a,b=resolve_bundle(full/'latest.json'),resolve_bundle(split/'latest.json')
        ma,mb=read_json(a+'.json'),read_json(b+'.json')
        self.assertEqual(ma,mb)
        self.assertEqual(ma['step'],m['step']+3)
        self.assertEqual(ma['validation_fingerprint'],m['validation_fingerprint'])
        self.assertEqual(set(read_json(full/'last_eval.json')['sources']),{'a','b'})
        self.assertEqual(read_json(full/'last_eval.json')['step'],ma['step'])
        baseline=read_json(full/'assessments'/f'step_{m["step"]:07d}.json')
        assessment=read_json(full/'assessments'/f'step_{ma["step"]:07d}.json')
        self.assertEqual(baseline['fingerprint'],assessment['fingerprint'])
        self.assertEqual(assessment['total'],1)
        self.assertEqual(file_sha256(resolve_bundle(source)),source_hash)
        self.assertNotIn('[stage-b-boundary]',again)
        for suffix in ['', '.optimizer.safetensors']:
            left,right=mx.load(a+suffix),mx.load(b+suffix)
            for key in left:self.assertTrue(mx.array_equal(left[key],right[key]).item(),key)
        with gzip.open(a+'.rank0.data_state.json.gz','rt') as x,gzip.open(b+'.rank0.data_state.json.gz','rt') as y:
            self.assertEqual(json.load(x),json.load(y))


if __name__=='__main__':unittest.main()
