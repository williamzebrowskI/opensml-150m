"""Resume equivalence and safeguards for the optional data review experiment."""
import copy
import gzip
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from tests import test_pipeline as fixtures
from tests.test_five_mac_training import saved, contract
from sml_v2.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v2.continuation import PRE_DATA_REVIEW_CODE, check_resume
from sml_v2.corpus import content_key, partition
from sml_v2.data_review import migrate_cursor, rebaseline_scheduler, reviewed_recipe
from sml_v2.lr_scheduler import DEFAULTS, adaptive_recipe, restore_scheduler
from sml_v2.recipe import validate
from sml_v2.stream import (PrefetchedStream, StreamingTokens, StreamExhausted,
                           stream_manifest, validation_batches, with_stream_settings)
from sml_v2.topology_transition import migrate_recipe


class DataReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.PipelineTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        fixtures.PipelineTests.tearDownClass.__func__(cls)

    def make_manifest(self, shuffle=0):
        settings = dict(prefetch_batches=2, read_attempts=3, dedup_window=100)
        if shuffle:
            settings['shuffle_buffer'] = shuffle
        return stream_manifest(self.config, self.tokenizer, settings)

    def metadata(self):
        meta = saved()
        meta['recipe'] = migrate_recipe(meta, [4, 3, 2, 2, 2])
        manifest = self.make_manifest()
        meta['recipe']['stream'] = copy.deepcopy(manifest['settings'])
        meta['v2_contract'] = dict(contract(meta, meta['recipe']), data=fingerprint(manifest),
                                  code=PRE_DATA_REVIEW_CODE)
        return meta, manifest

    def test_only_order_and_validation_change_with_verified_manifest(self):
        meta, old_manifest = self.metadata()
        original = copy.deepcopy(meta)
        recipe = reviewed_recipe(meta)
        manifest = with_stream_settings(old_manifest, recipe['stream'])
        proposed = dict(contract(meta, recipe), data=fingerprint(manifest))
        self.assertEqual(check_resume(meta, proposed, recipe, manifest), 'explicit-data-review-transition')
        self.assertEqual(meta, original)
        self.assertEqual(restore_scheduler(recipe, meta).state_dict(), meta['lr_scheduler'])
        current = dict(meta, recipe=recipe, v2_contract=proposed)
        self.assertEqual(reviewed_recipe(current), recipe)
        self.assertEqual(check_resume(current, proposed, recipe, manifest), 'unchanged')
        with self.assertRaises(ValueError):
            reviewed_recipe(current, shuffle_buffer=128)
        with self.assertRaises(ValueError):
            check_resume(meta, proposed, recipe)
        for kind in ('corpus', 'package', 'tokenizer', 'recipe', 'provenance', 'data', 'stream', 'capacity', 'order'):
            bad_recipe, bad_manifest, bad_contract = copy.deepcopy(recipe), copy.deepcopy(manifest), dict(proposed)
            if kind == 'corpus': bad_manifest['corpus']['seed'] += 1
            if kind == 'package': bad_manifest['datasets'] = 'other'
            if kind == 'stream': bad_manifest['settings']['dedup_window'] += 1
            if kind == 'capacity': bad_manifest['settings']['shuffle_buffer'] += 1
            if kind == 'order': bad_manifest['document_order'] = 'other'
            if kind == 'tokenizer': bad_contract['tokenizer'] = 'other'
            if kind == 'recipe': bad_recipe['peak_lr'] *= 2
            if kind == 'provenance': bad_recipe['data_review_transition']['parent_replica_hash'] = 'other'
            bad_contract.update(recipe=fingerprint(bad_recipe), data=fingerprint(bad_manifest))
            if kind == 'data': bad_contract['data'] = 'other'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                check_resume(meta, bad_contract, bad_recipe, bad_manifest)
        for size in (True, 1, -1, 10001):
            with self.assertRaises(ValueError): reviewed_recipe(meta, shuffle_buffer=size)

    def test_cursor_transition_preserves_unconsumed_tokens_and_offsets(self):
        manifest = self.make_manifest()
        stream = StreamingTokens(manifest, self.tokenizer, 'train', 32)
        self.addCleanup(stream.close)
        stream.batch(50)
        old = stream.state_dict()
        meta, _ = self.metadata()
        meta['recipe']['model']['max_seq_len'] = 32
        meta['tokens'] = sum(old['offsets'].values())
        new_manifest = self.make_manifest(8)
        migrated = migrate_cursor(old, meta, new_manifest)
        for key in ('sources', 'offsets', 'recent_document_hashes', 'split', 'seq_len'):
            self.assertEqual(migrated[key], old[key])
        self.assertNotIn('shuffle', old)
        resumed = StreamingTokens(new_manifest, self.tokenizer, 'train', 32, migrated)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.state_dict(), migrated)
        with self.assertRaises(ValueError): migrate_cursor(migrated, meta, new_manifest)

    def test_shuffled_prefetch_restart_is_exact_and_validation_unchanged(self):
        manifest = self.make_manifest(8)
        stream = PrefetchedStream(StreamingTokens(manifest, self.tokenizer, 'train', 32), 2)
        self.addCleanup(stream.close)
        stream.global_batch(2, [2, 1])
        state = json.loads(json.dumps(stream.state_dict()))
        self.assertTrue(any(s['documents'] for s in state['shuffle']['sources'].values()))
        expected = [stream.global_batch(2, [2, 1]) for _ in range(6)]
        resumed = PrefetchedStream(StreamingTokens(manifest, self.tokenizer, 'train', 32, state), 2)
        self.addCleanup(resumed.close)
        for batch in expected:
            np.testing.assert_array_equal(batch, resumed.global_batch(2, [2, 1]))
        self.assertEqual(stream.state_dict(), resumed.state_dict())
        for source in state['shuffle']['sources'].values():
            self.assertTrue(all(partition(content_key(text)) == 'train' for text in source['documents']))
        self.assertEqual(validation_batches(self.make_manifest(), self.tokenizer, 32, 2, 2)[1],
                         validation_batches(manifest, self.tokenizer, 32, 2, 2)[1])
        for change in ('missing', 'capacity', 'source', 'documents'):
            bad = copy.deepcopy(state)
            if change == 'missing': bad.pop('shuffle')
            if change == 'capacity': bad['shuffle']['capacity'] += 1
            if change == 'source': bad['shuffle']['sources']['a'] = None
            if change == 'documents': bad['shuffle']['sources']['a']['documents'] = ['']
            with self.subTest(change=change), self.assertRaises(ValueError):
                StreamingTokens(manifest, self.tokenizer, 'train', 32, bad)

    def test_buffer_drains_without_dropping_or_repeating_documents(self):
        from datasets import IterableDataset
        documents = [f'This is unique training document number {i}, about a reader and a library.' for i in range(40)]
        documents = [d for d in documents if partition(content_key(d)) == 'train']
        rows = [dict(text=d) for d in documents]
        with patch('sml_v2.stream.open_dataset', side_effect=lambda _: IterableDataset.from_generator(lambda: iter(rows))):
            stream = StreamingTokens(self.make_manifest(8), self.tokenizer, 'train', 32)
            self.addCleanup(stream.close)
            actual = [stream._document('a') for _ in documents]
            self.assertCountEqual(actual, documents)
            self.assertNotEqual(actual, documents)
            with self.assertRaises(StreamExhausted): stream._document('a')
            state = json.loads(json.dumps(stream.state_dict()))
            resumed = StreamingTokens(self.make_manifest(8), self.tokenizer, 'train', 32, state)
            self.addCleanup(resumed.close)
            with self.assertRaises(StreamExhausted): resumed._document('a')

    def test_rebaseline_preserves_lr_and_archives_parent_history(self):
        meta, _ = self.metadata()
        recipe = reviewed_recipe(meta)
        scheduler = restore_scheduler(recipe, meta)
        before = scheduler.state_dict()
        new = rebaseline_scheduler(scheduler, 2.95, meta['step'], meta['tokens'])
        self.assertEqual(new.state['best'], 2.95)
        for key in ('cap', 'evaluations', 'reductions', 'config_fingerprint'):
            self.assertEqual(new.state[key], before[key])
        self.assertEqual(recipe['data_review_transition']['parent_scheduler'], before)
        self.assertEqual(scheduler.state_dict(), before)
        for value in (float('nan'), -1):
            with self.assertRaises(ValueError): rebaseline_scheduler(scheduler, value, meta['step'], meta['tokens'])

    def test_launcher_selects_separate_run_and_refuses_parent_overwrite(self):
        import launch_five_mac as launch
        meta, manifest = self.metadata()
        proposed = with_stream_settings(manifest, reviewed_recipe(meta)['stream'])
        with patch.object(launch, 'verified_metadata', return_value=meta), \
                patch.object(launch, 'select_checkpoint', return_value=launch.LOW_LR_OUTPUT / 'latest.json') as select, \
                patch('sml_v2.stream.stream_manifest', return_value=proposed):
            with patch('sys.argv', ['launch', '--data-review']), patch('sys.stdout', new_callable=io.StringIO) as out:
                launch.main()
            select.assert_called_once_with(launch.DATA_REVIEW_OUTPUT.resolve(), None, launch.LOW_LR_OUTPUT)
            self.assertIn('"validation_tokens": 4194304', out.getvalue())
            self.assertIn('"new_validation_baseline_pending": true', out.getvalue())
            self.assertIn('No model loaded', out.getvalue())
            with patch('sys.argv', ['launch', '--data-review', '--output', str(launch.LOW_LR_OUTPUT)]), \
                    patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
                launch.main()
            with patch('sys.argv', ['launch', '--data-review', '--lr-floor', '1e-6']), \
                    patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
                launch.main()

    def test_tiny_worker_baseline_and_restart_match_uninterrupted(self):
        import mlx.core as mx
        from sml_v2.pretrain import main
        from sml_v2.checkpoint_bundle import resolve_bundle
        mx.set_default_device(mx.cpu)
        cfg = read_json(ROOT / 'configs/pilot.json')
        cfg['model'].update(vocab_size=300, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1,
                            mlp_multiple_of=8, max_seq_len=128, attention_impl='vanilla',
                            ce_impl='reference', ffn_impl='reference')
        cfg.update(batches=[1], grad_accum=1, target_tokens=4096, reserve_gib=0,
                   eval_batches_per_source=1, eval_batch_size=1, prompts=[], keep_checkpoints=10,
                   eval_every_tokens=128, sample_every_tokens=1000000)
        corpus = self.root / 'review-corpus.json'; atomic_json(corpus, self.config)
        def run(recipe, name, steps, resume=None, output=None):
            config = self.root / (name + '.json'); atomic_json(config, recipe)
            output = output or self.root / name
            args = ['worker', '--config', str(config), '--tokenizer', str(self.output),
                    '--save-dir', str(output), '--job-dir', str(self.root / (name + '-job')),
                    '--corpus', str(corpus), '--stop-after-steps', str(steps)]
            if resume: args += ['--resume', str(resume)]
            with patch.object(sys, 'argv', args), patch('sys.stdout', new_callable=io.StringIO) as log:
                main()
            return output, log.getvalue()
        base, _ = run(cfg, 'review-base', 1)
        initial = read_json(resolve_bundle(base / 'latest.json') + '.json')
        adaptive = adaptive_recipe(initial, DEFAULTS)
        parent, _ = run(adaptive, 'review-parent', 1, base / 'latest.json')
        old_weights = resolve_bundle(parent / 'latest.json')
        old = read_json(old_weights + '.json')
        old_hash = file_sha256(old_weights)
        recipe = reviewed_recipe(old, shuffle_buffer=8, eval_batches=4)
        full, log = run(recipe, 'review-full', 3, parent / 'latest.json')
        split, _ = run(recipe, 'review-split', 1, parent / 'latest.json')
        _, resumed_log = run(recipe, 'review-resume', 2, split / 'latest.json', split)
        self.assertEqual(log.count('[eval-baseline '), 1)
        self.assertNotIn('[eval-baseline ', resumed_log)
        self.assertIn('[eval-legacy ', resumed_log)
        a, b = resolve_bundle(full / 'latest.json'), resolve_bundle(split / 'latest.json')
        left, right = read_json(a + '.json'), read_json(b + '.json')
        self.assertEqual(left, right)
        self.assertEqual(left['legacy_validation']['fingerprint'], old['validation_fingerprint'])
        self.assertNotEqual(left['validation_fingerprint'], old['validation_fingerprint'])
        baseline_dir = next(p for p in full.glob('step_*') if read_json(p / 'model.safetensors.json')['step'] == old['step'])
        baseline = read_json(baseline_dir / 'model.safetensors.json')
        self.assertEqual(baseline['replica_hash'], old['replica_hash'])
        self.assertEqual(baseline['lr_scheduler']['cap'], old['lr_scheduler']['cap'])
        self.assertEqual(baseline['best_val_loss'], read_json(full / 'baseline_eval.json')['loss'])
        self.assertEqual(file_sha256(old_weights), old_hash)
        for suffix in ('', '.optimizer.safetensors'):
            first, second = mx.load(a + suffix), mx.load(b + suffix)
            for name in first:
                self.assertTrue(mx.array_equal(first[name], second[name]).item(), name)
        with gzip.open(a + '.rank0.data_state.json.gz', 'rt') as x, gzip.open(b + '.rank0.data_state.json.gz', 'rt') as y:
            self.assertEqual(json.load(x), json.load(y))


if __name__ == '__main__':
    unittest.main()
