import copy
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sml_v1.common import ROOT, atomic_json, file_sha256, load_corpus, quotas, read_json
from sml_v1.corpus import accepted, content_key, documents, identity, partition
from sml_v1.data import TokenPools, verify_data
from sml_v1.prepare_data import prepare
from sml_v1.recipe import event_due, learning_rate, tokens_per_update, validate
from sml_v1.tokenization import FIXTURES, Tokenizer
from sml_v1.train_tokenizer import fit, prepare_samples, read_rows


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.config = load_corpus(ROOT / 'configs/corpus.json')

    def test_weights(self):
        self.assertEqual(quotas(100, self.config['sources']), dict(web=55,dclm=25,wiki=10,cosmopedia=10))
        self.assertEqual(sum(quotas(103, self.config['sources']).values()), 103)

    def test_content_identity(self):
        self.assertEqual(content_key('a  caf\u00e9\n'), content_key('a cafe\u0301'))
        self.assertEqual(partition('0000000000000000'), 'final')
        self.assertEqual(partition('0000000000000064'), 'validation')
        self.assertEqual(partition('00000000000000c8'), 'tokenizer_dev')
        self.assertEqual(partition('000000000000012c'), 'train')

    def test_filters(self):
        text = 'The reader visits a library to learn about plants and animals. ' * 5
        source = self.config['sources'][1]
        self.assertIsNone(accepted(dict(text=text, edu_int_score=2), source, self.config))
        self.assertEqual(accepted(dict(text=text, edu_int_score=3), source, self.config), text)
        with self.assertRaises(ValueError):
            accepted(dict(text=text), source, self.config)
        source = self.config['sources'][3]
        self.assertIsNone(accepted(dict(text=text, format='code'), source, self.config))
        self.assertEqual(accepted(dict(text=text, format='textbook'), source, self.config), text)
        self.assertIsNone(accepted(dict(text=('Repeated line of unhelpful text\n' * 15), format='textbook'), source, self.config))

    def test_schedule(self):
        cfg = validate(read_json(ROOT / 'configs/pilot.json'))
        self.assertEqual(tokens_per_update(cfg), 106496)
        self.assertEqual(learning_rate(0, cfg), 0)
        self.assertAlmostEqual(learning_rate(5000000, cfg), .0003)
        self.assertAlmostEqual(learning_rate(300000000, cfg), .0003)
        self.assertAlmostEqual(learning_rate(500000000, cfg), .00003)
        self.assertAlmostEqual(learning_rate(900000000, cfg), .00003)
        self.assertTrue(event_due(99, 102, 100))
        self.assertFalse(event_due(102, 105, 100))
        cfg['batches'] = [4,3,4,3]
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_selected_pretraining_geometry(self):
        cfg = validate(read_json(ROOT / 'configs/pilot.json'))
        self.assertEqual(cfg['model']['max_seq_len'], 2048)
        self.assertEqual(cfg['model']['vocab_size'], 32000)
        self.assertEqual(cfg['batches'], [4, 3, 3, 3])
        self.assertEqual(cfg['grad_accum'], 4)
        self.assertEqual(tokens_per_update(cfg), 106496)
        self.assertEqual(cfg['target_tokens'], 500000000)
        self.assertEqual(cfg['data_mode'], 'hf_stream')
        self.assertEqual(cfg['stream']['prefetch_batches'], 4)

    def test_stream_settings_validation(self):
        for key, value in [('prefetch_batches', -1), ('read_attempts', 0), ('dedup_window', 0)]:
            cfg = read_json(ROOT / 'configs/pilot.json')
            cfg['stream'][key] = value
            with self.assertRaises(ValueError):
                validate(cfg)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='sml-v2-offline-tests-')
        cls.root = Path(cls.temp.name)
        local = cls.root / 'fixtures.jsonl'
        with local.open('w') as f:
            for i in range(6500):
                text = f'Document {i}: A curious reader explores a library, learning new words about birds, trees, history and the wider world. The story ends here. '
                f.write(json.dumps(dict(text=text, id=str(i))) + '\n')
        cls.config = dict(seed=17, shuffle_buffer=10, min_chars=1, max_bytes=10000,
                          max_rows_per_source=100000, sources=[
            dict(label='a', weight=70, text_field='text', local_jsonl=str(local)),
            dict(label='b', weight=30, text_field='text', local_jsonl=str(local))])
        cls.cache = cls.root / 'sample'
        cls.records = prepare_samples(cls.config, cls.cache, 30000, 2000)
        cls.output = cls.root / 'tokenizer'
        cls.tokenizer = fit(cls.config, cls.cache, cls.output, cls.records, vocab_size=300)
        cls.data = cls.root / 'data'
        prepare(cls.config, cls.tokenizer, cls.data, 10000, 1500, 512)
        cls.manifest = verify_data(cls.data, cls.tokenizer)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_roundtrip_and_no_structural_injection(self):
        for text in FIXTURES:
            ids = self.tokenizer.assert_roundtrip(text)
            self.assertTrue(all(i >= 4 for i in ids))
        self.assertEqual(self.tokenizer.vocab_size, 300)

    def test_frozen_reuse_and_drift(self):
        original = file_sha256(self.output / 'tokenizer.json')
        fit(self.config, self.cache, self.output, self.records, vocab_size=300)
        self.assertEqual(original, file_sha256(self.output / 'tokenizer.json'))
        with self.assertRaises(ValueError):
            fit(self.config, self.cache, self.output, self.records, vocab_size=301)
        with self.assertRaises(ValueError):
            prepare_samples(self.config, self.cache, 30001, 2000)

    def test_split_isolation_and_exact_dedup(self):
        seen = set()
        for record in self.records:
            for row in read_rows(self.cache / record['file']):
                self.assertEqual(partition(row['key']), record['split'])
                self.assertNotIn(row['key'], seen)
                seen.add(row['key'])
        seen.clear()
        for record in self.manifest['files']:
            for row in read_rows(self.data / record['index']):
                self.assertEqual(partition(row['key']), record['split'])
                self.assertNotIn(row['key'], seen)
                seen.add(row['key'])

    def test_cursor_and_token_weighting(self):
        import numpy as np
        pools = TokenPools(self.data, self.manifest, 'train', 16)
        pools.batch(100)
        self.assertEqual(pools.offsets, dict(a=1120,b=480))
        state = pools.state_dict()
        expected = pools.global_batch(2, [3,2,2,2])
        restored = TokenPools(self.data, self.manifest, 'train', 16, state)
        self.assertTrue(np.array_equal(expected, restored.global_batch(2, [3,2,2,2])))
        self.assertEqual(expected.shape, (2,2,9,16))
        self.assertTrue(np.array_equal(expected[0,:,:,1:], expected[1,:,:,:-1]))
        bad = copy.deepcopy(state); bad['offsets']['a'] += 1
        with self.assertRaises(ValueError):
            TokenPools(self.data, self.manifest, 'train', 16, bad)

    def test_reject_tampered_tokenizer(self):
        import shutil
        temp = self.root / 'tampered'
        shutil.copytree(self.output, temp)
        with (temp / 'tokenizer.json').open('a') as f:
            f.write(' ')
        with self.assertRaises(ValueError):
            Tokenizer(temp)

    def test_no_pool_recycling(self):
        pools = TokenPools(self.data, self.manifest, 'validation', 16)
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            for _ in range(10000):
                pools.row('b')

    def test_completed_sample_and_data_reuse(self):
        original = file_sha256(self.data / 'manifest.json')
        records = prepare_samples(self.config, self.cache, 30000, 2000)
        self.assertEqual(records, self.records)
        prepare(self.config, self.tokenizer, self.data, 10000, 1500, 512)
        self.assertEqual(original, file_sha256(self.data / 'manifest.json'))

    def test_staged_inputs_verified(self):
        import subprocess
        from sml_v1.launch import stage, verify_script
        source = stage(self.root / 'stage-job', read_json(ROOT / 'configs/pilot.json'), self.output, None, self.config)
        self.assertEqual(read_json(source / 'corpus.json'), self.config)
        result = subprocess.run([sys.executable, '-c', verify_script(source)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (source / 'recipe.json').write_text('{}')
        result = subprocess.run([sys.executable, '-c', verify_script(source)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def stream_manifest(self):
        from sml_v1.stream import stream_manifest
        return stream_manifest(self.config, self.tokenizer,
                               dict(prefetch_batches=2, read_attempts=3, dedup_window=100))

    def test_stream_cursor_mixture_and_fixed_validation(self):
        import numpy as np
        from sml_v1.stream import StreamingTokens, validation_batches
        manifest = self.stream_manifest()
        stream = StreamingTokens(manifest, self.tokenizer, 'train', 32)
        self.addCleanup(stream.close)
        stream.batch(100)
        self.assertEqual(stream.offsets, dict(a=2240, b=960))
        state = json.loads(json.dumps(stream.state_dict()))
        expected = stream.global_batch(4, [4,3,3,3])
        resumed = StreamingTokens(manifest, self.tokenizer, 'train', 32, state)
        self.addCleanup(resumed.close)
        actual = resumed.global_batch(4, [4,3,3,3])
        self.assertTrue(np.array_equal(actual, expected))
        self.assertEqual(actual.shape, (2,4,13,32))
        self.assertTrue(np.array_equal(actual[0,:,:,1:], actual[1,:,:,:-1]))
        self.assertTrue(all(partition(k) == 'train' for k in state['recent_document_hashes']))
        validation = StreamingTokens(manifest, self.tokenizer, 'validation', 32)
        self.addCleanup(validation.close)
        validation.batch(100)
        self.assertTrue(all(partition(k) == 'validation' for k in validation.seen))
        first = validation_batches(manifest, self.tokenizer, 32, 2, 2)
        second = validation_batches(manifest, self.tokenizer, 32, 2, 2)
        self.assertEqual(first[1], second[1])
        changed = copy.deepcopy(manifest); changed['settings']['dedup_window'] += 1
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            StreamingTokens(changed, self.tokenizer, 'train', 32, state)

    def test_prefetch_resume_uses_consumed_position(self):
        import numpy as np
        from sml_v1.stream import PrefetchedStream, StreamingTokens
        manifest = self.stream_manifest()
        stream = PrefetchedStream(StreamingTokens(manifest, self.tokenizer, 'train', 32), 2)
        self.addCleanup(stream.close)
        first = stream.global_batch(2, [3,2,2,2])
        saved = json.loads(json.dumps(stream.state_dict()))
        self.assertEqual(sum(saved['offsets'].values()), 576)
        expected = stream.global_batch(2, [3,2,2,2])
        resumed = PrefetchedStream(StreamingTokens(manifest, self.tokenizer, 'train', 32, saved), 2)
        self.addCleanup(resumed.close)
        self.assertTrue(np.array_equal(expected, resumed.global_batch(2, [3,2,2,2])))
        self.assertFalse(np.array_equal(first, expected))
        with self.assertRaisesRegex(ValueError, 'geometry'):
            resumed.global_batch(1, [1])

    def test_stream_retry_and_exhaustion(self):
        from datasets import IterableDataset
        from sml_v1.stream import SourceCursor
        failed = [False]
        def rows():
            for number in range(12):
                if number == 5 and not failed[0]:
                    failed[0] = True
                    raise EOFError('simulated interrupted read')
                yield {'number': number}
        with patch('sml_v1.stream.open_dataset', side_effect=lambda _: IterableDataset.from_generator(rows)), \
             patch('sml_v1.stream.time.sleep'):
            cursor = SourceCursor(dict(label='fixture'), 3)
            self.addCleanup(cursor.close)
            self.assertEqual([cursor.next()['number'] for _ in range(12)], list(range(12)))
            with self.assertRaisesRegex(RuntimeError, 'exhausted'):
                cursor.next()
        with patch('sml_v1.stream.open_dataset', side_effect=EOFError('still unavailable')) as opened, \
             patch('sml_v1.stream.time.sleep'):
            with self.assertRaises(EOFError):
                SourceCursor(dict(label='fixture'), 3).next()
            self.assertEqual(opened.call_count, 3)

    def test_stream_reads_hf_without_materializing(self):
        from sml_v1.stream import open_dataset
        remote = load_corpus(ROOT / 'configs/corpus.json')['sources'][0]
        with patch('datasets.load_dataset', return_value=object()) as loader:
            open_dataset(remote)
        loader.assert_called_once_with(remote['repo'], remote['config'], revision=remote['revision'],
                                       split='train', streaming=True)

    def test_stream_parquet_row_group_resume(self):
        import numpy as np
        import pyarrow.json as paj
        import pyarrow.parquet as pq
        from datasets import load_dataset
        from sml_v1.stream import StreamingTokens
        parquet = self.root / 'stream-fixture.parquet'
        pq.write_table(paj.read_json(self.root / 'fixtures.jsonl'), parquet, row_group_size=17)
        def open_parquet(_):
            return load_dataset('parquet', data_files=str(parquet), split='train', streaming=True)
        with patch('sml_v1.stream.open_dataset', side_effect=open_parquet):
            manifest = self.stream_manifest()
            stream = StreamingTokens(manifest, self.tokenizer, 'train', 32)
            self.addCleanup(stream.close)
            stream.batch(200)
            state = json.loads(json.dumps(stream.state_dict()))
            expected = stream.global_batch(4, [4,3,3,3])
            resumed = StreamingTokens(manifest, self.tokenizer, 'train', 32, state)
            self.addCleanup(resumed.close)
            self.assertTrue(np.array_equal(expected, resumed.global_batch(4, [4,3,3,3])))

    def test_stream_launcher_needs_no_prepared_data(self):
        from sml_v1.launch import main
        config = self.root / 'launch-stream.json'
        corpus = self.root / 'launch-corpus.json'
        recipe = read_json(ROOT / 'configs/pilot.json')
        recipe['model']['vocab_size'] = self.tokenizer.vocab_size
        atomic_json(config, recipe); atomic_json(corpus, self.config)
        absent = self.root / 'no-token-pool'
        with patch('sml_v1.launch.ROOT', self.root.resolve()), patch('sml_v1.launch.launch') as launch, \
             patch('sml_v1.data.verify_data', side_effect=AssertionError('must not read local pools')), \
             patch.object(sys, 'argv', ['launch', '--run', '--config', str(config), '--corpus', str(corpus),
                          '--tokenizer', str(self.output), '--save-dir', str(self.root / 'runs' / 'stream'),
                          '--data', str(absent)]):
            main()
        launch.assert_called_once()
        self.assertFalse(absent.exists())

    def test_prepare_refuses_streaming_recipe(self):
        from contextlib import redirect_stderr
        import io
        from sml_v1.prepare_data import main
        output = self.root / 'forbidden-token-pools'
        with patch.object(sys, 'argv', ['prepare', '--run', '--output', str(output)]), \
             redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as stopped:
                main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn('live HF streaming', error.getvalue())
        self.assertFalse(output.exists())

    def test_unequal_batch_gradient_weighting(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_map, tree_flatten
        from sml_v1.model import TransformerLM, TransformerConfig
        from sml_v1.pretrain import build_backward
        mx.set_default_device(mx.cpu)
        mx.random.seed(23)
        model = TransformerLM(TransformerConfig(vocab_size=300, d_model=16, n_heads=4,
            n_kv_heads=2, n_layers=1, mlp_multiple_of=8, max_seq_len=16, attention_impl='vanilla'))
        pools = TokenPools(self.data, self.manifest, 'train', 16)
        batch = pools.global_batch(2, [2,1,1,1])
        reference = build_backward(model, 2, compile_step=False)(mx.array(batch[0]), mx.array(batch[1]))[1]
        combined, start = None, 0
        for count in [2,1,1,1]:
            grads = build_backward(model, 2, compile_step=False)(mx.array(batch[0,:,start:start+count]), mx.array(batch[1,:,start:start+count]))[1]
            weighted = tree_map(lambda g: g * count / 5, grads)
            combined = weighted if combined is None else tree_map(lambda a,b:a+b, combined, weighted)
            start += count
        for (name, actual), (_, expected) in zip(tree_flatten(combined), tree_flatten(reference)):
            self.assertTrue(mx.allclose(actual, expected, rtol=2e-4, atol=2e-6).item(), name)

    def test_model_parameter_count(self):
        import mlx.core as mx
        from sml_v1.model import TransformerLM, TransformerConfig, count_parameters
        mx.set_default_device(mx.cpu)
        model = TransformerLM(TransformerConfig(**read_json(ROOT / 'configs/pilot.json')['model']))
        self.assertEqual(count_parameters(model), 150439168)

    def test_tiny_cpu_worker_checkpoint_resume(self):
        self.worker_checkpoint_resume('tokens')

    def test_tiny_cpu_stream_worker_checkpoint_resume(self):
        self.worker_checkpoint_resume('hf_stream')

    def test_tiny_cpu_adaptive_restart_matches_uninterrupted(self):
        import mlx.core as mx
        from sml_v1.pretrain import main
        from sml_v1.checkpoint_bundle import resolve_bundle
        from sml_v1.lr_scheduler import DEFAULTS, adaptive_recipe
        mx.set_default_device(mx.cpu)
        cfg = read_json(ROOT / 'configs/pilot.json')
        cfg['model'].update(vocab_size=300, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1,
                            mlp_multiple_of=8, max_seq_len=128, attention_impl='vanilla',
                            ce_impl='reference', ffn_impl='reference')
        cfg.update(batches=[1], grad_accum=1, target_tokens=4096, reserve_gib=0,
                   eval_batches_per_source=1, eval_batch_size=1, prompts=['The reader'],
                   eval_every_tokens=128, sample_every_tokens=1000000)
        config = self.root / 'adaptive-base.json'; atomic_json(config, cfg)
        corpus = self.root / 'adaptive-corpus.json'; atomic_json(corpus, self.config)
        source = self.root / 'adaptive-base'
        def run(config, output, name, steps, resume=None):
            args = ['worker', '--config', str(config), '--tokenizer', str(self.output),
                    '--save-dir', str(output), '--job-dir', str(self.root / name),
                    '--corpus', str(corpus), '--stop-after-steps', str(steps)]
            if resume:
                args += ['--resume', str(resume)]
            with patch.object(sys, 'argv', args):
                main()
        run(config, source, 'adaptive-base-job', 1)
        weights = resolve_bundle(source / 'latest.json')
        parent = read_json(weights + '.json')
        parent_hash = file_sha256(weights)
        # A deliberately huge threshold makes reduction timing deterministic.
        new = adaptive_recipe(parent, dict(DEFAULTS, patience=1, cooldown=1, min_delta=100.0))
        config = self.root / 'adaptive.json'; atomic_json(config, new)
        full, split = self.root / 'adaptive-full', self.root / 'adaptive-split'
        run(config, full, 'adaptive-full-job', 4, source / 'latest.json')
        run(config, split, 'adaptive-first-job', 2, source / 'latest.json')
        halfway = read_json(resolve_bundle(split / 'latest.json') + '.json')
        self.assertEqual(halfway['lr_scheduler']['reductions'], 1)
        self.assertEqual(halfway['lr_scheduler']['cooldown_remaining'], 1)
        run(config, split, 'adaptive-resumed-job', 2, split / 'latest.json')
        a, b = resolve_bundle(full / 'latest.json'), resolve_bundle(split / 'latest.json')
        left, right = read_json(a + '.json'), read_json(b + '.json')
        self.assertEqual(left, right)
        self.assertEqual(left['lr_scheduler']['reductions'], 2)
        self.assertEqual(left['lr_scheduler']['cap'], .00005)
        self.assertEqual(left['tokens'], 640)
        self.assertEqual(left['validation_fingerprint'], parent['validation_fingerprint'])
        self.assertEqual(left['recipe']['model'], cfg['model'])
        self.assertEqual(file_sha256(weights), parent_hash)
        for suffix in ('', '.optimizer.safetensors'):
            first, second = mx.load(a + suffix), mx.load(b + suffix)
            self.assertEqual(first.keys(), second.keys())
            for name in first:
                self.assertTrue(mx.array_equal(first[name], second[name]).item(), name)
        with gzip.open(a + '.rank0.data_state.json.gz', 'rt') as first, \
             gzip.open(b + '.rank0.data_state.json.gz', 'rt') as second:
            self.assertEqual(json.load(first), json.load(second))

    def test_tiny_cpu_stream_budget_extension(self):
        # Exercise the new schedule with real checkpoint/optimizer/cursor loading.
        import mlx.core as mx
        from sml_v1.pretrain import main
        from sml_v1.checkpoint_bundle import resolve_bundle
        from sml_v1.continuation import extend_recipe
        mx.set_default_device(mx.cpu)
        cfg = read_json(ROOT / 'configs/pilot.json')
        cfg['model'].update(vocab_size=300, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1,
                            mlp_multiple_of=8, max_seq_len=128, attention_impl='vanilla',
                            ce_impl='reference', ffn_impl='reference')
        cfg.update(batches=[1], grad_accum=1, target_tokens=256, reserve_gib=0,
                   eval_batches_per_source=1, eval_batch_size=1, prompts=['The reader'],
                   eval_every_tokens=128, sample_every_tokens=1000000)
        config = self.root / 'extension-pilot.json'; atomic_json(config, cfg)
        corpus = self.root / 'extension-corpus.json'; atomic_json(corpus, self.config)
        pilot = self.root / 'extension-pilot'
        def run(config, out, job, resume=None):
            args = ['worker', '--config', str(config), '--tokenizer', str(self.output),
                    '--save-dir', str(out), '--job-dir', str(job), '--stop-after-steps', '1',
                    '--corpus', str(corpus)]
            if resume:
                args += ['--resume', str(resume)]
            with patch.object(sys, 'argv', args):
                main()
        run(config, pilot, self.root / 'extension-pilot-job')
        source = resolve_bundle(pilot / 'latest.json')
        meta = read_json(source + '.json')
        source_hash = file_sha256(source)
        full_cfg = extend_recipe(meta, 512, rewarm_tokens=128)
        full_config = self.root / 'extension-full.json'; atomic_json(full_config, full_cfg)
        full = self.root / 'extension-full'
        run(full_config, full, self.root / 'extension-full-job', pilot / 'latest.json')
        full_weights = resolve_bundle(full / 'latest.json')
        saved = read_json(full_weights + '.json')
        self.assertEqual((saved['step'], saved['tokens']), (2, 256))
        self.assertEqual(saved['recipe']['target_tokens'], 512)
        self.assertEqual(saved['validation_fingerprint'], meta['validation_fingerprint'])
        self.assertEqual(file_sha256(source), source_hash)
        old_state = mx.load(source + '.optimizer.safetensors')
        new_state = mx.load(full_weights + '.optimizer.safetensors')
        self.assertEqual(int(new_state['step'].item()), int(old_state['step'].item()) + 1)
        with gzip.open(full_weights + '.rank0.data_state.json.gz', 'rt') as f:
            self.assertEqual(sum(json.load(f)['stream_state']['offsets'].values()), 256)
        run(full_config, full, self.root / 'extension-resume-job', full / 'latest.json')
        resumed = read_json(resolve_bundle(full / 'latest.json') + '.json')
        self.assertEqual((resumed['step'], resumed['tokens']), (3, 384))
        self.assertEqual(resumed['recipe'], full_cfg)

    def worker_checkpoint_resume(self, mode):
        # Synthetic 1-layer CPU fixture, never the actual model or live corpus.
        import mlx.core as mx
        from sml_v1.pretrain import main
        from sml_v1.checkpoint_bundle import resolve_bundle
        mx.set_default_device(mx.cpu)
        cfg = read_json(ROOT / 'configs/pilot.json')
        cfg['model'].update(vocab_size=300, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1,
                            mlp_multiple_of=8, max_seq_len=128, attention_impl='vanilla',
                            ce_impl='reference', ffn_impl='reference')
        cfg.update(data_mode=mode, batches=[1], grad_accum=1, target_tokens=256, reserve_gib=0,
                   eval_batches_per_source=1, eval_batch_size=1, prompts=['The reader'],
                   log_every=1, eval_every_tokens=128, sample_every_tokens=1000000)
        config = self.root / f'tiny-{mode}.json'; atomic_json(config, cfg)
        corpus = self.root / 'tiny-corpus.json'; atomic_json(corpus, self.config)
        out, job = self.root / f'tiny-run-{mode}', self.root / f'tiny-job-{mode}'
        args = ['worker', '--config', str(config), '--tokenizer', str(self.output), '--data', str(self.data),
                '--save-dir', str(out), '--job-dir', str(job), '--stop-after-steps', '1', '--corpus', str(corpus)]
        with patch.object(sys, 'argv', args):
            main()
        first = resolve_bundle(out / 'latest.json')
        meta = read_json(first + '.json')
        self.assertEqual((meta['step'],meta['tokens']), (1,128))
        with gzip.open(first + '.rank0.data_state.json.gz', 'rt') as f:
            self.assertEqual(sum(json.load(f)['stream_state']['offsets'].values()), 128)
        with patch.object(sys, 'argv', args + ['--resume', str(out / 'latest.json')]):
            main()
        final = read_json(resolve_bundle(out / 'latest.json') + '.json')
        self.assertEqual((final['step'],final['tokens']), (2,256))
        if mode == 'hf_stream':
            self.assertEqual(meta['validation_fingerprint'], final['validation_fingerprint'])
            self.assertEqual(len(final['validation_fingerprint']), 64)
        self.assertEqual(read_json(job / 'result.json')['status'], 'complete')
        resumed_weights = mx.load(resolve_bundle(out / 'latest.json'))
        other = self.root / f'uninterrupted-{mode}'
        uninterrupted = ['worker', '--config', str(config), '--tokenizer', str(self.output),
                         '--data', str(self.data), '--save-dir', str(other), '--job-dir', str(self.root / f'other-job-{mode}'),
                         '--corpus', str(corpus)]
        with patch.object(sys, 'argv', uninterrupted):
            main()
        baseline = mx.load(resolve_bundle(other / 'latest.json'))
        for name, actual in resumed_weights.items():
            self.assertTrue(mx.array_equal(actual, baseline[name]).item(), name)
        cfg['peak_lr'] *= 2
        atomic_json(config, cfg)
        with patch.object(sys, 'argv', args + ['--resume', str(out / 'latest.json')]):
            with self.assertRaisesRegex(ValueError, 'Resume rejected'):
                main()


if __name__ == '__main__':
    unittest.main()
