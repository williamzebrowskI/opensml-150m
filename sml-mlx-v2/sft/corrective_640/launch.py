"""Prepare/check/run the user-authorized 256-update experiment from checkpoint 640."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from sml_v2.common import atomic_json, read_json, fingerprint, lock
from sft.transfer_control.launch import preflight, Tee
from sft.corrective_640.protocol import DIR, OUTPUT, validate, contract, code_files, verify, plan


def restore(b, opt, path, frozen, cfg):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v2.checkpoint_bundle import resolve_bundle
    weights = resolve_bundle(path); meta = read_json(weights + '.json'); u = meta['additional_updates']
    if meta['contract'] != fingerprint(frozen) or not 0 <= u <= cfg['updates']:
        raise ValueError('Resume contract or cursor mismatch')
    if meta['step'] != cfg['source_step'] + u or meta['next_examples'] != {f: u * n for f, n in cfg['per_update'].items()}:
        raise ValueError('Resume family exposures mismatch')
    b.model.load_weights(weights, strict=True)
    opt.state = tree_unflatten(list(mx.load(weights + '.optimizer.safetensors').items()))
    # Mark the loaded, nonempty per-parameter state initialized. MLX retains
    # existing masters and moments; none is re-created from a gradient.
    opt.init(b.model.trainable_parameters())
    mx.eval(b.model.parameters(), opt.state)
    if int(opt.state['step'].item()) != u: raise ValueError('Optimizer cursor mismatch')
    return u, meta['training']


def check(cfg, frozen, data):
    import tempfile
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from sft.corrective_640 import engine
    from sft.corrective_640.evaluate import assess
    from sft.corrective_640.data import instruction_pass, visible_prefix
    from sft.reading_repair.generation import verify_generation
    from sml_v2.checkpoint_bundle import save_bundle
    # Validate EVERY target mask, EOS, context and choice boundary before training.
    b = engine.load(cfg); anchor = engine.load(cfg); checked = Counter()
    for split in ('train', 'dev', 'test'):
        for family, rows in data[split].items():
            if len({r['id'] for r in rows}) != len(rows): raise ValueError('Duplicate row identities: ' + split + ':' + family)
            for row in rows:
                if family == 'ranking': engine.choice_arrays(b, row, cfg['context'])
                else:
                    encoded = engine.encode(b, row, cfg['context'])
                    if encoded['y'][-1] != b.eos: raise ValueError('Missing supervised EOS')
                    if family == 'replay': engine.validate_conversation(b, row, cfg['context'])
                    if family == 'instruction' and not instruction_pass(row, row['answer']):
                        raise ValueError('Generated target violates its checker')
                checked[split + ':' + family] += 1
    # Compare normalized likelihood with independent scalar next-token scoring.
    row = data['train']['ranking'][0]; x, y, chars = engine.choice_arrays(b, row)
    raw, normalized = engine.choice_scores(b.model, x, y, chars)
    expected = []
    for choice in row['choices']:
        head = b.encode(row['ranking_prompt']); ids = b.encode(row['ranking_prompt'] + ' ' + choice); score = 0.
        for i in range(len(head), len(ids)):
            logits = b.logits(mx.array([ids[:i]], dtype=mx.int32))[0, -1].astype(mx.float32)
            score += float((logits[ids[i]] - mx.logsumexp(logits)).item())
        expected.append(score)
    raw_error = max(abs(a - c) for a, c in zip(raw.tolist(), expected))
    norm_error = max(abs(a - c / len(t)) for a, c, t in zip(normalized.tolist(), expected, row['choices']))
    if raw_error > .01 or norm_error > .001: raise ValueError('Independent likelihood check failed')
    anchor_x = mx.array([data['anchors'][0][:-1]], dtype=mx.int32)
    anchor_logits = anchor.logits(anchor_x).astype(mx.float32)
    reference = mx.stop_gradient(anchor_logits - mx.logsumexp(anchor_logits, axis=-1, keepdims=True))
    initial_kl = float(engine.reference_kl(b.model, anchor_x, reference).item())
    if abs(initial_kl) > 1e-5: raise ValueError('Parent KL should start at zero')
    opt = engine.optimizer(cfg); opt.init(b.model.trainable_parameters()); mx.eval(opt.state)
    first = engine.update(b, anchor, opt, engine.batch_at(data, cfg, 0), data['anchors'][0], cfg, 1)
    if int(opt.state['step'].item()) != 1: raise ValueError('Disposable optimizer did not advance')
    with tempfile.TemporaryDirectory(prefix='opensml-corrective640-check-') as temp:
        meta = dict(step=641, source_step=640, additional_updates=1, next_examples=cfg['per_update'],
                    contract=fingerprint(frozen), training=[first])
        saved = save_bundle(Path(temp) / 'resume', b.model, opt, meta, None, keep=1)
        restored = engine.load(cfg); restored_opt = engine.optimizer(cfg)
        u, _ = restore(restored, restored_opt, saved, frozen, cfg)
        if u != 1: raise ValueError('Disposable resume cursor mismatch')
        resume_moment_differences = {}
        resume_weight_differences = {}
        for label, left, right in (('model', b.model.parameters(), restored.model.parameters()), ('optimizer', opt.state, restored_opt.state)):
            aa = dict(tree_flatten(left)); bb = dict(tree_flatten(right))
            if aa.keys() != bb.keys() or any(not bool(mx.array_equal(aa[k], bb[k]).item()) for k in aa):
                raise ValueError('Restored checkpoint parameters/state differ before the next update')
        # Restore must be bitwise exact. A subsequent independent GPU backward
        # pass may round embedding scatter-adds differently; bound that effect.
        second_rows = engine.batch_at(data, cfg, 1)
        second = engine.update(b, anchor, opt, second_rows, data['anchors'][1], cfg, 2)
        resumed = engine.update(restored, anchor, restored_opt, second_rows, data['anchors'][1], cfg, 2)
        for label, left, right in (('model', b.model.parameters(), restored.model.parameters()), ('optimizer', opt.state, restored_opt.state)):
            aa = dict(tree_flatten(left)); bb = dict(tree_flatten(right))
            if aa.keys() != bb.keys(): raise ValueError('Resumed parameter/state keys differ')
            differences = {k: float(mx.max(mx.abs(aa[k] - bb[k])).item()) for k in aa
                           if not bool(mx.array_equal(aa[k], bb[k]).item())}
            for key, error in differences.items():
                is_embedding_weight = (label == 'model' and key == 'embed.weight') or (label == 'optimizer' and key == 'embed.weight.master')
                if is_embedding_weight:
                    magnitude = mx.maximum(mx.maximum(mx.abs(aa[key]), mx.abs(bb[key])), 1.1754943508222875e-38)
                    relative = float(mx.max(mx.abs(aa[key] - bb[key]) / magnitude).item())
                    if relative > 8 * 1.1920928955078125e-7 or error > second['lr'] * 1e-4:
                        raise ValueError('Embedding resume update differs beyond FP32 rounding')
                    resume_weight_differences[label + ':' + key] = dict(max_abs_error=error, max_elementwise_relative_error=relative)
                    continue
                if label != 'optimizer' or not key.endswith(('.m', '.v')):
                    raise ValueError('Resumed second update parameters/cursor differ: ' + repr(differences))
                scale = max(float(mx.max(mx.abs(aa[key])).item()), float(mx.max(mx.abs(bb[key])).item()))
                if error > 8 * 1.1920928955078125e-7 * scale:
                    raise ValueError('Resumed optimizer moments differ beyond FP32 rounding: ' + repr(differences))
                resume_moment_differences[key] = dict(max_abs_error=error, tensor_max_abs=scale)
        smoke = assess(restored, data, dict(cfg, max_new_tokens=32), smoke=True)
        decode = verify_generation(restored, [data['dev']['grounded'][0]['prompt'],
            'Return only these four words: quiet gardens grow slowly.'], limit=16)
        del restored, restored_opt
    del b, anchor, opt; gc.collect(); mx.clear_cache(); verify(frozen)
    return dict(all_rows_checked=dict(checked), answer_and_eos_masks_checked=True,
        scalar_likelihood_raw_error=raw_error, scalar_likelihood_normalized_error=norm_error,
        initial_parent_kl=initial_kl, first_disposable_update=first, second_disposable_update=second,
        restored_weights_and_optimizer_bitwise_equal=True,
        next_update_weights_and_masters_bitwise_equal=not bool(resume_weight_differences),
        next_update_numerically_equivalent=True, next_update_weight_differences=resume_weight_differences,
        next_update_embedding_elementwise_relative_tolerance=8 * 1.1920928955078125e-7,
        next_update_embedding_absolute_tolerance=second['lr'] * 1e-4,
        next_update_moment_differences=resume_moment_differences, next_update_moment_relative_tolerance=8 * 1.1920928955078125e-7,
        smoke_metrics=smoke['metrics'], decode_parity=decode,
        production_updates=0, source_640_unchanged=True)


def run(cfg, frozen, data, stop):
    import mlx.core as mx
    from sft.corrective_640 import engine
    from sft.corrective_640.evaluate import assess, retention_gate
    from sml_v2.checkpoint_bundle import save_bundle
    mx.random.seed(cfg['seed']); b = engine.load(cfg); anchor = engine.load(cfg)
    opt = engine.optimizer(cfg); opt.init(b.model.trainable_parameters()); mx.eval(opt.state)
    u = 0; training = []; saved = -1; bundle = None
    if (OUTPUT / 'latest.json').exists():
        u, training = restore(b, opt, OUTPUT / 'latest.json', frozen, cfg); saved = u
        bundle = OUTPUT / read_json(OUTPUT / 'latest.json')['bundle']
        print('[resume]', u, 'exact optimizer and task cursors', flush=True)
    evaluations = OUTPUT / 'evaluations'; evaluations.mkdir(exist_ok=True)
    cancelled = lambda: stop['requested'] or (OUTPUT / 'STOP').exists()

    def save():
        nonlocal saved, bundle
        meta = dict(step=cfg['source_step'] + u, source_step=cfg['source_step'], additional_updates=u,
            next_examples={f: u * n for f, n in cfg['per_update'].items()}, contract=fingerprint(frozen),
            training=training, training_format='plain-user-assistant-eos-v1', tokenizer=frozen['tokenizer'],
            source_model_sha256=cfg['source_model_sha256'], objective='corrective-family-ce-char-ranking-parent640-kl-v1',
            fresh_optimizer=True, experimental=True, automatic_promotion=False)
        bundle = Path(save_bundle(OUTPUT, b.model, opt, meta, None, keep=1000000, reserve_gib=10.))
        saved = u; print('[checkpoint]', meta['step'], 'retained; no pruning', flush=True)

    def evaluate():
        path = evaluations / f'update_{u:05d}.json'
        if path.exists():
            if read_json(path)['contract'] != fingerprint(frozen): raise ValueError('Development contract changed')
            return
        result = assess(b, data, cfg, cancelled)
        baseline = result['metrics'] if u == 0 else read_json(evaluations / 'update_00000.json')['metrics']
        result.update(update=u, step=cfg['source_step'] + u, bundle=bundle.name, contract=fingerprint(frozen),
                      retention_gate=retention_gate(result['metrics'], baseline, cfg), content_review_required=True)
        atomic_json(path, result)
        atomic_json(evaluations / f'review_{u:05d}.json', dict(status='unreviewed', evaluation=fingerprint(result),
            rubric='Inspect correct content, actual task completion, relevance, unsupported additions, repetition and actual-history follow-ups. Stopping alone does not pass.'))
        print('[grounded-eval]', u, json.dumps(result['metrics']), 'retention_gate=', result['retention_gate'],
              '; inspect saved answers, no automatic best', flush=True)

    try:
        if saved < 0: save()
        if u in cfg['evaluation_updates'] and not cancelled(): evaluate()
        print(f'[ready] preserved 640 + {u}/256; 15 examples/update; all replay assistant turns; fresh optimizer', flush=True)
        while u < cfg['updates'] and not cancelled():
            start = time.monotonic(); rows = engine.batch_at(data, cfg, u)
            result = engine.update(b, anchor, opt, rows, data['anchors'][u % len(data['anchors'])], cfg, u + 1)
            u += 1; result['seconds'] = time.monotonic() - start
            result['batch_hash'] = fingerprint({f: [r['id'] for r in rr] for f, rr in rows.items()}); training.append(result)
            if u == 1 or u % 8 == 0:
                print(f'[corrective {u}/256] step={cfg["source_step"] + u} loss={result["loss"]:.4f} rank={result["ranking_ce"]:.4f} '
                      f'KL={result["anchor_kl"]:.5f} lr={result["lr"]:.3e} seconds={result["seconds"]:.1f}', flush=True)
            if u % cfg['checkpoint_every'] == 0: save()
            if u in cfg['evaluation_updates'] and not cancelled(): evaluate()
        if saved != u: save()
        status = 'stopped' if cancelled() else 'complete'
    except (InterruptedError, KeyboardInterrupt):
        stop['requested'] = True; (OUTPUT / 'STOP').touch()
        if saved != u: save()
        status = 'stopped'
    except Exception as exc:
        atomic_json(OUTPUT / 'report.json', dict(status='error', update=u, last_saved_update=saved, error=repr(exc)))
        raise
    finally:
        del b, anchor, opt; gc.collect(); mx.clear_cache()
    verify(frozen)
    atomic_json(OUTPUT / 'report.json', dict(status=status, update=u, step=cfg['source_step'] + u,
        contract=fingerprint(frozen), automatic_promotion=False, public_benchmarks_evaluated=False,
        reserved_test_evaluated=False, content_review_required=True))
    print('[finished]', status, 'review development answers before selecting a candidate', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true'); action.add_argument('--check', action='store_true')
    action.add_argument('--run', action='store_true'); parser.add_argument('--clear-stop', action='store_true')
    args = parser.parse_args(); cfg = read_json(DIR / 'config.json'); validate(cfg)
    print(json.dumps(plan(cfg), indent=2), flush=True)
    if not (args.prepare or args.check or args.run): return
    preflight()
    with lock(ROOT / 'sft/.experiment.lock'), lock(DIR / '.launcher.lock'):
        if args.prepare:
            if (OUTPUT / 'contract.json').exists(): raise ValueError('Existing experiment cannot be re-prepared')
            from sft.corrective_640.data import prepare
            selection = prepare(cfg); print('[prepared]', json.dumps(selection['stats']), flush=True); return
        frozen = contract(cfg); data = read_json(DIR / 'prepared.json')
        if args.check:
            result = check(cfg, frozen, data)
            if contract(cfg) != frozen: raise ValueError('Inputs changed during readiness checks')
            atomic_json(DIR / 'readiness.json', dict(status='passed', contract=frozen, checks=result, production_training_started=False))
            print('[check-passed]', json.dumps(result), flush=True); return
        readiness = read_json(DIR / 'readiness.json')
        if readiness.get('status') != 'passed' or readiness['contract'] != frozen:
            raise ValueError('Readiness missing or stale; run --check first')
        if (OUTPUT / 'contract.json').exists() and read_json(OUTPUT / 'contract.json') != frozen:
            raise ValueError('Existing run contract changed')
        if (OUTPUT / 'report.json').exists() and read_json(OUTPUT / 'report.json').get('status') == 'complete':
            print('[already-complete] No additional training', flush=True); return
        if shutil.disk_usage(ROOT).free < 35 * 1024 ** 3: raise OSError('Need 35 GiB free for retained checkpoints')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        if args.clear_stop: (OUTPUT / 'STOP').unlink(missing_ok=True)
        if (OUTPUT / 'STOP').exists(): raise ValueError('STOP exists; use --clear-stop to resume')
        atomic_json(OUTPUT / 'contract.json', frozen); atomic_json(OUTPUT / 'config.json', cfg)
        for src in code_files() + [DIR / n for n in ('config.json', 'selection.json', 'prepared.json', 'review.json', 'review_samples.json')]:
            dest = OUTPUT / 'inputs' / src.relative_to(ROOT)
            if not dest.exists(): dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest)
        stop = {'requested': False}
        def request_stop(*_):
            stop['requested'] = True; (OUTPUT / 'STOP').touch()
            print('[stop] Finish current update and save; no automatic restart', flush=True)
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, request_stop)
        with (OUTPUT / 'training.log').open('a', buffering=1) as log:
            original = sys.stdout; sys.stdout = Tee(original, log)
            try: run(cfg, frozen, data, stop)
            finally: sys.stdout = original


if __name__ == '__main__': main()
