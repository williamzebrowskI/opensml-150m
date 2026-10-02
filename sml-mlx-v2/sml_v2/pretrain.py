"""V2 fresh/resumed pretraining worker. Launch through scripts/launch_pretrain.py."""

import argparse
import copy
from contextlib import ExitStack
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .common import atomic_json, file_sha256, fingerprint, load_corpus, read_json
from .checkpoint_bundle import resolve_bundle, save_bundle
from .continuation import check_resume
from .data import TokenPools, verify_data
from .jaccl_control import ControlChannel
from .lr_scheduler import restore_scheduler
from .lr_trial import end_step as lr_trial_end_step
from .model import TransformerConfig, TransformerLM, count_parameters
from .precision import MasterAdamW
from .recipe import event_due, learning_rate, tokens_per_update, validate
from .tokenization import Tokenizer


def initialize(model, recipe):
    # Keep residual branches modest in the deeper backbone. Norm scales remain one.
    values = []
    for name, value in tree_flatten(model.parameters()):
        if value.ndim == 2:
            std = .02 / math.sqrt(2 * recipe['model']['n_layers']) if name.endswith(('attn.proj.weight', 'ffn.down.weight')) else .02
            value = mx.random.normal(value.shape) * std
        values.append((name, value.astype(mx.bfloat16)))
    model.update(tree_unflatten(values))


def build_backward(model, accum, compile_step=True):
    value_grad = nn.value_and_grad(model, lambda x, y: model(x, targets=y)['loss'])
    def backward(xs, ys):
        total, value = None, mx.array(0., dtype=mx.float32)
        for micro in range(accum):
            loss, grads = value_grad(xs[micro], ys[micro])
            grads = tree_map(lambda g: g.astype(mx.float32), grads)
            total = grads if total is None else tree_map(lambda a, b: a + b, total, grads)
            value = value + loss.astype(mx.float32)
        return value / accum, tree_map(lambda g: g / accum, total)
    return mx.compile(backward, inputs=[model.state], outputs=[model.state]) if compile_step else backward


def state_digest(model, optimizer):
    result = hashlib.sha256()
    for prefix, state in (('model', model.parameters()), ('optimizer', optimizer.state)):
        for name, value in tree_flatten(state):
            result.update((prefix + name + str(value.shape) + str(value.dtype)).encode())
            result.update(np.asarray(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value).tobytes())
    return result.hexdigest()


def clip_gradients(grads, maximum):
    norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for _, g in tree_flatten(grads)))
    scale = mx.minimum(mx.array(1., dtype=mx.float32), maximum / (norm + 1e-6))
    return tree_map(lambda g: g * scale, grads), norm


def completion(model, tokenizer, prompt, limit=64):
    ids = tokenizer.encode(prompt)
    if not ids or len(ids) + limit > model.cfg.max_seq_len:
        raise ValueError('Sample prompt exceeds context budget')
    caches, generated = None, []
    inputs = mx.array([ids], dtype=mx.int32)
    for _ in range(limit):
        logits, caches = model.logits(inputs, caches)
        token = int(mx.argmax(logits[0, -1]).item())
        if token == tokenizer.eos:
            break
        generated.append(token)
        inputs = mx.array([[token]], dtype=mx.int32)
    return tokenizer.decode(generated)


def run(resources):
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('config', 'tokenizer', 'save-dir', 'job-dir'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--corpus', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after-steps', type=int, default=0)
    p.add_argument('--ring-control', choices=('all-sum', 'native'), default='all-sum')
    args = p.parse_args()
    args.job_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.job_dir / f'worker_rank{os.getenv("MLX_RANK", "0")}.json', dict(pid=os.getpid(), output=str(args.job_dir)))
    recipe = validate(read_json(args.config))
    trial_end = lr_trial_end_step(recipe)
    stage_b = recipe.get('stage_b_transition')
    from .stage_b import end_step as stage_b_end_step
    experiment_end = stage_b_end_step(recipe) if stage_b else trial_end
    if 'continuation' in recipe and not args.resume:
        raise ValueError('A continuation requires a source checkpoint; fresh initialization is forbidden')
    batches = recipe['batches']
    if len(batches) > 1 and os.environ.get('MLX_JACCL_RING') != '1':
        raise RuntimeError('Distributed v2 requires the verified JACCL ring launcher')
    # A stalled peer must not leave abandoned GPU workers running indefinitely.
    heartbeat = [time.monotonic()]
    finished = threading.Event()
    resources.callback(finished.set)
    def watchdog():
        while not finished.wait(10):
            if time.monotonic() - heartbeat[0] > 1800:
                print('[fatal] No progress for 30 minutes; exiting without restart', flush=True)
                os._exit(70)
    threading.Thread(target=watchdog, daemon=True).start()
    group = mx.distributed.init(backend='jaccl', strict=True) if len(batches) > 1 else mx.distributed.init(strict=False)
    rank, world = group.rank(), group.size()
    if world != len(batches):
        raise RuntimeError('World size differs from batch recipe')
    control = ControlChannel(mx, group, ring_mode=args.ring_control)
    print(f'[control] rank={rank} ring_mode={args.ring_control}', flush=True)
    def root_call(label, fn):
        value, error = None, None
        if rank == 0:
            try:
                value = fn()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'[:1000]
        status = control.broadcast(dict(error=error) if rank == 0 else None, label)
        if status['error']:
            raise RuntimeError(status['error'])
        return value
    tokenizer = Tokenizer(args.tokenizer)
    if recipe['model']['vocab_size'] != tokenizer.vocab_size:
        raise ValueError('Model vocabulary must match the frozen tokenizer')
    streaming = recipe.get('data_mode') == 'hf_stream'
    if streaming:
        from .stream import PrefetchedStream, StreamingTokens, stream_manifest, validation_batches
        if args.corpus is None:
            raise ValueError('Live streaming requires --corpus')
        corpus = load_corpus(args.corpus)
        if stage_b:
            from .stage_b import training_manifest
            manifest = training_manifest(corpus, tokenizer, recipe['stream'])
            if fingerprint(manifest) != stage_b['manifest']:
                raise ValueError('Stage B corpus differs from the saved recipe')
        else:
            manifest = stream_manifest(corpus, tokenizer, recipe['stream'])
    else:
        if args.data is None:
            raise ValueError('Local token-pool mode requires --data')
        manifest = root_call('verify-data', lambda: verify_data(args.data, tokenizer))
    data_fingerprint = control.broadcast(fingerprint(manifest) if rank == 0 else None, 'data-identity')
    code_identity = fingerprint({p.name: file_sha256(p) for p in Path(__file__).parent.glob('*.py')})
    contract = dict(recipe=fingerprint(recipe), tokenizer=tokenizer.fingerprint, data=data_fingerprint,
                    code=code_identity)
    contracts = control.exchange(contract, 'contract')
    if any(c != contract for c in contracts):
        raise ValueError('Worker recipes/tokenizers differ')
    mx.random.seed(recipe['seed'])
    model = TransformerLM(TransformerConfig(**recipe['model']))
    initialize(model, recipe)
    optimizer = MasterAdamW(learning_rate=recipe['peak_lr'], betas=tuple(recipe['betas']),
        eps=recipe['eps'], weight_decay=recipe['weight_decay'])
    step, tokens, best, ema, cursor = 0, 0, None, None, None
    metadata, transition = None, None
    if args.resume:
        weights = resolve_bundle(args.resume)
        metadata = read_json(weights + '.json')
        transition = check_resume(metadata, contract, recipe, manifest)
        if rank == 0:
            print(f'[resume] {transition}; restoring weights, FP32 optimizer, tokens and data cursor', flush=True)
        model.load_weights(weights)
        optimizer.init(model.trainable_parameters())
        optimizer.state = tree_unflatten(list(mx.load(weights + '.optimizer.safetensors').items()))
        with gzip.open(weights + '.rank0.data_state.json.gz', 'rt') as f:
            saved = json.load(f)
        step, tokens, best, ema = (metadata[k] for k in ('step', 'tokens', 'best_val_loss', 'ema'))
        if saved['step'] != step:
            raise ValueError('Cursor/checkpoint step mismatch')
        cursor = saved['stream_state']
        if sum(cursor['offsets'].values()) != tokens:
            raise ValueError('Saved token accounting does not match cursor')
        if transition == 'explicit-data-review-transition':
            from .data_review import migrate_cursor
            cursor = migrate_cursor(cursor, metadata, manifest)
        if transition == 'explicit-stage-b-transition':
            from .stage_b import migrate_cursor
            cursor = migrate_cursor(cursor, metadata, manifest)
    else:
        optimizer.init(model.trainable_parameters())
    scheduler = restore_scheduler(recipe, metadata)
    if scheduler is not None:
        states = control.exchange(scheduler.state_dict(), 'scheduler-resume-state')
        if any(s != scheduler.state_dict() for s in states):
            raise ValueError('Resumed scheduler states differ across ranks')
    mx.eval(model.parameters(), optimizer.state)
    if any(v.dtype != mx.bfloat16 for _, v in tree_flatten(model.parameters())):
        raise ValueError('Compute weights must remain BF16')
    if any(mx.issubdtype(v.dtype, mx.floating) and v.dtype != mx.float32 for _, v in tree_flatten(optimizer.state)):
        raise ValueError('Optimizer/master weights must remain FP32')
    review = recipe.get('data_review_transition')
    legacy_heldout, legacy_validation = None, None
    eval_manifest = manifest['reference'] if stage_b else manifest
    if streaming:
        exclusions = set() if stage_b else None
        if review:
            # Rebuild separately: expanding an earlier source can affect bounded
            # cross-source dedup, so prefixes of the larger set are not enough.
            from .stream import with_stream_settings
            legacy_manifest = with_stream_settings(eval_manifest, review['parent_stream'])
            legacy_heldout = root_call('legacy-heldout-stream', lambda: validation_batches(
                legacy_manifest, tokenizer, model.cfg.max_seq_len,
                review['legacy_eval_batches'], recipe['eval_batch_size'], exclusion_keys=exclusions))
            legacy_identity = control.broadcast(legacy_heldout[1] if rank == 0 else None, 'legacy-heldout-identity')
            if legacy_identity != review['legacy_validation_fingerprint']:
                raise ValueError('Original validation fingerprint changed; refusing incomparable legacy metric')
            if transition != 'explicit-data-review-transition':
                legacy_validation = copy.deepcopy(metadata.get('legacy_validation'))
                if not legacy_validation or legacy_validation.get('fingerprint') != legacy_identity:
                    raise ValueError('Missing or incompatible legacy validation history')
        heldout = root_call('heldout-stream', lambda: validation_batches(eval_manifest, tokenizer,
            model.cfg.max_seq_len, recipe['eval_batches_per_source'], recipe['eval_batch_size'],
            exclusion_keys=exclusions))
        validation_identity = control.broadcast(heldout[1] if rank == 0 else None, 'heldout-identity')
        if args.resume and transition != 'explicit-data-review-transition' and metadata.get('validation_fingerprint') != validation_identity:
            raise ValueError('Rebuilt held-out stream differs from the checkpoint; refusing changed evaluation')
        if rank == 0:
            print(f'[eval] fixed held-out batches rebuilt in RAM; fingerprint={validation_identity}', flush=True)
        def restore_stream():
            if stage_b:
                from .stage_b_data import StageBStreamingTokens
                stream = StageBStreamingTokens(manifest, tokenizer, 'train', model.cfg.max_seq_len,
                                               cursor, exclusions=exclusions)
            else:
                stream = StreamingTokens(manifest, tokenizer, 'train', model.cfg.max_seq_len, cursor)
            return PrefetchedStream(stream, recipe['stream']['prefetch_batches'])
        data = root_call('restore-stream', restore_stream)
        if rank == 0:
            resources.callback(data.close)
            print('[data] live HF streaming -> bounded RAM shuffle/tokenization/prefetch -> disjoint rank batches; '
                  'pending input is saved in resume checkpoints', flush=True)
            if stage_b and transition == 'explicit-stage-b-transition':
                print('[stage-b-boundary] preserved raw source positions; discarded unconsumed pre-filter input: '
                      + json.dumps(cursor['stage_b']['discarded_unconsumed_input']), flush=True)
    else:
        data = root_call('restore-cursor', lambda: TokenPools(args.data, manifest, 'train', model.cfg.max_seq_len, cursor))
        validation_identity = None
    update_tokens = tokens_per_update(recipe)
    def next_lr(at_tokens):
        base = learning_rate(at_tokens, recipe)
        return scheduler.rate(base) if scheduler is not None else base

    def log_scheduler(action):
        if rank == 0 and scheduler is not None:
            state = scheduler.state_dict()
            settings = recipe['plateau_scheduler']
            print(f'[lr-controller {step}] action={action} bad_evals={state["bad_evals"]}/{settings["patience"]} '
                  f'cooldown={state["cooldown_remaining"]} reductions={state["reductions"]} '
                  f'monitor_best={state["best"]:.6f} min_delta={settings["min_delta"]} '
                  f'cap={state["cap"]:.3e} base_next_lr={learning_rate(tokens + update_tokens, recipe):.3e} '
                  f'next_lr={next_lr(tokens + update_tokens):.3e}', flush=True)
    print(f'[ready] rank={rank} world={world} params={count_parameters(model):,} step={step} tokens={tokens:,} batch={batches[rank]} accum={recipe["grad_accum"]} seq={model.cfg.max_seq_len} tokens/update={update_tokens:,}', flush=True)
    if rank == 0 and 'continuation' in recipe:
        extension = recipe['continuation']
        print(f'[schedule] target={recipe["target_tokens"]:,} total tokens; '
              f'LR {extension["start_lr"]:.3e} -> {recipe["peak_lr"]:.3e} over '
              f'{extension["rewarm_tokens"]:,} tokens from token {extension["start_tokens"]:,}; '
              f'final decay starts at {int(recipe["target_tokens"] * recipe["decay_start_fraction"]):,}; '
              f'next_lr={next_lr(tokens + update_tokens):.3e}', flush=True)
    log_scheduler('restored' if metadata and 'plateau_scheduler' in metadata['recipe'] else 'initialized')
    if rank == 0 and trial_end is not None:
        t = recipe['lr_trial_transition']
        print(f'[lr-trial] arm={t["arm"]} start_lr={t["start_lr"]:.3e} target_lr={t["target_lr"]:.3e} '
              f'warmup_updates={t["warmup_steps"]} parent_step={t["parent_step"]} '
              f'stop_step={trial_end} remaining_updates={max(0, trial_end-step)}', flush=True)
    if rank == 0 and stage_b:
        print(f'[stage-b] weights={manifest["weights"]} lr={next_lr(tokens+update_tokens):.3e} '
              f'stop_step={experiment_end} remaining_updates={max(0, experiment_end-step)}; '
              'unchanged reference validation mixture', flush=True)
    backward = build_backward(model, recipe['grad_accum'])
    start_step, start_tokens, started = step, tokens, time.monotonic()
    last_print_time, last_print_tokens = started, tokens
    saved_step = -1

    def save(is_best=False):
        nonlocal saved_step
        hashes = control.exchange(state_digest(model, optimizer), 'replica-state')
        if len(set(hashes)) != 1:
            raise RuntimeError('Model/optimizer replicas disagree; checkpoint not committed')
        metadata = dict(step=step, tokens=tokens, best_val_loss=best, ema=ema, v2_contract=contract,
                        recipe=recipe, initialization='normal-0.02-residual-scaled-v1', replica_hash=hashes[0],
                        validation_fingerprint=validation_identity)
        if review:
            metadata['legacy_validation'] = copy.deepcopy(legacy_validation)
        if scheduler is not None:
            states = control.exchange(scheduler.state_dict(), 'scheduler-checkpoint-state')
            if any(s != scheduler.state_dict() for s in states):
                raise RuntimeError('Scheduler replicas disagree; checkpoint not committed')
            metadata['lr_scheduler'] = scheduler.state_dict()
        destination = root_call('commit-checkpoint', lambda: save_bundle(args.save_dir, model, optimizer,
            metadata, data.state_dict(), best=is_best, keep=recipe['keep_checkpoints'], reserve_gib=recipe['reserve_gib']))
        if rank == 0:
            print(f'[ckpt] {"best + " if is_best else ""}latest {destination}', flush=True)
            if stage_b:
                selection = data.state_dict()['stage_b']
                atomic_json(args.save_dir / 'data_selection.json',
                    dict(step=step, tokens=tokens, offsets=selection['offsets'],
                         selection_counts=selection['selection_counts'],
                         exclusions=selection['exclusions'],
                         discarded_unconsumed_input=selection['discarded_unconsumed_input']))
        saved_step = step

    def evaluate(batch_set=None, count=None):
        pools = None if streaming else TokenPools(args.data, manifest, 'validation', model.cfg.max_seq_len)
        losses = {}
        model.eval()
        try:
            for source in eval_manifest['weights']:
                values = []
                for index in range(count or recipe['eval_batches_per_source']):
                    x, y = (batch_set or heldout)[0][source][index] if streaming else pools.batch(recipe['eval_batch_size'], source)
                    values.append(float(model(mx.array(x), targets=mx.array(y))['loss'].item()))
                losses[source] = sum(values) / len(values)
        finally:
            model.train()
        mean = sum(losses[s] * w for s, w in eval_manifest['weights'].items()) / sum(eval_manifest['weights'].values())
        if not math.isfinite(mean):
            raise RuntimeError('Nonfinite validation loss')
        return dict(loss=mean, sources=losses)

    def evaluate_all():
        result = evaluate()
        if review:
            result['legacy'] = evaluate(legacy_heldout, review['legacy_eval_batches'])
        return result

    def record_legacy(result, baseline=False):
        nonlocal legacy_validation
        if not review:
            return
        loss = result['legacy']['loss']
        if baseline:
            legacy_validation = dict(fingerprint=review['legacy_validation_fingerprint'],
                                     baseline_loss=loss, best_loss=review['legacy_best_val_loss'])
        legacy_validation.update(last_loss=loss, last_step=step, last_tokens=tokens,
                                 best_loss=min(legacy_validation['best_loss'], loss))
        if rank == 0:
            print(f'[eval-legacy {step}] val_loss={loss:.6f} best={legacy_validation["best_loss"]:.6f} '
                  f'sources={result["legacy"]["sources"]}', flush=True)

    def stage_b_assessment():
        from .stage_b_eval import evaluate as assess
        result = assess(lambda prompt: completion(model, tokenizer, prompt, limit=32))
        atomic_json(args.save_dir / 'assessments' / f'step_{step:07d}.json', dict(step=step, tokens=tokens, **result))
        print(f'[reading-check {step}] exact={result["exact_matches"]}/{result["total"]}; '
              'raw answers saved separately from validation loss', flush=True)

    if transition == 'explicit-data-review-transition':
        # The larger held-out set has a new scale: establish its baseline using
        # the unchanged resumed weights before making any optimizer update.
        result = control.broadcast(root_call('baseline-evaluate', evaluate_all), 'baseline-result')
        best = result['loss']
        from .data_review import rebaseline_scheduler
        scheduler = rebaseline_scheduler(scheduler, best, step, tokens)
        record_legacy(result, baseline=True)
        if rank == 0:
            print(f'[eval-baseline {step}] val_loss={best:.6f} sources={result["sources"]}; new validation scale', flush=True)
            atomic_json(args.save_dir / 'baseline_eval.json', dict(step=step, tokens=tokens, **result))
        log_scheduler('new-validation-baseline-same-lr')
        save(is_best=True)
        # Exclude validation collection/baseline time from training throughput.
        started = last_print_time = time.monotonic()

    if not args.resume or transition in ('explicit-validation-scheduler-transition', 'explicit-five-rank-transition'):
        save()
    if transition in ('explicit-lr-trial-transition', 'explicit-stage-b-transition'):
        # Keep the evaluated parent as best if this experiment never improves it.
        save(is_best=True)
    if transition == 'explicit-stage-b-transition':
        root_call('stage-b-baseline-reading', stage_b_assessment)
    model.train()
    while tokens < recipe['target_tokens']:
        heartbeat[0] = time.monotonic()
        stop = control.broadcast((args.save_dir / 'STOP').exists() if rank == 0 else None, 'stop')
        if (stop or (args.stop_after_steps and step - start_step >= args.stop_after_steps)
                or (experiment_end is not None and step >= experiment_end)):
            break
        full = root_call('next-batch', lambda: data.global_batch(recipe['grad_accum'], batches))
        digest = control.broadcast(hashlib.sha256(full.tobytes()).hexdigest() if rank == 0 else None, 'batch-hash')
        with mx.stream(mx.cpu):
            wire = mx.array(full) if rank == 0 else mx.zeros((2, recipe['grad_accum'], sum(batches), model.cfg.max_seq_len), dtype=mx.int32)
            wire = mx.distributed.all_sum(wire, group=group, stream=mx.cpu)
            mx.eval(wire)
            array = np.asarray(wire)
        checks = control.exchange(hashlib.sha256(array.tobytes()).hexdigest() == digest, 'batch-verified')
        if not all(checks):
            raise RuntimeError('Batch transfer checksum failed')
        begin, end = sum(batches[:rank]), sum(batches[:rank+1])
        loss, grads = backward(mx.array(array[0, :, begin:end].copy()), mx.array(array[1, :, begin:end].copy()))
        mx.eval(loss, grads)
        metrics = control.exchange(float(loss.item()) if bool(mx.isfinite(loss).item()) else None, 'loss')
        if any(v is None for v in metrics):
            raise RuntimeError('Nonfinite loss; no update or checkpoint')
        if world > 1:
            grads = tree_map(lambda g: g * (world * batches[rank] / sum(batches)), grads)
            grads = nn.average_gradients(grads, group=group, all_reduce_size=32*1024**2, communication_stream=mx.cpu)
        grads, norm = clip_gradients(grads, recipe['grad_clip'])
        mx.eval(grads, norm)
        checks = control.exchange(bool(mx.isfinite(norm).item()), 'finite-gradients')
        if not all(checks):
            raise RuntimeError('Nonfinite gradients; update skipped, job stopped')
        lr = next_lr(tokens + update_tokens)
        optimizer.learning_rate = lr
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        before = tokens
        tokens += update_tokens; step += 1
        value = sum(v*b for v,b in zip(metrics, batches)) / sum(batches)
        ema = value if ema is None else .98 * ema + .02 * value
        if rank == 0 and (step == start_step + 1 or step % recipe['log_every'] == 0):
            now = time.monotonic()
            print(f'[step {step:7d}] loss={value:.4f} ema={ema:.4f} lr={lr:.3e} grad_norm={norm.item():.3f} tok/s={(tokens-last_print_tokens)/(now-last_print_time):,.0f} run_tok/s={(tokens-start_tokens)/(now-started):,.0f} tokens={tokens:,}', flush=True)
            last_print_time, last_print_tokens = now, tokens
        final = tokens >= recipe['target_tokens'] or (experiment_end is not None and step >= experiment_end)
        is_best = False
        did_evaluate = final or event_due(before, tokens, recipe['eval_every_tokens'])
        if did_evaluate:
            result = root_call('evaluate', evaluate_all)
            result = control.broadcast(result, 'eval-result')
            is_best = best is None or result['loss'] < best
            if is_best:
                best = result['loss']
            if scheduler is not None:
                action = scheduler.observe(result['loss'], step=step, tokens=tokens)
            record_legacy(result)
            if rank == 0:
                print(f'[eval {step}] val_loss={result["loss"]:.4f} val_ppl={math.exp(min(80,result["loss"])):.2f} sources={result["sources"]}', flush=True)
                atomic_json(args.save_dir / 'last_eval.json', dict(step=step, tokens=tokens, **result))
            if scheduler is not None:
                log_scheduler(action)
        if final or event_due(before, tokens, recipe['sample_every_tokens']):
            def samples():
                for prompt in recipe['prompts']:
                    print(f'[sample {step}] temperature=0 prompt={prompt!r}\n{completion(model, tokenizer, prompt)}', flush=True)
            root_call('samples', samples)
            if stage_b:
                root_call('stage-b-reading', stage_b_assessment)
        if final or is_best or (scheduler is not None and did_evaluate) or event_due(before, tokens, recipe['save_every_tokens']):
            save(is_best)
    if saved_step != step:
        save()
    root_call('completion-status', lambda: atomic_json(args.job_dir / 'result.json',
        dict(status='complete' if tokens >= recipe['target_tokens'] else 'stopped', step=step, tokens=tokens)))
    finished.set()
    print(f'[finished] rank={rank} step={step} tokens={tokens:,}; no automatic restart', flush=True)
    return bool(stage_b)


def main():
    with ExitStack() as resources:
        return run(resources)


if __name__ == '__main__':
    completed_stage_b = main()
    if completed_stage_b:
        # Every rank has acknowledged the saved completion status and resource
        # callbacks ran. HF reader pools can still block interpreter shutdown;
        # the bounded Stage B CLI must actually release its worker process.
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
