#!/usr/bin/env python3
"""Isolated, paired gradient-group benchmark. Never starts or changes production."""

import argparse
import datetime
import fcntl
import json
from pathlib import Path
import statistics

import jaccl_benchmark as bench


PRODUCTION = bench.ROOT / 'train/checkpoints/dolma_cosmo_152m_wide_jaccl_v1'


def protected_state(root):
    return {name: bench.common.sha256(root / name) if (root / name).exists() else None
            for name in ('latest.json', 'best.json', 'lr_control.json', 'STOP')}


def assert_comparable(first, other):
    for field in ('input_sha256', 'source_step', 'fixed_lr', 'batches', 'tokens_per_update', 'measured_steps'):
        if other[field] != first[field]:
            raise RuntimeError(f'Benchmark comparison changed {field}; do not interpret throughput')
    if not all(c and c['passed'] for c in other['gradient_checks']):
        raise RuntimeError('Missing real-gradient correctness checks')


def report(root, results):
    bench.common.atomic_json(root / 'summary.json', results)
    lines = ['# Gradient Group Benchmark', '',
             'Same checkpoint, 32/24/24/24 batches, sequence 256, accumulation 2, BF16 compute and FP32 gradients/state.',
             'Paired forward/reverse trials; identical cached real-data batches. Streaming, warmup, evaluation and checkpoint I/O excluded.',
             'This is not a production continuation or a model-quality evaluation.', '',
             '| Group MiB | Trial tokens/s | Mean tokens/s | Change vs 32 MiB |',
             '| ---: | --- | ---: | ---: |']
    baseline = statistics.mean(r['tokens_per_second'] for r in results if r['gradient_bucket_mib'] == 32) if results else 1
    for size in (32, 64, 128):
        values = [r['tokens_per_second'] for r in results if r['gradient_bucket_mib'] == size]
        if values:
            mean = statistics.mean(values)
            lines.append(f'| {size} | {", ".join(f"{v:,.0f}" for v in values)} | {mean:,.0f} | {(mean / baseline - 1) * 100:+.2f}% |')
    lines += ['', 'Gradient sync timing includes waiting for slower ranks; it is not pure link transfer time.',
              'Production checkpoint pointers, STOP file and learning-rate control are not modified.']
    (root / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(PRODUCTION / 'latest.json'))
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--resident-batches', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if not 20 <= args.steps <= 100 or not 5 <= args.warmup <= 20 or not 1 <= args.resident_batches <= 100:
        parser.error('Use steps 20..100, warmup 5..20 and resident batches 1..100')
    if not 120 <= args.timeout <= 1800:
        parser.error('Timeout must be 120..1800 seconds per trial')
    args.control_transport = 'thunderbolt'
    bench.INPUTS.mkdir(parents=True, exist_ok=True)
    lock = open(bench.INPUTS / '.launcher.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = protected_state(PRODUCTION)
    bench.preflight(gpu=True, control_transport=args.control_transport)
    destination = bench.prepare(args.checkpoint)
    info = bench.sync_input(destination, bench.INPUTS / 'prepared_gradient_groups.json', args.control_transport)
    root = bench.OUTPUTS / ('gradient_groups_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir(parents=True)
    bench.common.atomic_json(root / 'experiment.json', dict(input=str(destination), inputs=info,
                            options=vars(args), order=[32, 64, 128, 128, 64, 32], production_before=before))
    results = []
    try:
        args.gradient_bucket_mib = 32
        bench.launch(destination, root / 'cpu_correctness', [32, 24, 24, 24], args, cpu=True)
        for trial, size in enumerate((32, 64, 128, 128, 64, 32), start=1):
            bench.preflight(gpu=True, control_transport=args.control_transport)
            args.gradient_bucket_mib = size
            print(f'[trial] {trial}/6 gradient groups={size} MiB; production remains stopped', flush=True)
            value = bench.launch(destination, root / f'trial_{trial}_{size}mib', [32, 24, 24, 24], args)
            assert_comparable(results[0] if results else value, value)
            results.append(value)
            report(root, results)
        bench.preflight(gpu=True, control_transport=args.control_transport)
    except BaseException as exc:
        bench.common.atomic_json(root / 'failure.json', dict(error=str(exc), completed_trials=len(results)))
        raise
    finally:
        after = protected_state(PRODUCTION)
        bench.common.atomic_json(root / 'production_guard.json', dict(before=before, after=after, unchanged=before == after))
        report(root, results)
        if before != after:
            raise RuntimeError('Production control files changed during the test; inspect before resuming')
    print((root / 'summary.md').read_text(), flush=True)
    print(f'[done] All four Macs idle; production unchanged. Reports: {root}', flush=True)


if __name__ == '__main__':
    main()
