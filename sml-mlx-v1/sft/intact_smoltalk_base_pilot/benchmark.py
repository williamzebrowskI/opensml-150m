"""Benchmark a retained intact-smoltalk checkpoint without training it."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sml_v1.common import atomic_json, file_sha256, lock, read_json
from sft.intact_smoltalk_base_pilot.launch import OUTPUT
from sft.transfer_control.launch import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--suite', choices=['all', 'multiple-choice', 'ifeval'], default='all')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true')
    action.add_argument('--run', action='store_true')
    args = parser.parse_args()
    cfg = read_json(OUTPUT / 'config.json')
    update = args.step - cfg['checkpoint_start_step']
    if update not in cfg['evaluation_updates']:
        raise ValueError('Select an evaluated checkpoint')
    evaluation = OUTPUT / 'evaluations' / f'update_{update:05d}.json'
    ev = read_json(evaluation)
    report = read_json(OUTPUT / 'report.json')
    bundle = OUTPUT / ev['bundle']
    from sml_v1.checkpoint_bundle import resolve_bundle
    resolve_bundle(bundle)
    meta = read_json(bundle / 'model.safetensors.json')
    if (report['status'] != 'complete' or ev['contract'] != report['contract']
            or meta['contract'] != report['contract']
            or (meta['step'], meta['source_step'], meta['additional_updates'])
            != (args.step, cfg['source_step'], update)):
        raise ValueError('Checkpoint/run identity mismatch')
    sha = file_sha256(bundle / 'model.safetensors')
    model = f'intact-smoltalk-base-{args.step}'
    out = ROOT / 'diagnostics/intact_smoltalk_base_pilot_benchmarks_v1'
    suites = ['multiple-choice', 'ifeval'] if args.suite == 'all' else [args.suite]
    print(json.dumps(dict(model=model, source=str(bundle), sha256=sha,
                         suites=suites, output=str(out / model), training=False,
                         automatic_promotion=False), indent=2), flush=True)
    if not (args.check or args.run):
        return
    preflight()
    with lock(ROOT / 'sft/.experiment.lock'):
        identity = dict(bundle=str(bundle), sha256=sha, step=args.step,
                        development_evaluation_sha256=file_sha256(evaluation))
        pin = out / model / 'selection.json'
        if pin.exists() and read_json(pin) != identity:
            raise ValueError('Benchmark checkpoint selection changed')
        if not pin.exists():
            atomic_json(pin, identity)
        from evaluation.full_benchmarks import core, launch, spec
        spec.MODELS[model] = dict(bundle=str(bundle.relative_to(ROOT)),
                                 step=args.step, sha256=sha)
        core.RESULTS = launch.RESULTS = out
        for suite in suites:
            frozen = core.frozen(model, suite)
            frozen['code'][str(Path(__file__).resolve())] = file_sha256(__file__)
            for path in [OUTPUT / 'report.json', OUTPUT / 'contract.json', evaluation, pin]:
                frozen['protected'][str(path)] = file_sha256(path)
            if args.check:
                launch.check(model, suite, frozen)
                continue
            receipt = spec.DIR / f'readiness_{model}_{suite}.json'
            if not receipt.exists() or read_json(receipt)['contract'] != frozen:
                raise ValueError('Run --check first')
            launch.run(model, suite, frozen)
            result = read_json(out / model / suite / 'summary.json')
            if result['completed'] != result['expected']:
                break


if __name__ == '__main__':
    main()
