"""Full public benchmarks for the final corrective checkpoint 896."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v2.common import atomic_json, file_sha256, fingerprint, lock, read_json
from sft.transfer_control.launch import preflight

RUN = ROOT / 'runs/sft_corrective_640_v1'
BUNDLE = RUN / 'step_0000896_1f398b2e73b7'
SHA = '12dd43f3125ff21a73b99809eb45752f40b3039de98499ffd011ee98099ae7ce'
MODEL = 'corrective-896'
RESULTS = ROOT / 'diagnostics/corrective_896_benchmarks_v1'
EVALUATION = RUN / 'evaluations/update_00256.json'


def show_results(suite):
    path = RESULTS / MODEL / suite / 'summary.json'
    if not path.exists():
        print(f'{suite}: no saved results yet', flush=True)
        return
    result = read_json(path)
    print(f'\nCheckpoint 896 — {suite}: {result["status"]} '
          f'({result["completed"]:,}/{result["expected"]:,})', flush=True)
    if suite == 'multiple-choice':
        print(f'{"Benchmark":<16} {"Accuracy":>10} {"Normalized":>12}')
        for key, label in [('arc_easy', 'ARC-Easy'), ('arc_challenge', 'ARC-Challenge'),
                           ('piqa', 'PIQA'), ('hellaswag', 'HellaSwag')]:
            scores = result['tasks'][key]
            print(f'{label:<16} {scores["acc"]:>9.2%} {scores["acc_norm"]:>11.2%}')
    else:
        for kind in ('strict', 'loose'):
            scores = result[kind]
            print(f'{kind.capitalize()}: prompts {scores["prompt_accuracy"]:.2%}; '
                  f'instructions {scores["instruction_accuracy"]:.2%}')
    print(f'Results: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['all', 'multiple-choice', 'ifeval'], default='all')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true', help='Verify setup without running the full benchmarks')
    action.add_argument('--run', action='store_true', help='Check setup if needed, run or resume, then show scores')
    args = parser.parse_args()
    meta = read_json(BUNDLE / 'model.safetensors.json')
    evaluation = read_json(EVALUATION)
    report = read_json(RUN / 'report.json')
    contract = fingerprint(read_json(RUN / 'contract.json'))
    if file_sha256(BUNDLE / 'model.safetensors') != SHA:
        raise ValueError('Selected checkpoint weights changed')
    if (meta['step'], meta['source_step'], meta['additional_updates']) != (896, 640, 256):
        raise ValueError('Selected checkpoint ancestry mismatch')
    if (report['status'], report['step'], report['update']) != ('complete', 896, 256):
        raise ValueError('Expected completed 256-update training run')
    if not (meta['contract'] == evaluation['contract'] == report['contract'] == contract):
        raise ValueError('Training contract mismatch')
    if (evaluation['bundle'], evaluation['step'], evaluation['update']) != (BUNDLE.name, 896, 256):
        raise ValueError('Development evaluation identity mismatch')
    suites = ['multiple-choice', 'ifeval'] if args.suite == 'all' else [args.suite]
    print(json.dumps(dict(model=MODEL, source=str(BUNDLE), sha256=SHA, suites=suites,
                         output=str(RESULTS / MODEL), training=False,
                         automatic_promotion=False), indent=2), flush=True)
    if not (args.check or args.run):
        for suite in suites:
            show_results(suite)
        return
    preflight()
    with lock(ROOT / 'sft/.experiment.lock'):
        pin = RESULTS / MODEL / 'selection.json'
        identity = dict(bundle=str(BUNDLE), sha256=SHA, step=896,
                        development_evaluation_sha256=file_sha256(EVALUATION))
        if pin.exists() and read_json(pin) != identity:
            raise ValueError('Frozen benchmark selection changed')
        if not pin.exists():
            atomic_json(pin, identity)
        from evaluation.full_benchmarks import core, launch, spec
        spec.MODELS[MODEL] = dict(bundle=str(BUNDLE.relative_to(ROOT)), step=896, sha256=SHA)
        core.RESULTS = launch.RESULTS = RESULTS
        for suite in suites:
            frozen = core.frozen(MODEL, suite)
            frozen['code'][str(Path(__file__).resolve())] = file_sha256(__file__)
            for path in [RUN / 'contract.json', RUN / 'report.json', EVALUATION, pin]:
                frozen['protected'][str(path)] = file_sha256(path)
            receipt = spec.DIR / f'readiness_{MODEL}_{suite}.json'
            if args.check or not receipt.exists() or read_json(receipt)['contract'] != frozen:
                launch.check(MODEL, suite, frozen)
            if args.check:
                continue
            launch.run(MODEL, suite, frozen)
            show_results(suite)
            result = read_json(RESULTS / MODEL / suite / 'summary.json')
            if result['completed'] != result['expected']:
                break


if __name__ == '__main__':
    main()
