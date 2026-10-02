"""Exploratory public benchmarks for Constraint Completion 2048; no promotion."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sml_v2.common import file_sha256, fingerprint, lock, read_json
from sft.transfer_control.launch import preflight

RUN = ROOT / 'runs/sft_constraint_completion_1920_v1'
BUNDLE = RUN / 'step_0002048_1267d9f4cf7a'
SHA = 'e4e8a0de28e2141f6a580b49cc950aa64492c9f94d0d3aa98e6a26a2f439764b'
MODEL = 'constraint-completion-1920-2048'
RESULTS = ROOT / 'diagnostics/constraint_completion_1920_benchmarks_v1'
EVALUATION = RUN / 'evaluations/update_00128.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['all', 'multiple-choice', 'ifeval'], default='all')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    suites = ['multiple-choice', 'ifeval'] if args.suite == 'all' else [args.suite]

    if file_sha256(BUNDLE / 'model.safetensors') != SHA:
        raise ValueError('Pinned step 2048 weights changed')
    meta = read_json(BUNDLE / 'model.safetensors.json')
    report = read_json(RUN / 'report.json')
    evaluation = read_json(EVALUATION)
    if (meta['step'], meta['source_step'], meta['additional_updates']) != (2048, 1920, 128):
        raise ValueError('Checkpoint ancestry mismatch')
    if (report['status'], report['step'], report['update']) != ('complete', 2048, 128):
        raise ValueError('Expected the completed Constraint Completion run')
    if report['contract'] != meta['contract'] or fingerprint(read_json(RUN / 'contract.json')) != meta['contract']:
        raise ValueError('Saved training contract mismatch')
    if evaluation['bundle'] != BUNDLE.name or evaluation['update'] != 128:
        raise ValueError('Development evaluation checkpoint mismatch')

    diagnostic = dict(
        experimental=True,
        automatic_promotion=False,
        selection_note='User-requested exploratory evaluation of final step 2048; not a promoted or semantically approved model.',
    )
    print(json.dumps(dict(model=MODEL, source=str(BUNDLE), suites=suites,
                          output=str(RESULTS / MODEL), training=False, **diagnostic), indent=2), flush=True)
    if not (args.check or args.run):
        return
    preflight()
    with lock(ROOT / 'sft/.experiment.lock'):
        from evaluation.full_benchmarks import core, launch, spec

        spec.MODELS[MODEL] = dict(bundle=str(BUNDLE.relative_to(ROOT)), step=2048, sha256=SHA)
        launch.RESULTS = core.RESULTS = RESULTS
        for suite in suites:
            frozen = core.frozen(MODEL, suite)
            frozen.update(diagnostic)
            frozen['code'][str(Path(__file__).resolve())] = file_sha256(__file__)
            for path in (RUN / 'report.json', RUN / 'contract.json', EVALUATION):
                frozen['protected'][str(path)] = file_sha256(path)
            if args.check:
                launch.check(MODEL, suite, frozen)
                continue
            receipt = spec.DIR / f'readiness_{MODEL}_{suite}.json'
            if not receipt.exists() or read_json(receipt)['contract'] != frozen:
                raise ValueError('Run --check for this suite first')
            launch.run(MODEL, suite, frozen)
            result = read_json(RESULTS / MODEL / suite / 'summary.json')
            if result['completed'] != result['expected']:
                break


if __name__ == '__main__':
    main()
