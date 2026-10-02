"""User-selected base SFT 384: unchanged public benchmark protocol, inference only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v1.common import file_sha256, fingerprint, lock, read_json
from sft.transfer_control.launch import preflight

RUN = ROOT / 'runs/sft_intact_smoltalk_base_512_v1'
BUNDLE = RUN / 'step_0000384_02840558cc37'
SHA = '35f78b8e76b29dc6b3e1f7b8b49858117f8b81ed5ff6c5e9c768afb3899a1b00'
MODEL = 'intact-base-384'
RESULTS = ROOT / 'diagnostics/intact_base_384_review_v1/benchmarks'
EVALUATION = RUN / 'evaluations/update_00384.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['all', 'multiple-choice', 'ifeval'], default='all')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    meta = read_json(BUNDLE / 'model.safetensors.json')
    evaluation = read_json(EVALUATION)
    selection = read_json(RUN / 'frozen_selection.json')
    if file_sha256(BUNDLE / 'model.safetensors') != SHA or selection['sha256'] != SHA:
        raise ValueError('Selected checkpoint weights changed')
    if (meta['step'], meta['source_step'], meta['source_sft_step'], meta['additional_updates']) != (384, 73243, 128, 384):
        raise ValueError('Selected checkpoint ancestry mismatch')
    if meta['contract'] != fingerprint(read_json(RUN / 'contract.json')):
        raise ValueError('Original training contract mismatch')
    if (evaluation['bundle'] != BUNDLE.name or evaluation['update'] != 384
            or evaluation['contract'] != meta['contract'] or selection['step'] != 384
            or selection['development_evaluation_sha256'] != file_sha256(EVALUATION)):
        raise ValueError('Frozen development selection mismatch')
    suites = ['multiple-choice', 'ifeval'] if args.suite == 'all' else [args.suite]
    print(json.dumps(dict(model=MODEL, source=str(BUNDLE), sha256=SHA, suites=suites,
          output=str(RESULTS / MODEL), training=False, automatic_promotion=False), indent=2), flush=True)
    if not (args.check or args.run): return
    preflight()
    with lock(ROOT / 'sft/.experiment.lock'):
        from evaluation.full_benchmarks import core, launch, spec
        spec.MODELS[MODEL] = dict(bundle=str(BUNDLE.relative_to(ROOT)), step=384, sha256=SHA)
        core.RESULTS = launch.RESULTS = RESULTS
        for suite in suites:
            frozen = core.frozen(MODEL, suite)
            frozen['code'][str(Path(__file__).resolve())] = file_sha256(__file__)
            for path in (RUN / 'contract.json', EVALUATION, RUN / 'frozen_selection.json'):
                frozen['protected'][str(path)] = file_sha256(path)
            if args.check:
                launch.check(MODEL, suite, frozen)
                continue
            receipt = spec.DIR / f'readiness_{MODEL}_{suite}.json'
            if not receipt.exists() or read_json(receipt)['contract'] != frozen:
                raise ValueError('Run --check for this suite first')
            launch.run(MODEL, suite, frozen)
            result = read_json(RESULTS / MODEL / suite / 'summary.json')
            if result['completed'] != result['expected']: break


if __name__ == '__main__':
    main()
