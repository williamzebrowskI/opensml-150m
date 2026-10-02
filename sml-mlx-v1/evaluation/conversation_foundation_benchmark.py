"""Full ARC, PIQA, HellaSwag and IFEval for a saved Conversation Foundation step."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v1.common import atomic_json, file_sha256, lock, read_json
from sft.conversation_foundation_original.launch import accepts_contract

RUN = ROOT / 'runs/sft_conversation_foundation_v1'
RESULTS = ROOT / 'diagnostics/conversation_foundation_benchmarks_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--suite', choices=['all', 'multiple-choice', 'ifeval'], default='all')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.step <= 0:
        parser.error('Choose a saved post-training checkpoint')
    evaluation_path = RUN / f'evaluations/update_{args.step:05d}.json'
    evaluation = read_json(evaluation_path)
    bundle = RUN / evaluation['bundle']
    if bundle.parent != RUN or not bundle.name.startswith(f'step_{args.step:07d}_'):
        raise ValueError('Unexpected checkpoint path')
    meta = read_json(bundle / 'model.safetensors.json')
    contract = read_json(RUN / 'contract.json')
    cfg = contract['config']
    if (meta['step'] != args.step or meta['additional_updates'] != args.step
            or meta['source_step'] != 73243
            or meta['source_bundle'] != cfg['source_bundle']
            or meta['source_model_sha256'] != cfg['source_model_sha256']
            or meta['training_format'] != 'plain-user-assistant-eos-v1'
            or not accepts_contract(meta['contract'], contract)
            or evaluation['contract'] != meta['contract']):
        raise ValueError('Checkpoint ancestry or development evaluation mismatch')
    manifest = read_json(bundle / 'manifest.json')
    digest = file_sha256(bundle / 'model.safetensors')
    if digest != manifest['files']['model.safetensors']['sha256']:
        raise ValueError('Checkpoint weights changed')
    model = f'conversation-foundation-{args.step}'
    suites = ['multiple-choice', 'ifeval'] if args.suite == 'all' else [args.suite]
    print(json.dumps(dict(model=model, source=str(bundle), sha256=digest,
                         suites=suites, output=str(RESULTS / model),
                         training=False, automatic_promotion=False), indent=2), flush=True)
    if not (args.check or args.run):
        return
    from sft.transfer_control.launch import preflight
    preflight()
    with lock(ROOT / 'sft/.experiment.lock'):
        from evaluation.full_benchmarks import core, launch, spec
        spec.MODELS[model] = dict(bundle=str(bundle.relative_to(ROOT)),
                                 step=args.step, sha256=digest)
        core.RESULTS = launch.RESULTS = RESULTS
        pin = RESULTS / model / 'selection.json'
        identity = dict(bundle=str(bundle), sha256=digest, step=args.step,
                        development_evaluation_sha256=file_sha256(evaluation_path))
        if pin.exists() and read_json(pin) != identity:
            raise ValueError('Frozen benchmark selection changed')
        if not pin.exists():
            atomic_json(pin, identity)
        for suite in suites:
            frozen = core.frozen(model, suite)
            frozen['code'][str(Path(__file__).resolve())] = file_sha256(__file__)
            for path in (RUN / 'contract.json', evaluation_path, pin,
                         ROOT / 'sft/conversation_foundation_original/launch.py'):
                frozen['protected'][str(path)] = file_sha256(path)
            receipt = spec.DIR / f'readiness_{model}_{suite}.json'
            if args.check or not receipt.exists() or read_json(receipt)['contract'] != frozen:
                launch.check(model, suite, frozen)
            if args.check:
                continue
            launch.run(model, suite, frozen)
            result_path = RESULTS / model / suite / 'summary.json'
            print(f'Results: {result_path}', flush=True)
            result = read_json(result_path)
            if result['completed'] != result['expected']:
                break


if __name__ == '__main__':
    main()
