"""Full ARC, PIQA, HellaSwag and IFEval for a saved teacher-assisted SFT checkpoint."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v2.common import atomic_json, file_sha256, fingerprint, lock, read_json

WORK = ROOT / 'runs/teacher_chat_768_v1'
RESULTS = ROOT / 'diagnostics/teacher_chat_768_benchmarks_v1'


def freeze_benchmark(core, model, suite, contract):
    """Resolve the architecture pin from this run's verified 768 parent.

    Teacher runs pin parent weights rather than copying the base architecture
    hash. Keep the shared benchmark and saved training contracts unchanged.
    """
    parent = (ROOT / contract['config']['source_bundle']).resolve()
    parent_meta_path = parent / 'model.safetensors.json'
    parent_contract_path = parent.parent / 'contract.json'
    parent_manifest_path = parent / 'manifest.json'
    parent_weights = parent / 'model.safetensors'
    manifest = read_json(parent_manifest_path)
    provenance = {
        str(parent_weights): contract['config']['source_model_sha256'],
        str(parent_meta_path): manifest['files']['model.safetensors.json']['sha256'],
    }
    core.verify(provenance)
    meta = read_json(parent_meta_path)
    parent_contract = read_json(parent_contract_path)
    if (meta['step'] != 768 or meta['contract'] != fingerprint(parent_contract)
            or manifest['files']['model.safetensors']['sha256'] != provenance[str(parent_weights)]):
        raise ValueError('Parent checkpoint or parent contract changed')
    verify_architecture = core.verify_architecture
    base = ROOT / core.BASE / 'model.safetensors.json'
    architecture = verify_architecture(parent_contract, base)

    def resolve_architecture(saved_contract, saved_base):
        if saved_contract != contract or saved_base != base:
            raise ValueError('Unexpected architecture verification request')
        augmented = dict(saved_contract)
        if 'architecture' not in augmented:
            augmented['architecture'] = architecture
        return verify_architecture(augmented, saved_base)

    # Adapt only this manifest construction; restore the shared function even
    # if verification fails. No files or training identities are rewritten.
    core.verify_architecture = resolve_architecture
    try:
        frozen = core.frozen(model, suite)
    finally:
        core.verify_architecture = verify_architecture
    for path in (parent_manifest_path, parent_contract_path):
        provenance[str(path)] = file_sha256(path)
    frozen['protected'].update(provenance)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--arm', choices=['conservative', 'moderate', 'faster'], default='moderate')
    parser.add_argument('--suite', choices=['all', 'multiple-choice', 'ifeval'], default='all')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.step not in (832, 896, 1024, 1152, 1280):
        parser.error('Choose a teacher SFT checkpoint: 832, 896, 1024, 1152, or 1280')
    RUN = WORK / args.arm
    updates = args.step - 768
    evaluation_path = RUN / f'evaluations/update_{updates:05d}.json'
    evaluation = read_json(evaluation_path)
    bundle = (ROOT / evaluation['bundle']).resolve()
    if bundle.parent != RUN or not bundle.name.startswith(f'step_{args.step:07d}_'):
        raise ValueError('Unexpected checkpoint path')
    meta = read_json(bundle / 'model.safetensors.json')
    contract = read_json(RUN / 'contract.json')
    cfg = contract['config']
    if (meta['step'] != args.step or meta['additional_updates'] != updates
            or meta['source_step'] != 768
            or meta['arm'] != args.arm
            or evaluation['update'] != updates
            or meta['source_bundle'] != cfg['source_bundle']
            or meta['source_model_sha256'] != cfg['source_model_sha256']
            or meta['training_format'] != 'plain-user-assistant-eos-v1'
            or meta['contract'] != fingerprint(contract)
            or evaluation['contract'] != meta['contract']):
        raise ValueError('Checkpoint ancestry or development evaluation mismatch')
    manifest = read_json(bundle / 'manifest.json')
    digest = file_sha256(bundle / 'model.safetensors')
    if digest != manifest['files']['model.safetensors']['sha256']:
        raise ValueError('Checkpoint weights changed')
    model = f'teacher-chat-{args.arm}-{args.step}'
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
            frozen = freeze_benchmark(core, model, suite, contract)
            frozen['code'][str(Path(__file__).resolve())] = file_sha256(__file__)
            for path in (RUN / 'contract.json', evaluation_path, pin,
                         bundle / 'model.safetensors.json', bundle / 'manifest.json'):
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
