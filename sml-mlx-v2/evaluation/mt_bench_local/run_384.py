"""Run the existing local MT-Bench protocol on intact Smol-SmolTalk SFT 384."""
from pathlib import Path
import sys

import run as benchmark
from sml_v2.common import file_sha256, fingerprint, read_json
from sml_v2.tokenization import Tokenizer

KEY = '384'
benchmark.MODELS[KEY] = (
    'intact-base-384',
    'runs/sft_intact_smoltalk_base_512_v1/step_0000384_02840558cc37',
    '35f78b8e76b29dc6b3e1f7b8b49858117f8b81ed5ff6c5e9c768afb3899a1b00',
)
original_selected = benchmark.selected


def selected(key):
    if key != KEY:
        return original_selected(key)
    name, relative, digest = benchmark.MODELS[key]
    bundle = benchmark.ROOT / relative
    meta = read_json(bundle / 'model.safetensors.json')
    contract = read_json(bundle.parent / 'contract.json')
    if file_sha256(bundle / 'model.safetensors') != digest:
        raise ValueError('Checkpoint weights changed')
    if (meta['step'], meta['source_step'], meta['training_format']) != (
            384, 73243, 'plain-user-assistant-eos-v1'):
        raise ValueError('Checkpoint identity mismatch')
    if meta['contract'] != fingerprint(contract):
        raise ValueError('Original training contract mismatch')
    if file_sha256(benchmark.BASE) != contract['protected'][str(benchmark.BASE)]:
        raise ValueError('Base architecture changed')
    tok = Tokenizer(benchmark.ROOT / 'tokenizer/bytebpe32k_v1')
    base = read_json(benchmark.BASE)
    if tok.fingerprint != benchmark.TOKENIZER or tok.fingerprint != base['v2_contract']['tokenizer']:
        raise ValueError('Tokenizer mismatch')
    return dict(model_id=name, bundle=str(bundle), weights_sha256=digest,
                metadata_sha256=file_sha256(bundle / 'model.safetensors.json'),
                checkpoint_wrapper_sha256=file_sha256(Path(__file__)))


benchmark.selected = selected

if __name__ == '__main__':
    if '--model' not in sys.argv:
        sys.argv.extend(['--model', KEY])
    try:
        benchmark.main()
    except KeyboardInterrupt:
        print('\n[interrupted] Completed items saved. Rerun the same command to resume.')
        sys.exit(130)
