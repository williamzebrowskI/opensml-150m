"""Check cached versus full-prefix logits on two failing fresh probes."""
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v1.common import atomic_json, file_sha256, lock, read_json
from evaluation.intact_base_384_benchmark import BUNDLE, SHA, RUN


def main():
    from sft.transfer_control.launch import preflight
    preflight()
    with lock(ROOT / 'sft/.experiment.lock'):
        import mlx.core as mx
        from sft.intact_smoltalk_base_pilot.engine import load
        from sft.intact_smoltalk_base_pilot.data import visible_prefix
        if file_sha256(BUNDLE / 'model.safetensors') != SHA: raise ValueError('Selected weights changed')
        out = ROOT / 'diagnostics/intact_base_384_review_v1'
        rows = read_json(out / 'fresh_probes.json')['rows']
        b = load(read_json(RUN / 'config.json'), weights=BUNDLE / 'model.safetensors')
        checks = []
        for row in [r for r in rows if r['id'] in ('owner', 'latest-location')]:
            ids = b.encode(visible_prefix([dict(role='user', content=row['prompt'])]))
            output = []
            scores, cache = b.model.logits(mx.array([ids], dtype=mx.int32))
            scores = scores[:, -1, :]; mx.eval(scores, cache)
            for position in range(128):
                token = int(mx.argmax(scores[0]).item())
                if True:
                    reference = b.model.logits(mx.array([ids + output], dtype=mx.int32))[0][:, -1, :]
                    mx.eval(reference)
                    match = token == int(mx.argmax(reference[0]).item())
                    error = float(mx.max(mx.abs(scores - reference)).item())
                    checks.append(dict(id=row['id'],position=position,greedy_token_equal=match,
                                       max_abs_logit_difference=error))
                    if position in (0, 1, 7, 31, 63, 127): print(json.dumps(checks[-1]), flush=True)
                if token == b.eos: break
                output.append(token)
                scores, cache = b.model.step(mx.array([[token]], dtype=mx.int32), caches=cache)
                mx.eval(scores, cache)
        del b; gc.collect(); mx.clear_cache()
        if file_sha256(BUNDLE / 'model.safetensors') != SHA: raise ValueError('Selected weights changed')
        receipt = dict(status='passed' if all(r['greedy_token_equal'] for r in checks) else 'greedy_mismatch',
                       criterion='Greedy tokens agree at every one of the first 128 positions on each failing prompt; FP32 logits are not expected to be bit-identical across matrix shapes.',
                       checks=checks,weights_sha256=SHA,training=False,
                       max_abs_logit_difference=max(r['max_abs_logit_difference'] for r in checks))
        atomic_json(out / 'decode_audit.json', receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k!='checks'}), flush=True)


if __name__ == '__main__':
    main()
