"""Small frozen diagnostic prompts, native chat history and choice scoring; no training."""
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v1.common import atomic_json, file_sha256, lock, read_json
from evaluation.intact_base_384_benchmark import BUNDLE, SHA, RUN

OUT = ROOT / 'diagnostics/intact_base_384_review_v1'


def main():
    from sft.transfer_control.launch import preflight
    preflight()
    with lock(ROOT / 'sft/.experiment.lock'):
        import mlx.core as mx
        from sml_v1.checkpoint_bundle import resolve_bundle
        from sft.intact_smoltalk_base_pilot.engine import load, generate
        from evaluation.full_benchmarks.core import score_mc, verify
        weights = resolve_bundle(BUNDLE)
        if file_sha256(weights) != SHA: raise ValueError('Selected checkpoint changed')
        paths = [OUT / 'fresh_probes.json', Path(__file__).resolve(),
                 BUNDLE / 'manifest.json', BUNDLE / 'model.safetensors.json',
                 BUNDLE / 'model.safetensors', ROOT / 'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json']
        paths += list((ROOT / 'tokenizer/bytebpe32k_v1').glob('*'))
        protected = {str(p): file_sha256(p) for p in paths if p.is_file()}
        manifest = dict(protected=protected, weights_sha256=SHA, training=False,
                        generation_limit=384, decoding='native greedy cached, EOS only',
                        choice_scoring='Same raw and character-normalized likelihood scoring as public MC; plain Question/Answer prefix',
                        limitation='20 hand-authored diagnostic prompts, not a representative benchmark or training dataset.')
        mp = OUT / 'fresh_manifest.json'
        if mp.exists() and read_json(mp) != manifest: raise ValueError('Fresh diagnostic inputs changed')
        if not mp.exists(): atomic_json(mp, manifest)
        b = load(read_json(RUN / 'config.json'), weights=weights)
        b.decode_mode = 'cached'
        dest = OUT / 'fresh_answers.json'
        results = read_json(dest) if dest.exists() else []
        rows = read_json(OUT / 'fresh_probes.json')['rows']
        if [r['id'] for r in results] != [r['id'] for r in rows[:len(results)]]:
            raise ValueError('Fresh diagnostic cursor mismatch')
        for row in rows[len(results):]:
            actual = []; generations = []
            messages = row.get('messages', [dict(role='user', content=row['prompt'])])
            for message in messages:
                actual.append(message)
                g = generate(b, actual, 384)
                generations.append(g)
                actual.append(dict(role='assistant', content=g['text']))
            result = dict(**row, generations=generations, actual_messages=actual)
            if 'choices' in row:
                result['choice_scoring'] = score_mc(b, dict(id=row['id'], task=row['family'],
                      prompt='Question: ' + row['prompt'] + '\nAnswer:', choices=row['choices'], gold=row['gold']))
            results.append(result); atomic_json(dest, results)
            print('[probe]', len(results), '/', len(rows), row['id'], json.dumps(generations[-1]['text']), flush=True)
        del b; gc.collect(); mx.clear_cache(); verify(protected)
        atomic_json(OUT / 'fresh_integrity.json', dict(inputs_unchanged=True, completed=len(results),
                    expected=len(rows), training=False, manifest_sha256=file_sha256(mp)))


if __name__ == '__main__':
    main()
