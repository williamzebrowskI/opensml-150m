"""Prepare/check or run a matched two-base-model SFT diagnostic on one Mac."""
import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sml_v1.common import atomic_json, file_sha256, fingerprint, lock, read_json
from sft.transfer_control.data import records, audit, digest

DIR = Path(__file__).resolve().parent
OUTPUT = ROOT/'runs/sft_transfer_control_v1'
BASE_AUDIT = ROOT/'diagnostics/base_capability_20260924'
PARENT = ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors'
PARENT_SHA = 'ca9bd5d82005f28e77f319d3a5a29f29fb4e4f86e7ceea049f01162fd8aae80f'


def code_files():
    return sorted(list(DIR.glob('*.py')) + list((ROOT/'sml_v1').glob('*.py')) +
                  list((ROOT/'evaluation/base_audit').glob('*.py')))


def protected():
    saved = read_json(BASE_AUDIT/'protected_before.json')
    reference = read_json(BASE_AUDIT/'protocol.json')['reference']
    for name, sha in reference['hashes'].items(): saved[str(BASE_AUDIT/'reference'/name)] = sha
    if saved.get(str(PARENT)) != PARENT_SHA: raise ValueError('Unexpected pretrained parent')
    for path, sha in saved.items():
        if file_sha256(path) != sha: raise ValueError(f'Protected input changed: {path}')
    return saved


def prose_texts():
    # Existing evaluation-only passages. Never used in gradients, checkpoint
    # selection, or as a new general-knowledge benchmark.
    rows = [r for r in read_json(BASE_AUDIT/'items.json') if r['task'] == 'boolq']
    rows.sort(key=lambda r: digest(r['id']))
    return [r['prompt'].split('\nQuestion:')[0] for r in rows[:64]]


def contract(cfg):
    return dict(config=fingerprint(cfg), data=audit(), prose=digest(prose_texts()),
                code={str(p.relative_to(ROOT)): file_sha256(p) for p in code_files()},
                protected=protected(), reference_protocol=file_sha256(BASE_AUDIT/'protocol.json'),
                runtime={p: importlib.metadata.version(p) for p in ('mlx', 'mlx-lm', 'numpy', 'transformers', 'tokenizers')})


def preflight():
    lines = subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,command='], text=True)
    busy = []
    for line in lines.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3 or int(parts[0]) in (os.getpid(), os.getppid()): continue
        command = parts[2]
        if any(x in command for x in ('-m sml_v1.pretrain', '-m sft.', '/sft/', 'launch_five_mac.py', '/stage_b/launch.py')):
            # Only commands actually invoking a Python worker, not an editor or
            # an inspection shell whose quoted argument mentions the directory.
            if 'python' in command and not any(x in command for x in ('/bin/zsh -c ', '/bin/bash -c ', ' -c ')):
                busy.append(line.strip())
    if busy: raise RuntimeError('Other training/setup process active: '+repr(busy))


def check(cfg, frozen):
    import mlx.core as mx
    from sft.transfer_control.engine import load, encode, arrays, objective, prose_loss, generate, prefix
    preflight(); result = {}
    for arm in cfg['arms']:
        backend = load(arm)
        splits = {}
        for split in ('train', 'dev', 'test'):
            rr = records(split)
            encoded = [encode(backend, r) for r in rr]
            splits[split] = dict(examples=len(rr), answer_targets=sum(r['targets'] for r in encoded),
                                 max_length=max(len(r['x']) for r in encoded))
        for r in records('dev', 'familiar'): encode(backend, r)
        x, y = arrays([encode(backend, r) for r in records('train')[:2]], backend.pad)
        value = float(objective(backend.model, x, y, arm=='reference').item())
        if not math.isfinite(value): raise ValueError('Nonfinite forward loss')
        # Compare padded/unpadded logits on the same real first example.
        n = len(encode(backend, records('train')[0])['x'])
        short = backend.logits(x[:1, :n])[0, -1].astype(mx.float32)
        padded = backend.logits(x[:1])[0, n-1].astype(mx.float32)
        gap = float(mx.max(mx.abs(short-padded)).item())
        if gap > .2: raise ValueError('Unexpected padding sensitivity')
        probe = generate(backend, 'The cup is on the shelf. Where is the cup?', limit=4)
        drift = prose_loss(backend, prose_texts())
        result[arm] = dict(splits=splits, forward_loss=value, prose_baseline=drift,
                          padding_max_logit_difference=gap, generation_probe=probe,
                          format_example=prefix(records('train')[0]['prompt']), eos=backend.eos)
        print('[check]', arm, 'forward, generation, full tokenization and protected inputs passed', flush=True)
        del backend; gc.collect(); mx.clear_cache()
    assert contract(cfg) == frozen
    atomic_json(DIR/'readiness.json', dict(status='passed', contract=frozen, checks=result,
                                         pretrained_updates=0, reference_updates=0))


class Tee:
    def __init__(self, terminal, log): self.terminal, self.log = terminal, log
    def write(self, value):
        self.terminal.write(value); self.log.write(value); self.log.flush(); return len(value)
    def flush(self): self.terminal.flush(); self.log.flush()
    def __getattr__(self, key): return getattr(self.terminal, key)


def run_arm(arm, cfg, frozen, stop):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v1.checkpoint_bundle import save_bundle, resolve_bundle
    from sml_v1.precision import MasterAdamW
    from sml_v1.pretrain import clip_gradients
    from sft.transfer_control.engine import load, encode, arrays, gradients, learning_rate, evaluate, prose_loss
    out = OUTPUT/arm; out.mkdir(parents=True, exist_ok=True)
    report_path = out/'report.json'
    if report_path.exists() and read_json(report_path)['status'] == 'complete':
        print('[skip-completed]', arm, flush=True); return
    backend = load(arm)
    optimizer = MasterAdamW(learning_rate=0., betas=tuple(cfg['betas']), weight_decay=cfg['weight_decay'])
    train = records('train'); total = cfg['updates']; step = 0; best_key = None
    report = dict(status='running', arm=arm, training=[], evaluations=[], promoted=False)
    if (out/'latest.json').exists():
        weights = resolve_bundle(out/'latest.json'); metadata = read_json(weights+'.json')
        if metadata['contract'] != fingerprint(frozen): raise ValueError('Resume contract changed')
        backend.model.load_weights(weights); step = metadata['step']; best_key = metadata['best_key']
        if step:
            optimizer.state = tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
        report = metadata['report']; report['status'] = 'running'
        if (out/'best.json').exists():
            best_weights = resolve_bundle(out/'best.json')
            report['best'] = dict(step=read_json(best_weights+'.json')['step'], path=str(Path(best_weights).parent))
        mx.eval(backend.model.parameters(), optimizer.state)
        print('[resume]', arm, step, 'with exact optimizer and next example', flush=True)

    def save(best=False):
        report['step'] = step
        meta = dict(step=step, arm=arm, contract=fingerprint(frozen), best_key=best_key,
                    report=report, next_example=step*cfg['batch'], experimental=True,
                    training_format='plain-user-assistant-eos-v1', playground_compatible=False)
        path = save_bundle(out, backend.model, optimizer, meta, None, best=best, keep=1)
        report['latest'] = path
        if best: report['best'] = dict(step=step, path=path)
        atomic_json(report_path, report)
        print('[checkpoint]', arm, step, 'best + latest' if best else 'latest', flush=True)

    def assess(step):
        unseen = evaluate(backend, records('dev'), cfg['max_new_tokens'])
        familiar = evaluate(backend, records('dev', 'familiar'), cfg['max_new_tokens'], include_open=False)
        prose = prose_loss(backend, prose_texts())
        value = dict(step=step, unseen=unseen, familiar=familiar, prose=prose)
        if step == 0: report['baseline_prose'] = prose
        value['prose_change'] = prose-report['baseline_prose']
        atomic_json(out/'evaluations'/f'step_{step:07d}.json', value)
        row = dict(step=step, exact=unseen['exact'], both_correct=unseen['both_correct'],
                   familiar_exact=familiar['exact'], same_answer_pairs=unseen['same_answer_pairs'],
                   stopped=unseen['stopped'], assistant_nll=unseen['assistant_nll'], prose_change=value['prose_change'])
        report['evaluations'].append(row)
        print('[eval]', arm, json.dumps(row), flush=True)
        return [unseen['both_correct'], unseen['exact'], unseen['stopped'], -unseen['assistant_nll']]

    try:
        if not (out/'latest.json').exists():
            assess(0); save()
        print(f'[ready] {arm}: {step}/{total} updates; {len(train)} examples; no automatic extension', flush=True)
        while step < total:
            if (OUTPUT/'STOP').exists(): stop['requested'] = True
            if stop['requested']: break
            rows = train[step*cfg['batch']:(step+1)*cfg['batch']]
            if len(rows) != cfg['batch']: raise ValueError('Training coverage mismatch')
            encoded = [encode(backend, r) for r in rows]
            x, y = arrays(encoded, backend.pad)
            value, grad = gradients(backend.model, x, y, arm=='reference', cfg['microbatch'])
            grad, norm = clip_gradients(grad, cfg['clip_norm']); mx.eval(value, grad, norm)
            if not math.isfinite(float(value.item())) or not math.isfinite(float(norm.item())):
                raise ValueError('Nonfinite update; previous checkpoint retained')
            lr = learning_rate(step+1, cfg); optimizer.learning_rate = lr
            optimizer.update(backend.model, grad); mx.eval(backend.model.parameters(), optimizer.state)
            step += 1
            report['training'].append(dict(step=step, examples=step*cfg['batch'], loss=float(value.item()),
                                           lr=lr, grad_norm=float(norm.item()), batch_hash=digest([r['id'] for r in rows])))
            if step == 1 or step % 8 == 0: print(f'[control {arm} {step}/{total}] loss={value.item():.4f} lr={lr:.3e}', flush=True)
            if step in cfg['evaluation_steps']:
                key = assess(step); improved = best_key is None or key > best_key
                if improved: best_key = key
                save(improved)
            elif step % 16 == 0: save()
        if stop['requested']:
            report['status'] = 'stopped'; save(); return
        # Test groups/wording are not used for selecting a checkpoint. Best is a
        # trained development candidate even if it did not improve over baseline.
        selected = resolve_bundle(out/'best.json'); backend.model.load_weights(selected); mx.eval(backend.model.parameters())
        report['best'] = dict(step=read_json(selected+'.json')['step'], path=str(Path(selected).parent))
        test = evaluate(backend, records('test'), cfg['max_new_tokens'])
        atomic_json(out/'test_best.json', dict(selected=selected, evaluation=test, selection_unchanged=True))
        report.update(status='complete', step=step, test=str(out/'test_best.json'))
        atomic_json(report_path, report)
        print('[finished]', arm, 'reserved pair score', f'{test["both_correct"]:.1%}', flush=True)
    except Exception as exc:
        report.update(status='error', error=repr(exc)); atomic_json(report_path, report); raise
    finally:
        del backend, optimizer; gc.collect(); mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true'); mode.add_argument('--run', action='store_true')
    parser.add_argument('--clear-stop', action='store_true')
    args = parser.parse_args(); cfg = read_json(DIR/'config.json')
    if len(records('train')) != cfg['updates']*cfg['batch']: raise ValueError('Not exactly one pass')
    if cfg['arms'] != ['opensml', 'reference']: raise ValueError('This is a matched two-arm protocol')
    frozen = contract(cfg)
    print(json.dumps(dict(mode='run' if args.run else 'check' if args.check else 'plan',
                         arms=cfg['arms'], parent=str(PARENT), reference='SmolLM2-135M base, existing pinned copy',
                         output=str(OUTPUT), examples_per_model=len(records('train')), updates_per_model=cfg['updates'],
                         peak_lr=cfg['peak_lr'], final_lr=cfg['final_lr'], teacher=False, main_mac_only=True,
                         data='Original deterministic curriculum; generated on demand, no HF dataset download.',
                         purpose='Matched learning/transfer diagnostic, not a ready chat model.'), indent=2), flush=True)
    if not args.check and not args.run: return
    with lock(ROOT/'sft/.experiment.lock'):
        if args.check: check(cfg, frozen); return
        ready = read_json(DIR/'readiness.json')
        if ready['status'] != 'passed' or ready['contract'] != frozen: raise ValueError('Setup changed; rerun --check')
        preflight()
        if (OUTPUT/'contract.json').exists():
            if read_json(OUTPUT/'contract.json') != frozen: raise ValueError('Run code/config/input contract changed')
        else:
            atomic_json(OUTPUT/'contract.json', frozen)
            for p in code_files()+[DIR/'config.json', DIR/'README.md']:
                dest = OUTPUT/'input'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
        if args.clear_stop: (OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists(): raise ValueError('STOP exists; use --clear-stop to resume')
        stop = dict(requested=False)
        def handler(sig, frame):
            stop['requested'] = True; (OUTPUT/'STOP').touch()
            print('[stop] Saving after the current update/evaluation. The second arm will not start.', flush=True)
        handlers = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            with (OUTPUT/'console.log').open('a') as log:
                old = sys.stdout, sys.stderr; sys.stdout = Tee(old[0], log); sys.stderr = Tee(old[1], log)
                try:
                    for arm in cfg['arms']:
                        if stop['requested']: break
                        run_arm(arm, cfg, frozen, stop)
                    if not stop['requested']:
                        comparison = {}
                        for arm in cfg['arms']:
                            r = read_json(OUTPUT/arm/'report.json'); t = read_json(OUTPUT/arm/'test_best.json')['evaluation']
                            comparison[arm] = dict(status=r['status'], development=r['evaluations'],
                                                   reserved={k:t[k] for k in ('exact','both_correct','stopped','same_answer_pairs')})
                        atomic_json(OUTPUT/'comparison.json', comparison)
                        print('[complete] Both arms finished. Review comparison.json and raw answers; no playground change.', flush=True)
                finally: sys.stdout, sys.stderr = old
        finally:
            for s, h in handlers.items(): signal.signal(s, h)
            assert protected() == frozen['protected']


if __name__ == '__main__': main()
