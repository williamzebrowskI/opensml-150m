"""Stage verified v1 inputs; reuse the existing pinned wired-ring transport only."""

import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import uuid

from .common import ROOT, atomic_json, file_sha256, load_corpus, lock, read_json
from .recipe import tokens_per_update, validate

PARENT = ROOT.parent
PACKAGES = ('mlx', 'mlx-metal', 'numpy', 'tokenizers', 'datasets', 'huggingface-hub')


def network_helpers():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'scripts'))
    import check_ring_thunderbolt as wired
    import jaccl_benchmark as bench
    import benchmark_jaccl_ring as ring
    return wired, bench, ring


def stage(job, recipe, tokenizer, resume, corpus=None):
    source = job / 'input'
    source.mkdir(parents=True)
    shutil.copytree(ROOT / 'sml_v1', source / 'sml_v1', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(tokenizer, source / 'tokenizer')
    atomic_json(source / 'recipe.json', recipe)
    if corpus is not None:
        atomic_json(source / 'corpus.json', corpus)
    if resume:
        from .checkpoint_bundle import resolve_bundle
        bundle = Path(resolve_bundle(resume)).parent
        shutil.copytree(bundle, source / 'checkpoint')
    files = {}
    for path in source.rglob('*'):
        if path.is_symlink():
            raise ValueError('Staged inputs cannot include symbolic links')
        if path.is_file():
            files[str(path.relative_to(source))] = file_sha256(path)
    atomic_json(source / 'input_manifest.json', dict(files=files,
                packages={name: importlib.metadata.version(name) for name in PACKAGES}))
    return source


def verify_script(source):
    return f'''import hashlib,json,importlib.metadata
from pathlib import Path
p=Path({str(source)!r}).resolve()
m=json.loads((p/'input_manifest.json').read_text())
for name,sha in m['files'].items():
    f=p/name
    if not f.is_file() or f.is_symlink() or p not in f.resolve().parents:
        raise RuntimeError('Invalid staged file: '+name)
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''): h.update(chunk)
    if h.hexdigest()!=sha: raise RuntimeError('Hash mismatch: '+name)
for name,version in m['packages'].items():
    if importlib.metadata.version(name)!=version: raise RuntimeError('Package mismatch: '+name)
print('All v1 files and package versions verified')
'''


def idle_script():
    return '''import os,subprocess
rows=subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True).splitlines()
busy=[]
for row in rows:
    fields=row.strip().split(None,1)
    if len(fields)<2 or int(fields[0])==os.getpid(): continue
    cmd=fields[1]
    if ' -c ' in cmd: continue
    if any(s in cmd for s in ('-m sml_v1.pretrain','-m train.pretrain_jaccl','-m train.jaccl_worker','/train/train.py','benchmark_mlx_versions.py --run')):
        busy.append(row)
if busy: raise RuntimeError('Other training/benchmark workers active: '+str(busy))
print('No active training worker found')
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true', help='Explicitly copy inputs and start training')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/pilot.json')
    p.add_argument('--data', type=Path, default=ROOT / 'data/pilot_v1')
    p.add_argument('--corpus', type=Path, default=ROOT / 'configs/corpus.json')
    p.add_argument('--tokenizer', type=Path, default=ROOT / 'tokenizer/bytebpe32k_v1')
    p.add_argument('--save-dir', type=Path, default=ROOT / 'runs/pilot_v1')
    p.add_argument('--resume', type=Path, help='Explicit v1 latest.json or bundle; no old-run fallback')
    p.add_argument('--stop-after-steps', type=int, default=0, help='Bounded smoke run; saves then stops')
    p.add_argument('--clear-stop', action='store_true', help='Explicitly remove this output directory STOP marker')
    p.add_argument('--ring-control', choices=('all-sum', 'native'), default='all-sum',
                   help='Native requires a fixed MLX build and passes a startup correctness probe; default keeps workaround')
    p.add_argument('--extend-target-tokens', type=int,
                   help='Derive an explicit continuation from the resume checkpoint, with this TOTAL budget')
    p.add_argument('--rewarm-tokens', type=int, default=50_000_000,
                   help='Initial continuation LR ramp length; later resumes preserve the saved ramp')
    p.add_argument('--plateau-config', type=Path,
                   help='Explicit validation-scheduler continuation; preserve the source in a separate output')
    args = p.parse_args()
    recipe = validate(read_json(args.config))
    if len(recipe['batches']) == 5:
        p.error('Five ranks require scripts/launch_five_mac.py and a verified five-Mac network plan')
    if args.extend_target_tokens is not None:
        if not args.resume:
            p.error('--extend-target-tokens requires --resume')
        from .checkpoint_bundle import resolve_bundle
        from .continuation import extend_recipe
        weights = resolve_bundle(args.resume)
        metadata = read_json(weights + '.json')
        if 'continuation' not in metadata['recipe'] and args.save_dir.resolve() == Path(weights).parent.parent.resolve():
            p.error('Use a separate --save-dir for the continuation; preserve the pilot checkpoints')
        recipe = extend_recipe(metadata, args.extend_target_tokens, args.rewarm_tokens)
    if args.plateau_config is not None:
        if not args.resume or args.extend_target_tokens is not None:
            p.error('--plateau-config requires --resume and cannot also extend the token budget')
        from .checkpoint_bundle import resolve_bundle
        from .lr_scheduler import adaptive_recipe
        weights = resolve_bundle(args.resume)
        metadata = read_json(weights + '.json')
        if ('plateau_scheduler' not in metadata['recipe'] and
                args.save_dir.resolve() == Path(weights).parent.parent.resolve()):
            p.error('Use a separate --save-dir for the scheduler experiment; preserve the source run')
        recipe = adaptive_recipe(metadata, read_json(args.plateau_config))
    if 'continuation' in recipe and not args.resume:
        p.error('A continuation requires --resume')
    streaming = recipe.get('data_mode') == 'hf_stream'
    if args.stop_after_steps < 0:
        p.error('--stop-after-steps must be nonnegative')
    print(json.dumps(dict(recipe=recipe['name'], model=recipe['model'], batches=recipe['batches'],
        tokens_per_update=tokens_per_update(recipe), target_tokens=recipe['target_tokens'],
        data_mode=recipe.get('data_mode', 'tokens'),
        data=None if streaming else str(args.data), corpus=str(args.corpus) if streaming else None,
        tokenizer=str(args.tokenizer), output=str(args.save_dir),
        topology='wired JACCL ring' if len(recipe['batches']) == 4 else 'single host',
        resume=str(args.resume) if args.resume else None, ring_control=args.ring_control, run=args.run), indent=2))
    if 'continuation' in recipe:
        print('[continuation] ' + json.dumps(recipe['continuation'], sort_keys=True), flush=True)
    if 'plateau_scheduler' in recipe:
        print('[plateau-scheduler] ' + json.dumps(recipe['plateau_scheduler'], sort_keys=True), flush=True)
    if args.resume:
        # Reject incompatible code/recipes before any network or worker launch.
        from .checkpoint_bundle import resolve_bundle
        from .common import fingerprint
        from .continuation import check_resume
        metadata = read_json(resolve_bundle(args.resume) + '.json')
        proposed = dict(metadata['v2_contract'], recipe=fingerprint(recipe),
                        code=fingerprint({p.name: file_sha256(p) for p in (ROOT / 'sml_v1').glob('*.py')}))
        print('[resume-precheck] ' + check_resume(metadata, proposed, recipe), flush=True)
    if not args.run:
        print('Plan only. No SSH, GPU work, downloads or training. ' +
              ('Live HF streaming; no prepare_data step or saved corpus. Tokenizer must already be fitted.'
               if streaming else 'Local token pools must be prepared before training.'))
        return
    from .tokenization import Tokenizer
    from .data import verify_data
    tokenizer = Tokenizer(args.tokenizer)
    if tokenizer.vocab_size != recipe['model']['vocab_size']:
        raise ValueError('Vocabulary mismatch')
    if streaming:
        from .stream import stream_manifest
        stream_manifest(load_corpus(args.corpus), tokenizer, recipe['stream'])
    else:
        verify_data(args.data, tokenizer)
    output = args.save_dir.resolve()
    if not output.is_relative_to(ROOT / 'runs'):
        raise ValueError('V1 outputs must stay inside sml-mlx-v1/runs')
    if (output / 'latest.json').exists() and not args.resume:
        raise ValueError('Existing run: pass --resume explicitly or choose a new output')
    with lock(output / '.launcher.lock'):
        if args.clear_stop:
            (output / 'STOP').unlink(missing_ok=True)
        if (output / 'STOP').exists():
            raise ValueError('STOP is present; use --clear-stop only when ready to resume')
        launch(args, recipe, output)


def launch(args, recipe, output):
    if len(recipe['batches']) == 5:
        raise ValueError('Five ranks require the explicit five-Mac launcher; local fallback is forbidden')
    wired = bench = ring = mesh = None
    distributed = len(recipe['batches']) == 4
    python = str(Path(sys.executable).absolute())
    if distributed:
        wired, bench, ring = network_helpers()
        mesh = wired.mesh
        mapping = ring.read_ring_map()
        wired.check_mapping(mapping)
        wired.preflight(mapping)
        wired.verify_coordinator()
        order = mapping['physical_order']
        mode = 'ring-thunderbolt'
        python = mesh.PYTHON
    else:
        order = [0]
    def remote(physical, command, timeout=60):
        command = mesh.transport(physical, command, control_transport=mode) if distributed else command
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout
    for physical in order:
        print(remote(physical, [python, '-c', idle_script()]).strip())
    name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
    job = output / 'jobs' / name
    streaming = recipe.get('data_mode') == 'hf_stream'
    source = stage(job, recipe, args.tokenizer.resolve(), args.resume,
                   load_corpus(args.corpus) if streaming else None)
    if distributed:
        network = mesh.control_settings(mode)
        for physical in order[1:]:
            remote(physical, ['/bin/mkdir', '-p', str(source)])
            ssh = shlex.join(['/usr/bin/ssh', '-F', str(network['config'])])
            subprocess.run(['/usr/bin/rsync', '-a', '--partial', '-e', ssh, str(source) + '/',
                mesh.ssh_alias(physical, mode) + ':' + str(source) + '/'], check=True, timeout=600)
            print(f'[verified] mac-{physical+1}: {remote(physical, [python, "-c", verify_script(source)], 180).strip()}', flush=True)
        hosts = ring.hostfile('ring', mapping=mapping)
        for logical, physical in enumerate(order):
            hosts['hosts'][logical]['ssh'] = '127.0.0.1' if not logical else mesh.ssh_alias(physical, mode)
            hosts['hosts'][logical]['ips'] = ['127.0.0.1'] if not logical else []
        atomic_json(job / 'hosts.json', hosts)
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        env = bench.launch_environment(mode, 'jaccl-ring', port, job)
        prefix = [str(PARENT / '.venv/bin/mlx.launch'), '--backend', 'jaccl-ring', '--hostfile', str(job / 'hosts.json'),
                  '--cwd', str(source), '--starting-port', str(port), '--']
    else:
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(('MLX_', 'JACCL_', 'SML_JACCL_')) or key == 'PYTHONPATH':
                env.pop(key)
        prefix = []
    print(remote(0, [python, '-c', verify_script(source)], 180).strip())
    worker = ['/usr/bin/env', '-u', 'MLX_METAL_FAST_SYNCH', '-u', 'JACCL_RING', '/usr/bin/caffeinate', '-i',
              python, '-u', '-m', 'sml_v1.pretrain', '--config', str(source / 'recipe.json'),
              '--tokenizer', str(source / 'tokenizer'),
              '--save-dir', str(output), '--job-dir', str(job), '--stop-after-steps', str(args.stop_after_steps),
              '--ring-control', args.ring_control]
    if args.resume:
        worker += ['--resume', str(source / 'checkpoint')]
    if streaming:
        worker += ['--corpus', str(source / 'corpus.json')]
    else:
        worker += ['--data', str(args.data.resolve())]
    atomic_json(job / 'launch.json', dict(command=prefix + worker, physical_order=order,
                note=('Root alone streams/tokenizes HF data in RAM; no local corpus. ' if streaming else
                      'Root alone reads local data. ') + 'Workers use verified staged code and tokenizer.'))
    process = subprocess.Popen(prefix + worker, cwd=source, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        if not stopped:
            (output / 'STOP').touch()
            stopped = True
            print('[stop] Coordinated save/stop requested. Wait for [finished].', flush=True)
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    def cleanup():
        for logical, physical in enumerate(order):
            code = f'''import json,os,signal,subprocess
from pathlib import Path
p=Path({str(job / f'worker_rank{logical}.json')!r})
d=json.loads(p.read_text()) if p.exists() else {{}}
pid=int(d.get('pid',0))
r=subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],text=True,capture_output=True) if pid else None
if r and r.returncode==0 and '-m sml_v1.pretrain' in r.stdout and {str(job)!r} in r.stdout:
    os.kill(pid,signal.SIGKILL)
'''
            try:
                remote(physical, [python, '-c', code], 20)
            except Exception as exc:
                print(f'[warn] Cleanup unconfirmed on mac-{physical+1}: {exc}; worker watchdog still applies', flush=True)
    def kill_launcher():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
    timer = None
    if args.stop_after_steps:
        timer = threading.Timer(1800, kill_launcher)
        timer.daemon = True
        timer.start()
    try:
        with (output / 'train.log').open('a') as log, (job / 'train.log').open('w') as job_log:
            for line in process.stdout:
                print(line, end='', flush=True)
                for stream in (log, job_log):
                    stream.write(line); stream.flush()
                if re.search(r'Node with rank \d+ exited with code (?!0\b)', line):
                    raise RuntimeError('A worker exited; no automatic restart')
            status = process.wait()
        if status or not (job / 'result.json').exists():
            raise RuntimeError(f'Run failed (exit={status}); prior committed bundles are intact. See {job}')
        print(f'[verified-stop] {job / "result.json"}', flush=True)
    except BaseException:
        kill_launcher(); cleanup()
        raise
    finally:
        if timer:
            timer.cancel()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    main()
