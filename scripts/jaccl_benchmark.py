#!/usr/bin/env python3
"""Prepare or run isolated unequal-batch JACCL benchmarks from a complete checkpoint."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import jaccl_common as common
import configure_rdma_mesh as mesh
import check_jaccl

PYTHON = str(ROOT / '.venv/bin/python')
DEFAULT = ROOT / 'train/checkpoints/dolma_cosmo_152m_wide_full_v1/latest.json'
INPUTS = ROOT / 'train/checkpoints/jaccl_benchmark_inputs'
OUTPUTS = ROOT / 'train/checkpoints/jaccl_benchmarks'
CODE = [f'train/{name}.py' for name in (
    '__init__', 'model', 'train', 'precision', 'fast_loss', 'fast_swiglu', 'ffn_options', 'performance',
    'tokenization', 'checkpoint_bundle', 'jaccl_worker', 'jaccl_common', 'jaccl_control')]
CODE += ['train/data/__init__.py', 'train/data/stream.py']
CODE += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'tokenizer/cosmo2').glob('*.json'))]
CODE += [f'train/data/dolma_v1_7_balanced/{name}.txt' for name in
         ('math', 'code', 'web', 'science', 'flan', 'stackexchange', 'reddit', 'reference')]
CODE += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'train/data/dolma_v1_7_prose').glob('*.txt'))]


def setup_retry(operation, label):
    # Only setup/copy operations are retried. Never restart a failed GPU trial.
    for attempt in range(3):
        try:
            return operation()
        except (RuntimeError, subprocess.SubprocessError) as exc:
            if attempt == 2:
                raise
            print(f'[retry] {label}: {exc}; retrying setup {attempt + 2}/3', flush=True)
            time.sleep(2 ** (attempt + 1))


def remote_common(rank, expression, control_transport='wifi'):
    source = (ROOT / 'train/jaccl_common.py').read_text()
    code = f'exec(compile({source!r}, "jaccl_common.py", "exec")); print(json.dumps({expression}))'
    return json.loads(setup_retry(lambda: mesh.run(mesh.transport(rank, [PYTHON, '-c', code], control_transport=control_transport), timeout=180,
                               label=f'mac-{rank + 1} input inspection'), f'mac-{rank + 1} inspection').stdout)


def preflight(gpu=False, control_transport='wifi'):
    check_jaccl.check_hostfile(control_transport)
    states = []
    for rank in range(4):
        state = setup_retry(lambda: mesh.invoke(rank, 'inspect', control_transport=control_transport),
                            f'mac-{rank + 1} network preflight')
        mesh.configured_check(state, rank)
        states.append(state)
        if gpu:
            busy = remote_common(rank, 'training_processes()', control_transport)
            if busy:
                raise RuntimeError(f'mac-{rank + 1} has an active GPU job; stop it after a checkpoint first: {busy}')
        print(f'[preflight] mac-{rank + 1}: physical identity, {control_transport} control and mesh ports verified', flush=True)
    mesh.topology_check(states)
    if control_transport == 'thunderbolt':
        from configure_thunderbolt_control import verify_coordinator
        verify_coordinator()


def resolve_source(path):
    path = Path(path).absolute()
    if path.suffix == '.json':
        pointer = json.loads(path.read_text())
        name = pointer.get('bundle', '')
        if pointer.get('format') != 'sml-pretrain-bundle-v1' or Path(name).name != name or not name.startswith('step_'):
            raise ValueError('Expected a transactional production checkpoint pointer')
        path = path.parent / name
    if path.is_symlink():
        raise ValueError('Checkpoint must not be a symlink')
    manifest = json.loads((path / 'manifest.json').read_text())
    required = {'model.safetensors', 'model.safetensors.optimizer.safetensors',
                'model.safetensors.json', 'model.safetensors.rank0.data_state.json.gz'}
    if manifest.get('format') != 'sml-pretrain-bundle-v1' or not required <= manifest['files'].keys():
        raise ValueError('Need a complete model/optimizer/data-state production bundle')
    return path, manifest


def prepare(checkpoint):
    source, manifest = resolve_source(checkpoint)
    code_records = {name: common.file_record(ROOT / name) for name in CODE}
    identity = hashlib.sha256(json.dumps(dict(bundle=manifest, code=code_records), sort_keys=True).encode()).hexdigest()[:12]
    destination = INPUTS / f'step_{manifest["step"]:07d}_{identity}'
    if destination.exists():
        common.verify_input(destination)
        return destination
    INPUTS.mkdir(parents=True, exist_ok=True)
    needed = sum(v['bytes'] for v in manifest['files'].values()) + 5 * 1024**3
    if shutil.disk_usage(INPUTS).free < needed:
        raise OSError('Not enough free space for a pinned benchmark input plus 5 GiB reserve')
    staging = INPUTS / ('.prepare-' + uuid.uuid4().hex)
    staging.mkdir()
    try:
        (staging / 'bundle').mkdir()
        for name in [*manifest['files'], 'manifest.json']:
            if Path(name).name != name or (source / name).is_symlink():
                raise ValueError('Unsafe checkpoint filename')
            shutil.copy2(source / name, staging / 'bundle' / name)
        common.verify_files(staging / 'bundle', manifest['files'])
        for name in CODE:
            target = staging / 'code' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        common.verify_files(staging / 'code', code_records)
        files = {str(p.relative_to(staging)): common.file_record(p) for p in staging.rglob('*') if p.is_file()}
        common.atomic_json(staging / 'input.json', dict(format=common.INPUT_FORMAT, source=str(source),
                            step=manifest['step'], files=files))
        common.verify_input(staging)
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f'[snapshot] pinned step {manifest["step"]}: {destination}', flush=True)
    return destination


def sync_input(destination, prepared_file=None, control_transport='wifi'):
    local = common.verify_input(destination)
    prepared_file = prepared_file or ROOT / 'cluster/jaccl/prepared.json'
    previous = None
    if prepared_file.exists():
        candidate = Path(json.loads(prepared_file.read_text())['input'])
        if candidate.parent == destination.parent and candidate != destination and candidate.name.startswith('step_'):
            previous = candidate
    needed = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()) + 8 * 1024**3
    for rank in range(1, 4):
        code = ('import pathlib,shutil; '
                f'p=pathlib.Path({str(destination)!r}); '
                f'assert shutil.disk_usage({str(ROOT)!r}).free > {needed}, "Insufficient disk reserve"; '
                'p.mkdir(parents=True,exist_ok=True)')
        setup_retry(lambda: mesh.run(mesh.transport(rank, [PYTHON, '-c', code], control_transport=control_transport),
                    label=f'mac-{rank + 1} input directory'), f'mac-{rank + 1} input directory')
        print(f'[copy] mac-{rank + 1}: syncing model, FP32 optimizer, tokenizer and benchmark code...', flush=True)
        # Destination is a content-addressed benchmark directory. Never delete remote project files.
        transport = f'/usr/bin/ssh -F {shlex.quote(str(mesh.control_settings(control_transport)["config"]))}'
        # Immutable snapshots may share unchanged files, avoiding repeated 2 GiB
        # optimizer transfers for code-only changes. Never use --inplace here.
        reuse = ['--link-dest=' + str(previous)] if previous else []
        setup_retry(lambda: subprocess.run(['/usr/bin/rsync', '-a', '--partial', *reuse, '-e', transport, str(destination) + '/',
                        f'{mesh.ssh_alias(rank, control_transport)}:{destination}/'], check=True), f'mac-{rank + 1} copy')
        remote = remote_common(rank, f'verify_input({str(destination)!r})', control_transport)
        if remote != local:
            raise RuntimeError(f'mac-{rank + 1}: file hashes or MLX/dependency versions differ')
        print(f'[verified] mac-{rank + 1}: all input hashes and package versions match mac-1', flush=True)
    common.atomic_json(prepared_file, dict(input=str(destination), **local))
    return local


def launch_environment(control_transport, backend, port, output):
    from train.jaccl_ring_transport import MODE, PORT_ENV, JOB_ENV
    env = dict(os.environ, PATH=str(ROOT / 'cluster/jaccl/production_bin') + ':' + os.environ.get('PATH', ''))
    env['SML_JACCL_CONTROL_TRANSPORT'] = control_transport
    for name in ('MLX_RANK', 'MLX_WORLD_SIZE', 'MLX_HOSTFILE', 'MLX_IBV_DEVICES',
                 'MLX_JACCL_COORDINATOR', 'JACCL_COORDINATOR', 'MLX_METAL_FAST_SYNCH', 'MLX_JACCL_RING',
                 'JACCL_RING', 'JACCL_RANK', 'JACCL_IBV_DEVICES', 'PYTHONPATH', PORT_ENV, JOB_ENV):
        env.pop(name, None)
    if control_transport == MODE:
        if backend != 'jaccl-ring' or not port or not Path(output).is_absolute():
            raise ValueError('Wired ring benchmark requires a ring, port and absolute job directory')
        env[PORT_ENV], env[JOB_ENV] = str(port), str(output)
    return env


def launch(destination, output, batches, args, cpu=False):
    control_transport = getattr(args, 'control_transport', 'wifi')
    network = mesh.control_settings(control_transport)
    backend = getattr(args, 'collective_backend', 'jaccl')
    if backend not in ('jaccl', 'jaccl-ring'):
        raise ValueError('Only JACCL mesh or JACCL RDMA ring is supported')
    if backend == 'jaccl-ring' and (control_transport not in ('wifi', 'ring-thunderbolt') or len(batches) != 4):
        raise ValueError('Ring experiments require four ranks and explicit ring control')
    if control_transport == 'ring-thunderbolt' and backend != 'jaccl-ring':
        raise ValueError('Wired ring control requires jaccl-ring')
    hostfile = getattr(args, 'collective_hostfile', network['hostfile'])
    physical_order = getattr(args, 'physical_order', list(range(len(batches))))
    if (sorted(physical_order) != list(range(len(batches))) or physical_order[0] != 0
            or (backend != 'jaccl-ring' and physical_order != list(range(len(batches))))):
        raise ValueError('Invalid benchmark physical rank mapping')
    if physical_order != list(range(len(batches))) and len(set(batches[1:])) != 1:
        raise ValueError('Reordered ring peers require equal peer batches')

    def stop_remote_workers():
        for rank in range(1, len(batches)):
            code = ('import json,os,signal,subprocess; from pathlib import Path; '
                    f'p=Path({str(output / f"worker_rank{rank}.json")!r}); '
                    'd=json.loads(p.read_text()) if p.exists() else {}; pid=int(d.get("pid",0)); '
                    'r=subprocess.run(["/bin/ps","-p",str(pid),"-o","command="],text=True,capture_output=True) if pid else None; '
                    f'valid=r is not None and r.returncode==0 and "train.jaccl_worker" in r.stdout and {str(output)!r} in r.stdout; '
                    'os.kill(pid,signal.SIGTERM) if valid else None')
            try:
                mesh.run(mesh.transport(physical_order[rank], [PYTHON, '-c', code], control_transport=control_transport), timeout=20,
                         label=f'mac-{physical_order[rank] + 1} stop this benchmark worker')
            except Exception as exc:
                print(f'[warn] Worker cleanup could not be confirmed: {exc}. Rank watchdog remains enabled.', flush=True)
    port = None
    if len(batches) == 4:
        with socket.socket() as sock:
            sock.bind((network['listen'], 0))
            port = sock.getsockname()[1]
        command = [str(ROOT / '.venv/bin/mlx.launch'), '--backend', backend, '--hostfile',
                   str(hostfile), '--cwd', str(destination / 'code'),
                   '--starting-port', str(port), '--']
    else:
        command = []
    command += ['/usr/bin/env', '-u', 'MLX_METAL_FAST_SYNCH', '-u', 'JACCL_RING',
                *([] if backend == 'jaccl-ring' else ['-u', 'MLX_JACCL_RING']),
                '/usr/bin/caffeinate', '-i', PYTHON, '-u', '-m', 'train.jaccl_worker',
                '--input', str(destination), '--output', str(output), '--batches', ','.join(map(str, batches)),
                '--steps', str(args.steps), '--warmup', str(args.warmup)]
    if cpu:
        command.append('--cpu-test')
    command += ['--gradient-bucket-mib', str(getattr(args, 'gradient_bucket_mib', 32)),
                '--resident-batches', str(getattr(args, 'resident_batches', 0)),
                '--grad-accum', str(getattr(args, 'grad_accum', 2)),
                '--resident-microbatches', str(getattr(args, 'resident_microbatches', 0))]
    command += ['--expected-topology', 'ring' if backend == 'jaccl-ring' else 'mesh',
                '--comms-rounds', str(getattr(args, 'comms_rounds', 0))]
    if getattr(args, 'no_save_trial', False):
        command.append('--no-save-trial')
    if getattr(args, 'root_data_stream', False):
        command.append('--root-data-stream')
    if getattr(args, 'check_accumulation', False):
        command.append('--check-accumulation')
    if len(batches) == 4:
        boundary = command.index('--') + 1
        command = command[:boundary] + mesh.worker_command(command[boundary:], control_transport)
    env = launch_environment(control_transport, backend, port, output)
    output.mkdir(parents=True)
    timed_out = threading.Event()
    process = subprocess.Popen(command, cwd=destination / 'code', env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               bufsize=1, start_new_session=True)

    def terminate():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def deadline():
        timed_out.set()
        terminate()
    timer = threading.Timer(getattr(args, 'cpu_timeout', 120) if cpu else args.timeout, deadline)
    timer.daemon = True
    timer.start()
    try:
        from train.jaccl_training_state import worker_failed
        with (output / 'train.log').open('w') as log:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
                if worker_failed(line):
                    raise RuntimeError(f'Benchmark worker failed: {line.strip()}')
        status = process.wait()
    except BaseException:
        terminate()
        stop_remote_workers()
        raise
    finally:
        timer.cancel()
    if timed_out.is_set() or status != 0:
        stop_remote_workers()
        raise RuntimeError(f'Benchmark failed or timed out; not advancing to the next split. See {output / "train.log"}')
    if cpu:
        text = (output / 'train.log').read_text()
        if not all(f'[cpu-ok] rank={rank} ' in text for rank in range(4)):
            raise RuntimeError('Not all four CPU correctness checks completed')
        if getattr(args, 'comms_rounds', 0) and not all(
                f'[comms-ok] mode=framed rank={rank} rounds={args.comms_rounds} ' in text for rank in range(4)):
            raise RuntimeError('Not all four collective stress checks completed')
        if getattr(args, 'comms_rounds', 0) and not all(
                f'[rdma-payload-ok] rank={rank} ' in text for rank in range(4)):
            raise RuntimeError('Not all four large-payload sum checks completed')
        return None
    results = []
    for rank in range(len(batches)):
        path = output / f'rank{rank}.json'
        if rank:
            value = remote_common(physical_order[rank], f'json.loads(Path({str(path)!r}).read_text())', control_transport)
            common.atomic_json(path, value)
            subprocess.run(['/usr/bin/rsync', '-a', '-e',
                f'/usr/bin/ssh -F {network["config"]}',
                f'{mesh.ssh_alias(physical_order[rank], control_transport)}:{output}/rank{rank}.data_state.json.gz', str(output) + '/'], check=True)
        else:
            value = json.loads(path.read_text())
        if (not value.get('passed') or value['batches'] != batches or value['rank'] != rank
                or value.get('topology') != ('ring' if backend == 'jaccl-ring' else 'mesh')):
            raise RuntimeError(f'Invalid rank {rank} benchmark result')
        results.append(value)
    if len({v['model_optimizer_sha256'] for v in results}) != 1:
        raise RuntimeError('Replica checksums do not match')
    return dict(batches=batches, physical_order=physical_order, tokens_per_update=results[0]['tokens_per_update'],
                tokens_per_second=min(v['tokens_per_second'] for v in results),
                peak_gib=[v['peak_gib'] for v in results],
                checkpoint=None if getattr(args, 'no_save_trial', False) else str(output / 'checkpoint'),
                topology=results[0]['topology'], control_transport=control_transport,
                data_layout=results[0]['data_layout'],
                measured_steps=args.steps, passed=True,
                gradient_bucket_mib=results[0]['gradient_bucket_mib'],
                input_sha256=[v['input_sha256'] for v in results],
                global_input_sha256=[v.get('global_input_sha256') for v in results],
                gradient_checks=[v['gradient_check'] for v in results],
                grad_accum=results[0]['grad_accum'],
                warmup=results[0]['warmup'],
                accumulation_checks=[v['accumulation_check'] for v in results],
                model_optimizer_sha256=results[0]['model_optimizer_sha256'],
                fixed_lr=results[0]['fixed_lr'], source_step=results[0]['source_step'])


def report(root, results):
    common.atomic_json(root / 'summary.json', results)
    lines = ['# JACCL Throughput Comparison', '',
             'Fresh-stream experiments, not production continuation. Warmup, evaluation and checkpoint I/O excluded from measured throughput.', '',
             '| Batches | Tokens/update | Tokens/sec | Peak MLX GiB by rank |',
             '| --- | ---: | ---: | --- |']
    for row in results:
        lines.append(f'| {"/".join(map(str,row["batches"]))} | {row["tokens_per_update"]:,} | {row["tokens_per_second"]:,.0f} | {", ".join(f"{x:.2f}" for x in row["peak_gib"])} |')
    if results:
        best = max(results, key=lambda r: r['tokens_per_second'])
        lines += ['', f'Fastest measured configuration: {best["batches"]}.',
                  'Different global batches change optimization; throughput alone does not select the best learning recipe.']
    (root / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true', help='Copy and verify inputs only (default); no GPU use')
    mode.add_argument('--run', action='store_true', help='Run local baseline, then the requested GPU splits')
    mode.add_argument('--cpu-test', action='store_true', help='Check real JACCL weighted gradients on CPU only')
    parser.add_argument('--checkpoint', default=str(DEFAULT), help='Complete production bundle or latest.json')
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument('--splits', help='Increasing subset of 8,12,16,20,24; Ultra stays at 32 (default: 8,12,16)')
    shapes.add_argument('--cases', help='Ordered explicit splits, e.g. 32/24/24/24,40/24/24/24; Ultra=32,40,48')
    parser.add_argument('--steps', type=int, default=100, help='Measured updates per case (1..500)')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup updates per case (1..20)')
    parser.add_argument('--timeout', type=int, default=1800, help='Maximum seconds per GPU case (60..7200)')
    args = parser.parse_args()
    cases = (common.parse_cases(args.cases) if args.cases is not None else
             [[32, n, n, n] for n in common.parse_splits(args.splits if args.splits is not None else '8,12,16')])
    if not 1 <= args.steps <= 500 or not 1 <= args.warmup <= 20 or not 60 <= args.timeout <= 7200:
        parser.error('Invalid bounded experiment limits')
    INPUTS.mkdir(parents=True, exist_ok=True)
    lock = open(INPUTS / '.launcher.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight(gpu=args.run)
    destination = prepare(args.checkpoint)
    info = sync_input(destination)
    print(f'[ready] verified step {info["step"]} model + optimizer + code on all four Macs', flush=True)
    if not args.run and not args.cpu_test:
        print('Preparation complete. No GPU training was launched.', flush=True)
        return
    root = OUTPUTS / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    common.atomic_json(root / 'experiment.json', dict(input=str(destination), input_sha256=info['input_sha256'],
                                                     options=vars(args), source_step=info['step']))
    if args.cpu_test:
        launch(destination, root / 'cpu_correctness', [32, 8, 8, 8], args, cpu=True)
        print(f'All four ranks passed all supported weighted-gradient tests. Log: {root}', flush=True)
        return
    results = []
    try:
        for batches in [[32], *cases]:
            # Recheck before each case, not just before lengthy checkpoint transfers.
            preflight(gpu=True)
            result = launch(destination, root / ('batch_' + '_'.join(map(str, batches))), batches, args)
            results.append(result)
            report(root, results)
    except BaseException as exc:
        common.atomic_json(root / 'failure.json', dict(error=str(exc), completed_cases=len(results)))
        report(root, results)
        raise
    print((root / 'summary.md').read_text(), flush=True)
    print(f'Reports: {root}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
