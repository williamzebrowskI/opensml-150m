#!/usr/bin/env python3
"""Opt-in mesh/ring experiments. Never rewires, changes routes, or resumes production."""

import argparse
import datetime
import fcntl
import hashlib
import json
import ipaddress
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import uuid

import jaccl_benchmark as bench
from benchmark_gradient_groups import protected_state, assert_comparable as compare_gradients

mesh = bench.mesh
PRODUCTION = bench.ROOT / 'train/checkpoints/dolma_cosmo_152m_prose_continuation_v1'
POINTER = bench.INPUTS / 'ring_experiment.json'
RING_EDGES = (mesh.EDGES[0], mesh.EDGES[3], mesh.EDGES[5], mesh.EDGES[2])
BATCHES = [32, 24, 24, 24]
RING_MAP = bench.ROOT / 'cluster/jaccl/ring_benchmark_map.json'


def port_inventory(state):
    mapping = dict(re.findall(r'Hardware Port: ([^\n]+)\nDevice: (\S+)', state['hardware_ports']))
    return {mapping.get('Thunderbolt ' + str(bus.get('receptacle_1_tag', {}).get('receptacle_id_key'))):
            (bus['domain_uuid_key'], {p['domain_uuid_key'] for p in bus.get('_items', []) if 'domain_uuid_key' in p})
            for bus in state['thunderbolt']['SPThunderboltDataType'] if 'domain_uuid_key' in bus}


def discover_ring(states):
    ports = [port_inventory(s) for s in states]
    owners = {}
    for rank, nodes in enumerate(ports):
        for iface, (uuid, _) in nodes.items():
            if iface is not None:
                mesh.require(uuid not in owners, 'Duplicate Thunderbolt domain identity')
                owners[uuid] = (rank, iface)
    cables = []
    for rank, nodes in enumerate(ports):
        for iface, (uuid, peers) in nodes.items():
            for peer in peers:
                if peer not in owners:
                    continue  # A display is not an inter-Mac cable.
                other, remote = owners[peer]
                mesh.require(other != rank and ports[other][remote][1] == {uuid} and peers == {peer},
                             'Ambiguous or nonreciprocal Thunderbolt connection')
                mesh.require(iface in ('en3', 'en4', 'en5') and remote in ('en3', 'en4', 'en5'),
                             'Unpinned Thunderbolt port; verify hardware before using it')
                if rank < other:
                    cables.append([rank, iface, other, remote])
    adjacency = {r: set() for r in range(4)}
    for a, _, b, _ in cables:
        adjacency[a].add(b)
        adjacency[b].add(a)
    mesh.require(len(cables) == 4 and all(len(v) == 2 for v in adjacency.values()),
                 'Need exactly four reciprocal cables and two different neighbors per Mac')
    order = [0, min(adjacency[0])]
    while len(order) < 4:
        following = adjacency[order[-1]] - {order[-2]}
        node = next(iter(following))
        mesh.require(node not in order, 'Ring does not connect all four Macs')
        order.append(node)
    mesh.require(0 in adjacency[order[-1]], 'Ring is not closed')
    return dict(edges=sorted(cables), physical_order=order)


def read_ring_map():
    if not RING_MAP.exists():
        return None
    value = json.loads(RING_MAP.read_text())
    order = value['physical_order']
    mesh.require(sorted(order) == list(range(4)) and order[0] == 0, 'Invalid ring rank order')
    mesh.require(len(value['edges']) == 4, 'Invalid ring cable map')
    return value


def edges(topology, mapping=None):
    if topology not in ('mesh', 'ring'):
        raise ValueError('Unknown physical topology')
    return mesh.EDGES if topology == 'mesh' else (mapping['edges'] if mapping else RING_EDGES)


def hostfile(topology, mapping=None):
    order = mapping['physical_order'] if mapping else list(range(4))
    matrix = [[None] * 4 for _ in range(4)]
    for a, ai, b, bi in edges(topology, mapping):
        a, b = order.index(a), order.index(b)
        matrix[a][b], matrix[b][a] = 'rdma_' + ai, 'rdma_' + bi
    return dict(backend='jaccl-ring' if topology == 'ring' else 'jaccl', hosts=[
        dict(ssh='127.0.0.1' if rank == 0 else mesh.ssh_alias(order[rank]),
             ips=[mesh.HOSTS[0][2]] if rank == 0 else [], rdma=matrix[rank])
        for rank in range(4)])


def inspect_host(rank, mac4_via_mac2=False, control_transport='wifi'):
    mesh.require(control_transport in ('wifi', 'ring-thunderbolt'), 'Unsupported ring inventory transport')
    mesh.require(not mac4_via_mac2 or control_transport == 'wifi', 'Setup relay is Wi-Fi-only')
    # Send a self-contained read-only collector: no dependency on remote project scripts.
    source = (bench.ROOT / 'scripts/configure_rdma_mesh.py').read_text()
    common = (bench.ROOT / 'train/jaccl_common.py').read_text()
    destinations = tuple(h[2] for i, h in enumerate(mesh.HOSTS) if i != rank) if control_transport == 'wifi' else ()
    code = f'''import json, os, importlib.metadata as metadata
from pathlib import Path
m = {{"__name__": "ring_read_only"}}
c = {{"__name__": "ring_common"}}
exec(compile({source!r}, "mesh_read_only.py", "exec"), m)
exec(compile({common!r}, "common_read_only.py", "exec"), c)
run = m["run"]
launcher = metadata.distribution("mlx").locate_file("mlx/_distributed_utils/launch.py").read_text()
state = dict(rank={rank}, interfaces=m["interfaces"](run(["/sbin/ifconfig"]).stdout),
    management={{ip: m["route"](ip) for ip in {destinations!r}}},
    ssh_connection=os.environ.get("SSH_CONNECTION", ""),
    rdma=run(["/usr/bin/rdma_ctl", "status"]).stdout.strip(),
    thunderbolt=json.loads(run(["/usr/sbin/system_profiler", "SPThunderboltDataType", "-json", "-timeout", "30"], timeout=45).stdout),
    hardware_ports=run(["/usr/sbin/networksetup", "-listallhardwareports"]).stdout,
    packages={{p: metadata.version(p) for p in c["PACKAGES"]}},
    ring_supported="jaccl-ring" in launcher and "MLX_JACCL_RING=1" in launcher,
    launcher_sha256=c["sha256"](metadata.distribution("mlx").locate_file("mlx/_distributed_utils/launch.py")),
    busy=c["training_processes"]())
print(json.dumps(state))
'''
    if mac4_via_mac2 and rank == 3:
        from repair_ring_network import transport
        command = transport(rank, [bench.PYTHON, '-c', code], relay=True)
    else:
        command = mesh.transport(rank, [bench.PYTHON, '-c', code], control_transport=control_transport)
    result = mesh.run(command, timeout=90,
                      label=f'mac-{rank + 1} read-only ring inventory ' +
                            ('via mac-2 setup relay' if mac4_via_mac2 and rank == 3 else f'over {control_transport}'))
    return json.loads(result.stdout)


def validate_states(states, topology, require_idle=True, mapping=None, allow_bridges=False, require_wifi=True):
    mesh.require(len(states) == 4, 'Need all four inventories')
    ports = []
    for rank, state in enumerate(states):
        devs = state['interfaces']
        mesh.require(state['rank'] == rank and devs['en0']['mac'] == mesh.HOSTS[rank][3],
                     f'mac-{rank + 1}: physical identity mismatch; do not move cables by name alone')
        if require_wifi:
            mesh.require([mesh.HOSTS[rank][2], '255.255.255.0'] in devs['en1']['ipv4'],
                         f'mac-{rank + 1}: pinned Wi-Fi address changed')
            wanted = {h[2] for i, h in enumerate(mesh.HOSTS) if i != rank}
            mesh.require(set(state['management']) == wanted and all(
                route.get('interface') == 'en1' for route in state['management'].values()),
                'This test requires Wi-Fi management; no automatic fallback')
        mesh.require(state['rdma'] == 'enabled' and state['ring_supported'],
                     f'mac-{rank + 1}: RDMA or installed jaccl-ring support missing')
        mesh.require(state['packages'] == states[0]['packages'] and
                     state['launcher_sha256'] == states[0]['launcher_sha256'], 'MLX/dependency versions differ')
        if require_idle:
            mesh.require(not state['busy'], f'mac-{rank + 1}: stop production yourself before testing: {state["busy"]}')
        needed = {ai if rank == a else bi for a, ai, b, bi in edges(topology, mapping) if rank in (a, b)}
        for i, iface in enumerate(('en3', 'en4', 'en5')):
            dev = devs[iface]
            mesh.require(dev['mac'] == mesh.PORT_MACS[rank][i], 'Thunderbolt port identity changed')
            if iface in needed:
                mesh.require(dev['up'] and dev['active'], f'mac-{rank + 1}: {iface} must be connected')
            else:
                mesh.require(not dev['active'], f'mac-{rank + 1}: remove the diagonal cable on {iface} for a physical ring')
        for name, dev in devs.items():
            if set(dev['members']) & {'en3', 'en4', 'en5'}:
                mesh.require(allow_bridges or not dev['up'], f'mac-{rank + 1}: active bridge {name}; repair before testing')
        hardware_mapping = dict(re.findall(r'Hardware Port: ([^\n]+)\nDevice: (\S+)', state['hardware_ports']))
        nodes = {}
        for item in state['thunderbolt']['SPThunderboltDataType']:
            iface = hardware_mapping.get('Thunderbolt ' + str(item.get('receptacle_1_tag', {}).get('receptacle_id_key')))
            if iface in needed:
                nodes[iface] = (item['domain_uuid_key'],
                    {p['domain_uuid_key'] for p in item.get('_items', []) if 'domain_uuid_key' in p})
        ports.append(nodes)
    for a, ai, b, bi in edges(topology, mapping):
        mesh.require(ai in ports[a] and bi in ports[b], 'Missing Thunderbolt port inventory')
        mesh.require(ports[a][ai][1] == {ports[b][bi][0]} and ports[b][bi][1] == {ports[a][ai][0]},
                     f'Wrong cable: mac-{a + 1} {ai} must connect to mac-{b + 1} {bi}')


def require_ipv4(states, topology, mapping=None):
    missing = [f'mac-{r + 1} {iface}' for r, state in enumerate(states)
               for iface in {ai if r == a else bi for a, ai, b, bi in edges(topology, mapping) if r in (a, b)}
               if not state['interfaces'][iface]['ipv4']]
    mesh.require(not missing, 'RDMA requires IPv4 on each selected port; missing: ' + ', '.join(missing)
                 + '. For ring wiring run scripts/configure_ring_ipv4.py --apply first.')
    for a, ai, b, bi in edges(topology, mapping):
        left, right = (states[r]['interfaces'][i]['ipv4'][0] for r, i in ((a, ai), (b, bi)))
        ln, rn = (ipaddress.IPv4Network(f'{ip}/{mask}', strict=False) for ip, mask in (left, right))
        mesh.require(ln == rn and left[0] != right[0],
                     f'Ring peer IPv4 mismatch: mac-{a + 1} {ai} {left[0]} <-> mac-{b + 1} {bi} {right[0]}. '
                     'Run configure_ring_ipv4.py --apply --align-peers; do not launch collectives yet.')


def preflight(topology, require_idle=True, mapping=None):
    states = [inspect_host(rank) for rank in range(4)]
    validate_states(states, topology, require_idle, mapping)
    if mapping:
        mesh.require(discover_ring(states) == mapping, 'Cables changed since remapping; run --remap-ring again')
    require_ipv4(states, topology, mapping)
    print(f'[preflight] all four physical identities, {topology} cables, RDMA and Wi-Fi management verified', flush=True)
    return states


def assert_comparable(first, other):
    compare_gradients(first, other)
    for field in ('grad_accum', 'warmup', 'gradient_bucket_mib', 'control_transport', 'data_layout'):
        mesh.require(first[field] == other[field], f'Comparison changed {field}')
    mesh.require(len(other['input_sha256']) == 4 and all(other['input_sha256']), 'Missing cached-input hashes')


def report(root, results):
    bench.common.atomic_json(root / 'results.json', results)
    lines = ['# Four-Mac JACCL Ring Test', '',
             '32/24/24/24; accumulation 4; sequence 256; BF16 compute, FP32 gradients/optimizer.',
             'Identical pinned checkpoint and cached fresh-stream inputs; fixed checkpoint LR.',
             'Wi-Fi SSH/coordinator for BOTH topologies; tensor collectives use Thunderbolt RDMA.',
             'Ring uses an isolated host-control compatibility module (int32 rank-slot all_sum); mesh uses native all_gather.',
             'All other pinned input files are unchanged. This compares those working configurations, not identical collective implementations.',
             'Warmup, dataset fetching, evaluation and checkpoint I/O excluded. Not production continuation.', '',
             '| Topology | Trial tokens/s | Mean tokens/s | Change vs mesh |',
             '| --- | --- | ---: | ---: |']
    base = [r['tokens_per_second'] for r in results if r['topology'] == 'mesh']
    baseline = statistics.mean(base) if base else None
    for topology in ('mesh', 'ring'):
        values = [r['tokens_per_second'] for r in results if r['topology'] == topology]
        if values:
            mean = statistics.mean(values)
            change = f'{(mean / baseline - 1) * 100:+.2f}%' if baseline else 'pending'
            lines.append(f'| {topology} | {", ".join(f"{v:,.0f}" for v in values)} | {mean:,.0f} | {change} |')
    lines += ['', 'Two sequential trials per topology. Rewire to mesh and repeat if the difference is small;',
              'thermal drift and run-to-run noise are not eliminated. A short pass is not overnight reliability proof.',
              'CPU framed-collective stress, independent weighted-gradient reference, real-gradient repeat checks,',
              'finite loss/gradients, memory bounds and within-trial model/optimizer replica hashes must pass.',
              'Production files remain unchanged. Test-trained weights are not saved or used for production.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')


def load_experiment(path):
    path = Path(path).resolve()
    mesh.require(path.parent == bench.OUTPUTS.resolve() and path.name.startswith('ring_'),
                 'Experiment must be in the isolated ring benchmark directory')
    value = json.loads((path / 'experiment.json').read_text())
    destination = Path(value['input']).resolve()
    mesh.require(destination.parent == bench.INPUTS.resolve(), 'Invalid pinned input directory')
    mesh.require(bench.common.verify_input(destination) == value['input_info'], 'Pinned input or packages changed')
    return value, destination


def prepare_ring_control(source):
    """Derive an immutable input, changing ONLY the audited host-control module."""
    original = bench.common.verify_input(source)
    manifest = json.loads((source / 'input.json').read_text())
    name = 'code/train/jaccl_control.py'
    replacement = bench.ROOT / 'train/jaccl_control.py'
    record = bench.common.file_record(replacement)
    mesh.require(name in manifest['files'], 'Pinned control module is missing')
    identity = hashlib.sha256(json.dumps([original['input_sha256'], record], sort_keys=True).encode()).hexdigest()[:12]
    destination = bench.INPUTS / f'step_{manifest["step"]:07d}_ringctl_{identity}'
    updated = dict(manifest, files={**manifest['files'], name: record})
    if not destination.exists():
        staging = bench.INPUTS / ('.ring-control-' + uuid.uuid4().hex)
        try:
            # Immutable checkpoint/code files may share storage. Never write into
            # a linked file: detach the replacement and atomically replace JSON.
            shutil.copytree(source, staging, copy_function=os.link)
            (staging / name).unlink()
            shutil.copy2(replacement, staging / name)
            bench.common.atomic_json(staging / 'input.json', updated)
            bench.common.verify_input(staging)
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    derived = bench.common.verify_input(destination)
    mesh.require(json.loads((destination / 'input.json').read_text()) == updated,
                 'Unexpected changes in compatibility snapshot')
    mesh.require(bench.common.verify_input(source) == original, 'Original snapshot changed')
    return destination, dict(implementation='rank_slots_int32_all_sum_v1',
        original_input=str(source), original_input_sha256=original['input_sha256'],
        derived_input=str(destination), derived_input_sha256=derived['input_sha256'],
        changed_file=name, before=manifest['files'][name], after=record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--inspect', choices=('mesh', 'ring'), help='Read-only inventory; no model or collectives')
    modes.add_argument('--baseline', action='store_true', help='Run stopped-mesh baseline BEFORE disconnecting cables')
    modes.add_argument('--ring', action='store_true', help='Run physical ring against the pinned baseline')
    modes.add_argument('--check-ring', action='store_true', help='CPU/RDMA checks only, after baseline and rewiring')
    modes.add_argument('--remap-ring', action='store_true', help='Discover cables and save benchmark-only map; no network changes or tests')
    parser.add_argument('--checkpoint', default=str(PRODUCTION / 'latest.json'))
    parser.add_argument('--experiment', help='Existing experiment directory; defaults to last successful baseline')
    args = parser.parse_args()
    if args.remap_ring:
        bench.INPUTS.mkdir(parents=True, exist_ok=True)
        with (bench.INPUTS / '.launcher.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            states = [inspect_host(r) for r in range(4)]
            mapping = discover_ring(states)
            validate_states(states, 'ring', mapping=mapping, allow_bridges=True)
            audit = bench.ROOT / 'diagnostics' / ('ring_remap_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            audit.mkdir(parents=True)
            bench.common.atomic_json(audit / 'inventory.json', states)
            bench.common.atomic_json(audit / 'previous_map.json', read_ring_map())
            bench.common.atomic_json(RING_MAP, mapping)
            print('[remap] ' + ' -> '.join(f'mac-{r + 1}' for r in mapping['physical_order'] + [0]))
            for a, ai, b, bi in mapping['edges']:
                print(f'  mac-{a + 1} {ai} <-> mac-{b + 1} {bi}')
            print(f'[saved] Benchmark-only map: {RING_MAP}\nAudit: {audit}')
            blockers = [f'mac-{r + 1}: {name}' for r, s in enumerate(states)
                        for name, dev in s['interfaces'].items()
                        if dev.get('up') and set(dev['members']) & {'en3', 'en4', 'en5'}]
            if blockers:
                print('[blocked] Active bridges: ' + ', '.join(blockers))
                print('Before testing, run: bash ' + str(bench.ROOT / 'scripts/isolate_thunderbolt_bridges.sh') + ' --apply')
            print('No network settings or production files changed. No benchmark or training started.')
        return
    mapping = read_ring_map() if not args.baseline and args.inspect != 'mesh' else None
    if args.inspect:
        states = preflight(args.inspect, require_idle=False, **({'mapping': mapping} if mapping else {}))
        for state in states:
            print(f'mac-{state["rank"] + 1}: MLX={state["packages"]["mlx"]}; active jobs={len(state["busy"])}')
        return
    bench.INPUTS.mkdir(parents=True, exist_ok=True)
    with (bench.INPUTS / '.launcher.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        topology = 'mesh' if args.baseline else 'ring'
        inventory = preflight(topology, **({'mapping': mapping} if mapping else {}))
        if args.baseline:
            mesh.require(not args.experiment, '--baseline creates a new experiment; do not pass --experiment')
            source, _ = bench.resolve_source(args.checkpoint)
            production = source.parent
            before = protected_state(production)
            destination = bench.prepare(args.checkpoint)
            info = bench.common.verify_input(destination)
            root = bench.OUTPUTS / ('ring_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            root.mkdir(parents=True)
            experiment = dict(input=str(destination), input_info=info, production=str(production), before=before)
            bench.common.atomic_json(root / 'experiment.json', experiment)
            results = []
        else:
            root = Path(args.experiment or json.loads(POINTER.read_text())['experiment']).resolve()
            experiment, destination = load_experiment(root)
            production, before = Path(experiment['production']), experiment['before']
            results = json.loads((root / 'results.json').read_text())
            mesh.require(len([r for r in results if r['topology'] == 'mesh']) == 2,
                         'Complete both mesh baseline trials before rewiring')
            mesh.require(args.check_ring or not any(r['topology'] == 'ring' for r in results),
                         'This experiment already has ring results; create a new baseline for a repeat')
        mesh.require(protected_state(production) == before, 'Production changed since the baseline; create a fresh baseline')
        run = root / (topology + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        run.mkdir()
        bench.common.atomic_json(run / 'inventory.json', inventory)
        compatibility = None
        if topology == 'ring':
            destination, compatibility = prepare_ring_control(destination)
            bench.common.atomic_json(run / 'control_compatibility.json', compatibility)
            print('[ring-control] using checked int32 rank-slot reductions; native ring all_gather is not used', flush=True)
        hosts = run / 'hosts.json'
        bench.common.atomic_json(hosts, hostfile(topology, mapping))
        options = argparse.Namespace(control_transport='wifi', collective_hostfile=hosts,
            collective_backend='jaccl-ring' if topology == 'ring' else 'jaccl',
            steps=60, warmup=10, timeout=900, cpu_timeout=240, comms_rounds=100,
            grad_accum=4, gradient_bucket_mib=32, resident_batches=8, no_save_trial=True, root_data_stream=True)
        options.physical_order = mapping['physical_order'] if mapping else list(range(4))
        try:
            bench.sync_input(destination, bench.INPUTS / 'prepared_ring.json', 'wifi')
            preflight(topology, mapping=mapping)
            bench.launch(destination, run / 'cpu_checks', BATCHES, options, cpu=True)
            if not args.check_ring:
                for trial in (1, 2):
                    preflight(topology, mapping=mapping)
                    value = bench.launch(destination, run / f'trial_{trial}', BATCHES, options)
                    value['control_compatibility'] = compatibility
                    assert_comparable(results[0] if results else value, value)
                    results.append(value)
                    report(root, results)
            preflight(topology, mapping=mapping)
        except BaseException as exc:
            bench.common.atomic_json(run / 'failure.json', dict(error=str(exc)))
            raise
        finally:
            after = protected_state(production)
            bench.common.atomic_json(run / 'production_guard.json', dict(before=before, after=after, unchanged=before == after))
            mesh.require(before == after, 'Production files changed during testing; inspect before resuming')
        if args.baseline:
            bench.common.atomic_json(POINTER, dict(experiment=str(root)))
            print('[next] Baseline complete. Remove ONLY Mac-1 <-> Mac-3 and Mac-2 <-> Mac-4 cables. Then run --ring.')
        print(f'[done] Tests stopped; production not started. Reports: {root / "report.md"}', flush=True)
        if (root / 'report.md').exists():
            print((root / 'report.md').read_text())


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        raise SystemExit(f'ERROR: {exc}. No automatic restart or network repair was attempted.')
