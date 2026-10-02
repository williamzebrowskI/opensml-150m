#!/usr/bin/env python3
"""Verify wired ring SSH and job-scoped coordinator tunnels; never change networking."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import socket
import subprocess

import benchmark_jaccl_ring as ring
from train.jaccl_common import atomic_json
from train.jaccl_ring_transport import MODE, PORT_ENV, ssh_command

mesh = ring.mesh
ROOT = mesh.ROOT
MAPPING = dict(physical_order=[0, 1, 3, 2], edges=[
    [0, 'en4', 2, 'en5'], [0, 'en5', 1, 'en5'],
    [1, 'en4', 3, 'en5'], [2, 'en4', 3, 'en4']])
ADDRESSES = [dict(en5='10.77.0.1', en4='10.77.0.9'),
             dict(en5='10.77.0.2', en4='10.77.0.17'),
             dict(en5='10.77.0.10', en4='10.78.204.1'),
             dict(en5='10.77.0.18', en4='10.78.204.2')]
ROUTES = [{'10.77.0.2': 'en5', '10.77.0.10': 'en4'},
          {'10.77.0.1': 'en5', '10.77.0.18': 'en4'},
          {'10.77.0.9': 'en5'}, {'10.77.0.17': 'en5'}]
SSH_ENDPOINTS = [None, ('10.77.0.1', '10.77.0.2'),
                 ('10.77.0.9', '10.77.0.10'), ('10.77.0.17', '10.77.0.18')]


def check_mapping(mapping):
    mesh.require(mapping == MAPPING,
                 'Wired ring control is pinned to the verified Mac-1/Mac-2/Mac-4/Mac-3 ring; '
                 'cables or mapping changed. Inspect before configuring new control paths.')


def check_ssh_config():
    for physical in range(1, 4):
        raw = mesh.run(['/usr/bin/ssh', '-G', '-F', str(mesh.control_settings(MODE)['config']),
                        mesh.ssh_alias(physical, MODE)]).stdout
        options = dict(line.split(None, 1) for line in raw.splitlines() if ' ' in line)
        expected = dict(hostname=SSH_ENDPOINTS[physical][1], hostkeyalias=mesh.HOSTS[physical][1],
                        user='williamzebrowski', addressfamily='inet', port='22', batchmode='yes',
                        stricthostkeychecking='true', updatehostkeys='false', forwardagent='no',
                        controlmaster='false', controlpersist='no', exitonforwardfailure='yes')
        if physical < 3:
            expected.update(bindaddress=SSH_ENDPOINTS[physical][0])
        expected['proxyjump'] = mesh.ssh_alias(1, MODE) if physical == 3 else 'none'
        mesh.require(all(options.get(k, 'none') == value for k, value in expected.items())
                     and options.get('proxycommand', 'none') == 'none',
                     f'mac-{physical + 1}: unexpected SSH options; refusing a possible non-wired path')
        mesh.require(not any(line.startswith(('localforward ', 'remoteforward ')) for line in raw.splitlines()),
                     'Coordinator forwarding must be job-scoped, not permanently configured')


def validate_connection(physical, connection):
    fields = connection.split()
    expected = SSH_ENDPOINTS[physical]
    mesh.require(len(fields) == 4 and (fields[0], fields[2]) == expected and fields[3] == '22',
                 f'mac-{physical + 1}: SSH did not arrive over the expected Thunderbolt link')


def validate_wired_states(states, mapping):
    check_mapping(mapping)
    ring.validate_states(states, 'ring', mapping=mapping, require_wifi=False)
    mesh.require(ring.discover_ring(states) == mapping, 'Physical ring changed')
    ring.require_ipv4(states, 'ring', mapping)
    for physical, state in enumerate(states):
        for interface, ip in ADDRESSES[physical].items():
            mesh.require([ip, '255.255.255.252'] in state['interfaces'][interface]['ipv4'],
                         f'mac-{physical + 1}: expected {ip}/30 on {interface}; repair IPv4 before launching')
        if physical:
            validate_connection(physical, state.get('ssh_connection', ''))


def validate_routes(physical, routes):
    mesh.require(set(routes) == set(ROUTES[physical]) and all(
        routes[ip].get('interface') == interface for ip, interface in ROUTES[physical].items()),
        f'mac-{physical + 1}: a wired control route changed; refusing Wi-Fi fallback')


def preflight(mapping):
    check_mapping(mapping)
    check_ssh_config()
    states = [ring.inspect_host(r, control_transport=MODE) for r in range(4)]
    validate_wired_states(states, mapping)
    # Check both directions, including the jump host's outbound Mac-4 link.
    source = mesh.SELF.read_text()
    for physical, state in enumerate(states):
        code = ('import json; s={"__name__":"route_audit"}; '
                f'exec({source!r},s); print(json.dumps({{ip:s["route"](ip) for ip in {list(ROUTES[physical])!r}}}))')
        result = mesh.run(mesh.transport(physical, [mesh.PYTHON, '-c', code], control_transport=MODE),
                          timeout=45, label=f'mac-{physical + 1} wired ring route audit')
        state['wired_routes'] = json.loads(result.stdout)
        validate_routes(physical, state['wired_routes'])
    print('[preflight] four identities, ring cables, RDMA and wired SSH verified; '
          'Mac-4 via Mac-2; Wi-Fi not used for control', flush=True)
    return states


def receive_exact(connection, size):
    value = bytearray()
    while len(value) < size:
        part = connection.recv(min(65536, size - len(value)))
        if not part:
            raise RuntimeError('Coordinator probe connection closed early')
        value.extend(part)
    return bytes(value)


def verify_coordinator():
    records = []
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen(4)
        server.settimeout(20)
        port = server.getsockname()[1]
        for physical in range(1, 4):
            nonce = secrets.token_bytes(32)
            payload = nonce * 32768  # 1 MiB through each exact launch-time tunnel.
            digest = hashlib.sha256(payload).digest()
            code = f'''import hashlib,json,os,socket,subprocess
port={port}
rows=[line.split() for line in subprocess.check_output(
    ["/usr/sbin/netstat","-an","-p","tcp"],text=True).splitlines()]
listeners=[r[3] for r in rows if len(r)>5 and r[-1]=="LISTEN" and r[3].endswith("."+str(port))]
assert listeners==["127.0.0.1."+str(port)], ("Unsafe coordinator listener",listeners)
data=bytes.fromhex({nonce.hex()!r})*32768
with socket.create_connection(("127.0.0.1",port),timeout=15) as c:
    c.sendall(data)
    reply=bytearray()
    while len(reply)<32:
        part=c.recv(32-len(reply))
        assert part, "Truncated coordinator response"
        reply.extend(part)
    assert bytes(reply)==hashlib.sha256(data).digest(), "Coordinator checksum differs"
print(json.dumps(dict(physical={physical},bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
    ssh_connection=os.environ.get("SSH_CONNECTION",""),loopback_only=True)))
'''
            arguments = ['-tt', mesh.ssh_alias(physical, MODE), shlex.join([mesh.PYTHON, '-c', code])]
            command = ssh_command(arguments, {PORT_ENV: str(port)}, ROOT)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                with server.accept()[0] as connection:
                    connection.settimeout(15)
                    mesh.require(receive_exact(connection, len(payload)) == payload,
                                 f'mac-{physical + 1}: coordinator payload changed')
                    connection.sendall(digest)
                out, err = process.communicate(timeout=20)
                mesh.require(process.returncode == 0, f'Coordinator tunnel failed: {err}')
                record = json.loads(out.strip())
                validate_connection(physical, record['ssh_connection'])
                mesh.require(record['physical'] == physical and record['sha256'] == digest.hex()
                             and record['loopback_only'], 'Invalid coordinator probe result')
                records.append(record)
                print(f'[control-ok] mac-{physical + 1}: wired SSH and loopback-only coordinator tunnel; '
                      '1 MiB checksum verified', flush=True)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate(timeout=5)
    return records


def main():
    audit = ROOT / 'diagnostics' / ('ring_thunderbolt_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    audit.mkdir(parents=True)
    print(f'Wired ring check; no network changes or training. Audit: {audit}', flush=True)
    states = preflight(ring.read_ring_map())
    atomic_json(audit / 'inventory.json', states)
    records = verify_coordinator()
    atomic_json(audit / 'result.json', dict(status='passed', control_transport=MODE, probes=records,
                                          network_changed=False, training_started=False))
    print('[verified] All inter-Mac control paths use Thunderbolt. No bridges or global IP forwarding enabled.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f'ERROR: {exc}. No network repair or training restart was attempted.')
