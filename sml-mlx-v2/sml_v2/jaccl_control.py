"""Bounded, sequenced host control messages; gradients still use JACCL reductions."""

import hashlib
import json
import os
import time

import numpy as np


WORDS = 4096
HEADER = 16
MAGIC = 0x534D4C43
VERSION = 1
CAPACITY = WORDS - HEADER
MAX_RING_CONTROL_RANKS = 64
GATHER_PROBE_SIZES = (1, 17, WORDS, 262147)


class ControlError(RuntimeError):
    pass


def ring_control_slots(frame, rank, world):
    """One writer per int32 word: adding zeros preserves even signed hash bits."""
    frame = np.asarray(frame)
    if frame.shape != (WORDS,) or frame.dtype != np.int32:
        raise ControlError('Ring control requires a fixed-size int32 frame')
    if not 1 <= world <= MAX_RING_CONTROL_RANKS or not 0 <= rank < world:
        raise ControlError('Invalid or excessive ring control group size/rank')
    slots = np.zeros((world, WORDS), dtype=np.int32)
    slots[rank] = frame
    return slots.reshape(-1)


def encode_frame(value, sequence, label, rank):
    body = json.dumps(dict(sequence=sequence, label=label, rank=rank, value=value),
                      allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(body) > CAPACITY:
        raise ControlError(f'Control payload exceeds {CAPACITY} bytes: {label}')
    frame = np.zeros(WORDS, dtype=np.int32)
    frame[:8] = [MAGIC, VERSION, sequence, rank, len(body), 0, 0, 0]
    frame[8:16] = np.frombuffer(hashlib.sha256(body).digest(), dtype='<i4')
    frame[HEADER:HEADER + len(body)] = np.frombuffer(body, dtype=np.uint8)
    return frame


def decode_frame(frame, sequence, label, rank):
    frame = np.asarray(frame)
    if frame.shape != (WORDS,) or frame.dtype != np.int32:
        raise ControlError(f'Invalid control frame shape/dtype at {label}')
    if frame[:4].tolist() != [MAGIC, VERSION, sequence, rank]:
        raise ControlError(f'Control sequence/header mismatch at {label}, seq={sequence}, '
                           f'peer={rank}: {frame[:5].tolist()}')
    if np.any(frame[5:8] != 0):
        raise ControlError(f'Corrupt reserved control header at {label}, peer={rank}')
    size = int(frame[4])
    if not 0 < size <= CAPACITY:
        raise ControlError(f'Invalid bounded control length {size} at {label}, peer={rank}')
    values = frame[HEADER:HEADER + size]
    if np.any(values < 0) or np.any(values > 255) or np.any(frame[HEADER + size:] != 0):
        raise ControlError(f'Corrupt control payload at {label}, peer={rank}')
    body = values.astype(np.uint8).tobytes()
    if not np.array_equal(frame[8:16], np.frombuffer(hashlib.sha256(body).digest(), dtype='<i4')):
        raise ControlError(f'Control checksum mismatch at {label}, peer={rank}')
    try:
        message = json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise ControlError(f'Invalid control JSON at {label}, peer={rank}') from exc
    if (message.get('sequence'), message.get('label'), message.get('rank')) != (sequence, label, rank):
        raise ControlError(f'Control operation mismatch at {label}, peer={rank}')
    return message['value']


def gather_probe_values(rank, size, round_index, dtype):
    """Exactly representable, rank-distinct values across both directional halves."""
    return ((np.arange(size, dtype=np.int32) % 1021) - 510 +
            rank * 2048 + round_index * 131072).astype(dtype)


def verify_native_all_gather(mx, group):
    """Run only at a future job's startup, before data/model loading.

    Never trust a version number alone: test odd tails, large transfers, both
    control and floating dtypes, and successive inputs to detect stale regions.
    The known-good all_sum shares each verdict so every rank fails together.
    """
    rank, world = group.rank(), group.size()
    with mx.stream(mx.cpu):
        for dtype in (np.int32, np.float32):
            for size in GATHER_PROBE_SIZES:
                for round_index in range(2):
                    local = gather_probe_values(rank, size, round_index, dtype)
                    gathered = mx.distributed.all_gather(mx.array(local), group=group, stream=mx.cpu)
                    mx.eval(gathered)
                    actual = np.asarray(gathered)
                    expected = np.concatenate([gather_probe_values(r, size, round_index, dtype)
                                               for r in range(world)])
                    passed = actual.dtype == expected.dtype and np.array_equal(actual, expected)
                    failures = mx.distributed.all_sum(mx.array([int(not passed)], dtype=mx.int32),
                                                      group=group, stream=mx.cpu)
                    mx.eval(failures)
                    if int(failures.item()):
                        raise ControlError(
                            f'Native all_gather validation failed: dtype={np.dtype(dtype).name}, '
                            f'size={size}, round={round_index}. Training has not started. '
                            'Use --ring-control all-sum or an MLX build containing PR #4443 '
                            'on every Mac; do not upgrade a running training environment.')
    print(f'[all-gather-ok] rank={rank} world={world} int32/float32, odd/large sizes, 2 rounds', flush=True)


class ControlChannel:
    def __init__(self, mx, group, ring_mode='all-sum'):
        if ring_mode not in ('all-sum', 'native'):
            raise ValueError(f'Unknown ring control mode: {ring_mode}')
        self.mx, self.group = mx, group
        self.rank, self.world = group.rank(), group.size()
        self.stream = mx.default_stream(mx.cpu)
        self.sequence = 0
        self.ring = os.environ.get('MLX_JACCL_RING') == '1'
        self.ring_mode = ring_mode
        if self.ring and ring_mode == 'native':
            verify_native_all_gather(mx, group)

    def exchange(self, value, label):
        sequence = self.sequence
        # All control operations have the SAME wire dtype and allocation size.
        # A peer-provided length must never determine an MLX allocation.
        try:
            envelope = {'value': value}
            frame = encode_frame(envelope, sequence, label, self.rank)
        except (ValueError, TypeError, OverflowError) as exc:
            frame = encode_frame({'error': str(exc)[:1000]}, sequence, label, self.rank)
        except ControlError as exc:
            frame = encode_frame({'error': str(exc)[:1000]}, sequence, label, self.rank)
        with self.mx.stream(self.stream):
            # PR #4443 fixes native ring gathering upstream. Keep the workaround
            # by default for installed 0.32.2; native mode must pass the probe.
            # Gradient reductions are unchanged.
            if self.ring and self.ring_mode == 'all-sum':
                slots = ring_control_slots(frame, self.rank, self.world)
                result = self.mx.distributed.all_sum(self.mx.array(slots), group=self.group, stream=self.stream)
            else:
                result = self.mx.distributed.all_gather(self.mx.array(frame), group=self.group, stream=self.stream)
            self.mx.eval(result)
            frames = np.array(result, copy=True).reshape(self.world, WORDS)
        values = [decode_frame(frames[r], sequence, label, r) for r in range(self.world)]
        self.sequence += 1
        for r, payload in enumerate(values):
            if 'error' in payload:
                raise ControlError(f'Control encoding failed on rank {r} at {label}: {payload["error"]}')
        return [payload['value'] for payload in values]

    def broadcast(self, value, label):
        values = self.exchange(value if self.rank == 0 else None, label)
        if any(v is not None for v in values[1:]):
            raise ControlError(f'Non-root broadcast contribution at {label}')
        return values[0]


def stress_test(mx, group, mode, rounds):
    """CPU/RDMA only: no model, optimizer, dataset, or checkpoint writes."""
    rank, world = group.rank(), group.size()
    control = ControlChannel(mx, group)
    started = time.monotonic()
    sequence = 0

    def broadcast(value, label):
        nonlocal sequence
        if mode == 'framed':
            return control.broadcast(value, label)
        with mx.stream(mx.cpu):
            payload = json.dumps(value).encode() if rank == 0 else b''
            size = mx.distributed.all_sum(mx.array([len(payload)], dtype=mx.int64), group=group, stream=mx.cpu)
            mx.eval(size)
            length = int(size.item())
            if not 0 < length <= CAPACITY:
                raise ControlError(f'LEGACY invalid length={length} rank={rank} round={sequence} label={label}')
            data = mx.array(list(payload), dtype=mx.uint8) if rank == 0 else mx.zeros((length,), dtype=mx.uint8)
            data = mx.distributed.all_sum(data, group=group, stream=mx.cpu)
            mx.eval(data)
            return json.loads(bytes(data.tolist()))

    def gather(value, label):
        if mode == 'framed':
            return control.exchange(value, label)
        with mx.stream(mx.cpu):
            result = mx.distributed.all_gather(mx.array(value, dtype=mx.float32), group=group, stream=mx.cpu)
            mx.eval(result)
            return result.reshape(world, len(value)).tolist()

    for sequence in range(rounds):
        label = str(sequence)
        if rank == sequence % world and sequence % 11 == 0:
            time.sleep(.002)
        flags = gather([0.0], 'stop/' + label)
        if flags != [[0.0]] * world:
            raise ControlError(f'Stop flag mismatch rank={rank} round={sequence}: {flags}')
        if sequence % 10 == 0:
            assert broadcast({'error': None}, 'lr-status/' + label) == {'error': None}
            assert broadcast({'lr': 3e-5}, 'lr/' + label) == {'lr': 3e-5}
        assert broadcast({'error': None}, 'data-status/' + label) == {'error': None}
        with mx.stream(mx.cpu):
            data = mx.full((2, 2, 104, 256), sequence % 49152 if rank == 0 else 0, dtype=mx.int32)
            result = mx.distributed.all_sum(data, group=group, stream=mx.cpu)
            mx.eval(result)
            if not bool(mx.all(result == sequence % 49152).item()):
                raise ControlError(f'Batch corruption rank={rank} round={sequence}')
            for count in (16384, 262144, 17):
                data = mx.full((count,), rank + 1, dtype=mx.float32)
                result = mx.distributed.all_sum(data, group=group, stream=mx.cpu)
                mx.eval(result)
                if not bool(mx.all(result == sum(range(1, world + 1))).item()):
                    raise ControlError(f'Gradient corruption rank={rank} round={sequence} count={count}')
        values = [1.0, 3.25 + rank/32, 1.25, 10.0, .03125, .5, .125]
        metrics = gather(values, 'metrics/' + label)
        if metrics != [[1.0, 3.25 + r/32, 1.25, 10.0, .03125, .5, .125] for r in range(world)]:
            raise ControlError(f'Metrics corruption rank={rank} round={sequence}: {metrics}')
        if sequence % 50 == 0:
            assert broadcast([3.5, {'math': {'loss': 1.125}}], 'eval/' + label) == [3.5, {'math': {'loss': 1.125}}]
        if (sequence + 1) % 500 == 0:
            print(f'[comms-check] mode={mode} rank={rank} rounds={sequence+1}/{rounds}', flush=True)
    print(f'[comms-ok] mode={mode} rank={rank} rounds={rounds} cpu_only=True', flush=True)
    return dict(status='comms_checked', mode=mode, rounds=rounds, seconds=time.monotonic()-started)
