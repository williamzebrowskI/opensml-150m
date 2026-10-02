"""CPU-only simulated collectives; never initialize MLX or contact the rack."""

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sml_v2.jaccl_control import (
    ControlChannel, ControlError, WORDS, encode_frame, gather_probe_values,
    verify_native_all_gather,
)


class FakeMx:
    cpu = 'cpu'
    int32 = np.int32
    array = staticmethod(np.array)
    eval = staticmethod(lambda *args: None)
    stream = staticmethod(lambda *args: nullcontext())
    default_stream = staticmethod(lambda *args: 'cpu')

    def __init__(self, world=4, rank=0, corruption=None, peer_failure=False):
        self.world, self.rank = world, rank
        self.corruption, self.peer_failure = corruption, peer_failure
        self.gathers, self.sums = 0, 0
        self.distributed = SimpleNamespace(all_gather=self.gather, all_sum=self.sum)

    def gather(self, local, **kwargs):
        self.gathers += 1
        # Each simulated peer has distinct values, including the second half.
        result = np.concatenate([local + (r - self.rank) * 2048 for r in range(self.world)])
        if self.corruption == 'upper-half':
            for r in range(self.world):
                result[r * len(local) + len(local) // 2:(r + 1) * len(local)] = 0
        elif self.corruption == 'tail':
            result[-1] += 1
        elif self.corruption == 'rank-order':
            result = result.reshape(self.world, -1)[::-1].reshape(-1)
        elif self.corruption == 'stale' and self.gathers % 2 == 0:
            result -= 131072
        elif self.corruption == 'dtype':
            result = result.astype(np.float64)
        elif self.corruption == 'shape':
            result = result[:-1]
        return result

    def sum(self, local, **kwargs):
        self.sums += 1
        return local + int(self.peer_failure)


def group(rank=0, world=4):
    return SimpleNamespace(rank=lambda: rank, size=lambda: world)


class NativeGatherTests(unittest.TestCase):
    def test_correct_rank_distinct_data_passes_on_every_rank(self):
        for rank in range(4):
            mx = FakeMx(rank=rank)
            verify_native_all_gather(mx, group(rank))
            self.assertEqual(mx.gathers, 16)
            self.assertEqual(mx.sums, 16)

    def test_corruption_blocks_training(self):
        for corruption in ('upper-half', 'tail', 'rank-order', 'stale', 'dtype', 'shape'):
            with self.subTest(corruption=corruption):
                with self.assertRaisesRegex(ControlError, 'Training has not started'):
                    verify_native_all_gather(FakeMx(corruption=corruption), group())

    def test_other_ranks_failure_is_not_ignored(self):
        with self.assertRaises(ControlError):
            verify_native_all_gather(FakeMx(peer_failure=True), group())

    def test_probe_float_values_are_exact(self):
        for rank in range(4):
            for round_index in range(2):
                integers = gather_probe_values(rank, 262147, round_index, np.int32)
                floats = gather_probe_values(rank, 262147, round_index, np.float32)
                np.testing.assert_array_equal(integers, floats.astype(np.int32))

    def channel(self, mode='all-sum', corruption=False):
        mx = FakeMx()
        frames = np.concatenate([encode_frame({'value': r}, 0, 'test', r) for r in range(4)])
        if corruption:
            frames[WORDS + 8] ^= 1
        with patch.dict('os.environ', {'MLX_JACCL_RING': '1'}), \
             patch('sml_v2.jaccl_control.verify_native_all_gather') as probe:
            channel = ControlChannel(mx, group(), ring_mode=mode)
            self.assertEqual(probe.call_count, int(mode == 'native'))
        mx.distributed.all_gather = lambda value, **kw: self.record(mx, 'gathers', frames)
        mx.distributed.all_sum = lambda value, **kw: self.record(mx, 'sums', frames)
        return channel, mx

    @staticmethod
    def record(mx, key, value):
        setattr(mx, key, getattr(mx, key) + 1)
        return value

    def test_default_keeps_workaround(self):
        channel, mx = self.channel()
        self.assertEqual(channel.exchange(0, 'test'), list(range(4)))
        self.assertEqual((mx.gathers, mx.sums), (0, 1))

    def test_native_keeps_frame_protocol(self):
        channel, mx = self.channel('native')
        self.assertEqual(channel.exchange(0, 'test'), list(range(4)))
        self.assertEqual((mx.gathers, mx.sums), (1, 0))
        self.assertEqual(channel.sequence, 1)

    def test_native_still_checks_checksums(self):
        channel, _ = self.channel('native', corruption=True)
        with self.assertRaisesRegex(ControlError, 'checksum mismatch'):
            channel.exchange(0, 'test')

    def test_failed_probe_never_returns_native_channel(self):
        with patch.dict('os.environ', {'MLX_JACCL_RING': '1'}), \
             patch('sml_v2.jaccl_control.verify_native_all_gather', side_effect=ControlError('bad')):
            with self.assertRaises(ControlError):
                ControlChannel(FakeMx(), group(), ring_mode='native')

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            ControlChannel(FakeMx(), group(), ring_mode='typo')


if __name__ == '__main__':
    unittest.main()
