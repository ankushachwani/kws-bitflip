"""Unit tests for the bit-flip primitive.

A flip must change exactly one bit of a weight's b-bit two's-complement encoding and leave the value inside
the b-bit signed range. Run with:  python3 tests/test_bitflip.py   (or pytest).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor'))
from bitflip import flip_value, _flip_deltas, load_model


def _popcount_lowbits(x, b):
    return bin(x & ((1 << b) - 1)).count('1')


def test_flip_changes_exactly_one_bit():
    for b in (4, 8):
        lo, hi = -(1 << (b - 1)), (1 << (b - 1)) - 1
        for v in range(lo, hi + 1):
            for bit in range(b):
                v2 = flip_value(v, bit, b)
                # exactly one bit differs in the b-bit field
                xor = (v & ((1 << b) - 1)) ^ (v2 & ((1 << b) - 1))
                assert bin(xor).count('1') == 1, (b, v, bit, v2)
                assert xor == (1 << bit), (b, v, bit, v2)
                # value stays in the b-bit signed range
                assert lo <= v2 <= hi, (b, v, bit, v2)
    print('ok: flip changes exactly one bit and stays in range')


def test_flip_is_involution():
    for b in (4, 8):
        lo, hi = -(1 << (b - 1)), (1 << (b - 1)) - 1
        for v in range(lo, hi + 1):
            for bit in range(b):
                assert flip_value(flip_value(v, bit, b), bit, b) == v
    print('ok: flipping the same bit twice restores the value')


def test_flip_deltas_match():
    rng = np.random.RandomState(0)
    for b in (4, 8):
        lo, hi = -(1 << (b - 1)), (1 << (b - 1)) - 1
        W = rng.randint(lo, hi + 1, size=(5, 7)).astype(np.int8)
        d = _flip_deltas(W, b)
        for bit in range(b):
            for idx in np.ndindex(W.shape):
                assert W[idx] + d[bit][idx] == flip_value(int(W[idx]), bit, b)
    print('ok: _flip_deltas agrees with flip_value')


def test_on_real_weights_in_range():
    for variant, b in (('int8', 8), ('w4a8', 4), ('w4a4', 4)):
        m = load_model('s', variant)
        lo, hi = -(1 << (b - 1)), (1 << (b - 1)) - 1
        for i in m['wops']:
            W = m['qops'][i]['Wq']
            assert W.dtype == np.int8
            Wf = W.reshape(-1)
            before = int(Wf[0]); raw_before = int(Wf[0]) & 0xFF
            Wf[0] = flip_value(before, 1, b)
            # the stored int8 byte differs in exactly one of its low b bits
            xor = (raw_before ^ (int(Wf[0]) & 0xFF)) & ((1 << b) - 1)
            assert bin(xor).count('1') == 1
            assert lo <= int(Wf[0]) <= hi
        print(f'ok: real {variant} weights, flip keeps values in [{lo},{hi}]')


if __name__ == '__main__':
    test_flip_changes_exactly_one_bit()
    test_flip_is_involution()
    test_flip_deltas_match()
    test_on_real_weights_in_range()
    print('ALL TESTS PASSED')
