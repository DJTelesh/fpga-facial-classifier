#!/usr/bin/env python3
"""
quantize.py  --  float weights -> int8 weights + bit-exact integer reference model

Reads weights/fer3_float.npz (written by train.py), quantizes it, and writes
weights/fer3_int8.npz. Then runs the integer model on the val and test sets and
compares it against the float model.

infer_int() is the golden reference for the RTL. It uses only integer operations the
hardware can do (int8 x int8 multiplies, int32 accumulate, one multiply + arithmetic
right shift to requantize, compare for argmax). If the RTL disagrees with infer_int()
on any input, the RTL is wrong.

Integer pipeline, per image:
    x_q  = pixel - 128                                   int8,  [-128, 127]
    acc1 = W1_q @ x_q + b1_q                             int32
    h_q  = clip((acc1 * M1 + 2^(S1-1)) >> S1, 0, 127)    ReLU + requantize to [0, 127]
    acc2 = W2_q @ h_q + b2_q                             int32
    pred = argmax(acc2)          ties go to the LOWEST index -- the RTL must match this

The (acc * M + 2^(S-1)) >> S step is floor(v + 0.5) rounding (contract section 4),
done with an integer multiply and an arithmetic shift, so there is no divider.
Full definition: arithmetic_contract.md at the repo root.

Usage:
    python3 quantize.py
"""

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FLOAT_WEIGHTS = ROOT / "weights" / "fer3_float.npz"
INT_WEIGHTS = ROOT / "weights" / "fer3_int8.npz"

W_MAX = 127          # weights clamp to +/-127 (contract section 4)
H_MAX = 127          # post-ReLU activations live in [0, 127]
M_BITS = 15          # requantize multiplier is unsigned and fits in 15 bits
B_LIMIT = 2**24      # abs(bias) must stay below this (contract section 4)
INT32_MAX = 2**31 - 1

S_X = 1 / 128        # input scale: train.py uses (pixel - 128) / 128


def round_half_up(v):
    """floor(v + 0.5), NOT np.round (which rounds half to even)."""
    return np.floor(v + 0.5)


def quantize_weights(w):
    """Symmetric per-tensor int8: scale = max|w| / 127."""
    s = np.max(np.abs(w)) / W_MAX
    q = np.clip(round_half_up(w / s), -W_MAX, W_MAX).astype(np.int8)
    return q, s


def multiplier_and_shift(m_real):
    """Approximate m_real as M / 2^S with M < 2^M_BITS, using the largest S that fits."""
    s = 0
    while round_half_up(m_real * 2 ** (s + 1)) < 2**M_BITS:
        s += 1
    return int(round_half_up(m_real * 2**s)), s


def float_hidden(x_uint8, f):
    x = (x_uint8.astype(np.float32) - 128) / 128
    return np.maximum(0, x @ f["w1"].T + f["b1"])


def infer_float(x_uint8, f):
    return (float_hidden(x_uint8, f) @ f["w2"].T + f["b2"]).argmax(1)


def quantize(f, x_calib):
    w1_q, s_w1 = quantize_weights(f["w1"])
    w2_q, s_w2 = quantize_weights(f["w2"])

    # Hidden activation scale: largest post-ReLU value seen on the training set maps
    # to 127. Calibrated on train only, so val/test stay unseen.
    s_h = float_hidden(x_calib, f).max() / H_MAX

    # Biases go in the accumulator, so they use the accumulator's scale.
    b1_q = round_half_up(f["b1"] / (S_X * s_w1)).astype(np.int32)
    b2_q = round_half_up(f["b2"] / (s_h * s_w2)).astype(np.int32)

    m1, shift1 = multiplier_and_shift(S_X * s_w1 / s_h)

    # Limits from arithmetic_contract.md sections 4-6. The accumulator and
    # requantize bit widths are only safe while these hold.
    for name, b in (("b1_q", b1_q), ("b2_q", b2_q)):
        assert np.abs(b).max() < B_LIMIT, f"{name} breaks the 2^24 bias limit"
    assert 2 ** (M_BITS - 1) <= m1 < 2**M_BITS, f"M1={m1} out of range"
    assert 1 <= shift1 <= 40, f"S1={shift1} out of range"

    return {
        "w1_q": w1_q, "b1_q": b1_q, "m1": np.int32(m1), "shift1": np.int32(shift1),
        "w2_q": w2_q, "b2_q": b2_q,
        # scales are for documentation only; infer_int() never touches them
        "s_x": S_X, "s_w1": s_w1, "s_h": s_h, "s_w2": s_w2,
    }


def infer_int(x_uint8, q, return_intermediates=False):
    """Bit-exact integer model. x_uint8 is (N, 2304) raw pixels."""
    # int64 in numpy only so a bug shows up as a failed assert, not a silent wrap.
    x_q = x_uint8.astype(np.int64) - 128

    acc1 = x_q @ q["w1_q"].astype(np.int64).T + q["b1_q"]
    assert np.abs(acc1).max() <= INT32_MAX, "layer 1 accumulator overflowed int32"

    m1, shift1 = int(q["m1"]), int(q["shift1"])
    h_q = np.clip((acc1 * m1 + (1 << (shift1 - 1))) >> shift1, 0, H_MAX)

    acc2 = h_q @ q["w2_q"].astype(np.int64).T + q["b2_q"]
    assert np.abs(acc2).max() <= INT32_MAX, "layer 2 accumulator overflowed int32"

    pred = acc2.argmax(1)   # numpy argmax returns the first (lowest) index on ties
    if return_intermediates:
        return pred, {"x_q": x_q, "acc1": acc1, "h_q": h_q, "acc2": acc2}
    return pred


def main():
    d = np.load(ROOT / "fer3.npz")
    f = dict(np.load(FLOAT_WEIGHTS))

    q = quantize(f, d["x_train"])
    np.savez(INT_WEIGHTS, **q)

    digest = hashlib.sha256()
    for k in ("w1_q", "b1_q", "m1", "shift1", "w2_q", "b2_q"):
        digest.update(np.ascontiguousarray(q[k]).tobytes())

    print("=" * 72)
    print("Quantization parameters")
    print("=" * 72)
    print(f"  s_w1 {q['s_w1']:.6g}   s_h {q['s_h']:.6g}   s_w2 {q['s_w2']:.6g}")
    m1, shift1 = int(q["m1"]), int(q["shift1"])
    exact = S_X * q["s_w1"] / q["s_h"]
    print(f"  requantize        : (acc1 * {m1}) >> {shift1}"
          f"   = x {m1 / 2**shift1:.8g}  (exact {exact:.8g})")
    print(f"  b1_q range        : [{q['b1_q'].min()}, {q['b1_q'].max()}]")
    print(f"  b2_q range        : [{q['b2_q'].min()}, {q['b2_q'].max()}]")
    print(f"  saved             : {INT_WEIGHTS.relative_to(ROOT)}")
    print(f"  sha256 (int data) : {digest.hexdigest()}")
    print()

    print("=" * 72)
    print("Float vs integer model")
    print("=" * 72)
    print(f"  {'split':<6} {'float acc':>10} {'int acc':>10} {'agree':>8}"
          f" {'max |acc1|':>12} {'max |acc2|':>12}")
    for split in ("val", "test"):
        x, y = d[f"x_{split}"], d[f"y_{split}"]
        p_float = infer_float(x, f)
        p_int, mid = infer_int(x, q, return_intermediates=True)
        print(f"  {split:<6} {(p_float == y).mean():>10.4f} {(p_int == y).mean():>10.4f}"
              f" {(p_float == p_int).mean():>8.4f}"
              f" {np.abs(mid['acc1']).max():>12,} {np.abs(mid['acc2']).max():>12,}")
    print(f"  int32 max is {INT32_MAX:,}")


if __name__ == "__main__":
    main()
