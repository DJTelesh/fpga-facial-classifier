#!/usr/bin/env python3
"""
quantize.py  --  float weights -> integer weights + bit-exact integer reference model

Reads weights/fer3_float.npz (written by train.py), quantizes it to DATA_WIDTH bits, and
writes weights/fer3_int{DATA_WIDTH}.npz. Then runs the integer model on the val and test
sets and compares it against the float model.

infer_int() is the golden reference for the RTL. It uses only integer operations the
hardware can do (b-bit x b-bit multiplies, ACC_WIDTH accumulate, one multiply + arithmetic
right shift to requantize, compare for argmax). If the RTL disagrees with infer_int()
on any input, the RTL is wrong.

Integer pipeline, per image (b = DATA_WIDTH, H = 2^(b-1) - 1):
    x_q  = (pixel - 128) >> max(0, 8 - b)                b bits
    acc1 = W1_q @ x_q + b1_q                             ACC_WIDTH (32 if b <= 8, else 48)
    h_q  = clip((acc1 * M1 + 2^(S1-1)) >> S1, 0, H)      ReLU + requantize to [0, H]
    acc2 = W2_q @ h_q + b2_q                             ACC_WIDTH
    pred = argmax(acc2)          ties go to the LOWEST index -- the RTL must match this

The (acc * M + 2^(S-1)) >> S step is floor(v + 0.5) rounding (contract section 4),
done with an integer multiply and an arithmetic shift, so there is no divider.
Full definition: arithmetic_contract.md at the repo root.

Usage:
    python3 quantize.py               # int8, the contracted baseline
    python3 quantize.py --bits 16     # int16 hardware configuration
    python3 quantize.py --sweep       # accuracy vs DATA_WIDTH, nothing written
"""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FLOAT_WEIGHTS = ROOT / "weights" / "fer3_float.npz"

N_IN, N_HIDDEN = 2304, 64
M_BITS = 15          # requantize multiplier is unsigned and fits in 15 bits
SWEEP_BITS = (2, 3, 4, 5, 6, 7, 8, 12, 16)


@dataclass(frozen=True)
class Fmt:
    """Everything arithmetic_contract.md sections 3-6 derive from DATA_WIDTH."""
    bits: int

    def __post_init__(self):
        assert 2 <= self.bits <= 16, "contract v2 covers DATA_WIDTH 2..16"

    @property
    def w_max(self):        # weights clamp to +/- w_max (section 4)
        return 2 ** (self.bits - 1) - 1

    @property
    def h_max(self):        # post-ReLU activations live in [0, h_max]
        return self.w_max

    @property
    def in_shift(self):     # pixels drop this many LSBs on the way in (section 3)
        return max(0, 8 - self.bits)

    @property
    def x_max(self):        # largest |x_q|; the input is the only value reaching -2^(n-1)
        return 2 ** (7 - self.in_shift)

    @property
    def s_x(self):          # input scale, so x_q * s_x ~= (pixel - 128) / 128
        return 2 ** self.in_shift / 128

    @property
    def acc_width(self):
        return 32 if self.bits <= 8 else 48

    @property
    def b_limit(self):      # abs(bias) must stay below this (section 4)
        return 2 ** (self.acc_width - 8)

    def acc_bounds(self):
        """Worst-case |acc| per layer (section 5): N * max|x| * max|w| + bias limit."""
        return {
            "layer 1": N_IN * self.x_max * self.w_max + self.b_limit,
            "layer 2": N_HIDDEN * self.h_max * self.w_max + self.b_limit,
        }

    def weight_dtype(self):
        return np.int8 if self.bits <= 8 else np.int16

    def acc_dtype(self):
        return np.int32 if self.acc_width == 32 else np.int64


def int_weights_path(bits):
    return ROOT / "weights" / f"fer3_int{bits}.npz"


def round_half_up(v):
    """floor(v + 0.5), NOT np.round (which rounds half to even)."""
    return np.floor(v + 0.5)


def quantize_weights(w, fmt):
    """Symmetric per-tensor: scale = max|w| / w_max."""
    s = np.max(np.abs(w)) / fmt.w_max
    q = np.clip(round_half_up(w / s), -fmt.w_max, fmt.w_max).astype(fmt.weight_dtype())
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


def quantize(f, x_calib, fmt):
    w1_q, s_w1 = quantize_weights(f["w1"], fmt)
    w2_q, s_w2 = quantize_weights(f["w2"], fmt)

    # Hidden activation scale: largest post-ReLU value seen on the training set maps
    # to h_max. Calibrated on train only, so val/test stay unseen.
    s_h = float_hidden(x_calib, f).max() / fmt.h_max

    # Biases go in the accumulator, so they use the accumulator's scale.
    b1_q = round_half_up(f["b1"] / (fmt.s_x * s_w1)).astype(np.int64)
    b2_q = round_half_up(f["b2"] / (s_h * s_w2)).astype(np.int64)

    m1, shift1 = multiplier_and_shift(fmt.s_x * s_w1 / s_h)

    # Limits from arithmetic_contract.md sections 4-6. The accumulator and
    # requantize bit widths are only safe while these hold.
    for name, b in (("b1_q", b1_q), ("b2_q", b2_q)):
        assert np.abs(b).max() < fmt.b_limit, f"{name} breaks the bias limit"
    for layer, bound in fmt.acc_bounds().items():
        assert bound.bit_length() + 1 <= fmt.acc_width, f"{layer} can overflow ACC_WIDTH"
    assert 2 ** (M_BITS - 1) <= m1 < 2**M_BITS, f"M1={m1} out of range"
    assert 1 <= shift1 <= 40, f"S1={shift1} out of range"

    return {
        "w1_q": w1_q, "b1_q": b1_q.astype(fmt.acc_dtype()),
        "m1": np.int32(m1), "shift1": np.int32(shift1),
        "w2_q": w2_q, "b2_q": b2_q.astype(fmt.acc_dtype()),
        "bits": np.int32(fmt.bits), "acc_width": np.int32(fmt.acc_width),
        # scales are for documentation only; infer_int() never touches them
        "s_x": fmt.s_x, "s_w1": s_w1, "s_h": s_h, "s_w2": s_w2,
    }


def infer_int(x_uint8, q, return_intermediates=False):
    """Bit-exact integer model. x_uint8 is (N, 2304) raw pixels."""
    fmt = Fmt(int(q["bits"]))
    acc_max = 2 ** (fmt.acc_width - 1) - 1

    # int64 in numpy only so a bug shows up as a failed assert, not a silent wrap.
    x_q = (x_uint8.astype(np.int64) - 128) >> fmt.in_shift

    acc1 = x_q @ q["w1_q"].astype(np.int64).T + q["b1_q"].astype(np.int64)
    assert np.abs(acc1).max() <= acc_max, f"layer 1 accumulator overflowed int{fmt.acc_width}"

    m1, shift1 = int(q["m1"]), int(q["shift1"])
    h_q = np.clip((acc1 * m1 + (1 << (shift1 - 1))) >> shift1, 0, fmt.h_max)

    acc2 = h_q @ q["w2_q"].astype(np.int64).T + q["b2_q"].astype(np.int64)
    assert np.abs(acc2).max() <= acc_max, f"layer 2 accumulator overflowed int{fmt.acc_width}"

    pred = acc2.argmax(1)   # numpy argmax returns the first (lowest) index on ties
    if return_intermediates:
        return pred, {"x_q": x_q, "acc1": acc1, "h_q": h_q, "acc2": acc2}
    return pred


def int_weights_sha256(q):
    """The weight fingerprint in arithmetic_contract.md section 8.

    Hashes only the integer arrays the hardware uses, so it does not change if the
    .npz container or the documentation-only float scales change.
    """
    digest = hashlib.sha256()
    for k in ("w1_q", "b1_q", "m1", "shift1", "w2_q", "b2_q"):
        digest.update(np.ascontiguousarray(q[k]).tobytes())
    return digest.hexdigest()


def save_if_changed(path, q):
    """np.savez stamps the time into the file, so rewriting identical weights would
    still show up as a binary change in git. Only write when the contents differ."""
    if path.exists():
        old = dict(np.load(path))
        if old.keys() == q.keys() and all(np.array_equal(old[k], q[k]) for k in q):
            return False
    np.savez(path, **q)
    return True


def report(d, f, fmt):
    q = quantize(f, d["x_train"], fmt)
    path = int_weights_path(fmt.bits)
    written = save_if_changed(path, q)

    print("=" * 72)
    print(f"Quantization parameters, DATA_WIDTH = {fmt.bits}")
    print("=" * 72)
    print(f"  weights / acts    : [-{fmt.w_max}, {fmt.w_max}] / [0, {fmt.h_max}]"
          f"   input >> {fmt.in_shift}   ACC_WIDTH {fmt.acc_width}")
    print(f"  s_w1 {q['s_w1']:.6g}   s_h {q['s_h']:.6g}   s_w2 {q['s_w2']:.6g}")
    m1, shift1 = int(q["m1"]), int(q["shift1"])
    exact = fmt.s_x * q["s_w1"] / q["s_h"]
    print(f"  requantize        : (acc1 * {m1}) >> {shift1}"
          f"   = x {m1 / 2**shift1:.8g}  (exact {exact:.8g})")
    print(f"  b1_q range        : [{q['b1_q'].min()}, {q['b1_q'].max()}]")
    print(f"  b2_q range        : [{q['b2_q'].min()}, {q['b2_q'].max()}]")
    print(f"  saved             : {path.relative_to(ROOT)}"
          f"{'' if written else '  (unchanged, not rewritten)'}")
    print(f"  sha256 (int data) : {int_weights_sha256(q)}")
    print()

    print("=" * 72)
    print("Accumulator bound (contract section 5)")
    print("=" * 72)
    for layer, bound in fmt.acc_bounds().items():
        print(f"  {layer}: worst case {bound:>18,}   needs {bound.bit_length() + 1}"
              f" bits signed, ACC_WIDTH {fmt.acc_width}")
    print()

    print("=" * 72)
    print("Float vs integer model")
    print("=" * 72)
    print(f"  {'split':<6} {'float acc':>10} {'int acc':>10} {'agree':>8}"
          f" {'max |acc1|':>16} {'max |acc2|':>16}")
    for split in ("val", "test"):
        x, y = d[f"x_{split}"], d[f"y_{split}"]
        p_float = infer_float(x, f)
        p_int, mid = infer_int(x, q, return_intermediates=True)
        print(f"  {split:<6} {(p_float == y).mean():>10.4f} {(p_int == y).mean():>10.4f}"
              f" {(p_float == p_int).mean():>8.4f}"
              f" {np.abs(mid['acc1']).max():>16,} {np.abs(mid['acc2']).max():>16,}")


def sweep(d, f):
    """Accuracy vs DATA_WIDTH (proposal section 5.3). Writes nothing."""
    print("=" * 72)
    print("Accuracy vs DATA_WIDTH   (agree = same prediction as the float model)")
    print("=" * 72)
    p_float = {s: infer_float(d[f"x_{s}"], f) for s in ("val", "test")}
    print(f"  {'float':>5} {'':>4} {(p_float['val'] == d['y_val']).mean():>9.4f}"
          f" {(p_float['test'] == d['y_test']).mean():>9.4f}")
    print(f"  {'bits':>5} {'acc':>4} {'val acc':>9} {'test acc':>9}"
          f" {'val agree':>10} {'test agree':>11}")
    for bits in SWEEP_BITS:
        fmt = Fmt(bits)
        q = quantize(f, d["x_train"], fmt)
        row = f"  {bits:>5} {fmt.acc_width:>4}"
        acc, agree = [], []
        for s in ("val", "test"):
            p = infer_int(d[f"x_{s}"], q)
            acc.append((p == d[f"y_{s}"]).mean())
            agree.append((p == p_float[s]).mean())
        print(row + f" {acc[0]:>9.4f} {acc[1]:>9.4f} {agree[0]:>10.4f} {agree[1]:>11.4f}")
    print("  chance = 0.3333")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8, help="DATA_WIDTH (default 8)")
    ap.add_argument("--sweep", action="store_true",
                    help="print accuracy for every width in SWEEP_BITS; writes nothing")
    args = ap.parse_args()

    d = np.load(ROOT / "fer3.npz")
    f = dict(np.load(FLOAT_WEIGHTS))
    if args.sweep:
        sweep(d, f)
    else:
        report(d, f, Fmt(args.bits))


if __name__ == "__main__":
    main()
