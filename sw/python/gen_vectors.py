#!/usr/bin/env python3
"""
gen_vectors.py -- dump golden-model values as hex files for the SystemVerilog testbenches.

Reads the integer weights and runs infer_int() on a few VALIDATION images. Never test.
    python sw/python/gen_vectors.py              # int8  -> test_vectors/
    python sw/python/gen_vectors.py --bits 16    # int16 -> test_vectors/int16/
Run quantize.py with the same --bits first.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from quantize import Fmt, infer_int, int_weights_path, int_weights_sha256   # golden model

ROOT = Path(__file__).resolve().parents[2]
N_IMAGES = 4                            # how many validation images to dump


def write_hex(path, values, bits):
    """One value per line, two's complement, fixed width: -1 -> 'ff' at 8 bits."""
    mask = (1 << bits) - 1
    digits = -(-bits // 4)              # ceil(bits / 4)
    with open(path, "w") as f:
        for v in np.asarray(values).ravel():
            f.write(f"{int(v) & mask:0{digits}x}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8, help="DATA_WIDTH (default 8)")
    args = ap.parse_args()

    fmt = Fmt(args.bits)
    d = np.load(ROOT / "fer3.npz")
    wpath = int_weights_path(fmt.bits)
    q = dict(np.load(wpath))
    assert int(q["bits"]) == fmt.bits, f"{wpath.name} was not quantized to {fmt.bits} bits"

    # int8 stays at the top level so existing testbench paths keep working.
    out = ROOT / "test_vectors" if fmt.bits == 8 else ROOT / "test_vectors" / f"int{fmt.bits}"
    out.mkdir(parents=True, exist_ok=True)

    # Shared by every test. Row-major: W1[n][i] is on line n*2304 + i.
    write_hex(out / "w1.hex", q["w1_q"], fmt.bits)
    write_hex(out / "b1.hex", q["b1_q"], fmt.acc_width)

    # Run the golden model on the first few validation images.
    x = d["x_val"][:N_IMAGES]
    _, mid = infer_int(x, q, return_intermediates=True)

    for k in range(N_IMAGES):
        write_hex(out / f"img{k}_x.hex",    mid["x_q"][k],  fmt.bits)        # 2304 inputs
        write_hex(out / f"img{k}_acc1.hex", mid["acc1"][k], fmt.acc_width)   # 64 sums

    # Sanity check one neuron by hand, independently of infer_int's internals.
    n = 0
    by_hand = int(q["b1_q"][n]) + int(np.dot(q["w1_q"][n].astype(np.int64),
                                             mid["x_q"][0].astype(np.int64)))
    assert by_hand == mid["acc1"][0, n], "hand-computed acc1 disagrees with infer_int"

    # Record which data and weights these vectors belong to (contract section 8).
    # Same fingerprints quantize.py prints and the contract records.
    data_sha = hashlib.sha256((ROOT / "fer3.npz").read_bytes()).hexdigest()
    (out / "manifest.txt").write_text(
        f"dataset             : fer3.npz\n"
        f"dataset sha256      : {data_sha}\n"
        f"weights             : weights/{wpath.name}\n"
        f"int weight sha256   : {int_weights_sha256(q)}\n"
        f"DATA_WIDTH          : {fmt.bits}\n"
        f"ACC_WIDTH           : {fmt.acc_width}\n"
        f"images              : x_val[0:{N_IMAGES}]\n"
    )
    print(f"wrote vectors for {N_IMAGES} images to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
