#!/usr/bin/env python3
"""
gen_vectors.py -- dump golden-model values as hex files for the SystemVerilog testbenches.

Reads the int8 weights and runs infer_int() on a few VALIDATION images. Never test.
Run from the repo root:  python sw/python/gen_vectors.py
"""
import hashlib
from pathlib import Path

import numpy as np

from quantize import infer_int          # the golden model

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "test_vectors"
N_IMAGES = 4                            # how many validation images to dump


def write_hex(path, values, bits):
    """One value per line, two's complement, fixed width: -1 -> 'ff' at 8 bits."""
    mask = (1 << bits) - 1
    digits = bits // 4
    with open(path, "w") as f:
        for v in np.asarray(values).ravel():
            f.write(f"{int(v) & mask:0{digits}x}\n")


def main():
    d = np.load(ROOT / "fer3.npz")
    wpath = ROOT / "weights" / "fer3_int8.npz"
    q = dict(np.load(wpath))
    OUT.mkdir(exist_ok=True)

    # Shared by every test. Row-major: W1[n][i] is on line n*2304 + i.
    write_hex(OUT / "w1.hex", q["w1_q"], 8)
    write_hex(OUT / "b1.hex", q["b1_q"], 32)

    # Run the golden model on the first few validation images.
    x = d["x_val"][:N_IMAGES]
    _, mid = infer_int(x, q, return_intermediates=True)

    for k in range(N_IMAGES):
        write_hex(OUT / f"img{k}_x.hex",    mid["x_q"][k],  8)    # 2304 inputs
        write_hex(OUT / f"img{k}_acc1.hex", mid["acc1"][k], 32)   # 64 expected sums

    # Sanity check one neuron by hand, independently of infer_int's internals.
    n = 0
    by_hand = int(q["b1_q"][n]) + int(np.dot(q["w1_q"][n].astype(np.int64),
                                             mid["x_q"][0].astype(np.int64)))
    assert by_hand == mid["acc1"][0, n], "hand-computed acc1 disagrees with infer_int"

    # Record which weights these vectors belong to (contract section 8).
    sha = hashlib.sha256(wpath.read_bytes()).hexdigest()
    (OUT / "manifest.txt").write_text(
        f"weights file : weights/fer3_int8.npz\n"
        f"sha256       : {sha}\n"
        f"images       : x_val[0:{N_IMAGES}]\n"
    )
    print(f"wrote vectors for {N_IMAGES} images to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()