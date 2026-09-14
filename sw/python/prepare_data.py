#!/usr/bin/env python3
"""
prepare_data.py  --  ECE 4999 week 2

Loads the FER-2013 CSV, audits it, builds the balanced three-class subset defined in
arithmetic_contract.md section 1, and writes fer3.npz.

Audits first, subsets second. Every count printed here should go in the results file --
the class balance is something an interviewer can reasonably ask about and "I counted it"
is a better answer than "I read it somewhere".

Usage:
    python3 prepare_data.py --csv /path/to/fer2013.csv --out fer3.npz

FER-2013 CSV schema (three columns):
    emotion  int 0-6
    pixels   2304 space-separated ints, row-major, 48x48 grayscale
    Usage    "Training" | "PublicTest" | "PrivateTest"
"""

import argparse
import hashlib
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Frozen by arithmetic_contract.md section 1. Changing these changes the contract.
# ---------------------------------------------------------------------------

FER_LABEL_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

# (FER-2013 source label, project class index, project class name)
CLASS_MAP = [
    (3, 0, "smiling"),
    (6, 1, "neutral"),
    (5, 2, "surprised"),
]

TILE = 48
N_PIXELS = TILE * TILE  # 2304

SEED = 4999  # fixed, recorded, so the subset is reproducible


def load_csv(path):
    """Read the FER-2013 CSV into (pixels uint8 [N,2304], labels int [N], usage str [N])."""
    labels, pixels, usage = [], [], []
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        if len(header) != 3:
            sys.exit(f"Unexpected header (want 3 columns, got {len(header)}): {header}")
        for lineno, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            # The pixel field may be quoted; split on the first and last comma only.
            first = line.index(",")
            last = line.rindex(",")
            emo = int(line[:first])
            pix = line[first + 1:last].strip().strip('"')
            use = line[last + 1:].strip()
            vals = np.fromstring(pix, dtype=np.int32, sep=" ")
            if vals.size != N_PIXELS:
                sys.exit(f"Line {lineno}: expected {N_PIXELS} pixels, got {vals.size}")
            labels.append(emo)
            pixels.append(vals.astype(np.uint8))
            usage.append(use)
    return np.stack(pixels), np.array(labels), np.array(usage)


def audit(labels, usage, pixels):
    """Print the full class histogram per split. Verify assumptions, do not trust them."""
    print("=" * 68)
    print("FER-2013 audit")
    print("=" * 68)
    print(f"total rows        : {len(labels)}")
    print(f"pixel dtype/range : {pixels.dtype}, [{pixels.min()}, {pixels.max()}]")
    print(f"tile shape        : {TILE}x{TILE} = {N_PIXELS} values")
    print()

    splits = ["Training", "PublicTest", "PrivateTest"]
    unknown = set(usage) - set(splits)
    if unknown:
        print(f"WARNING: unexpected Usage values present: {unknown}")

    header = f"{'label':<10}" + "".join(f"{s:>13}" for s in splits) + f"{'total':>9}"
    print(header)
    print("-" * len(header))
    for lab in range(7):
        row = f"{FER_LABEL_NAMES[lab]:<10}"
        tot = 0
        for s in splits:
            n = int(((labels == lab) & (usage == s)).sum())
            tot += n
            row += f"{n:>13}"
        print(row + f"{tot:>9}")
    print("-" * len(header))
    row = f"{'ALL':<10}"
    for s in splits:
        row += f"{int((usage == s).sum()):>13}"
    print(row + f"{len(labels):>9}")
    print()


def build_subset(pixels, labels, usage, balance=True):
    """Select the three project classes, remap labels, optionally balance by subsampling."""
    rng = np.random.default_rng(SEED)
    out = {}

    # Standard FER-2013 protocol: train on Training, validate on PublicTest,
    # report on PrivateTest. PrivateTest is the split published baselines quote.
    split_names = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}

    print("=" * 68)
    print(f"Three-class subset (balance={balance}, seed={SEED})")
    print("=" * 68)

    for key, usage_name in split_names.items():
        idx_per_class = []
        for fer_lab, proj_idx, _name in CLASS_MAP:
            idx = np.flatnonzero((labels == fer_lab) & (usage == usage_name))
            idx_per_class.append(idx)

        if balance:
            n = min(len(i) for i in idx_per_class)
            idx_per_class = [rng.permutation(i)[:n] for i in idx_per_class]

        xs, ys = [], []
        for (_fer, proj_idx, _name), idx in zip(CLASS_MAP, idx_per_class):
            xs.append(pixels[idx])
            ys.append(np.full(len(idx), proj_idx, dtype=np.int64))
        x = np.concatenate(xs)
        y = np.concatenate(ys)

        # Shuffle so that training order is not class-ordered.
        perm = rng.permutation(len(y))
        out[f"x_{key}"] = x[perm]
        out[f"y_{key}"] = y[perm]

        counts = {name: int((y == pi).sum()) for _f, pi, name in CLASS_MAP}
        print(f"{key:<6} ({usage_name:<11}) n={len(y):>6}   {counts}")

    print()
    chance = 1.0 / len(CLASS_MAP)
    print(f"chance accuracy = {chance:.4f}" + ("  (exact, balanced)" if balance else "  (approx)"))
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to fer2013.csv")
    ap.add_argument("--out", default="fer3.npz")
    ap.add_argument("--no-balance", action="store_true",
                    help="keep natural class proportions instead of subsampling")
    args = ap.parse_args()

    pixels, labels, usage = load_csv(args.csv)
    audit(labels, usage, pixels)
    data = build_subset(pixels, labels, usage, balance=not args.no_balance)

    # Record what produced this file.
    data["seed"] = np.array(SEED)
    data["class_names"] = np.array([n for _f, _p, n in CLASS_MAP])
    data["fer_labels"] = np.array([f for f, _p, _n in CLASS_MAP])

    np.savez_compressed(args.out, **data)

    with open(args.out, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    print(f"wrote {args.out}")
    print(f"sha256 {digest}")
    print()
    print("Record the sha256 in the results file. Test vectors will be keyed to it.")


if __name__ == "__main__":
    main()
