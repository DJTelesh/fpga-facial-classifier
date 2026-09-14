#!/usr/bin/env python3
"""
model_budget.py  --  ECE 4999 week 2

Recomputes, from the layer sizes, every number the proposal currently estimates:
parameter count, on-chip memory footprint, the accumulator bound in the arithmetic
contract, and the throughput ceiling.

These are COMPUTED, not measured. Everything printed here is an upper bound that assumes
perfect weight delivery. The whole point of the parallelism sweep is that the assumption
breaks somewhere, and the measured curve falls below these numbers. Label them as such
in the report.

Usage:
    python3 model_budget.py                      # frozen topology
    python3 model_budget.py --layers 2304 64 3
    python3 model_budget.py --layers 1024 64 3 --tile 32
"""

import argparse
import math

# Cyclone V SE 5CSEMA5F31 (A5), confirmed from the device handbook.
M10K_BLOCKS = 397
M10K_BITS_EACH = 10240
MLAB_KBITS = 480
DSP_BLOCKS = 87
MULT_18x18 = 174
FPGA_CLK_HZ = 50_000_000

# Fraction of on-chip memory assumed usable for weights. The rest goes to activation
# buffers, the display path, and whatever the fitter wants. This is an assumption, not
# a measurement -- replace it with the Quartus fitter report once week 4 synthesizes.
USABLE_FRACTION = 0.5

INT8_MIN, INT8_MAX = -128, 127
W_MAX = 127          # weights clamp to +/-127 (contract section 4)
INT32_MAX = 2**31 - 1


def human_bits(b):
    return f"{b/1024:,.0f} Kb" if b >= 1024 else f"{b} b"


def report(layers, tile, widths=(16, 8)):
    print("=" * 72)
    print("Topology")
    print("=" * 72)
    shape = " -> ".join(str(n) for n in layers)
    print(f"  input tile        : {tile} x {tile} = {tile*tile} values")
    if tile * tile != layers[0]:
        print(f"  WARNING: tile gives {tile*tile} inputs but layer 0 is {layers[0]}")
    print(f"  layers            : {shape}")
    print()

    weights = sum(layers[i] * layers[i + 1] for i in range(len(layers) - 1))
    biases = sum(layers[i + 1] for i in range(len(layers) - 1))
    macs = weights  # dense: one MAC per weight per inference

    print(f"  weights           : {weights:,}")
    print(f"  biases            : {biases:,}")
    print(f"  MACs / inference  : {macs:,}")
    print()
    for i in range(len(layers) - 1):
        w = layers[i] * layers[i + 1]
        print(f"    layer {i+1}: {layers[i]:>5} -> {layers[i+1]:<5}  {w:>10,} weights"
              f"  ({100*w/weights:5.1f}% of total)")
    print()

    # ---------------------------------------------------------------- memory
    total_bits = M10K_BLOCKS * M10K_BITS_EACH + MLAB_KBITS * 1024
    usable_bits = total_bits * USABLE_FRACTION
    print("=" * 72)
    print("On-chip memory")
    print("=" * 72)
    print(f"  M10K              : {M10K_BLOCKS} blocks x {M10K_BITS_EACH} b"
          f" = {human_bits(M10K_BLOCKS*M10K_BITS_EACH)}")
    print(f"  MLAB              : {human_bits(MLAB_KBITS*1024)}")
    print(f"  total             : {human_bits(total_bits)}")
    print(f"  assumed usable    : {human_bits(usable_bits)}  ({USABLE_FRACTION:.0%} -- ASSUMPTION)")
    print()
    for wbits in widths:
        wb = weights * wbits
        bb = biases * 32          # biases are int32 regardless (contract section 4)
        tot = wb + bb
        blocks = math.ceil(tot / M10K_BITS_EACH)
        print(f"  @ {wbits:>2}-bit weights : {human_bits(wb)} weights + {human_bits(bb)} bias"
              f" = {human_bits(tot)}")
        print(f"                      {100*tot/total_bits:5.1f}% of total,"
              f" {100*tot/usable_bits:5.1f}% of assumed usable,"
              f" >= {blocks} M10K blocks if perfectly packed")
    print()

    # ----------------------------------------------------- accumulator bound
    print("=" * 72)
    print("Accumulator bound (arithmetic contract section 5)")
    print("=" * 72)
    worst = 0
    for i in range(len(layers) - 1):
        n = layers[i]
        # Layer 1 sees the input, which reaches -128. Later layers see post-ReLU
        # activations in [0, 127].
        x_max = 128 if i == 0 else 127
        bound = n * x_max * W_MAX
        worst = max(worst, bound)
        bits = bound.bit_length() + 1  # +1 for sign
        print(f"  layer {i+1}: {n:>5} terms x {x_max} x {W_MAX} = {bound:>15,}"
              f"   needs {bits} bits signed")
    margin = INT32_MAX / worst
    print(f"  int32 max         : {INT32_MAX:,}")
    print(f"  worst case uses   : {100*worst/INT32_MAX:.2f}% of range (margin {margin:.0f}x)")
    if margin < 2:
        print("  *** int32 IS NOT SAFE for this topology. Widen the accumulator or")
        print("      reintroduce saturation, and update the contract. ***")
    else:
        print("  int32 is safe. Overflow is structurally impossible, so the accumulator")
        print("  needs no saturation logic and addition stays order-independent.")
    print()

    # -------------------------------------------------------- throughput ceiling
    print("=" * 72)
    print("Throughput ceiling -- COMPUTED, assumes one MAC/unit/cycle and perfect feeding")
    print("=" * 72)
    sweep = [1, 2, 4, 8, 16, 32, 64, 128]
    print(f"  clock {FPGA_CLK_HZ/1e6:.0f} MHz")
    print()
    print(f"  {'NUM_MACS':>9} {'cycles/inf':>12} {'inf/sec':>12} {'subjects @30fps':>17}")
    print("  " + "-" * 52)
    for nm in sweep:
        cycles = math.ceil(macs / nm)
        ips = FPGA_CLK_HZ / cycles
        print(f"  {nm:>9} {cycles:>12,} {ips:>12,.0f} {ips/30:>17,.0f}")
    print()
    print(f"  DSP blocks available : {DSP_BLOCKS} ({MULT_18x18} 18x18 multipliers)")
    print(f"  NUM_MACS <= {MULT_18x18} is the hard ceiling at 16-bit unless the fitter")
    print("  packs narrower multipliers. Whether it does at 8-bit is a week-10 measurement.")
    print()
    print("  Weight delivery required, per cycle, to sustain these rates:")
    for nm in (8, 32, 64, 128):
        for wbits in widths:
            print(f"    NUM_MACS={nm:>3} @ {wbits:>2}-bit : {nm*wbits:>5} bits/cycle"
                  f" = {nm*wbits/8:>5.0f} bytes/cycle")
    print()
    print("  A single M10K port delivers at most its configured width per cycle. The")
    print("  number of weights deliverable per cycle is set by how the array is")
    print("  partitioned across blocks, not by the total bit count. This is the")
    print("  bottleneck the week-11 sweep is designed to find.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[2304, 64, 3])
    ap.add_argument("--tile", type=int, default=48)
    args = ap.parse_args()
    report(args.layers, args.tile)


if __name__ == "__main__":
    main()
