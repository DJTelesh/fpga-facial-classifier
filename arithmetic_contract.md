# Arithmetic Contract

This document defines the exact integer arithmetic of the inference accelerator. The
Python golden model (`sw/python/quantize.py`, `infer_int()`), the ARM Cortex-A9
software baseline, and the RTL must all implement **this** arithmetic and produce
**bit-identical** results. When two implementations disagree, whichever one departs
from this document is wrong.

Changing anything in this document changes the contract. Invalidate and regenerate
every file keyed to it (weights, test vectors, measured results), and record the change
in section 9.

**Status:** v1, int8 baseline configuration.

---

## 1. Dataset

Source: FER-2013 (`fer2013.csv`), 48x48 grayscale, pixel values `uint8` in [0, 255].
Built by `sw/python/prepare_data.py`, which writes `fer3.npz`.

| Project index | Name      | FER-2013 label |
|---------------|-----------|----------------|
| 0             | smiling   | 3 (happy)      |
| 1             | neutral   | 6 (neutral)    |
| 2             | surprised | 5 (surprise)   |

| Split | FER-2013 `Usage` | Per class | Total |
|-------|------------------|-----------|-------|
| train | Training         | 3171      | 9513  |
| val   | PublicTest       | 415       | 1245  |
| test  | PrivateTest      | 416       | 1248  |

- Classes are balanced by randomly subsampling each class down to the smallest one,
  then each split is shuffled. RNG seed is `4999`.
- Chance accuracy is exactly 1/3.
- Images are flattened row-major: pixel `(r, c)` is input index `i = 48*r + c`.
- `fer3.npz` sha256: `1112e5950e94fc85376f96a0f73993c0a0bcb6a52e4c2276aab75434de4eded4`
- The test split is used only to report final numbers. Nothing (training, epoch
  selection, calibration) may be tuned on it.

## 2. Network topology

```
2304 inputs  ->  Dense(64) + ReLU  ->  Dense(3)  ->  argmax
```

| Layer | Shape of W   | Weights | Biases | MACs    |
|-------|--------------|---------|--------|---------|
| 1     | 64 x 2304    | 147,456 | 64     | 147,456 |
| 2     | 3 x 64       | 192     | 3      | 192     |
|       | **total**    | 147,648 | 67     | 147,648 |

`W[n][i]` is the weight from input `i` to neuron `n`. That index order is the *logical*
layout used by the weight and test-vector files. How the RTL partitions weights across
M10K blocks is a design choice and is not part of this contract.

## 3. Input encoding

```
x_q[i] = pixel[i] - 128          int8, range [-128, 127]
```

- The input scale is `s_x = 1/128`, so `x_q * s_x` equals the float input
  `(pixel - 128) / 128` used in training **exactly**, with no rounding.
- In hardware, `pixel - 128` is the same as inverting the pixel's MSB and reading the
  byte as two's complement. No subtractor is needed.
- Input is the only int8 value that can reach -128. Every other int8 value in the
  datapath is limited to +/-127.

## 4. Number formats and rounding

All integers are two's complement.

| Quantity                   | Type   | Range                  |
|----------------------------|--------|------------------------|
| Input `x_q`                | int8   | [-128, 127]            |
| Weights `W1_q`, `W2_q`     | int8   | [-127, 127] (clamped)  |
| Biases `b1_q`, `b2_q`      | int32  | abs(b) < 2^24          |
| Accumulators `acc1`, `acc2`| int32  | see section 5          |
| Hidden activations `h_q`   | int8   | [0, 127]               |
| Requantize multiplier `M1` | uint15 | [2^14, 2^15)           |
| Requantize shift `S1`      | uint6  | [1, 40]                |

**Rounding.** Every float-to-integer conversion, and every requantization, uses
round-half-up:

```
round(v) = floor(v + 0.5)
```

Do **not** use round-half-to-even (`np.round`, C `rint`, IEEE default). The two differ
on exact .5 cases and would make the golden model disagree with the RTL.

**Weight quantization** is symmetric, per tensor (one scale per layer):

```
s_w  = max(abs(W)) / 127
W_q  = clamp(round(W / s_w), -127, +127)
```

Weights clamp to +/-127, not -128, so the range is symmetric and `-W_q` never overflows.

**Bias quantization.** A bias is added directly into the accumulator, so it uses the
accumulator's scale:

```
b1_q = round(b1 / (s_x  * s_w1))
b2_q = round(b2 / (s_h  * s_w2))
```

The quantizer must reject any bias with abs(b_q) >= 2^24. Section 5 depends on that
limit.

## 5. Accumulator bound

A dense layer with `N` inputs has worst case

```
abs(acc) <= N * max(abs(x)) * max(abs(w)) + max(abs(b))
```

| Layer | N    | max abs(x) | max abs(w) | Product term | + bias limit | Total bound    | Bits (signed) |
|-------|------|------------|------------|--------------|--------------|----------------|---------------|
| 1     | 2304 | 128        | 127        | 37,453,824   | 2^24         | ≤ 54,231,040   | 27            |
| 2     | 64   | 127        | 127        | 1,032,256    | 2^24         | ≤ 17,809,472   | 26            |

Both bounds are under 2^26, so worst-case accumulators need 27 bits signed. int32 has a
margin of about 40x.

Consequences:

- **Overflow is structurally impossible.** Accumulators need no saturation logic and no
  overflow detection.
- **Addition order is irrelevant.** Two's complement addition that never overflows is
  associative, so any MAC-array width, tree shape, or partial-sum order gives the same
  answer bit for bit. The parallelism sweep may split the sum across any number of
  lanes.
- The golden model asserts that no real accumulator exceeds int32. On the current
  weights the largest observed values are about 1.4M (layer 1) and about 38K (layer 2).

If the topology, input width, or weight width ever changes, recompute this section with
`sw/python/model_budget.py` before touching anything else.

## 6. Hidden-layer requantization (ReLU + rescale)

`acc1` has scale `s_x * s_w1`. Layer 2 needs `h_q` in [0, 127] with scale `s_h`, so
each hidden value is multiplied by `s_x * s_w1 / s_h`. That real factor is approximated
as a fixed-point multiplier and shift:

```
M1 / 2^S1  ~=  s_x * s_w1 / s_h        M1 in [2^14, 2^15), S1 as large as that allows

h_q[n] = clamp( (acc1[n] * M1 + 2^(S1-1)) >> S1 , 0, 127 )
```

- `>>` is an **arithmetic** right shift, which is floor division by `2^S1`. Adding
  `2^(S1-1)` first makes the whole step exactly `floor(v + 0.5)` from section 4. The
  datapath contains no divider.
- The lower clamp at 0 is the ReLU. The upper clamp at 127 saturates rare values above
  the calibration maximum.
- **Hardware shortcut (allowed):** if `acc1[n] <= 0`, then `h_q[n] = 0`, and the
  multiply can be skipped. This gives identical results. A non-positive `acc1` can
  never round up to a positive value, because `acc1*M1 + 2^(S1-1) < 2^S1` whenever
  `acc1 <= 0`.
- **Product width:** with `0 < acc1 < 2^26` and `M1 < 2^15`, the product plus the
  rounding constant is under 2^42. The intermediate must be **at least 42 bits
  unsigned** (43 signed). This is 64 multiplies per inference and needs one wide
  multiplier, not an 18x18 DSP.
- **Calibration:** `s_h = max(float ReLU output over the train split) / 127`. It uses
  the train split only.

Layer 2 has no requantization. `acc2` is used directly in section 7.

## 7. Output

```
pred = argmax(acc2[0], acc2[1], acc2[2])
```

The comparison is signed. **Ties go to the lowest index.** A strictly-greater-than
comparison scanning from index 0 upward implements this. `pred` is a 2-bit value in
{0, 1, 2}, with meanings as in section 1.

## 8. Verification

- **Golden model:** `infer_int()` in `sw/python/quantize.py` is the reference
  implementation of sections 3–7. It uses only integer arithmetic.
- **Criterion:** bit-exact. For every test vector, the RTL and the ARM baseline must
  match the golden model on `pred`. The testbench should also compare `acc1`, `h_q`
  and `acc2`, so a mismatch is located to a layer. There is no tolerance: an
  off-by-one in any value is a failure.
- **Keying:** every test-vector set records the sha256 of `fer3.npz` (section 1) and
  of the integer weight data printed by `quantize.py`. Vectors generated from different
  weights are not valid against this build.
- **Float agreement is informational only.** The int8 model is expected to track the
  float model closely, but it is not required to match it. Report the agreement
  rate; do not tune the contract to it.

Current build (regenerated by `train.py` + `quantize.py`, not frozen by this document):

| Item                     | Value |
|--------------------------|-------|
| `M1`, `S1`               | 30446, 28 |
| int weight data sha256   | `f1c6b8841e66e8dcac6067a555d03d4c64da9caec6737564c05c4f52ca21c9b2` |
| val acc, float / int8    | 67.47% / 67.63% (agreement 99.20%) |
| test acc, float / int8   | 68.51% / 68.59% (agreement 99.28%) |

## 9. Open items and change log

Open:

- **Precision variants.** `model_budget.py` also sizes a 16-bit weight configuration.
  Each variant in the precision sweep needs its own version of section 4, a recomputed
  section 5, and its own golden model. Until then, only int8 is contracted.
- **Per-channel weight scales** (one scale per neuron) would likely improve accuracy
  but would need one `M1` per neuron. This is not adopted.

Change log:

| Version | Change |
|---------|--------|
| v1      | Initial contract: int8 weights/activations, int32 accumulate, single-multiplier requantize, lowest-index tie-break. |
