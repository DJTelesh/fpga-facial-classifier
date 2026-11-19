# Arithmetic Contract

This document defines the exact integer arithmetic of the inference accelerator. The
Python golden model (`sw/python/quantize.py`, `infer_int()`), the ARM Cortex-A9
software baseline, and the RTL must all implement **this** arithmetic and produce
**bit-identical** results. When two implementations disagree, whichever one departs
from this document is wrong.

Changing anything in this document changes the contract. Invalidate and regenerate
every file keyed to it (weights, test vectors, measured results), and record the change
in section 9.

**Status:** v2, parameterized by `DATA_WIDTH` (written `b` below). The hardware
configurations are **b = 8** (baseline) and **b = 16**. Widths 2–7 and 12 exist only in
the golden model, for the accuracy curve. `quantize.py --bits b` builds any of them.

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

Pixels are always 8 bits, because that is what the camera and dataset provide.

```
x_q[i] = (pixel[i] - 128) >> in_shift        in_shift = max(0, 8 - b)
s_x    = 2^in_shift / 128
```

- **b >= 8:** `in_shift = 0`, so `x_q = pixel - 128` in [-128, 127], sign-extended to
  b bits. `x_q * s_x` equals the float input `(pixel - 128) / 128` used in training
  **exactly**, with no rounding. A wider input would add no information.
- **b < 8:** the arithmetic shift keeps the top b bits of the pixel. This is
  **truncation**, a deliberate exception to the rounding rule in section 4: it is free
  in hardware (just wiring), whereas rounding would need saturation (255 would round up
  out of range).
- In hardware, `pixel - 128` is the same as inverting the pixel's MSB and reading the
  byte as two's complement. No subtractor is needed.
- The input is the only value that can reach `-2^(m-1)` (with `m = min(b, 8)`). Every
  other value in the datapath is limited to `+/-W_MAX`.

## 4. Number formats and rounding

All integers are two's complement. Everything below is derived from `b`:

```
W_MAX     = 2^(b-1) - 1                  (127 at b = 8, 32767 at b = 16)
ACC_WIDTH = 32 if b <= 8, 48 if 9 <= b <= 16
```

| Quantity                   | Type           | Range                          | b = 8         | b = 16            |
|----------------------------|----------------|--------------------------------|---------------|-------------------|
| Input `x_q`                | int b          | section 3                      | [-128, 127]   | [-128, 127]       |
| Weights `W1_q`, `W2_q`     | int b          | [-W_MAX, W_MAX] (clamped)      | +/-127        | +/-32767          |
| Biases `b1_q`, `b2_q`      | int ACC_WIDTH  | abs(b) < 2^(ACC_WIDTH-8)       | < 2^24        | < 2^40            |
| Accumulators `acc1`, `acc2`| int ACC_WIDTH  | see section 5                  | int32         | int48             |
| Hidden activations `h_q`   | int b          | [0, W_MAX]                     | [0, 127]      | [0, 32767]        |
| Requantize multiplier `M1` | uint15         | [2^14, 2^15)                   |               |                   |
| Requantize shift `S1`      | uint6          | [1, 40]                        |               |                   |

**Rounding.** Every float-to-integer conversion, and every requantization, uses
round-half-up:

```
round(v) = floor(v + 0.5)
```

Do **not** use round-half-to-even (`np.round`, C `rint`, IEEE default). The two differ
on exact .5 cases and would make the golden model disagree with the RTL.

**Weight quantization** is symmetric, per tensor (one scale per layer):

```
s_w  = max(abs(W)) / W_MAX
W_q  = clamp(round(W / s_w), -W_MAX, +W_MAX)
```

Weights clamp to `+/-W_MAX`, not `-2^(b-1)`, so the range is symmetric and `-W_q` never
overflows.

**Bias quantization.** A bias is added directly into the accumulator, so it uses the
accumulator's scale:

```
b1_q = round(b1 / (s_x  * s_w1))
b2_q = round(b2 / (s_h  * s_w2))
```

The quantizer must reject any bias with abs(b_q) >= 2^(ACC_WIDTH-8). Section 5 depends
on that limit.

## 5. Accumulator bound

A dense layer with `N` inputs has worst case

```
abs(acc) <= N * max(abs(x)) * max(abs(w)) + max(abs(b))
```

| b  | Layer | N    | max abs(x) | max abs(w) | Product term   | + bias limit | Total bound            | Bits (signed) | ACC_WIDTH |
|----|-------|------|------------|------------|----------------|--------------|------------------------|---------------|-----------|
| 8  | 1     | 2304 | 128        | 127        | 37,453,824     | 2^24         | ≤ 54,231,040           | 27            | 32        |
| 8  | 2     | 64   | 127        | 127        | 1,032,256      | 2^24         | ≤ 17,809,472           | 26            | 32        |
| 16 | 1     | 2304 | 128        | 32767      | 9,663,381,504  | 2^40         | ≤ 1,109,175,009,280    | 42            | 48        |
| 16 | 2     | 64   | 32767      | 32767      | 68,715,282,496 | 2^40         | ≤ 1,168,226,910,272    | 42            | 48        |

At b = 8 the worst case needs 27 bits signed, so int32 has a margin of about 40x. At
b = 16 it needs 42 bits, so int48 has a margin of about 120x. **int32 is not enough at
b = 16, and this is not just theoretical:** on real validation images, layer 2 reaches
about 2.5 x 10^9, past int32's 2.1 x 10^9.

Consequences:

- **Overflow is structurally impossible.** Accumulators need no saturation logic and no
  overflow detection.
- **Addition order is irrelevant.** Two's complement addition that never overflows is
  associative, so any MAC-array width, tree shape, or partial-sum order gives the same
  answer bit for bit. The parallelism sweep may split the sum across any number of
  lanes.
- The golden model asserts that no real accumulator exceeds ACC_WIDTH, and the
  quantizer refuses any width whose worst case does not fit. On the current weights the
  largest observed values are about 1.4M (layer 1) and 38K (layer 2) at b = 8, and
  about 358M and 2.5G at b = 16.

`quantize.py --bits b` prints this table for any width. If the topology or input width
ever changes, recompute it before touching anything else. (`model_budget.py`'s
accumulator section still assumes b = 8.)

## 6. Hidden-layer requantization (ReLU + rescale)

`acc1` has scale `s_x * s_w1`. Layer 2 needs `h_q` in [0, W_MAX] with scale `s_h`, so
each hidden value is multiplied by `s_x * s_w1 / s_h`. That real factor is approximated
as a fixed-point multiplier and shift:

```
M1 / 2^S1  ~=  s_x * s_w1 / s_h        M1 in [2^14, 2^15), S1 as large as that allows

h_q[n] = clamp( (acc1[n] * M1 + 2^(S1-1)) >> S1 , 0, W_MAX )
```

- `>>` is an **arithmetic** right shift, which is floor division by `2^S1`. Adding
  `2^(S1-1)` first makes the whole step exactly `floor(v + 0.5)` from section 4. The
  datapath contains no divider.
- The lower clamp at 0 is the ReLU. The upper clamp at W_MAX saturates rare values
  above the calibration maximum.
- **Hardware shortcut (allowed):** if `acc1[n] <= 0`, then `h_q[n] = 0`, and the
  multiply can be skipped. This gives identical results. A non-positive `acc1` can
  never round up to a positive value, because `acc1*M1 + 2^(S1-1) < 2^S1` whenever
  `acc1 <= 0`.
- **Product width:** the product plus the rounding constant needs 15 more bits than
  the positive part of `acc1`. At b = 8, `0 < acc1 < 2^26`, so the intermediate must be
  **at least 42 bits unsigned** (43 signed). At b = 16, `0 < acc1 < 2^41`, so it must be
  **at least 57 bits unsigned**. Either way it is only 64 multiplies per inference, done
  by one wide multiplier, not an 18x18 DSP.
- **Calibration:** `s_h = max(float ReLU output over the train split) / W_MAX`. It uses
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
- **Float agreement is informational only.** The integer model is expected to track the
  float model closely, but it is not required to match it. Report the agreement
  rate; do not tune the contract to it.
- **Each width has its own weights and vectors.** int8: `weights/fer3_int8.npz` and
  `test_vectors/`. int16: `weights/fer3_int16.npz` and `test_vectors/int16/`. Each
  `manifest.txt` records `DATA_WIDTH` and `ACC_WIDTH`.

Current build (regenerated by `train.py` + `quantize.py`, not frozen by this document):

| Item                          | b = 8 | b = 16 |
|-------------------------------|-------|--------|
| `M1`, `S1`                    | 30446, 28 | 30446, 28 |
| int weight data sha256        | `f1c6b8841e66e8dcac6067a555d03d4c64da9caec6737564c05c4f52ca21c9b2` | `5e3b3d0c47b7b4e66af3a81c6fca21173a49ba7f5fce6868b3a95fb5172d12fc` |
| val acc, float / int (agree)  | 67.47% / 67.63% (99.20%) | 67.47% / 67.47% (100%) |
| test acc, float / int (agree) | 68.51% / 68.59% (99.28%) | 68.51% / 68.51% (100%) |

## 9. Open items and change log

Open:

- **Only b = 8 and b = 16 are hardware configurations.** The other widths are defined
  so the golden model can measure accuracy, but no RTL or test vectors target them.
- **Per-channel weight scales** (one scale per neuron) would likely improve accuracy
  but would need one `M1` per neuron. This is not adopted.

Change log:

| Version | Change |
|---------|--------|
| v1      | Initial contract: int8 weights/activations, int32 accumulate, single-multiplier requantize, lowest-index tie-break. |
| v2      | Parameterized by `DATA_WIDTH` b (2–16): `W_MAX`, `ACC_WIDTH` (32 / 48), bias limit and input shift derived from b; int16 added as a hardware configuration. **b = 8 is bit-identical to v1** (same weight hash, same vectors), so nothing keyed to v1 is invalidated. |
