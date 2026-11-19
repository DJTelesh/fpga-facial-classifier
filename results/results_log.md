# Results log

One entry per measurement. Newest at the bottom. Never edit an old entry; add a new one.

---

## TODO date (week 2) — Float vs int8 accuracy
- Config: 2304→64→3, ReLU, Adam lr=0.001, best-val epoch kept. Weights sha256: TODO
- Train acc (float): 0.88
- Val acc: float 0.6747, int8 0.6763, prediction agreement 99.20%
- Caveats: val set is 1,245 images, so 0.16% = 2 images. Read as "no measurable loss
  from int8", not "int8 is better". Model overfits (train 0.88 vs val ~0.67).

## 2026-10-22 — MAC unit, directed tests
- Config: hw/rtl/mac_unit.sv, DATA_WIDTH=8, ACC_WIDTH=32. Icarus Verilog 13.0. Commit: TODO
- Result: 5/5 PASS (reset, hold, lb-requires-en, bias load incl. >8-bit bias,
  mixed-sign accumulate, −128 × −127)

## 2026-10-22 — MAC unit vs golden model
- Config: same commit. Vectors from gen_vectors.py, weights sha256: TODO (manifest.txt)
- Input: x_val[0]
- Result: 64/64 layer-1 accumulators bit-exact against infer_int()
- Max |acc1| seen: ~814,000 (contract bound ~54M)

## 2026-10-22 — Layer-1 latency, one MAC (SIMULATED)
- 64 neurons × (1 bias + 2304 MAC + 1 idle) = 147,584 cycles → 2.95 ms at 50 MHz
- Matches feasibility estimate (~338 inf/s at NUM_MACS=1)
- Caveats: testbench-driven, not the real FSM. Layer 1 only. Simulation, not hardware.

## 2026-10-22 — Quartus synthesis, mac_unit standalone
- Config: 5CSEMA5F31C6 (confirmed against chip label), Quartus Prime Lite TODO version,
  Analysis & Synthesis only, top-level = mac_unit. Commit: TODO
- Result: 1 DSP block, 32 registers, 84 pins
- Caveats: pins are an artifact of mac_unit being top-level (every port became a pin).
  Single multiplier, so says nothing yet about DSP packing with many MACs.

## 2026-11-19 — Accuracy vs DATA_WIDTH (golden model)
- Config: arithmetic contract v2, `quantize.py --sweep`. Float weights
  weights/fer3_float.npz (train.py, seed 0, best-val epoch). Per-tensor symmetric weight
  scales, s_h calibrated on the train max. For b < 8 the input is the top b bits of the
  pixel (truncation). ACC_WIDTH 32 for b ≤ 8, 48 above. Commit: 4a3f8ed
- Result:

  | bits | val acc | test acc | val agree | test agree |
  |------|---------|----------|-----------|------------|
  | float| 0.6747  | 0.6851   | —         | —          |
  | 2    | 0.3333  | 0.3333   | 0.3108    | 0.3317     |
  | 3    | 0.5807  | 0.5994   | 0.6916    | 0.6979     |
  | 4    | 0.6498  | 0.6651   | 0.8458    | 0.8550     |
  | 5    | 0.6723  | 0.6827   | 0.9309    | 0.9319     |
  | 6    | 0.6691  | 0.6883   | 0.9663    | 0.9599     |
  | 7    | 0.6715  | 0.6859   | 0.9823    | 0.9888     |
  | 8    | 0.6763  | 0.6859   | 0.9920    | 0.9928     |
  | 12   | 0.6747  | 0.6851   | 1.0000    | 1.0000     |
  | 16   | 0.6747  | 0.6851   | 1.0000    | 1.0000     |

  (agree = same prediction as the float model)
- Reading: accuracy holds down to ~5 bits, drops at 4, and is chance at 2 (predicts one
  class). Agreement with float falls steadily below 16 bits and is the sharper metric.
- Caveats: n = 1,245 (val) / 1,248 (test), so accuracy has ~±1.3 pt standard error; 5–8
  bits cannot be separated from float by accuracy alone. Golden model only: the hardware
  configurations are 8 and 16 bits. One trained model with per-tensor scales; per-channel
  scales would likely move the knee lower. Nothing was tuned on test.

## 2026-11-19 — MAC unit vs golden model, DATA_WIDTH 8 and 16
- Config: hw/rtl/mac_unit.sv identical for both widths (DATA_WIDTH/ACC_WIDTH = 8/32 and
  16/48, set with `iverilog -P` overrides on tb_mac_unit.sv). Icarus Verilog 14.0 (devel),
  OSS CAD Suite 2026-11-19. Vectors: test_vectors/ (int8, weights sha256 f1c6b884…) and
  test_vectors/int16/ (int16, weights sha256 5e3b3d0c…). Commit: 4a3f8ed
- Input: x_val[0:4]
- Result: 256/256 layer-1 accumulators bit-exact against infer_int() at BOTH widths.
  Directed tests 5/5 at both widths.
- Max |acc| over val + test (golden model): int8 acc1 1,388,056, acc2 37,670.
  int16 acc1 357,785,919, acc2 2,514,147,016. The int16 acc2 exceeds int32
  (2,147,483,647) on real data, so ACC_WIDTH 48 is required, not just a worst-case margin.
- Caveats: one MAC lane, layer 1 only, simulation. Not re-synthesized at 16 bits, so DSP
  use at 16 bits is still unmeasured.