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