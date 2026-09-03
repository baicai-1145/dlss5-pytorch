# Phase 6.5 — Task 7 Report (enc3 fix + tail gain + full-frame verification)

## 7a — COMBINED-C solidified into `semantic_fill.py`
- `B14_MXSCALE_BYTES = 1024`: b14 (merge2) misc no longer leaks the MX-scale
  table into `merges.2.norm.{weight,bias}`; forced canonical gamma=1 / beta=0.
- enc3 (b15-21) `mlp.2.weight` zero-padded tail refilled with Kaiming
  (std=√(1/fan_in)); scoped **only** to `enc.3.` prefix (first unscoped
  attempt broke enc2 std 1.556→4.011 and was reverted).
- enc3 norm fall-through now leaves params in `unfilled` so the global
  LN-canonical fallback fills gamma=1/beta=0 (was: zero-filled → near-identity).
- Regression: order_scan C00 enc3 std **21.738 → 3.316** (matches COMBINED-C);
  all 12 candidates now healthy; C03 still best (bn_avg 65.7 vs target 88).

## 7b — tail gain calibration (`tail_gain_calib.py`)
**Key finding**: the tail head emits a **residual delta**, not a frame estimate.
The v1 chain fed `res + bias` into `resolve()` as a frame estimate; its
negative-luma fallback (`model_luma ≤ 1e-5 → upgraded = original`, 99.97%
of pixels after bias) silently discarded the model output — the structural
cause of full-frame PSNR being stuck at pass-through level.

Correct chain (consistent with Phase 5 "out = in + tail"):
```
final = clip(proxy + g * (res + bias), 0, 1)
```
Sweep on 3 standard crops (judge = official `after`):

| g    | final_std | corr   | PSNR (dB) |
|------|-----------|--------|-----------|
| 0.50 | 0.1674    | +0.9613| 16.92     |
| 0.70 | 0.1874    | +0.9545| 19.46     |
| **0.80** | **0.1935** | **+0.9507** | **20.03** |
| 0.85 | 0.1966    | +0.9488| 20.02     |
| 1.00 | 0.2058    | +0.9429| 19.00     |
| 1.60 | 0.2234    | +0.9185| 12.63     |

**Adopted g\* = 0.8** (peak of the curve; 1.6 is NOT optimal in the composed
domain — the raw-domain std-matching heuristic was misleading).
Written to `phase4/.tmp/amp_calib_v2.json` (frozen bias unchanged from v1).

## 7c — full-frame verification (`final_infer.py`, cap2_live frame 0)
200 crops (10×20, 96×96 stride 96), covered region 960×1920:

| metric | pass-through | replica v2 |
|--------|--------------|------------|
| PSNR   | 7.45 dB      | **10.55 dB (+3.10)** |
| corr   | +0.330       | +0.400     |

Visualization: `phase4/.tmp/final_vis_v2.png`.

## Honest gap analysis (why 10.55 ≠ 20)
1. **dec1 collapse (pre-existing)**: `calib_model.py` truncates the encoder
   skip at `x = x[:, :lo]` (fuse conv disabled) — skips contribute nothing.
   Input差异 vanishes at dec0→dec1 (max|Δ| 6.36→0.051); dec4→tail output is
   **input-independent** (std clamped 0.1640, mean −0.1826 on every crop,
   zeroing the whole input changes output by 6.6e-4). Per-crop residual corr
   across 100 sampled crops: mean +0.334 ± 0.482 (35/100 < 0.3) — the 3
   calibration crops were favourable outliers.
2. **expand norms carry MX-scale tables** (b48/56/62/66 misc mean −1.2..−4.3,
   healthy reference b22 = +0.839) and norm.weight ≡ norm.bias (same slice).
   Not fixed here (Task 7 scope = 7a only); next-task candidate together with
   the dec1 skip path.

## Artifacts
- `phase4/tail_gain_calib.py`, `phase4/.tmp/amp_calib_v2.json`
- `phase4/final_infer.py` (v2 calibration + additive-residual composition)
- `phase4/.tmp/final_frame0.npy`, `phase4/.tmp/final_vis_v2.png`
- Commit: `fa9ec33` "Phase 6.5: enc3 root-cause fix (b14 MX-scale leak into LN
  gamma) + tail gain calibration"
