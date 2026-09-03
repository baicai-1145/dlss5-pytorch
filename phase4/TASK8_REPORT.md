# Phase 6.6 — Task 8 Report (dec1 collapse repair + expands norm zone fix + final calibration)

## 8a — dec1 skip/fuse repair (fuse weights DO exist in the blob)

**Root cause chain** (evidence-based, byte-exact):
1. `calib_model.py` truncated the concat at `x = x[:, :lo]` with the fuse conv
   disabled (`if False`) — the skip was discarded entirely.
2. The expand LayerNorm received garbage (misc slices with mean −1.2..−8.9,
   `norm.weight ≡ norm.bias` — same slice consumed twice, fill bug).
3. After dec1 the model output became **input-independent** (zeroing the
   entire input changed tail output by 6.6e-4 L1).

**Formula decode** (`phase1/BLOB_FORMAT.md`: `up = swin(c) + 2c² + γc`,
cubin suffix `_upsample`): the up records carry a **2c² fuse GEMM** over
`concat([up(x), skip])` plus a γc fp16 bias. Verified in-blob:

| block | fuse E4 zone | std | fuse bias | n | mean |
|---|---|---|---|---|---|
| b48 (c=256) | [360448:491520] | 0.0418 | [820224:820736] | 256 | +0.784 |
| b56 (c=128) | [98304:131072] | 0.0540 | [229888:230112] | 112 | +0.727 |
| b62 (c=64) | [66048:69632] **3584/8192 vals (incomplete)** | — | [69632:69728] | 48 | ≈0 |
| b66 (c=32) | [11264:13312] | 0.1771 | [22716:22780] | 32 | +0.719 |

bias val counts (240/112/48/32) match the `γc` B-field residuals exactly
(480/224/96/64 bytes). b62's fuse is truncated in the record → Kaiming
fallback for the missing tail.

**The expand GEMM (c→4c) has no blob bytes** (4c² does not exist in `up`).
Structural placeholder, Kaiming-filled; the trained fuse carries the signal.

**Implementation**:
- `patch_ops.py`: new `UpFuse(in_dim, out_dim)` — expand → pixel_shuffle →
  concat skip → `fuse = Linear(2c→c, bias)` (channels-last GEMM).
- `calib_model.py`: `self.expands` are now `UpFuse`; forward passes the skip;
  legacy `x[:, :lo]` truncation removed.
- `semantic_fill.py`: direct-from-blob fuse loading (UP_ZONES per block),
  incomplete-fuse → Kaiming; expand GEMM → Kaiming; expand LN → canonical
  γ=1/β=0 (blob source region not locatable; same precedent as b14/enc3).

**Verification** (3-crop):
- tail input sensitivity (zeroed whole input): **0.000655 → 0.174 L1** (×266)
- dec1 std 1.25 → 2.42; dec4 0.040 → 1.19 (collapse gone); tail std 0.164 → 0.40
- order_scan regression: all 12 candidates enc3 = 3.609 (no 7a regression);
  C03 still best (bn_avg 65.6 vs target 88)

## 8b — expands norm zone fix
The old fill consumed the same misc slice for `norm.weight` and `norm.bias`
and those slices were MX-scale tables (b62/b66) or inner-swin γ1/β1 (b48/b56).
Now canonical γ=1/β=0 via the fallback; the `≡` duplication bug is gone
(UpFuse loading no longer touches misc at all).

## 8c — final calibration: crop-mean readout (`amp_calib_v3.json`)

Honest finding: after repair the residual is genuinely input-dependent, but
its **per-pixel component is still uninformative** (corr(r_pixel, true_res) ≈
0, ch0 −0.061). The signal lives at the **per-crop mean** level:
corr(u_ch0, true_delta_ch0) = **−0.845**, luma −0.783 — an anti-correlated
scene-descriptor readout, stable across crops (u_ch0 range [0.138, 0.245]
vs true delta mean range [−0.886, −0.039]).

Composition: `final = clip(P + Σ_ch [wU·u + wV·v] + b, 0, 1)` with u = per-crop
residual mean (broadcast), v = residual − u. Fit on a 41-crop grid disjoint
from the 3 legacy calibration crops.

| metric | pass-through | task 7 (v2) | task 8 (v3) |
|---|---|---|---|
| full-frame PSNR | 7.45 dB | 10.55 dB | **11.96 dB** |
| full-frame corr | +0.330 | +0.400 | **+0.573** |
| held-out (159 crops, unseen in fit) | — | — | **12.03 dB / +0.579** |
| legacy 3 calibration crops | 11.13 dB | 20.03 dB | 20.03 dB |

- Stability: alternating-half cross-validation — frame PSNR gap 0.03 dB.
  (ch1 wU/b are collinear-ill-conditioned: u_ch1 is near-constant; the fitted
  *combination* wU·u+b is stable, the individual coefficients are not.)
- The 3 legacy crops still read 20.03 dB (they are favourable content);
  the honest full-frame number is 11.96 dB.
- Cross-frame generalization untested (single-frame capture).

## Artifacts
- `phase3/dlss5/patch_ops.py` (UpFuse), `phase3/dlss5/calib_model.py` (skip fuse)
- `phase4/semantic_fill.py` (UP_ZONES fuse loading)
- `phase4/dec1_repair_verify.py`, `phase4/cropmean_calib.py`
- `phase4/.tmp/amp_calib_v3.json`, `phase4/.tmp/residual_fullframe_v3.npy`
- `phase4/.tmp/final_vis_v3.png` (4-up)
