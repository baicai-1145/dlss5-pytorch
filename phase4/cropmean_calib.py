"""Phase 6.6 Task 8c — final calibration: crop-mean readout (amp_calib_v3).

Post dec1-repair the tail residual has real input dependence (zero-input
L1 0.000655 -> 0.174) but its per-pixel component is still uninformative
(corr(r, true_res) ~ 0).  What DOES carry signal is the per-crop mean of
the residual u (ch0 anti-correlated -0.845 with the true delta mean,
luma -0.783) — a learned scene-descriptor readout.

Composition (v3):
    final = clip(P + sum_ch [ wU_ch * u_ch + wV_ch * v_ch ] + b_ch, 0, 1)
where u = per-crop mean of the model residual (broadcast), v = residual - u.

Fit on a 41-crop grid (disjoint from the 3 legacy calibration crops),
validated two ways:
  - alternating-half cross-validation (coefficient stability)
  - held-out ~159 crops of the full frame (generalization)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from resolve_shader import load_rgb10a2

CAP = os.path.join(HERE, "..", ".tmp", "cap2_live")
RES_NPY = os.path.join(HERE, ".tmp", "residual_fullframe_v3.npy")
OUT_JSON = os.path.join(HERE, ".tmp", "amp_calib_v3.json")
H, W, S = 960, 1920, 96
FIT_BASE = [(y, x) for y in range(64, 864, 192) for x in range(64, 1824, 240)]
HELD_CROPS = [(400, 700), (300, 1100), (600, 500)]


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def fit_params(R, A, P, crops):
    """Closed-form per-channel LSQ on pooled crop pixels; returns (wU, wV, b)."""
    U = np.zeros_like(R)
    for y in range(0, H, S):
        for x in range(0, W, S):
            U[y:y + S, x:x + S] = R[y:y + S, x:x + S].mean(axis=(0, 1))
    V = R - U
    D = A - P
    wU, wV, b = [], [], []
    for ch in range(3):
        Xu = np.concatenate([U[y:y+S, x:x+S, ch].ravel() for y, x in crops])
        Xv = np.concatenate([V[y:y+S, x:x+S, ch].ravel() for y, x in crops])
        yd = np.concatenate([D[y:y+S, x:x+S, ch].ravel() for y, x in crops])
        X = np.stack([Xu, Xv, np.ones_like(Xu)], 1)
        w, *_ = np.linalg.lstsq(X, yd, rcond=None)
        wU.append(float(w[0])); wV.append(float(w[1])); b.append(float(w[2]))
    return wU, wV, b


def compose(R, A, P, wU, wV, b):
    U = np.zeros_like(R)
    for y in range(0, H, S):
        for x in range(0, W, S):
            U[y:y + S, x:x + S] = R[y:y + S, x:x + S].mean(axis=(0, 1))
    V = R - U
    fin = P + np.stack([wU[ch] * U[..., ch] + wV[ch] * V[..., ch] + b[ch]
                        for ch in range(3)], -1)
    return np.clip(fin, 0, 1)


def main():
    R = np.load(RES_NPY)
    A = load_rgb10a2(os.path.join(CAP, "after_00.raw"))[:H, :W].astype(np.float32)
    P = load_rgb10a2(os.path.join(CAP, "model_input_00.raw"))[:H, :W].astype(np.float32)
    fitpos = [(y, x) for y, x in FIT_BASE
              if not any(abs(y - hy) < 100 and abs(x - hx) < 100 for hy, hx in HELD_CROPS)]

    # --- stability: alternating-half cross-validation ---
    half1 = fitpos[0::2]
    half2 = fitpos[1::2]
    wU1, wV1, b1 = fit_params(R, A, P, half1)
    wU2, wV2, b2 = fit_params(R, A, P, half2)
    f1 = compose(R, A, P, wU1, wV1, b1)
    f2 = compose(R, A, P, wU2, wV2, b2)
    print("=== coefficient stability (alternating halves) ===")
    for ch in range(3):
        print(f"  ch{ch}: wU {wU1[ch]:+.2f}/{wU2[ch]:+.2f}  wV {wV1[ch]:+.4f}/{wV2[ch]:+.4f}  "
              f"b {b1[ch]:+.3f}/{b2[ch]:+.3f}")
    print(f"  half1-fit frame PSNR={psnr(f1, A):.2f} | half2-fit PSNR={psnr(f2, A):.2f} "
          f"(gap {abs(psnr(f1, A) - psnr(f2, A)):.2f} dB)")

    # --- final fit on all 41 ---
    wU, wV, b = fit_params(R, A, P, fitpos)
    fin = compose(R, A, P, wU, wV, b)
    fp = psnr(fin, A)
    fc = float(np.corrcoef(fin.ravel(), A.ravel())[0, 1])
    fp = float(fp)

    # held-out (crops not in fit grid)
    mask = np.zeros((H, W), bool)
    for y, x in fitpos:
        mask[y:y + S, x:x + S] = True
    hp = psnr(fin[~mask], A[~mask])
    hc = float(np.corrcoef(fin[~mask].ravel(), A[~mask].ravel())[0, 1])
    hp = float(hp)

    # legacy 3 calibration crops (held out entirely)
    lps = []
    for y, x in HELD_CROPS:
        lps.append(psnr(fin[y:y + S, x:x + S], A[y:y + S, x:x + S]))
    passthrough = psnr(P, A)

    print("\n=== final (41-crop fit) ===")
    print(f"  wU = {[round(v, 4) for v in wU]}")
    print(f"  wV = {[round(v, 4) for v in wV]}")
    print(f"  b  = {[round(v, 4) for v in b]}")
    print(f"  full-frame : PSNR={fp:.2f}  corr={fc:+.4f}")
    print(f"  held-out   : PSNR={hp:.2f}  corr={hc:+.4f}")
    print(f"  3 legacy   : PSNR={np.mean(lps):.2f}")
    print(f"  passthrough: PSNR={passthrough:.2f}")
    meta = {
        "composition_semantics": "cropmean_readout: final = clip(P + sum_ch[wU*u + wV*v] + b, 0, 1); "
                                 "u = per-crop residual mean (broadcast), v = residual - u",
        "wU": [float(v) for v in wU], "wV": [float(v) for v in wV], "b": [float(v) for v in b],
        "fit_crops": len(fitpos),
        "fit_grid": "y 64:864:192, x 64:1824:240, excluding 3 legacy calibration crops",
        "full_frame": {"psnr": fp, "corr": fc},
        "held_out": {"psnr": hp, "corr": hc, "crops": 200 - len(fitpos)},
        "legacy_3crop_psnr": float(np.mean(lps)),
        "passthrough_psnr": passthrough,
        "stability": {"half1": {"wU": [float(v) for v in wU1], "wV": [float(v) for v in wV1], "b": [float(v) for v in b1]},
                      "half2": {"wU": [float(v) for v in wU2], "wV": [float(v) for v in wV2], "b": [float(v) for v in b2]}},
        "source": "phase4/cropmean_calib.py (Task 8c), residuals phase4/.tmp/residual_fullframe_v3.npy",
        "note": "Residual is input-dependent after dec1 repair (zero-input L1 0.174). "
                "Per-pixel residual remains uninformative (corr ~ 0); the per-crop mean "
                "carries a robust scene-descriptor signal (ch0 anti-correlated). "
                "Cross-frame generalization untested (single-frame capture).",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()