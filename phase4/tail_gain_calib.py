"""Phase 6.5 Task 7b — tail output gain calibration (additive-residual chain).

Key finding (7b): the tail head emits a RESIDUAL delta, not a frame estimate.
The previous composition path fed `model_est = res + bias` straight into
resolve(), where the negative luma fallback (`model_luma <= 1e-5 ->
upgraded = original`) silently discarded the model output — this is why
full-frame PSNR was stuck at the pass-through level (7.45 dB).

Correct semantics (confirmed by sweep below and consistent with the
Phase 5 commit "residual semantics: out = in + tail"):

    final = clip(proxy + g * (res + bias), 0, 1)

Sweep g ∈ {0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.6, 2.0} on 3 standard crops
against the official `after` frame. Writes amp_calib_v2.json with the
adopted gain and composition semantics.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "phase3"))

from phase3.dlss5.calib_model import DLSS5NetCalib
from phase4.semantic_fill import load_all, fill_model
from phase4.resolve_shader import load_rgb10a2
from phase4.final_infer import load_depth_raw, load_motion_raw, AMP_JSON

CAP_DIR = os.path.join(HERE, "..", ".tmp", "cap2_live")
BLOB = os.path.join(HERE, "..", "weights_blob.bin")
OUT_JSON = os.path.join(HERE, ".tmp", "amp_calib_v2.json")
CROPS = [(400, 700), (300, 1100), (600, 500)]
SIZE = 96
GAIN_GRID = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.6, 2.0]


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(peak * peak / max(mse, 1e-12))


def main():
    print("=== 7b: tail gain calibration (additive-residual chain) ===\n")

    bef = load_rgb10a2(os.path.join(CAP_DIR, "model_input_00.raw"))
    aft = load_rgb10a2(os.path.join(CAP_DIR, "after_00.raw"))
    dep = load_depth_raw("depth_00.raw")
    mv = load_motion_raw("motion_00.raw")

    with open(AMP_JSON) as f:
        bias = np.array(json.load(f)["frozen_bias_only"], dtype=np.float32)
    print(f"[calib] frozen bias from v1: {[round(x, 4) for x in bias.tolist()]}")

    torch.manual_seed(42)
    model = DLSS5NetCalib().eval()
    fill_model(model, load_all(), blob_full=open(BLOB, "rb").read())

    raws, proxies, afters = [], [], []
    for y0, x0 in CROPS:
        c = lambda a: a[y0:y0 + SIZE, x0:x0 + SIZE]
        rgb = c(bef).transpose(2, 0, 1)[None].astype(np.float32)
        d = c(dep)[None, None].astype(np.float32)
        v = c(mv).transpose(2, 0, 1)[None].astype(np.float32)
        dmed = float(np.median(d))
        dn = np.clip(d / max(dmed, 1e-6), 0, 4) / 4.0
        with torch.no_grad():
            res = model(
                torch.from_numpy(rgb.copy()).float(),
                torch.from_numpy(dn.astype(np.float32)),
                torch.from_numpy((v * 0.02).astype(np.float32)),
                torch.from_numpy(rgb.copy()).float(),
            )[0].numpy().astype(np.float32)
        raws.append(np.transpose(res, (1, 2, 0)))     # (S,S,3) residual delta
        proxies.append(c(bef).astype(np.float32))
        afters.append(c(aft).astype(np.float32))

    # ---- sweep g in the additive-residual domain ----
    print(f"\n{'g':>5s}  {'final_std':>9s}  {'corr':>7s}  {'PSNR':>7s}")
    results = []
    for g in GAIN_GRID:
        stds, corrs, psnrs = [], [], []
        for res, proxy, after in zip(raws, proxies, afters):
            final = np.clip(proxy + g * (res + bias.reshape(1, 1, 3)), 0.0, 1.0)
            stds.append(float(final.std()))
            corrs.append(float(np.corrcoef(final.ravel(), after.ravel())[0, 1]))
            psnrs.append(psnr(final, after))
        r = {"g": g, "final_std": float(np.mean(stds)),
             "corr": float(np.mean(corrs)), "psnr": float(np.mean(psnrs))}
        results.append(r)
        print(f"{g:5.2f}  {r['final_std']:9.4f}  {r['corr']:+7.4f}  {r['psnr']:7.2f}")

    # passthrough baseline
    base_psnr = float(np.mean([psnr(p, a) for p, a in zip(proxies, afters)]))
    base_corr = float(np.mean([np.corrcoef(p.ravel(), a.ravel())[0, 1]
                               for p, a in zip(proxies, afters)]))
    print(f"\npassthrough baseline: PSNR={base_psnr:.2f} dB  corr={base_corr:+.4f}")

    # Adopt best PSNR (tie-break: corr); require >0.1 dB over g=1.0 to move off 1.0
    best = max(results, key=lambda r: (r["psnr"], r["corr"]))
    g1 = next(r for r in results if r["g"] == 1.0)
    if best["g"] != 1.0 and best["psnr"] - g1["psnr"] <= 0.1:
        best = g1
    g_star = best["g"]
    print(f"\nADOPTED g* = {g_star}  (PSNR {best['psnr']:.2f} dB, corr {best['corr']:+.4f}; "
          f"vs passthrough {best['psnr'] - base_psnr:+.2f} dB)")

    meta = {
        "frozen_bias_only": bias.tolist(),
        "tail_gain": float(g_star),
        "composition_semantics": "additive_residual: final = clip(proxy + g*(res+bias), 0, 1)",
        "source": "phase4/tail_gain_calib.py (Task 7b)",
        "sweep": results,
        "passthrough_baseline": {"psnr": base_psnr, "corr": base_corr},
        "note": "Prior v1 composition fed res+bias into resolve() as a frame "
                "estimate; the negative-luma fallback discarded it (PSNR stuck "
                "at pass-through). The tail head outputs a residual delta; "
                "additive composition is the correct chain (also consistent "
                "with the Phase 5 'out = in + tail' finding).",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()