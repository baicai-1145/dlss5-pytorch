"""Phase 6.6 Task 8 — dec1 collapse repair verification.

Checks:
  1. per-stage activation std (enc/dec chain) vs pre-fix baseline
  2. tail input sensitivity: zeroed-input vs normal input, L1 diff on
     tail output (pre-fix: 6.6e-4; target > 0.01)
  3. 3-crop composed PSNR/corr vs official after (additive-residual chain)
  4. 100-crop residual corr distribution (pre-fix: mean +0.334, median < 0.5)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "phase3"))
sys.path.insert(0, os.path.join(HERE, "..", "phase3", "dlss5"))

from dlss5.calib_model import DLSS5NetCalib
from semantic_fill import load_all, fill_model
from final_infer import load_depth_raw, load_motion_raw
from resolve_shader import load_rgb10a2

CAP = os.path.join(HERE, "..", ".tmp", "cap2_live")
BLOB = os.path.join(HERE, "..", "weights_blob.bin")


def psnr(a, b):
    return 10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12))


def main():
    bef = load_rgb10a2(os.path.join(CAP, "model_input_00.raw"))
    aft = load_rgb10a2(os.path.join(CAP, "after_00.raw"))
    dep = load_depth_raw("depth_00.raw")
    mv = load_motion_raw("motion_00.raw")

    torch.manual_seed(42)
    m = DLSS5NetCalib().eval()
    fill_model(m, load_all(), blob_full=open(BLOB, "rb").read())

    stats = {}
    hooks = []
    names = {0: "dec0", 1: "dec1", 2: "dec2", 3: "dec3", 4: "dec4"}
    for i, st in enumerate(m.dec):
        hooks.append(st.register_forward_hook(
            lambda mod, i, o, k=names[i]: stats.__setitem__(k, o.detach().float())))
    hooks.append(m.tail.register_forward_hook(
        lambda mod, i, o: stats.__setitem__("tail", o.detach().float())))

    def fwd(y0, x0, zero=False):
        c = lambda a: a[y0:y0 + 96, x0:x0 + 96]
        frame = bef.copy() if zero else bef
        if zero:
            frame[y0:y0 + 96, x0:x0 + 96] = 0.0
        rgb = c(frame).transpose(2, 0, 1)[None].astype(np.float32)
        d = c(dep)[None, None].astype(np.float32)
        v = c(mv).transpose(2, 0, 1)[None].astype(np.float32)
        dmed = float(np.median(d))
        dn = np.clip(d / max(dmed, 1e-6), 0, 4) / 4.0
        with torch.no_grad():
            return m(torch.from_numpy(rgb.copy()).float(),
                     torch.from_numpy(dn.astype(np.float32)),
                     torch.from_numpy((v * 0.02).astype(np.float32)),
                     torch.from_numpy(rgb.copy()).float())[0].numpy().astype(np.float32)

    # ---- 1. stage std profile (crop 400,700) ----
    print("=== stage std (crop 400,700) ===  [pre-fix: dec0 77.8, dec1 1.25, dec4 0.040, tail 0.164]")
    fwd(400, 700)
    for k in ("dec0", "dec1", "dec2", "dec3", "dec4", "tail"):
        if k in stats:
            print(f"  {k:5s} std={stats[k].std():.4f}")

    # ---- 2. tail input sensitivity (crop 400,700; whole-input zeroed) ----
    r1 = fwd(400, 700)
    r2 = fwd(400, 700, zero=True)
    l1 = float(np.abs(r1 - r2).mean())
    print(f"\n=== tail input sensitivity ===")
    print(f"  zeroed-whole-input L1 diff = {l1:.6f}   (pre-fix 0.000655, target > 0.01)")

    # ---- 3. 3-crop composed metrics (additive-residual, g from amp_calib_v2) ----
    import json
    with open(os.path.join(HERE, ".tmp", "amp_calib_v2.json")) as f:
        meta = json.load(f)
    bias = np.array(meta["frozen_bias_only"], dtype=np.float32)
    g = float(meta["tail_gain"])
    print(f"\n=== 3-crop composed (g={g}) ===  [pre-fix: PSNR 20.03 @g=0.8]")
    for y0, x0 in [(400, 700), (300, 1100), (600, 500)]:
        res = fwd(y0, x0)
        c = lambda a: a[y0:y0 + 96, x0:x0 + 96]
        proxy = c(bef).astype(np.float32)
        final = np.clip(proxy + g * (np.transpose(res, (1, 2, 0)) + bias.reshape(1, 1, 3)), 0, 1)
        print(f"  crop({y0},{x0}): PSNR={psnr(final, c(aft).astype(np.float32)):6.2f}  "
              f"res_std={res.std():.4f}")

    # ---- 4. 100-crop residual corr distribution ----
    print("\n=== 100-crop residual corr ===  [pre-fix: mean +0.334, 35/100 < 0.3]")
    cors = []
    for y in range(0, 960 - 95, 96):
        for x in range(0, 1920 - 95, 192):
            res = fwd(y, x)
            delta = np.transpose(res, (1, 2, 0)) + bias.reshape(1, 1, 3)
            off = aft[y:y + 96, x:x + 96] - bef[y:y + 96, x:x + 96]
            cors.append(float(np.corrcoef(delta.ravel(), off.ravel())[0, 1]))
    cors = np.array(cors)
    print(f"  n={len(cors)}  mean={cors.mean():+.3f}  median={np.median(cors):+.3f}  "
          f"min={cors.min():+.3f}  max={cors.max():+.3f}  frac<0.3={float((cors < 0.3).mean()):.2f}")


if __name__ == "__main__":
    main()