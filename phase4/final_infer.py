"""End-to-end DLSS-NR replica pipeline: raw frame + depth + mvec
  -> PyTorch forward (96x96 sliding window, stride 96, no overlap)
  -> bias-only calibration (frozen from phase4/.tmp/amp_calib.json)
  -> resolve composition (forward implementation from phase4/resolve_shader.py)
  -> final HDR-linear frame.

Reads:
  - one R10G10B10A2_UNORM frame (raw uint32 or PNG, anything load_rgb10a2
    accepts)
  - one depth R11G11B10F frame + one motion RG16_SFLOAT frame

Writes:
  - phase4/.tmp/final_frame0.npy       (H, W, 3) float32 -- final output
  - phase4/.tmp/final_vis.png          4-up: input | official after |
                                               replica | diffx5

The bias-only calibration coefficients are loaded from amp_calib.json
(``frozen_bias_only``); if that file is missing we fall back to the
precomputed values reported in phase4/ALIGN_REPORT.md with a loud warning.

Memory discipline:
  * one replica model in memory, used by every crop
  * stitched output written via numpy.memmap so the full frame never lives
    in RAM as a contiguous block
  * crops are 96x96; 1050/96 ~= 11 rows, 1920/96 = 20 cols => 220 crops
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from phase3.dlss5.calib_model import DLSS5NetCalib
from phase4.semantic_fill import load_all, fill_model
from phase4.resolve_shader import load_rgb10a2, resolve

CAP_DIR = os.path.join(HERE, "..", ".tmp", "cap2_live")
BLOB = os.path.join(HERE, "..", "weights_blob.bin")
AMP_JSON = os.path.join(HERE, ".tmp", "amp_calib.json")
W, H = 1920, 1050
SIZE = 96
STRIDE = 96            # non-overlapping sliding window


# ---------------------------------------------------------------------------
# Calibration loader (never hard-code the bias vector in two places).
# ---------------------------------------------------------------------------

def load_bias_calibration() -> np.ndarray:
    if not os.path.exists(AMP_JSON):
        print(f"[warn] {AMP_JSON} missing -- falling back to hard-coded bias")
        return np.array([-0.2593214437365532,
                         -0.016486987471580505,
                         -0.07616311684250832], dtype=np.float32)
    with open(AMP_JSON) as f:
        meta = json.load(f)
    bias = np.array(meta["frozen_bias_only"], dtype=np.float32)
    print(f"[calib] loaded frozen bias from {AMP_JSON}: {bias.tolist()}")
    return bias


# ---------------------------------------------------------------------------
# Decoders for depth + motion (copied from phase4/align_official.py).
# ---------------------------------------------------------------------------

def load_depth_raw(name: str) -> np.ndarray:
    """R11G11B10F: R = 6bit mantissa + 5bit exponent (unsigned).
    Returns linear distance float32."""
    u = np.fromfile(os.path.join(CAP_DIR, name), dtype=np.uint32).reshape(H, W)
    r11 = (u & 0x7FF).astype(np.uint16)
    e = (r11 >> 6) & 0x1F
    m = r11 & 0x3F
    val = np.where(e == 0, (m / 64.0) * 2.0 ** -14,
                   (1.0 + m / 64.0) * 2.0 ** (e.astype(np.float32) - 15.0))
    return val.astype(np.float32)


def load_motion_raw(name: str) -> np.ndarray:
    """RG16_SFLOAT (low-endian int16) -> pixel-units float32."""
    m = np.fromfile(os.path.join(CAP_DIR, name), dtype="<f2").reshape(H, W, 2)
    return m.astype(np.float32) * np.array([1920.0, 1050.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Sliding-window forward pass with memmap output.
# ---------------------------------------------------------------------------

def forward_sliding_window(
    model, color, depth, mv, out_path: str,
    bias: np.ndarray, white_point: float = 1.0, passthrough: bool = True,
) -> np.ndarray:
    """Run the replica on every 96x96 window of the frame, write to a
    memmap at out_path, return the in-memory-mapped array (lazy)."""
    n_rows = (H - SIZE) // STRIDE + 1     # 11
    n_cols = (W - SIZE) // STRIDE + 1     # 20
    n_crops = n_rows * n_cols
    print(f"  sliding window: {n_rows} rows x {n_cols} cols = {n_crops} crops")

    # memmap the full output -- never loaded as a contiguous in-RAM array
    final = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(H, W, 3))

    crops = []
    for yi in range(n_rows):
        for xi in range(n_cols):
            y, x = yi * STRIDE, xi * STRIDE
            crops.append((y, x, yi, xi))
    t0 = time.time()
    for k, (y, x, yi, xi) in enumerate(crops):
        rgb = color[y:y + SIZE, x:x + SIZE].transpose(2, 0, 1)[None].copy()
        d = depth[y:y + SIZE, x:x + SIZE][None, None].copy()
        v = mv[y:y + SIZE, x:x + SIZE].transpose(2, 0, 1)[None].copy()

        dmed = float(np.median(d))
        dn = np.clip(d / max(dmed, 1e-6), 0, 4) / 4.0

        with torch.no_grad():
            res = model(
                torch.from_numpy(rgb).float(),
                torch.from_numpy(dn.astype(np.float32)),
                torch.from_numpy((v * 0.02).astype(np.float32)),
                torch.from_numpy(rgb).float(),
            )[0].numpy().astype(np.float32)             # (3, S, S)

        # bias-only calibration: delta_hat = res + b
        res = res + bias.reshape(3, 1, 1)

        # resolve composition: with passthrough=1 and original==proxy (the
        # cap2_live path), the forward composition collapses; we use the
        # general resolve() function for safety. original == proxy here.
        proxy = rgb.transpose(0, 2, 3, 1)[0]            # (S, S, 3)
        composed = resolve(
            proxy.astype(np.float32),
            np.transpose(res, (1, 2, 0)),                 # (S, S, 3) for resolve
            proxy.astype(np.float32),
            white_point=white_point,
            transfer_strength=1.0,
            colour_strength=1.0,
            max_ratio=2.0,
            passthrough=passthrough,
        )
        final[y:y + SIZE, x:x + SIZE] = composed
        if (k + 1) % 50 == 0 or k == n_crops - 1:
            print(f"    crop {k + 1}/{n_crops} "
                  f"({(time.time() - t0) / (k + 1):.2f}s/crop)")
    final.flush()
    return final


# ---------------------------------------------------------------------------
# Visualisation.
# ---------------------------------------------------------------------------

def make_vis(input_frame, official_after, replica_out, out_path,
             diff_gain: float = 5.0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available -- skipping visualisation")
        return

    # use luminance for display-friendly viz; clip HDR to [0, 1] for input/after
    def lum(x):
        return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])

    # pick a centre crop 768x768 for readable panels
    y0 = (H - 768) // 2; x0 = (W - 768) // 2
    crop = lambda a: a[y0:y0 + 768, x0:x0 + 768]

    in_lum = np.clip(lum(crop(input_frame)), 0, 1)
    off_lum = np.clip(lum(crop(official_after)), 0, 1)
    rep_lum = np.clip(lum(crop(replica_out)), 0, 1)
    diff_lum = np.clip((crop(official_after) - crop(replica_out)) * diff_gain + 0.5, 0, 1)
    diff_lum = lum(diff_lum)

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    for a, im, t in zip(ax,
                         [in_lum, off_lum, rep_lum, diff_lum],
                         ["input (proxy)", "official after",
                          "replica output", f"diff (off-rep) x{diff_gain:g} +0.5"]):
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        a.set_title(t, fontsize=12)
        a.axis("off")
    fig.suptitle(f"DLSS-NR replica -- centre crop 768x768 (frame 0)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=72)
    plt.close(fig)
    print(f"[vis] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out-npy", default=os.path.join(HERE, ".tmp", "final_frame0.npy"))
    ap.add_argument("--out-vis", default=os.path.join(HERE, ".tmp", "final_vis.png"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_npy), exist_ok=True)

    bias = load_bias_calibration()

    print(f"loading frame {args.frame} ...")
    bef = load_rgb10a2(f"{os.path.join(CAP_DIR, f'model_input_{args.frame:02d}.raw')}")
    dep = load_depth_raw(f"depth_{args.frame:02d}.raw")
    mv = load_motion_raw(f"motion_{args.frame:02d}.raw")
    aft = load_rgb10a2(f"{os.path.join(CAP_DIR, f'after_{args.frame:02d}.raw')}")
    print(f"  bef  shape={bef.shape} mean={bef.mean():.3f}")
    print(f"  dep  range=[{dep.min():.3f}, {dep.max():.3f}] median={np.median(dep):.4f}")
    print(f"  mv   range=[{mv.min():.3f}, {mv.max():.3f}]")
    print(f"  aft  shape={aft.shape} mean={aft.mean():.3f}")

    print("loading replica (147.7M) ...")
    t0 = time.time()
    torch.manual_seed(42)
    model = DLSS5NetCalib().eval()
    by = load_all()
    fill_model(model, by, blob_full=open(BLOB, "rb").read())
    print(f"  ready in {time.time() - t0:.0f}s")

    print("running sliding-window forward + bias-only calibration + resolve ...")
    final = forward_sliding_window(model, bef, dep, mv, args.out_npy, bias)

    # ---- full-frame metrics ----------------------------------------------
    # Only the [0:H-stride, 0:W-stride] sub-rectangle is fully covered by
    # crops (the rightmost/bottommost strips < SIZE pixels are not produced).
    cov_rows = (H // STRIDE) * STRIDE
    cov_cols = (W // STRIDE) * STRIDE
    final_cov = np.asarray(final[:cov_rows, :cov_cols])
    aft_cov = aft[:cov_rows, :cov_cols]
    bef_cov = bef[:cov_rows, :cov_cols]

    diff = aft_cov - final_cov
    mse = float(np.mean(diff ** 2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    corr_full = float(np.corrcoef(final_cov.ravel(), aft_cov.ravel())[0, 1])
    corr_raw_full = float(np.corrcoef(bef_cov.ravel(), aft_cov.ravel())[0, 1])
    raw_diff = aft_cov - bef_cov
    raw_mse = float(np.mean(raw_diff ** 2))
    raw_psnr = 10 * np.log10(1.0 / max(raw_mse, 1e-12))
    print("\n  full-frame metrics (covered region, "
          f"{cov_rows}x{cov_cols} = {cov_rows * cov_cols:,} pixels):")
    print(f"    corr(pass-through input, official after) = {corr_raw_full:+.3f}")
    print(f"    corr(replica output, official after)    = {corr_full:+.3f}")
    print(f"    PSNR(replica, official)                 = {psnr:.2f} dB")
    print(f"    PSNR(passthrough input, official)       = {raw_psnr:.2f} dB  "
          f"(upper bound if model were a no-op)")

    # ---- visualise -------------------------------------------------------
    make_vis(bef, aft, np.asarray(final), args.out_vis)


if __name__ == "__main__":
    main()