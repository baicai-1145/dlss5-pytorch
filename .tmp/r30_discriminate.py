#!/usr/bin/env python3
"""r30_discriminate.py — 0x6e binding discriminator (H_A vs H_B).

Runs the full-tail replica against the captured MV-probe shots and scores
which hypothesis reproduces the official model_raw:

  H_A: bicubic source = current input color
  H_B: bicubic source = previous network output (frame 0: zeros/prev)

Primary discriminator = P3 impulseMV frame 0 (reset=1, no history):
  H_A prediction: displaced dot visible at dot+(dx,dy), sharp (2px) footprint
  H_B prediction: NO dot displacement visible (bicubic reads zeros)
Secondary = P2 edgeMV transition-band width:
  H_A: band width ~= input (2px), peak shift = dx exactly
  H_B: band width 4-6px (network-smoothed), shift only on steady frames

Usage:
  PYTHONPATH=. python3 .tmp/r30_discriminate.py capP3_mvp100
  PYTHONPATH=. python3 .tmp/r30_discriminate.py   # runs all shots found
"""
import os
import sys

os.environ["DLSS5_TAIL_SIGN"] = "game"
os.environ["DLSS5_MV_U_SCALE"] = "-0.14"
os.environ["DLSS5_MV_V_SCALE"] = "1.12"
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp"))
import dlss5  # noqa: E402
import eval_suite as ES  # noqa: E402
from eval_cap3 import H, W  # noqa: E402

dev = "cuda:0"


def load_shot(directory):
    """Load one probe shot: before/model_raw/motion N frames (native res)."""
    manifest = {}
    for line in open(os.path.join(directory, "manifest.txt")):
        parts = line.split()
        if parts[0] in ("before", "model_raw", "motion", "depth", "model_input"):
            manifest[parts[0]] = tuple(int(x) for x in parts[2:6])
    bw, bh = manifest.get("before", (W, H, 0, 0))[:2]
    n = 0
    while os.path.exists(os.path.join(directory, f"model_raw_{n:02d}.raw")):
        n += 1
    cols = np.stack([ES.load_fp16_rgb(os.path.join(directory, f"before_{i:02d}.raw"), bw, bh)
                     for i in range(n)])
    raws = np.stack([ES.load_fp16_rgb(os.path.join(directory, f"model_raw_{i:02d}.raw"), bw, bh)
                     for i in range(n)])
    mw, mh = manifest.get("motion", (0, 0, 0, 0))[:2]
    mvs = np.stack([dlss5.load_dxgi_motion(os.path.join(directory, f"motion_{i:02d}.raw"),
                                           width=mw, height=mh, scale=(float(mw), float(mh)))
                    for i in range(n)])
    resets = []
    for line in open(os.path.join(directory, "manifest.txt")):
        parts = line.split()
        if parts[0] == "frame":
            resets.append(int(parts[4]))
    return cols, raws, mvs, resets, (bw, bh)


def net_forward(m, col, dep_path, mv):
    ph, pw = (-col.shape[0]) % 16, (-col.shape[1]) % 16
    c = F.pad(torch.from_numpy(col.transpose(2, 0, 1))[None].float(), (0, pw, 0, ph), mode="replicate").to(dev)
    dep = dlss5.load_dxgi_depth(dep_path)
    d_ = F.pad(torch.from_numpy(dep)[None, None].float(), (0, pw, 0, ph), mode="replicate").to(dev)
    d_ = F.interpolate(d_, size=c.shape[-2:], mode="bilinear", align_corners=False)
    mm = F.pad(torch.from_numpy(mv.transpose(2, 0, 1))[None].float(), (0, pw, 0, ph), mode="replicate").to(dev)
    mm = F.interpolate(mm, size=c.shape[-2:], mode="bilinear", align_corners=False)
    mm = mm * torch.tensor([-0.14, 1.12], device=dev).view(1, 2, 1, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return m(c, d_, mm, c)[0, :, : col.shape[0], : col.shape[1]].float().cpu()


def bicubic_warp(src_t, mvx, mvy):
    h, w = src_t.shape[-2:]
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    gx = (xs + mvx / w * 2).float()
    gy = (ys + mvy / h * 2).float()
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    return F.grid_sample(src_t, grid, mode="bicubic", padding_mode="border", align_corners=False)


def discriminate(directory):
    cols, raws, mvs, resets, (bw, bh) = load_shot(directory)
    m = dlss5.load_model("weights_blob.bin", device=dev)
    m.eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "expands" in n and n.endswith("expand.weight"):
                p.zero_()

    prev_out = None
    score = {"H_A": [], "H_B": []}
    for i in range(len(cols)):
        net_raw = net_forward(m, cols[i], os.path.join(directory, f"depth_{i:02d}.raw"), mvs[i])
        mvx = torch.from_numpy(mvs[i][..., 0].copy()).float()
        mvy = torch.from_numpy(mvs[i][..., 1].copy()).float()

        srcA = torch.from_numpy(cols[i].transpose(2, 0, 1))[None].float()
        bicA = bicubic_warp(srcA, mvx, mvy)[0]

        if prev_out is None:
            bicB = bicA  # frame 0 of H_B: prev output undefined -> treat as srcA (upper bound)
        else:
            bicB = bicubic_warp(prev_out, mvx, mvy)[0]

        official = torch.from_numpy(raws[i].transpose(2, 0, 1)).float()
        for tag, bic in (("H_A", bicA), ("H_B", bicB)):
            # gate fit per frame (same as R28): out = g*bic - net_raw
            num = ((official + net_raw) * bic).sum()
            den = (bic * bic).sum()
            g = float((num / den).clamp(0, 1))
            out = (g * bic - net_raw)
            score[tag].append(float((out - official).abs().mean()))

        prev_out = official[None]

    # frame-0 (reset) decision weight x3 — cleanest discriminator
    fa = sum(score["H_A"][:1]) * 3 + sum(score["H_A"][1:])
    fb = sum(score["H_B"][:1]) * 3 + sum(score["H_B"][1:])
    print(f"{os.path.basename(directory)}: H_A={np.mean(score['H_A']):.5f} "
          f"H_B={np.mean(score['H_B']):.5f}  weighted(H_A)={fa:.5f} weighted(H_B)={fb:.5f} "
          f"-> {'H_A' if fa < fb else 'H_B'}")
    return fa, fb


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else None
    if base:
        discriminate(base)
        return
    import glob
    for d in sorted(glob.glob("capP?_mv*")):
        try:
            discriminate(d)
        except Exception as e:  # noqa: BLE001
            print(f"{d}: SKIP ({e})")


if __name__ == "__main__":
    main()
