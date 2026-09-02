#!/usr/bin/env python3
"""Forward smoke test — calibrated DLSS5 model (byte-exact, Phase 3).

Usage:
    python phase3/scripts/test_forward.py --calib          # calibrated model
    python phase3/scripts/test_forward.py --calib --load   # + load blob weights

Reports total params vs blob byte total, fp32 forward shape/NaN, and (with GPU)
fp16/bf16 timing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BLOB_BYTES = 147_683_778


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true", help="use calibrated model")
    ap.add_argument("--load", action="store_true", help="load blob weights (naive E4M3)")
    ap.add_argument("--size", type=int, nargs=2, default=(270, 480))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    H, W = args.size
    if args.calib:
        from dlss5.calib_model import DLSS5NetCalib
        from dlss5.weights_loader import load_weights_calib, parse_records
        model = DLSS5NetCalib()
        tag = "calib"
        if args.load:
            load_weights_calib(model, report=True)
    else:
        from dlss5.model import DLSS5Net, default_config
        from dlss5.weights_loader import load_weights, parse_records
        model = DLSS5Net(default_config())
        tag = "skeleton"
        if args.load:
            load_weights(model, report=True)

    total = sum(p.numel() for p in model.parameters())
    blob = sum(r["B"] for r in parse_records()) if os.path.exists(
        "/root/dlss5-pytorch/weights_blob.bin") else BLOB_BYTES
    print(f"\n[{tag}] total params = {total:,} ({total/1e6:.4f}M)")
    print(f"blob weight bytes = {blob:,} ({blob/1e6:.4f}M)  "
          f"match={total == blob}  diff={total - blob:,}")

    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model = model.to(dev).eval()
    with torch.no_grad():
        c = torch.rand(1, 3, H, W, device=dev)
        d = torch.rand(1, 1, H, W, device=dev)
        mv = torch.rand(1, 2, H, W, device=dev)
        ct = torch.rand(1, 3, H, W, device=dev)
        t0 = time.time()
        out = model(c, d, mv, ct)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) * 1e3
        print(f"forward ({dev}) {dt:.0f} ms  out={tuple(out.shape)}  "
              f"NaN={torch.isnan(out).sum().item()}  Inf={torch.isinf(out).sum().item()}")
        print(f"out absmean={out.abs().mean().item():.3e}")


if __name__ == "__main__":
    main()
