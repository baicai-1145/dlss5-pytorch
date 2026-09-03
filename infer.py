#!/usr/bin/env python3
"""infer.py — High-level CLI for DLSS5 PyTorch inference.

Runs DLSS5 Neural Rendering on raw DXGI buffers or standard images.

Examples:
    # Inference on captured raw frame
    python3 infer.py \\
        --color .tmp/cap2_live/cap2_f0_before.raw \\
        --depth .tmp/cap2_live/cap2_f0_depth.raw \\
        --motion .tmp/cap2_live/cap2_f0_motion.raw \\
        --weights weights_blob.bin \\
        --device cpu \\
        --crop 192 192 \\
        --output output.png \\
        --srgb

    # Full-resolution 1920x1056 inference (recommended on CUDA / RTX 3090/4090/5090)
    python3 infer.py \\
        --color .tmp/cap2_live/cap2_f0_before.raw \\
        --depth .tmp/cap2_live/cap2_f0_depth.raw \\
        --motion .tmp/cap2_live/cap2_f0_motion.raw \\
        --weights weights_blob.bin \\
        --device cuda \\
        --output output_full.png \\
        --srgb
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
from PIL import Image

import dlss5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NVIDIA DLSS5 Neural Rendering PyTorch replica inference."
    )
    parser.add_argument("--color", "-c", required=True, help="Path to input color frame (raw DXGI or image)")
    parser.add_argument("--depth", "-d", default=None, help="Path to depth buffer (raw DXGI or image)")
    parser.add_argument("--motion", "-m", default=None, help="Path to motion vector buffer (raw DXGI)")
    parser.add_argument("--weights", "-w", default="weights_blob.bin", help="Path to weights_blob.bin (default: weights_blob.bin)")
    parser.add_argument("--output", "-o", default="output.png", help="Path to save output image (default: output.png)")
    parser.add_argument("--device", default="auto", help="Execution device: 'auto', 'cuda', 'cpu', or 'mps'")
    parser.add_argument("--width", type=int, default=1920, help="Native buffer width (default: 1920)")
    parser.add_argument("--height", type=int, default=1050, help="Native buffer height (default: 1050)")
    parser.add_argument("--crop", nargs=2, type=int, default=None, metavar=("H", "W"), help="Crop center region of size (H, W) for fast preview")
    parser.add_argument("--srgb", action="store_true", help="Apply filmic tone curve and convert linear HDR output to sRGB")
    parser.add_argument("--calibrate", action="store_true", help="Apply frozen output-head affine calibration (recommended when comparing to official DLL output)")
    parser.add_argument("--save-hdr", default=None, help="Optional path to save raw float32 output as .npy")
    return parser.parse_args()


def load_inputs(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads and normalizes color, depth, and motion buffers."""
    # 1. Color
    if args.color.endswith(".raw"):
        color = dlss5.load_dxgi_color(args.color, width=args.width, height=args.height)
    else:
        im = Image.open(args.color).convert("RGB")
        color = np.array(im, dtype=np.float32) / 255.0

    H, W, _ = color.shape

    # 2. Depth
    if args.depth and os.path.isfile(args.depth):
        if args.depth.endswith(".raw"):
            depth = dlss5.load_dxgi_depth(args.depth, width=W, height=H)
        else:
            depth = np.array(Image.open(args.depth).convert("L"), dtype=np.float32) / 255.0
    else:
        # Synthesize default flat depth
        depth = np.ones((H, W), dtype=np.float32) * 0.5

    # 3. Motion
    if args.motion and os.path.isfile(args.motion):
        motion = dlss5.load_dxgi_motion(args.motion, width=W, height=H)
    else:
        # Synthesize stationary zero motion
        motion = np.zeros((H, W, 2), dtype=np.float32)

    return color, depth, motion


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    print(f"[DLSS5] Target device: {device}")
    t0 = time.time()
    print(f"[DLSS5] Loading weights from '{args.weights}'...")
    model = dlss5.load_model(args.weights, device=device, verbose=False)
    print(f"[DLSS5] Model ready in {time.time() - t0:.2f}s (147.7M parameters)")

    color, depth, motion = load_inputs(args)

    if args.crop:
        ch, cw = args.crop
        H, W, _ = color.shape
        sy = max((H - ch) // 2, 0)
        sx = max((W - cw) // 2, 0)
        color = color[sy : sy + ch, sx : sx + cw]
        depth = depth[sy : sy + ch, sx : sx + cw]
        motion = motion[sy : sy + ch, sx : sx + cw]
        print(f"[DLSS5] Cropped center patch: {color.shape[0]}x{color.shape[1]}")
    else:
        print(f"[DLSS5] Input resolution: {color.shape[1]}x{color.shape[0]}")

    print("[DLSS5] Running neural rendering inference...")
    t_inf = time.time()
    out = dlss5.infer_frame(model, color, depth, motion, device=device, affine_calibrate=args.calibrate)
    dt = time.time() - t_inf
    print(f"[DLSS5] Inference completed in {dt:.2f}s ({out.shape[1]}x{out.shape[0]})")

    if args.save_hdr:
        np.save(args.save_hdr, out)
        print(f"[DLSS5] Saved raw HDR float32 array to '{args.save_hdr}'")

    if args.srgb:
        out_vis = dlss5.linear_to_srgb(out)
    else:
        out_vis = out

    dlss5.save_image(out_vis, args.output)
    print(f"[DLSS5] Saved output image to '{args.output}'")


if __name__ == "__main__":
    main()
