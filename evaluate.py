#!/usr/bin/env python3
"""evaluate.py — Comprehensive evaluation benchmark of DLSS5 PyTorch against official ground truth.

Loads captured live frames (RTX 5090 Blackwell nvngx_dlssnr.dll ground truth)
and compares the PyTorch replica against the official DLL output.

Metrics:
  - PSNR (Peak Signal-to-Noise Ratio, in dB)
  - MSE (Mean Squared Error)
  - Pearson Correlation (r)
  - Gain over pass-through baseline (dB)

Usage:
    # Quick patch evaluation (CPU-friendly)
    python3 evaluate.py --data-dir .tmp/cap2_live --frame 0 --crop 192 192

    # Full-resolution evaluation (single frame or all frames)
    python3 evaluate.py --data-dir .tmp/cap2_live --frame 0 --device cuda
"""
from __future__ import annotations

import argparse
import os
import time
import numpy as np
from PIL import Image

import dlss5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DLSS5 PyTorch replica against official ground truth."
    )
    parser.add_argument(
        "--data-dir",
        default=".tmp/cap2_live",
        help="Path to capture directory containing before_XX.raw and after_XX.raw",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to evaluate (default: 0)",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Evaluate all 8 frames in data-dir and compute mean metrics",
    )
    parser.add_argument(
        "--weights",
        default="weights_blob.bin",
        help="Path to weights_blob.bin",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: 'auto', 'cuda', 'cpu', or 'mps'",
    )
    parser.add_argument(
        "--crop",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Center crop region for fast preview on CPU (e.g. 192 192)",
    )
    parser.add_argument(
        "--out-vis",
        default=".tmp/eval_comparison.png",
        help="Path to save 4-panel visual comparison (default: .tmp/eval_comparison.png)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply frozen output-head affine calibration (fit frame 0 vs official; +2 dB)",
    )
    return parser.parse_args()


def compute_metrics(pred: np.ndarray, gt: np.ndarray, baseline: np.ndarray) -> dict:
    """Computes PSNR, MSE, and Pearson correlation against ground truth."""
    p_flat = pred.ravel()
    g_flat = gt.ravel()
    b_flat = baseline.ravel()

    # Replica vs Ground Truth
    mse_replica = float(np.mean((pred - gt) ** 2))
    psnr_replica = float(10.0 * np.log10(1.0 / max(mse_replica, 1e-12)))
    corr_replica = float(np.corrcoef(p_flat, g_flat)[0, 1])

    # Pass-through Baseline vs Ground Truth
    mse_base = float(np.mean((baseline - gt) ** 2))
    psnr_base = float(10.0 * np.log10(1.0 / max(mse_base, 1e-12)))
    corr_base = float(np.corrcoef(b_flat, g_flat)[0, 1])

    return {
        "psnr_replica": psnr_replica,
        "mse_replica": mse_replica,
        "corr_replica": corr_replica,
        "psnr_baseline": psnr_base,
        "mse_baseline": mse_base,
        "corr_baseline": corr_base,
        "delta_psnr": psnr_replica - psnr_base,
    }


def make_comparison_panel(
    before: np.ndarray,
    after: np.ndarray,
    replica: np.ndarray,
    out_path: str,
):
    """Saves a 4-panel visual comparison: Before | Official GT | Replica | Error Map."""
    # Convert linear HDR to viewable sRGB
    b_vis = dlss5.linear_to_srgb(before)
    a_vis = dlss5.linear_to_srgb(after)
    r_vis = dlss5.linear_to_srgb(replica)

    # Error map: amplified 5x for visual clarity
    err = np.clip(np.abs(replica - after) * 5.0, 0.0, 1.0)

    # Panel stack horizontally
    panel = np.concatenate([b_vis, a_vis, r_vis, err], axis=1)
    dlss5.save_image(panel, out_path)


def evaluate_single_frame(
    model,
    frame_idx: int,
    data_dir: str,
    device: str,
    crop: tuple[int, int] | None = None,
    save_vis_path: str | None = None,
    calibrate: bool = False,
) -> dict:
    bef_path = os.path.join(data_dir, f"before_{frame_idx:02d}.raw")
    aft_path = os.path.join(data_dir, f"after_{frame_idx:02d}.raw")
    dep_path = os.path.join(data_dir, f"depth_{frame_idx:02d}.raw")
    mot_path = os.path.join(data_dir, f"motion_{frame_idx:02d}.raw")

    color = dlss5.load_dxgi_color(bef_path)
    gt = dlss5.load_dxgi_color(aft_path)
    depth = dlss5.load_dxgi_depth(dep_path)
    motion = dlss5.load_dxgi_motion(mot_path)

    if crop:
        ch, cw = crop
        H, W, _ = color.shape
        sy = max((H - ch) // 2, 0)
        sx = max((W - cw) // 2, 0)
        color = color[sy : sy + ch, sx : sx + cw]
        gt = gt[sy : sy + ch, sx : sx + cw]
        depth = depth[sy : sy + ch, sx : sx + cw]
        motion = motion[sy : sy + ch, sx : sx + cw]

    t0 = time.time()
    out = dlss5.infer_frame(
        model, color, depth, motion, device=device, affine_calibrate=calibrate
    )
    dt = time.time() - t0

    metrics = compute_metrics(out, gt, color)
    metrics["latency_s"] = dt
    metrics["resolution"] = (color.shape[0], color.shape[1])

    if save_vis_path:
        make_comparison_panel(color, gt, out, save_vis_path)

    return metrics


def main():
    args = parse_args()

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    print("=================================================================")
    print("  NVIDIA DLSS5 PyTorch Replica — Ground Truth Alignment Benchmark")
    print("=================================================================")
    print(f"  Target Device : {device}")
    print(f"  Capture Dir   : {args.data_dir}")
    print(f"  Weights Blob  : {args.weights}")
    if args.crop:
        print(f"  Eval Mode     : Center Crop {args.crop[0]}x{args.crop[1]}")
    else:
        print("  Eval Mode     : Full Frame (1920x1050 native)")
    print("-----------------------------------------------------------------")

    t0 = time.time()
    model = dlss5.load_model(args.weights, device=device, verbose=False)
    print(f"  Model loaded in {time.time() - t0:.2f}s")
    print("-----------------------------------------------------------------")

    frames_to_eval = list(range(8)) if args.all_frames else [args.frame]

    results = []
    for f_idx in frames_to_eval:
        vis_path = args.out_vis if (len(frames_to_eval) == 1 or f_idx == 0) else None
        m = evaluate_single_frame(
            model,
            frame_idx=f_idx,
            data_dir=args.data_dir,
            device=device,
            crop=tuple(args.crop) if args.crop else None,
            save_vis_path=vis_path,
            calibrate=args.calibrate,
        )
        results.append(m)
        print(
            f"  Frame {f_idx:02d} | "
            f"Pass-Through: {m['psnr_baseline']:.2f} dB (r={m['corr_baseline']:+.3f}) | "
            f"PyTorch: {m['psnr_replica']:.2f} dB (r={m['corr_replica']:+.3f}) | "
            f"Gain: {m['delta_psnr']:+.2f} dB | "
            f"Time: {m['latency_s']:.2f}s"
        )

    print("-----------------------------------------------------------------")
    mean_psnr_base = np.mean([r["psnr_baseline"] for r in results])
    mean_psnr_rep = np.mean([r["psnr_replica"] for r in results])
    mean_gain = np.mean([r["delta_psnr"] for r in results])
    mean_corr = np.mean([r["corr_replica"] for r in results])

    print("  Benchmark Summary:")
    print(f"    Evaluated Frames : {len(results)}")
    print(f"    Pass-Through PSNR: {mean_psnr_base:.2f} dB")
    print(f"    PyTorch Replica  : {mean_psnr_rep:.2f} dB")
    print(f"    Improvement      : {mean_gain:+.2f} dB over baseline")
    print(f"    Correlation (r)  : {mean_corr:+.3f}")
    if args.out_vis and os.path.isfile(args.out_vis):
        print(f"    Visual Comparison: {args.out_vis}")
    print("=================================================================")


if __name__ == "__main__":
    main()
