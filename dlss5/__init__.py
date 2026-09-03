"""DLSS5 PyTorch — Reverse-engineered PyTorch replica of NVIDIA DLSS5 Neural Rendering.

Reconstructed from leaked nvngx_dlssnr.dll (DLSS 310.8.0, sm_120 architecture).
147.7M parameters with MXFP8 / FP8 / FP16 decoded weights.

Usage:
    import dlss5

    # 1. Load model with official weights blob
    model = dlss5.load_model("weights_blob.bin", device="cuda")

    # 2. Run full-resolution or patch inference
    output = dlss5.infer_frame(model, color, depth, motion)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from .model import DLSS5Net, DLSS5Config
from .loader import load_model, load_weights
from .postprocess import (
    apply_residual,
    linear_to_srgb,
    apply_out_affine,
    fit_out_affine,
    DEFAULT_OUT_AFFINE_A,
    DEFAULT_OUT_AFFINE_B,
)
from .data_utils import load_dxgi_color, load_dxgi_depth, load_dxgi_motion, save_image


def infer_frame(
    model: DLSS5Net,
    color: np.ndarray | torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    motion: np.ndarray | torch.Tensor,
    ref: np.ndarray | torch.Tensor | None = None,
    device: str | torch.device | None = None,
    divisibility: int = 32,
    apply_postprocess: bool = True,
    affine_calibrate: bool = False,
) -> np.ndarray:
    """Runs DLSS5 NR inference on a single frame at full resolution.

    Handles padding to model divisibility (default: 32) and applies calibrated
    additive residual postprocessing.

    Args:
        model: DLSS5Net instance.
        color: (H, W, 3) float array in [0, 1].
        depth: (H, W) float array normalized in [0, 1].
        motion: (H, W, 2) float array in pixel units.
        ref: Optional reference frame (defaults to color).
        device: Target execution device. If None, uses model parameter device.
        divisibility: Spatial divisibility requirement (default: 32).
        apply_postprocess: If True, applies calibrated additive residual postprocessing.
        affine_calibrate: If True, additionally applies the frozen output-head affine
            calibration (fit on frame 0 vs official DLL; +2 dB on live captures).
            Use when comparing against official output or producing final frames.

    Returns:
        (H, W, 3) float32 array in [0, 1].
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model = model.to(device)

    model.eval()

    # Convert to tensors
    if isinstance(color, np.ndarray):
        c_t = torch.from_numpy(color.transpose(2, 0, 1)).unsqueeze(0).float()
    else:
        c_t = color.float()
        if c_t.ndim == 3:
            c_t = c_t.permute(2, 0, 1).unsqueeze(0)

    if isinstance(depth, np.ndarray):
        d_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    else:
        d_t = depth.float()
        if d_t.ndim == 2:
            d_t = d_t.unsqueeze(0).unsqueeze(0)

    if isinstance(motion, np.ndarray):
        m_t = torch.from_numpy(motion.transpose(2, 0, 1)).unsqueeze(0).float() * 0.02
    else:
        m_t = motion.float() * 0.02
        if m_t.ndim == 3:
            m_t = m_t.permute(2, 0, 1).unsqueeze(0)

    r_t = c_t if ref is None else (
        torch.from_numpy(ref.transpose(2, 0, 1)).unsqueeze(0).float()
        if isinstance(ref, np.ndarray) else ref.float()
    )

    _, _, H, W = c_t.shape
    pad_h = (divisibility - (H % divisibility)) % divisibility
    pad_w = (divisibility - (W % divisibility)) % divisibility

    if pad_h > 0 or pad_w > 0:
        c_t = F.pad(c_t, (0, pad_w, 0, pad_h), mode="replicate")
        d_t = F.pad(d_t, (0, pad_w, 0, pad_h), mode="replicate")
        m_t = F.pad(m_t, (0, pad_w, 0, pad_h), mode="replicate")
        r_t = F.pad(r_t, (0, pad_w, 0, pad_h), mode="replicate")

    c_t, d_t, m_t, r_t = c_t.to(device), d_t.to(device), m_t.to(device), r_t.to(device)

    with torch.no_grad():
        res_t = model(c_t, d_t, m_t, r_t)

    # Crop back to original dimensions
    res = res_t[:, :, :H, :W].squeeze(0).permute(1, 2, 0).cpu().numpy()
    c_np = color if isinstance(color, np.ndarray) else color.cpu().numpy()

    if apply_postprocess:
        out = apply_residual(c_np, res)
        if affine_calibrate:
            out = apply_out_affine(c_np, out)
        return out
    return res


__all__ = [
    "DLSS5Net",
    "DLSS5Config",
    "load_model",
    "load_weights",
    "infer_frame",
    "apply_residual",
    "linear_to_srgb",
    "apply_out_affine",
    "fit_out_affine",
    "DEFAULT_OUT_AFFINE_A",
    "DEFAULT_OUT_AFFINE_B",
    "load_dxgi_color",
    "load_dxgi_depth",
    "load_dxgi_motion",
    "save_image",
]
