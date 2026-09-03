"""dlss5.postprocess — Composition and display mapping utilities for DLSS5 NR.

Provides:
  - Additive residual composition: final = clip(input + gain * (res + bias), 0, 1).
    Avoids mosaic blocking artifacts by applying smooth, per-pixel edits.
  - Display mapping / Tone mapping for visual inspection of linear HDR frames.
"""
from __future__ import annotations

from typing import Tuple, Union
import numpy as np
import torch

# Calibrated defaults from Phase 6.5 (Task 7b)
DEFAULT_BIAS = np.array([-0.25932145, -0.01648699, -0.07616311], dtype=np.float32)
DEFAULT_GAIN = 0.8

# Phase 7 output-head calibration (full-frame affine fit on frame 0, validated 8/8):
#   official_delta ≈ A * replica_delta + B  (per channel, linear domain)
# Raises PSNR 9.36 -> 11.38 dB on held-out frames. See .tmp/diag_full.py protocol.
DEFAULT_OUT_AFFINE_A = np.array([0.11124780, 6.15148878, 1.12952542], dtype=np.float32)
DEFAULT_OUT_AFFINE_B = np.array([-0.21836868, 2.24499130, 0.17110443], dtype=np.float32)


def apply_out_affine(
    color_in: np.ndarray,
    model_out: np.ndarray,
    a: np.ndarray = DEFAULT_OUT_AFFINE_A,
    b: np.ndarray = DEFAULT_OUT_AFFINE_B,
) -> np.ndarray:
    """Output-head affine calibration: maps replica delta onto official delta scale.

    final = clip(color_in + a * (model_out - color_in) + b, 0, 1) per channel,
    where model_out is the already-blended apply_residual() output. Run a quick
    fit (fit_out_affine) on one reference frame to (re)derive a/b for a given
    weights build; defaults above are frozen for the current blob.
    """
    delta = model_out - color_in
    out = color_in + np.asarray(a, np.float32).reshape(1, 1, 3) * delta \
        + np.asarray(b, np.float32).reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)


def fit_out_affine(
    color_in: np.ndarray,
    model_out: np.ndarray,
    official_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares per-channel fit of official_delta ≈ a * replica_delta + b.

    Args:
        color_in: Input frame (H, W, 3) float [0, 1].
        model_out: Replica output (after apply_residual), same shape.
        official_out: Official DLL output (ground truth), same shape.

    Returns:
        (a, b): per-channel arrays of shape (3,).
    """
    a_out = np.zeros(3, np.float32)
    b_out = np.zeros(3, np.float32)
    for i in range(3):
        x = (model_out[..., i] - color_in[..., i]).ravel()
        y = (official_out[..., i] - color_in[..., i]).ravel()
        A = np.stack([x, np.ones_like(x)], 1)
        (aa, bb), *_ = np.linalg.lstsq(A, y, rcond=None)
        a_out[i], b_out[i] = aa, bb
    return a_out, b_out


def apply_residual(
    color_in: Union[np.ndarray, torch.Tensor],
    residual: Union[np.ndarray, torch.Tensor],
    gain: float = DEFAULT_GAIN,
    bias: Union[np.ndarray, Tuple[float, float, float]] = DEFAULT_BIAS,
) -> Union[np.ndarray, torch.Tensor]:
    """Applies model residual to input frame using calibrated additive composition.

    Args:
        color_in: Input color frame (H, W, 3) or (B, 3, H, W), float in [0, 1].
        residual: Model residual output, same spatial/channel dimensions.
        gain: Calibrated residual scalar gain (default: 0.8).
        bias: Calibrated per-channel bias correction (RGB).

    Returns:
        Denoised / relit frame clipped to [0, 1].
    """
    if isinstance(color_in, torch.Tensor):
        if not isinstance(bias, torch.Tensor):
            bias_t = torch.tensor(bias, device=color_in.device, dtype=color_in.dtype)
        else:
            bias_t = bias.to(device=color_in.device, dtype=color_in.dtype)

        if color_in.ndim == 4:  # (B, 3, H, W)
            bias_t = bias_t.view(1, 3, 1, 1)
        elif color_in.ndim == 3 and color_in.shape[-1] == 3:  # (H, W, 3)
            bias_t = bias_t.view(1, 1, 3)

        out = color_in + gain * (residual + bias_t)
        return torch.clamp(out, 0.0, 1.0)

    # Numpy branch
    b = np.asarray(bias, dtype=np.float32)
    if color_in.ndim == 3 and color_in.shape[-1] == 3:  # (H, W, 3)
        b = b.reshape(1, 1, 3)
    elif color_in.ndim == 3 and color_in.shape[0] == 3:  # (3, H, W)
        b = b.reshape(3, 1, 1)
    elif color_in.ndim == 4:  # (B, 3, H, W)
        b = b.reshape(1, 3, 1, 1)

    out = color_in + gain * (residual + b)
    return np.clip(out, 0.0, 1.0)


def linear_to_srgb(hdr: np.ndarray, white_point: float = 0.31) -> np.ndarray:
    """Converts linear HDR buffer to sRGB display space using OptiScaler white point mapping."""
    l = np.clip(hdr / max(white_point, 1e-6), 0.0, None)
    srgb = np.where(
        l <= 0.0031308,
        12.92 * l,
        1.055 * np.power(np.maximum(l, 1e-12), 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)
