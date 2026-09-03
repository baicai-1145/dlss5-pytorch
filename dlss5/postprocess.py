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
