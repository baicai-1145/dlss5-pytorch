"""dlss5.data_utils — Readers and writers for DXGI capture formats and image files."""
from __future__ import annotations

import os
from typing import Tuple
import numpy as np
from PIL import Image


def load_dxgi_color(path: str, width: int = 1920, height: int = 1050) -> np.ndarray:
    """Reads DXGI_FORMAT_R10G10B10A2_UNORM raw buffer as float32 RGB array in [0, 1].

    Returns:
        Array of shape (height, width, 3) in float32.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Color buffer file not found: {path}")
    u = np.fromfile(path, dtype=np.uint32).reshape(height, width)
    r = ((u >> 0) & 0x3FF) / 1023.0
    g = ((u >> 10) & 0x3FF) / 1023.0
    b = ((u >> 20) & 0x3FF) / 1023.0
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def load_dxgi_depth(path: str, width: int = 1920, height: int = 1050) -> np.ndarray:
    """Reads DXGI_FORMAT_R11G11B10_FLOAT raw buffer and decodes linear distance.

    Returns:
        Array of shape (height, width) in float32 normalized by robust median clipping.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Depth buffer file not found: {path}")
    u = np.fromfile(path, dtype=np.uint32).reshape(height, width)
    r11 = (u & 0x7FF).astype(np.uint16)
    e = (r11 >> 6) & 0x1F
    m = r11 & 0x3F
    val = np.where(
        e == 0,
        (m / 64.0) * (2.0 ** -14),
        (1.0 + m / 64.0) * np.power(2.0, e.astype(np.float32) - 15.0),
    ).astype(np.float32)
    # Robust normalization:
    dmed = float(np.median(val))
    norm_depth = np.clip(val / max(dmed, 1e-6), 0.0, 4.0) / 4.0
    return norm_depth.astype(np.float32)


def load_dxgi_motion(
    path: str, width: int = 1920, height: int = 1050, scale: Tuple[float, float] = (1920.0, 1050.0)
) -> np.ndarray:
    """Reads DXGI_FORMAT_R16G16_FLOAT motion vector buffer in pixel units.

    Returns:
        Array of shape (height, width, 2) in float32.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Motion buffer file not found: {path}")
    m = np.fromfile(path, dtype="<f2").reshape(height, width, 2).astype(np.float32)
    scale_arr = np.array([scale[0], scale[1]], dtype=np.float32)
    return m * scale_arr


def save_image(arr: np.ndarray, path: str):
    """Saves a (H, W, 3) float array in [0, 1] as an 8-bit image."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    u8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(u8).save(path)
