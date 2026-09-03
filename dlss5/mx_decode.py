"""dlss5.mx_decode — E4M3 and Microscaling (MXFP8) decoding utilities.

Decodes:
  - FP8 (E4M3): 1 sign bit, 4 exponent bits, 3 mantissa bits (bias = 7).
  - MXFP8: E4M3 weights paired with scale bytes (shared exponent per 32 elements).
    Median-8 adaptive shift: v = W * 2^(S - median(S) - 8).
  - FP16: IEEE 754 half-precision floats (little-endian).
"""
from __future__ import annotations

import numpy as np


def e4m3_decode(u8: np.ndarray) -> np.ndarray:
    """Decode an array of uint8 values as IEEE/OCP E4M3 FP8."""
    u8 = np.asarray(u8, dtype=np.uint8)
    e = (u8 >> 3) & 0xF
    m = u8 & 7
    sgn = np.where(u8 & 0x80, -1.0, 1.0)
    # Normal vs subnormal
    v = np.where(
        e == 0,
        (m / 16.0) * (2.0 ** -6),
        (1.0 + m / 8.0) * np.power(2.0, e.astype(np.float64) - 7.0),
    )
    # NaN/Inf representation in E4M3 (e=15, m=7 is NaN; mapped to 0.0)
    return sgn * np.where((e == 15) & (m == 7), 0.0, v)


def mx_decode_pairs(pairs: np.ndarray) -> np.ndarray:
    """MXFP8 decode: pair of (weight_byte, scale_byte).

    Formula: v = W * 2^(S - median(S) - 8.0).
    The median(S) provides per-tensor adaptive scale centering, and -8 scales
    E4M3 full-range down to standard neural network weight magnitudes.
    """
    w = e4m3_decode(pairs[:, 0])
    s = pairs[:, 1].astype(np.float64)
    med = np.median(s) if len(s) > 0 else 0.0
    return w * np.power(2.0, s - med - 8.0)


def fp16_decode(raw: bytes | np.ndarray) -> np.ndarray:
    """Decode raw bytes as little-endian IEEE float16."""
    if isinstance(raw, bytes):
        if len(raw) < 2:
            return np.zeros(0, dtype=np.float32)
        n = (len(raw) // 2) * 2
        with np.errstate(invalid="ignore"):
            v = np.frombuffer(raw[:n], dtype="<f2").astype(np.float32)
        return np.nan_to_num(v, nan=0.0, posinf=65504.0, neginf=-65504.0)
    arr = np.asarray(raw, dtype=np.uint8)
    if len(arr) < 2:
        return np.zeros(0, dtype=np.float32)
    n = (len(arr) // 2) * 2
    with np.errstate(invalid="ignore"):
        v = np.frombuffer(arr[:n].tobytes(), dtype="<f2").astype(np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=65504.0, neginf=-65504.0)
