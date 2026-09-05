"""dlss5.mx_decode — E4M3 and Microscaling (MXFP8) decoding utilities.

Decodes:
  - FP8 (E4M3): 1 sign bit, 4 exponent bits, 3 mantissa bits (bias = 7).
  - MXFP8: E4M3 weights paired with scale bytes (shared exponent per 32 elements).
    Median-8 adaptive shift: v = W * 2^(S - median(S) - 8).
  - FP16: IEEE 754 half-precision floats (little-endian).
"""
from __future__ import annotations

import os

import numpy as np


def e4m3_decode(u8: np.ndarray) -> np.ndarray:
    """Decode an array of uint8 values as IEEE/OCP E4M3 FP8.

    DLSS5_E4_SCALE env (default 0.25): round-5 calibration.  With the naive
    x1.0 decode the encoder activations grow ~2x per block from c64 onward
    (enc1 mean 80.6, enc2 75.3, enc3 102.2; tail pre-tanh 198 vs official
    ~0.16) even though each block is LN-renormalized - consistent with all
    E4 weights carrying an unaccounted global scale.  Measured grid (flat
    probes, pretanh = tail.conv pre-tanh mean|.|):

      es    enc1   enc2   enc3   dec1   pretanh   tmax   flat dcR
      1.0   80.6   75.3  102.2  125.4   197.9    2000    +0.049 locked
      0.5    4.9    7.2   10.9   22.2     8.35    43.2   +0.0245 locked
      0.25   1.2    4.4    3.9*  10.8     1.14     4.9   +0.0092 (~official)
      0.125  0.6    4.3    3.4    6.9     0.274    0.9   +0.001 (too cold)

      * with mlp.2 garbage-row hygiene: enc3 0.54, dec1 0.83.

    es=0.25 is the default.  mx_decode_pairs divides this out so the MX
    path keeps its own (k, median) calibration.
    """
    s = float(os.environ.get("DLSS5_E4_SCALE", "0.25"))
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
    v = np.where((e == 15) & (m == 7), 0.0, v)
    return sgn * v * s


def mx_decode_pairs(pairs: np.ndarray) -> np.ndarray:
    """MXFP8 decode: pair of (weight_byte, scale_byte).

    Formula: v = W * 2^(S - median(S) - K).
    The median(S) provides per-tensor adaptive scale centering, and K scales
    E4M3 full-range down to standard neural network weight magnitudes.

    K is a calibration parameter (default 8, historical guess).  Round-5:
    tail-saturation arithmetic says the replica's feature stack runs ~100x
    too hot vs the official (official pre-tanh ~+/-0.16, replica 40-184),
    consistent with a global MX fold offset error.  DLSS5_MX_K env override
    used for the k-in-{4..12} scan.
    """
    k = float(os.environ.get("DLSS5_MX_K", "8"))
    w = e4m3_decode(pairs[:, 0])
    es = float(os.environ.get("DLSS5_E4_SCALE", "0.25"))
    if es != 1.0:
        w = w / es   # keep MX path calibrated by (k, median) only
    s = pairs[:, 1].astype(np.float64)
    med = np.median(s) if len(s) > 0 else 0.0
    return w * np.power(2.0, s - med - k)


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
