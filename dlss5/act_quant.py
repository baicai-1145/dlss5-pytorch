"""E4M3 activation quantization (R33) — PyTorch equivalent of the SASS
F2FP.SATFINITE.E4M3.F16 instruction found at every fp8-GEMM input side
(R33 SASS decode: 388 conversions in 10 clusters inside the fp8
simple_blend kernel, each cluster = one quantization point before an
HMMA stage).

The official DLL runs the fp8 kernel variants: activations are quantized
to E4M3 (saturating at ±448) before every tensor-core GEMM. Our replica
has been computing in bf16/fp16 — this module injects the same rounding
to test whether FP8 activation quantization explains part of the corr
ceiling.
"""
from __future__ import annotations

import os

import torch

_E4M3_MAX = 448.0          # 1.75 * 2^8 (SATFINITE saturation)
_E4M3_MANT_BITS = 3        # 3 mantissa bits -> 8 quantization levels per binade
_E4M3_MIN_NORM = 2 ** -6   # smallest normal exponent (bias 7)


def _e4m3_quantize(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even quantization to the E4M3 grid, SATFINITE."""
    sign = torch.sign(x)
    a = x.abs()
    a = a.clamp(max=_E4M3_MAX)
    # subnormals: exponent -6, mantissa 3 bits -> step 2^-9
    normal = a >= _E4M3_MIN_NORM
    # quantize mantissa to 3 bits: scale into [1,2) binade grid
    exp = torch.floor(torch.log2(a.clamp(min=1e-30)))
    exp = torch.where(normal, exp, torch.full_like(exp, -6.0))
    frac = a / torch.pow(2.0, exp)
    # 3 mantissa bits: 8 steps in [1,2)
    q = torch.round((frac - 1.0) * 8.0) / 8.0 + 1.0
    # handle frac == 2 boundary (rounding up into the next binade)
    carry = q >= 2.0
    q = torch.where(carry, torch.full_like(q, 1.0), q)
    exp = torch.where(carry, exp + 1.0, exp)
    out = sign * q * torch.pow(2.0, exp)
    # subnormal branch: quantize on the 2^-9 grid directly
    sub = torch.where(
        normal,
        out,
        sign * torch.round(a / 2**-9) * 2**-9,
    )
    return torch.where(normal, out, sub)


class E4M3ActQuant(torch.nn.Module):
    """Activation quantizer matching the DLL fp8 path.

    Enabled via DLSS5_ACT_FP8=1. Points (per the SASS clusters): the
    input of every tensor-core GEMM in a swin block — qkv input, proj
    input, ffn1 input, ffn2 input — i.e. norm1 output, attention context,
    norm2 output, FFN hidden.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def is_enabled() -> bool:
        return os.environ.get("DLSS5_ACT_FP8", "0") == "1"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_enabled() or not torch.is_floating_point(x):
            return x
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return x
        return _e4m3_quantize(x.float()).to(x.dtype)
