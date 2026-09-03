"""dlss5.mp_cubic_silu — NVIDIA's MpCubicSiluActivation, recovered from DLSS5 PTX.

The official fused kernels do NOT use GELU. The representative low-precision
path clamps the input and evaluates a cubic polynomial approximation of SiLU:

    t = clamp(x, -4, +4)
    p = (-0.0559082) * abs(t) + 0.447266
    a = t * p + 0.894531
    y = x * a

Constants are half-precision rounded in the recovered PTX path. The activation
is SiLU-shaped but with amplitude ~1.789x SiLU at saturation (a→1.789),
crossing y=x slope-1 at large |x|. Source: madebyollin gist, PTX of
cc_tinlayout_fused_* kernels, MpCubicSiluActivation template.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Half-precision-rounded constants recovered from PTX
_C1 = -0.0559082   # |t| coefficient
_C2 = 0.447266     # p bias
_C3 = 0.894531     # a bias (= 2 * C2 to fp16 round)


def mp_cubic_silu(x: torch.Tensor) -> torch.Tensor:
    """Polynomial-cubic SiLU approximation used by DLSS5 fused kernels."""
    t = x.clamp(-4.0, 4.0)
    p = _C1 * t.abs() + _C2
    a = t * p + _C3
    return x * a


class MpCubicSiLU(nn.Module):
    """Module wrapper for drop-in replacement of nn.GELU / nn.SiLU."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mp_cubic_silu(x)

    def extra_repr(self) -> str:
        return "cubic-poly SiLU (PTX-recovered)"
