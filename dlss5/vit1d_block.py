"""1D ViT control encoder — from scratch (no timm).

Matches cubin naming: cc_vit_1d_attention / cc_vit_1d_qkv / cc_vit_1d_ffn_contract /
cc_vit_1d_ffn_expand / cc_vit_1d_repack_1d_to_2d / cc_vit_1d_repack_2d_to_1d.

Encodes UI/control/temporal state as a short 1D token sequence; result is re-packed
back to the 2D bottleneck feature map (cc_vit_1d_repack_1d_to_2d) and injected into
the decoder's lowest stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mp_cubic_silu import MpCubicSiLU


class ViT1DBlock(nn.Module):
    """Standard pre-LN transformer encoder block (norm -> MSA -> res, norm -> MLP -> res)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), MpCubicSiLU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ViT1DEncoder(nn.Module):
    """1D ViT encoder over the bottleneck feature map.

    2D bottleneck (B, dim, H, W) -> tokens via mean pooling over H*W
    -> `depth` ViT1DBlocks -> (B, num_tokens, dim).
    (The exact repack is cubin-dependent; Phase 5 pinpoints it. This skeleton uses
    mean pool; it's differentiable and matches a spatial-context readout.)
    """

    def __init__(self, dim: int, depth: int = 9, heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            [ViT1DBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)  # 2D -> 1 token (spatial context)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))  # optional class/context token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, dim, H, W)  ->  (B, 1+?, dim) token sequence."""
        B, C, H, W = x.shape
        ctx = self.pool(x).flatten(2).transpose(1, 2)     # (B, 1, dim)
        cls = self.cls.expand(B, -1, -1)                  # (B, 1, dim)
        toks = torch.cat([cls, ctx], dim=1)               # (B, 2, dim)
        for blk in self.blocks:
            toks = blk(toks)
        return toks
