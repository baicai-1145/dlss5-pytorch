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


class _CubicSelfAttention(nn.Module):
    """Self-attention with the official score activation: clamp(±4) + MpCubicSiLU,
    no softmax (SASS: no exp/max/sum anywhere in the DLL).

    Parameter names deliberately match nn.MultiheadAttention
    (in_proj_weight / in_proj_bias / out_proj.weight / out_proj.bias) so any
    weight mapping written against MHA stays valid.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        attn = (q * self.scale) @ k.transpose(-2, -1)   # scores
        attn = mp_cubic_silu(attn)                      # clamp(±4) + cubic, no softmax
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class ViT1DBlock(nn.Module):
    """Pre-LN transformer block with the official cubic-score attention
    (norm -> MSA -> res, norm -> MLP -> res)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CubicSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), MpCubicSiLU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
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
