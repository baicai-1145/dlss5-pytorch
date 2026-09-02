"""Patch ops — from scratch (no timm).

- ConvStem      : conv-projection front-end (optional, replaces patchify / Linear patch embed)
- PatchMerging  : 2x2 merge + Linear   (dim*4 -> dim_next)   [cc_tinlayout-style downsample]
- PatchExpanding: Linear + pixel-unshuffle (1 pixel -> 4), dim/4 expansion
                  (decoder upsample; REPORT: cc_dec_input_upsample_1024_512 etc.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """3-stage conv stem: conv3x3-bn-relu at patch/4 res. (config-gated; default off)"""

    def __init__(self, in_chans: int, out_dim: int, norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_dim // 2, kernel_size=3, stride=1, padding=1, bias=not norm)
        self.norm1 = nn.LayerNorm(out_dim // 2) if norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_dim // 2, out_dim, kernel_size=3, stride=1, padding=1, bias=not norm)
        self.norm2 = nn.LayerNorm(out_dim) if norm else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if isinstance(self.norm1, nn.LayerNorm):
            x = x.permute(0, 2, 3, 1); x = self.norm1(x).permute(0, 3, 1, 2)
        x = self.act(x)
        x = self.conv2(x)
        if isinstance(self.norm2, nn.LayerNorm):
            x = x.permute(0, 2, 3, 1); x = self.norm2(x).permute(0, 3, 1, 2)
        x = self.act(x)
        return x


class PatchMerging(nn.Module):
    """2x2 patch merging: rearrange (C -> 4C), LayerNorm, linear 4C -> dim_next."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.norm = nn.LayerNorm(4 * in_dim)
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  ->  (B, out_dim, H/2, W/2)."""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()          # (B, H, W, C)
        x = x.view(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()    # (B, H/2, W/2, 2, 2, C)
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)                            # (B, H/2, W/2, out_dim)
        return x.permute(0, 3, 1, 2)


class PatchExpanding(nn.Module):
    """Decoder upsample: linear in_dim -> (4*out_dim), then pixel-shuffle (space-to-depth inverse)
    to double spatial res with 1/4 channels. (cc_dec_input_upsample_1024_512: 1024->512)

    x: (B, C, H, W) -> (B, out_dim, 2H, 2W) with optional 1x1 fuse conv after concat skip.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.norm = nn.LayerNorm(in_dim)
        self.expand = nn.Linear(in_dim, 4 * out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  ->  (B, out_dim, 2H, 2W)."""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = self.norm(x)
        x = self.expand(x)                       # (B, H, W, 4*out)
        x = x.permute(0, 3, 1, 2)                # (B, 4*out, H, W)
        x = F.pixel_shuffle(x, 2)                # (B, out, 2H, 2W)
        return x

