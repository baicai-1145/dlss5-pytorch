"""Swin Transformer block / stage — from scratch (no timm).

Matches cubin naming: cc_tinlayout_fused_swin_{h}h_{d}_{n} / cc_split_swin_16h_512.
A stage = N SwinBlocks (window attention, alternating shifted/unshifted, relative
position bias) with depth N at hidden dim d.

Window attention uses the standard cyclic-shift + attention-mask formulation.
The mask is cached per (H, W) so repeated forward calls don't rebuild it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mp_cubic_silu import MpCubicSiLU


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """x: (B, C, H, W)  ->  windows (B*nH*nW, ws, ws, C)."""
    B, C, H, W = x.shape
    x = x.view(B, C, H // ws, ws, W // ws, ws)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int, C: int) -> torch.Tensor:
    """windows: (B*nH*nW, ws, ws, C)  ->  (B, C, H, W)."""
    B = windows.shape[0] // (H // ws * W // ws)
    x = windows.view(B, H // ws, W // ws, ws, ws, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(B, C, H, W)


def _shift_attn_mask(H: int, W: int, ws: int, shift: int, device, dtype) -> torch.Tensor:
    """Attention mask for cyclic-shifted windows.

    Returns (nW, ws*ws, ws*ws): 0 inside a window (relative to original grid),
    -100 across the shifted boundary.  nW = ceil(H/ws)*ceil(W/ws) after padding.
    """
    Hp = (H + ws - 1) // ws * ws
    Wp = (W + ws - 1) // ws * ws
    nH, nW = Hp // ws, Wp // ws
    img = torch.zeros((1, Hp, Wp, 1), device=device, dtype=dtype)
    # regions relative to the un-shifted grid (shifted by +shift so roll back aligns)
    h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    cnt = 0
    for hs in h_slices:
        for ws_ in w_slices:
            img[:, hs, ws_, :] = cnt
            cnt += 1
    mask_windows = window_partition(img.permute(0, 3, 1, 2), ws)  # (nW, ws, ws, 1)
    mask_windows = mask_windows.view(-1, ws * ws)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).to(dtype)
    return attn_mask


class WindowAttention(nn.Module):
    """Multi-head window attention with learnable relative position bias.

    Params: qkv (3*C*C), attn proj (C*C), rel-pos bias table (2ws-1)^2 * heads.
    """

    def __init__(self, dim: int, heads: int, window_size: int, qkv_bias: bool = True,
                 qk_scale=None, proj_bias: bool = True, rel_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.ws = window_size
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(0.0)
        self.rel_bias = rel_bias

        # relative position bias: (2*ws-1, 2*ws-1, heads)
        if rel_bias:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) ** 2, heads))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # precomputed relative position index (shared across all windows/stages with same ws)
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flat = coords.flatten(1)  # (2, ws*ws)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, L, L)
        rel = rel.permute(1, 2, 0).contiguous()                 # (L, L, 2)
        rel += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        idx = rel.sum(-1)  # (L, L)
        self.register_buffer("rel_index", idx, persistent=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B_win*nW?, n, dim); for the padded/rolled grid: (B, nW*ws*ws, dim)."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, hd)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, heads, N, N)
        if self.rel_bias:
            bias = self.relative_position_bias_table[self.rel_index.view(-1)].view(
                N, N, -1).permute(2, 0, 1)  # (heads, N, N)
            attn = attn + bias.unsqueeze(0)
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B // nW, nW, self.heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class SwinBlock(nn.Module):
    """One Swin transformer block (window MSA + MLP, both with residual).

    Params: dim, heads, window_size, shift; plus byte-budget knobs
    (ffn_hidden, qkv_bias, proj_bias, ffn1_bias, ffn2_bias, ln_mode, rel_bias).
    """

    def __init__(self, dim: int, heads: int, window_size: int, shift: bool,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, qk_scale=None,
                 proj_bias: bool = True, rel_bias: bool = True,
                 ffn_hidden: Optional[int] = None,
                 ffn1_bias: bool = True, ffn2_bias: bool = True,
                 ln_mode: str = "gb"):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.ws = window_size
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim, bias="b" in ln_mode)
        self.attn = WindowAttention(dim, heads, window_size, qkv_bias, qk_scale,
                                    proj_bias=proj_bias, rel_bias=rel_bias)
        self.norm2 = nn.LayerNorm(dim, bias="b" in ln_mode)
        hidden = ffn_hidden if ffn_hidden is not None else int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=ffn1_bias), MpCubicSiLU(),
            nn.Linear(hidden, dim, bias=ffn2_bias))
        self._mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def _attn_mask(self, H: int, W: int, device, dtype) -> torch.Tensor | None:
        if not self.shift:
            return None
        key = (H, W, self.ws, dtype)
        m = self._mask_cache.get(key)
        if m is None or m.device != device:
            m = _shift_attn_mask(H, W, self.ws, self.shift_size, device, dtype)
            self._mask_cache[key] = m
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) [channel-last internally]."""
        B, C, H, W = x.shape
        pad_h = (self.ws - H % self.ws) % self.ws
        pad_w = (self.ws - W % self.ws) % self.ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # to channel-last for attention
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        hp, wp = H + pad_h, W + pad_w
        shortcut = x

        x = self.norm1(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x = window_partition(x.permute(0, 3, 1, 2), self.ws).view(-1, self.ws * self.ws, C)
        attn_mask = self._attn_mask(hp, wp, x.device, x.dtype)
        x = self.attn(x, attn_mask)
        x = window_reverse(x, self.ws, hp, wp, C).permute(0, 2, 3, 1)  # (B, hp, wp, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        if pad_h or pad_w:
            x = x[:, :H, :W, :]
        return x.permute(0, 3, 1, 2)  # (B, C, H, W)


class SwinStage(nn.Module):
    """Stage of `depth` SwinBlocks, alternating unshifted/shifted windows.

    dim == number of channels; heads == num heads (head_dim = dim/heads).
    """

    def __init__(self, dim: int, depth: int, heads: int, window_size: int = 8,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, qk_scale=None,
                 proj_bias: bool = True, rel_bias: bool = True,
                 ffn_hidden: Optional[int] = None,
                 ffn1_bias: bool = True, ffn2_bias: bool = True,
                 ln_mode: str = "gb", block_cfg: Optional[list] = None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} not divisible by heads {heads}"
        self.dim = dim
        cfgs = block_cfg or [dict(ffn_hidden=ffn_hidden, ffn1_bias=ffn1_bias,
                                  ffn2_bias=ffn2_bias, ln_mode=ln_mode,
                                  proj_bias=proj_bias, rel_bias=rel_bias)] * depth
        self.blocks = nn.ModuleList([
            SwinBlock(dim, heads, window_size, shift=(i % 2 == 1), mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, **cfgs[i % len(cfgs)])
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
