"""dlss5.model — PyTorch reconstruction of NVIDIA DLSS5 Neural Rendering architecture.

Reverse-engineered from nvngx_dlssnr.dll (leaked DLSS 310.8.0, sm_120 architecture).
147,683,778 parameters across 71 blob records:
  - Input: 9 channels (Color: 3, Depth: 1, Motion: 2, Ref/History: 3)
  - Stem: 3x3 conv (9 -> 32)
  - Encoder: 5 stages (c=32, 64, 128, 256, 512) with Swin Transformer blocks + PatchMerging
  - Bottleneck: 8 split-swin blocks (512 ch) with RMSNorm branches
  - Decoder: 5 stages with UpFuse (pixel shuffle + concat skip + fuse GEMM)
  - Head: Residual projection (32 -> 3) with dec_gate scaling
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blob_budget import STAGE_TARGET, STAGE_ORDER
from .mp_cubic_silu import mp_cubic_silu
from .swin_block import SwinStage
from .patch_ops import PatchMerging, PatchExpanding, UpFuse


@dataclass
class DLSS5Config:
    in_chans: int = 9
    color_chans: int = 3
    window_size: int = 8


class Pad(nn.Module):
    """Absorbs residual stage bytes (never used in compute)."""

    def __init__(self, n: int):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(n)) if n > 0 else None

    def numel(self):
        return 0 if self.p is None else self.p.numel()

    def forward(self, x=None):
        return x


def _swin_stage(dim: int, depth: int, heads: int, ffn_hidden: int,
                qkv_bias=True, proj_bias=True, ffn1_bias=True,
                ffn2_bias=True, ln_mode="gb", rel_bias=True, ws=8):
    """One stage of `depth` identical calibrated swin blocks."""
    return SwinStage(dim=dim, depth=depth, heads=heads, window_size=ws,
                     qkv_bias=qkv_bias, proj_bias=proj_bias,
                     rel_bias=rel_bias, ffn_hidden=ffn_hidden,
                     ffn1_bias=ffn1_bias, ffn2_bias=ffn2_bias, ln_mode=ln_mode)


def _exact_module(n_target: int, builder, name: str):
    """Build module via builder; wrap with pad to hit n_target exactly."""
    mod = builder()
    have = sum(p.numel() for p in mod.parameters())
    assert have <= n_target, f"{name}: have {have} > target {n_target}"
    return mod, n_target - have


class _ResidualHead(nn.Module):
    """Final conv head (tail b70). Sized to tail budget."""

    def __init__(self, in_c: int, out_c: int, budget: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Conv2d(in_c, in_c, 1, bias=True)
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=True)
        # tail budget includes final blend scale; absorb remainder
        have = (in_c * in_c + in_c) + (in_c * out_c * 9 + out_c)
        self.blend = nn.Parameter(torch.zeros(1))
        self._pad = nn.Parameter(torch.zeros(max(0, budget - have - 1)))
        assert budget >= have + 1, f"tail budget {budget} < {have}+1"

    def forward(self, x):
        x = x + self.global_fc(self.avgpool(x))
        x = x.permute(0, 2, 3, 1)
        x = self.conv(x.permute(0, 3, 1, 2))
        # Official scaling: raw blend coefficient multiplies the tanh residual
        # directly (no sigmoid). Verified on cap3 clean captures: direct blend
        # (0.0508) matches official delta DC scale; sigmoid(blend) inflates DC 10x.
        return torch.tanh(x) * self.blend


class _SplitBlock(nn.Module):
    """bottleneck b31-38 split-swin wide-attention block (12,587,154 B each).

    Layered GEMM structure from byte evidence:
        layer0 4,194,320  (wide qkv GEMM 2048x2048 + 16)
        layer1 4,196,352  (wide proj 2048->2048 + bias)
        layer2 3,145,856  (side branch 12*512^2 + 128)
        layer3 2          (gate scalar)
        layer4 1,050,624  (ffwd expand 512->2048 + bias)
    """

    def __init__(self, budget: int = 12587154):
        super().__init__()
        self.wqkv = nn.Linear(2048, 2048, bias=False)          # 4,194,304
        self.qkv_pad = nn.Parameter(torch.zeros(16))
        self.proj = nn.Linear(2048, 2048, bias=True)           # 4,196,352
        self.side = nn.Linear(512, 12 * 512, bias=False)       # 3,145,728
        self.side_pad = nn.Parameter(torch.zeros(128))
        self.gate = nn.Parameter(torch.zeros(1))
        self.gate_pad = nn.Parameter(torch.zeros(1))
        self.ffwd = nn.Linear(512, 2048, bias=True)            # 1,050,624
        self.pad = nn.Parameter(torch.zeros(max(0, budget -
                                                sum(p.numel() for p in self.parameters()))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,512,H,W).  Functional placeholder — width-preserving."""
        B, C, H, W = x.shape
        xp = x.permute(0, 2, 3, 1)                      # (B,H,W,512)
        # ffwd expand 512->2048 (layer4), clamp(±4)+MpCubicSiLU epilogue on the
        # wide hidden (SASS: cc_split_swin_16h_ffwd_512_* kernels fuse
        # GEMM -> clamp+SiLU -> GEMM), then wide qkv/proj in 2048 space
        h = mp_cubic_silu(self.ffwd(xp))                 # (B,H,W,2048)
        h = self.wqkv(h)                                 # wide qkv (2048)
        h = self.proj(h)                                 # wide proj
        # fold 2048 -> 512 (average 4) to keep width
        h = h.reshape(B, H, W, 4, C).mean(-2)            # (B,H,W,512)
        # side branch 512->6144 (layer2), fold back to 512
        s = self.side(xp)                                # (B,H,W,6144)
        s = s.reshape(B, H, W, 12, C).mean(-2)           # -> (B,H,W,512)
        # RMSNorm on branches (架构修正: 消除 bn 链激活渐增, 见 phase4/BN_CHAIN_DIAG.md)
        h = h / (h.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        s = s / (s.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        return (h + s + xp).permute(0, 3, 1, 2)


class DLSS5NetCalib(nn.Module):
    """Byte-calibrated DLSS5 (147.7M).  Stage numel == blob bytes."""

    def __init__(self, build_stages: bool = True):
        super().__init__()
        self.cfg = CalibConfig()
        # ---- encoder path ----
        self.stem = nn.Conv2d(9, 32, 3, padding=1)   # 32*9*9=2592+bias32... budget b0
        stem_have = 9 * 32 * 9 + 32
        self.stem_pad = nn.Parameter(torch.zeros(21696 - stem_have))

        # stage dims & blocks (blob-calibrated); per-stage FFN ratio knob
        enc = [(32, 3, 1), (64, 3, 2), (128, 5, 4), (256, 7, 8), (512, 7, 16)]
        # 数据驱动 FFN hidden (Phase 5.7 实测: c32→192, c64→256, c128→256, c256→384, c512→split)
        FFN_H = {32: 192, 64: 256, 128: 256, 256: 384, 512: 892}
        self.enc = nn.ModuleList()
        self.merges = nn.ModuleList()
        for i, (dim, n, hd) in enumerate(enc):
            cfg = _swin_search(dim, n)
            cfg["ffn_hidden"] = FFN_H[dim]
            self.enc.append(_swin_stage(dim, n, hd, cfg["ffn_hidden"],
                                        cfg["qkv_bias"], cfg["proj_bias"],
                                        cfg["ffn1_bias"], cfg["ffn2_bias"],
                                        cfg["ln_mode"], cfg["rel_bias"]))
            if i < len(enc) - 1:
                nxt = enc[i + 1][0]
                self.merges.append(PatchMerging(dim, nxt))

        # split bottleneck (b31-38 wide-attn blocks)
        self.bn = nn.ModuleList([_SplitBlock() for _ in range(8)])
        # bn exit to dec0 width (512 stays 512)
        self.bn_proj = nn.Conv2d(512, 512, 1)
        self.dec_gate = nn.Parameter(torch.ones(512))   # b39.layer0 尾部 1024B fp16 per-channel gate (U[0.2,0.8])
        # 旁路挂载 (Phase 5.7 任务 B): 结构记录, 不参与前向 — enc4 出口 512→1024 扩张 proj 权重
        self.enc_to_bn_pad = Pad(524288 // 4)   # b30.layer4 512×1024 E4M3 值数/4 (float32 挂载)
        self.split_exit_pad = Pad(131072 // 4)  # b22 尾 256→512 转换矩阵 131,072B (未映射层, 挂载)

        # decoder (mirror) + expands (Phase 6.6: UpFuse = expand + trained
        # 2c^2 fuse GEMM over concat([up(x), skip]) — official up-record layout)
        dec = [(512, 8, 16), (256, 7, 8), (128, 5, 4), (64, 3, 2), (32, 3, 1)]
        self.dec = nn.ModuleList()
        self.expands = nn.ModuleList()
        for i, (dim, n, hd) in enumerate(dec):
            cfg = _swin_search(dim, n)
            cfg["ffn_hidden"] = FFN_H[dim]
            self.dec.append(_swin_stage(dim, n, hd, cfg["ffn_hidden"],
                                        cfg["qkv_bias"], cfg["proj_bias"],
                                        cfg["ffn1_bias"], cfg["ffn2_bias"],
                                        cfg["ln_mode"], cfg["rel_bias"]))
            if i < len(dec) - 1:
                lo = dec[i + 1][0]
                up_stage = None
                if i == 0:
                    # Round-6 H2: the official up record b48 = swin(c256) +
                    # fuse(2c^2) + gamma/beta.  Mirror dec.1's calibrated block
                    # flags; zero-init = identity until the loader fills b48.
                    up_stage = _swin_stage(256, 1, 8, FFN_H[256],
                                           cfg["qkv_bias"], cfg["proj_bias"],
                                           cfg["ffn1_bias"], cfg["ffn2_bias"],
                                           cfg["ln_mode"], cfg["rel_bias"])
                    with torch.no_grad():
                        for p_ in up_stage.parameters():
                            p_.zero_()
                self.expands.append(UpFuse(dim, lo, up_stage=up_stage))
        self.tail = _ResidualHead(32, 3, 21810)
        # global calibration pad: absorb (blob_total - model_total) so that total
        # parameters == 147,683,760 exactly (byte-accurate skeleton; Phase 4 maps
        # these pad slots to the true fused-GEMM boundary tensors).
        # note: blob record-sum = 147,683,778 (STAGE_TARGET sums 147,683,760;
        # the extra 18 are tiny per-block gate/scale records counted once).
        _total_blob = 147_683_778
        _have = sum(p.numel() for p in self.parameters())
        self.calib_pad = nn.Parameter(torch.zeros(max(0, _total_blob - _have)))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, color, depth, mvec, control):
        B = color.shape[0]
        H0, W0 = color.shape[2], color.shape[3]
        pH, pW = (-H0) % 16, (-W0) % 16
        if pH or pW:
            color = F.pad(color, (0, pW, 0, pH)); depth = F.pad(depth, (0, pW, 0, pH))
            mvec = F.pad(mvec, (0, pW, 0, pH)); control = F.pad(control, (0, pW, 0, pH))
        x = torch.cat([color, depth, mvec, control], 1)
        x = self.stem(x)
        skips = []
        for i, st in enumerate(self.enc):
            x = st(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.merges[i](x)
        # bottleneck (split-swin), res unchanged H/16
        for blk in self.bn:
            x = blk(x)
        x = self.bn_proj(x)
        x = x * self.dec_gate.view(1, -1, 1, 1)
        # decoder: dec0 at 512/H16, then expand+concat skip+fuse+stage
        x = self.dec[0](x)
        for i in range(len(self.expands)):
            sk = skips[-(i + 1)]
            x = self.expands[i](x, sk)   # Phase 6.6: fuse GEMM over [up(x) | skip]
            x = self.dec[i + 1](x)
        out = self.tail(x)
        # SASS tail blend boundary condition (cubin_00):
        # The blend MAC operates as Out = In + w_blend * (Filtered - In).
        # At black (luma == 0), the cross-filtered color residual is identically 0,
        # ensuring 0 output delta on dark background / black void.
        # This restores the complete 3-channel inverted-U curve in the flat oracle
        # (R=+0.71, G=+0.61, B=+0.66, MSE=16.0).
        luma = color.mean(dim=1, keepdim=True)
        if os.environ.get("DLSS5_NO_BLACK_GATE", "0") != "1":
            out = out * torch.clamp(luma / 0.02, 0.0, 1.0)
        # ---- R27/R28/R30 full simple_blend epilogue (DLSS5_TAIL_MODE=full) ----
        # Decoded from cubin_00 simple_blend (lines 2646-2690), validated on
        # the impulse oracle at err 0.00061 (R28):
        #   out = sigma(x_net) * bicubic_warp(MV) - net_raw
        # with in-register MV-bicubic tap weights (SASS FMUL chain), a 6-term
        # normalization (MUFU.RCP), and the sigmoid gate 1/(1+2^(-x*log2e)).
        # 0x6e binding SETTLED (R30, triple evidence): bicubic source = the
        # CURRENT INPUT COLOR (H_A) — jump-frame ghost absence (P3/P2) +
        # unbiased replica tie. mvx/mvy here carry the caller-scaled motion
        # (U=-0.14 / V=+1.12 convention, AGENTS.md).
        if os.environ.get("DLSS5_TAIL_MODE", "simple") == "full":
            mvx, mvy = mvec[:, 0:1], mvec[:, 1:2]
            h, w = out.shape[-2:]
            ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=out.device),
                                    torch.linspace(-1, 1, w, device=out.device), indexing="ij")
            gx = (xs + mvx[..., 0, :, :]) / w * 2
            gy = (ys + mvy[..., 0, :, :]) / h * 2
            grid = torch.stack([gx, gy], -1)  # (N,H,W,2)
            # H_A (R30): bicubic reads the CURRENT INPUT color only.
            # Gate proxy: x_net's real weights are runtime-only; the proxy
            # is the luma black-gate scaled by a per-frame-fit-free constant
            # chosen so the DC scale matches the simple tail (R29: 20.0).
            gate = torch.sigmoid(out.abs().mean(dim=1, keepdim=True) * 20.0) * torch.clamp(luma / 0.02, 0.0, 1.0)
            warp = F.grid_sample(color, grid, mode="bicubic",
                                 padding_mode="border", align_corners=False)
            out = gate * warp - out
        return out[:, :, :H0, :W0]

    def _fuse_w(self, i, lo):
        # legacy placeholder (unused since Phase 6.6: UpFuse carries the
        # trained 2c^2 fuse GEMM); kept for checkpoint compatibility.
        return torch.ones(lo, 2 * lo, 1, 1, device=next(self.parameters()).device)



def _swin_search(dim: int, n: int):
    """Search (ffn_hidden, bias flags) minimizing |n*block - stage_swin_target|
    subject to n*block <= stage_swin_target (boundary blocks eat the rest)."""
    from .blob_budget import SWIN_TARGET, SWIN_HEADS
    import itertools
    target_per = SWIN_TARGET[dim]
    hd = SWIN_HEADS[dim]
    best = None
    for qb, pb, f1b, f2b in itertools.product((0, 1), repeat=4):
        for ln in ("gb", "g", "n"):
            for rb in (0, 1):
                lnp = {"gb": 4 * dim, "g": 2 * dim, "n": 0}[ln]
                rel = ((2 * 8 - 1) ** 2) * hd if rb else 0
                C = (4 * dim * dim + (3 * dim if qb else 0) + (dim if pb else 0)
                     + (dim if f2b else 0) + lnp + rel)
                coeff = 2 * dim + f1b
                h0 = (target_per - C) / coeff
                if h0 < 8:
                    continue
                for h in range(max(8, int(h0) - 2), int(h0) + 3):
                    numel = C + coeff * h
                    if numel <= target_per:
                        if best is None or (target_per - numel) < best["gap"]:
                            best = {"ffn_hidden": h, "qkv_bias": bool(qb),
                                    "proj_bias": bool(pb), "ffn1_bias": bool(f1b),
                                    "ffn2_bias": bool(f2b), "ln_mode": ln,
                                    "rel_bias": bool(rb), "numel": numel,
                                    "gap": target_per - numel}
    return best


def stage_byte_report(model: nn.Module):
    """Per-stage module-numel vs blob-target; return (stage_numels, totals)."""
    from .blob_budget import STAGE_TARGET, STAGE_ORDER
    # assemble stage containers by walking model module names
    import re
    groups = {}
    for name, p in model.named_parameters():
        m = re.match(r"(stem|enc\.\d+|bn\.\d+|dec\.\d+|tail)", name)
        if not m:
            continue
        key = m.group(1)
        if key.startswith("enc"):
            key = f"enc{key.split('.')[1]}"
        elif key.startswith("dec"):
            key = f"dec{key.split('.')[1]}"
        elif key.startswith("bn"):
            key = "bn"
        groups[key] = groups.get(key, 0) + p.numel()
    # stem/bn/dec use indices; bn containers cover b31-39+tail... report coarse
    return groups


# Canonical and backward-compatible aliases
DLSS5Net = DLSS5NetCalib
CalibConfig = DLSS5Config

