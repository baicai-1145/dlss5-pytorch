"""DLSS5 (nvngx_dlssnr.dll) PyTorch reimplementation — Phase 3 skeleton.

Package layout (per REPORT.md §5):
    model.py        DLSS5Net — Swin U-Net + 1D ViT control path + residual head
    swin_block.py   Swin Transformer block/stage (window attn, shifted alternation)
    vit1d_block.py  1D ViT block/encoder (control/UI mask context)
    patch_ops.py    ConvStem / PatchMerging / PatchExpanding
"""
from .model import DLSS5Net, default_config

__all__ = ["DLSS5Net", "default_config"]
__version__ = "0.3.0"
