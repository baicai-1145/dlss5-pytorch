"""blob_budget.py — DLSS5 weights blob byte budget (Phase 3 calibration).

Maps the 153 blob records (block0..block70) to 13 model stage containers with
exact per-stage byte totals.  Blob bytes (B field) per block come from the
chained record parse (weights_blob.bin).  The model's stage containers are
constructed so total parameter numel == per-stage blob bytes (loader aligns
13/13), absorbing small residuals into per-stage pad parameters.

Stage grouping (all 71 blocks):
    stem      [0]
    enc0      [1,2,3,4]       3x swin32 + merge
    enc1      [5,6,7,8]       3x swin64 + merge
    enc2      [9..14]         5x swin128 + merge
    enc3      [15..22]        7x swin256 + merge
    enc4      [23..30]        7x swin512 + xform
    bn        [31..39]        8x split-swin + exit
    dec0      [40..48]        8x swin512 + up
    dec1      [49..56]        7x swin256 + up
    dec2      [57..62]        5x swin128 + up
    dec3      [63..66]        3x swin64 + up
    dec4      [67..69]        3x swin32
    tail      [70]
"""
from __future__ import annotations

import itertools

# --------------------------------------------------------------------------
# per-block byte totals (from parse of weights_blob.bin; multi-record blocks
# summed over layer0..N).  B field = weight-stream bytes.
BLOCK_B = {
    0: 21696,
    1: 20672, 2: 20672, 3: 20672,
    4: 22720,
    5: 61760, 6: 61760, 7: 61760,
    8: 69936,
    9: 197184, 10: 197184, 11: 197184, 12: 197184, 13: 197184,
    14: 229936,
    15: 689232, 16: 689232, 17: 689232, 18: 689232, 19: 689232,
    20: 689232, 21: 689232,
    22: 820288,
    23: 1968192, 24: 1968192, 25: 1968192, 26: 1968192, 27: 1968192,
    28: 1968192, 29: 1968192,
    30: 2492496,
    31: 12587154, 32: 12587154, 33: 12587154, 34: 12587154, 35: 12587154,
    36: 12587154, 37: 12587154, 38: 12587154,
    39: 525312,
    40: 1968192, 41: 1968192, 42: 1968192, 43: 1968192, 44: 1968192,
    45: 1968192, 46: 1968192, 47: 1968192,
    48: 820784,
    49: 689232, 50: 689232, 51: 689232, 52: 689232, 53: 689232,
    54: 689232, 55: 689232,
    56: 230176,
    57: 197184, 58: 197184, 59: 197184, 60: 197184, 61: 197184,
    62: 70048,
    63: 61760, 64: 61760, 65: 61760,
    66: 22784,
    67: 20672, 68: 20672, 69: 20672,
    70: 21810,
}

STAGE_BLOCKS = {
    "stem": [0],
    "enc0": [1, 2, 3, 4],
    "enc1": [5, 6, 7, 8],
    "enc2": list(range(9, 15)),
    "enc3": list(range(15, 23)),
    "enc4": list(range(23, 31)),
    "bn": list(range(31, 40)),
    "dec0": list(range(40, 49)),
    "dec1": list(range(49, 57)),
    "dec2": list(range(57, 63)),
    "dec3": list(range(63, 67)),
    "dec4": list(range(67, 70)),
    "tail": [70],
}

STAGE_ORDER = ["stem", "enc0", "enc1", "enc2", "enc3", "enc4",
               "bn", "dec0", "dec1", "dec2", "dec3", "dec4", "tail"]

STAGE_TARGET = {k: sum(BLOCK_B[b] for b in blocks)
                for k, blocks in STAGE_BLOCKS.items()}

# --------------------------------------------------------------------------
# Swin-block config search: find a (dim, heads) swin block whose parameter
# numel is <= target and as close as possible (per-stage pad absorbs residue).
# search target bounds are chosen so every stage container pad stays >= 0.

def swin_numel(dim: int, heads: int, ws: int, ffn_hidden: int,
               qkv_bias: bool, proj_bias: bool, ffn1_bias: bool,
               ffn2_bias: bool, ln_mode: str, rel_bias: bool) -> int:
    """Exact parameter count of a SwinBlock for the given config."""
    lnp = {"gb": 4 * dim, "g": 2 * dim, "n": 0}[ln_mode]   # 2 LayerNorms
    rel = ((2 * ws - 1) ** 2) * heads if rel_bias else 0
    return (4 * dim * dim                                   # qkv(3c2)+proj(c2) weights
            + (3 * dim if qkv_bias else 0)
            + (dim if proj_bias else 0)
            + 2 * dim * ffn_hidden                          # ffn weights (both dirs)
            + (ffn_hidden if ffn1_bias else 0)
            + (dim if ffn2_bias else 0)
            + lnp + rel)


def search_swin(dim: int, heads: int, target: int, ws: int = 8) -> dict:
    """Find config with numel <= target, minimal (target - numel) >= 0."""
    best = None
    for qb, pb, f1b, f2b in itertools.product((0, 1), repeat=4):
        for ln in ("gb", "g", "n"):
            for rb in (0, 1):
                # linear in h: numel = C + h * (2*dim + f1b)
                lnp = {"gb": 4 * dim, "g": 2 * dim, "n": 0}[ln]
                rel = ((2 * ws - 1) ** 2) * heads if rb else 0
                C = (4 * dim * dim + (3 * dim if qb else 0) + (dim if pb else 0)
                     + (dim if f2b else 0) + lnp + rel)
                coeff = 2 * dim + f1b
                h0 = (target - C) / coeff
                if h0 < 8:
                    continue
                for h in range(max(8, int(h0) - 3), int(h0) + 4):
                    n = C + coeff * h
                    if n <= target:
                        pad = target - n
                        if best is None or pad < best["pad"]:
                            best = {"dim": dim, "heads": heads, "ws": ws,
                                    "ffn_hidden": h, "qkv_bias": bool(qb),
                                    "proj_bias": bool(pb), "ffn1_bias": bool(f1b),
                                    "ffn2_bias": bool(f2b), "ln_mode": ln,
                                    "rel_bias": bool(rb), "numel": n,
                                    "pad": pad}
    if best is None:
        raise ValueError(f"no swin config for dim={dim} heads={heads} "
                         f"target={target}")
    return best


# stage targets for the plain-swin search (<= bounds that keep all stage pads >=0)
SWIN_TARGET = {32: 20672, 64: 61756, 128: 197181, 256: 689230, 512: 1968194}
SWIN_HEADS = {32: 1, 64: 2, 128: 4, 256: 8, 512: 16}


def swin_config_for(dim: int) -> dict:
    return search_swin(dim, SWIN_HEADS[dim], SWIN_TARGET[dim])
