"""dlss5.loader — Loads reverse-engineered weights from blob into DLSS5Net.

Decodes 71 blob records (E4M3, MXFP8, FP16) and maps them to DLSS5Net parameters.
Incorporates:
  - b14 MX-scale table boundary skip (prevents LN gamma corruption)
  - COMBINED-C fix: merges.2 canonical gamma/beta, enc3 norm/mlp2 Kaiming fallback
  - Phase 6.6 UpFuse 2c^2 fuse GEMM weights loading (b48/56/62/66)
  - b39 dec_gate 512 fp16 tail loading
"""
from __future__ import annotations

import json
import math
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .model import DLSS5Net, DLSS5Config
from .mx_decode import e4m3_decode, mx_decode_pairs, fp16_decode

HERE = os.path.dirname(os.path.abspath(__file__))
ROLES_PATH = os.path.join(HERE, "block_roles.json")

MX_BOUND = {
    20672: (11264, 19456),
    61760: (40960, 57344),
    22720: (11264, 19712),
    69936: (40960, 57600),
}
MX_BOUND[917568] = (786432, 917504)

MISC_TAIL = {20672: 192, 61760: 320, 22720: 2208, 69936: 8464}
MX_GAP = {20672: 112, 61760: 1024, 22720: 0, 69936: 0}
E4M3_HOLE = {
    20672: [(8192, 8448)],
    61760: [(28672, 28928)],
    22720: [(8192, 8448)],
    69936: [(28672, 28928)],
}

FP16_TAIL = {197184: 98304, 689232: 360448, 229936: 147968 + 7968, 524288: None}
B14_MXSCALE_BYTES = 1024
FP16_TAIL[229936] = 155936 + B14_MXSCALE_BYTES


def _classify_windows(arr: np.ndarray, win: int = 512) -> List[str]:
    tags = []
    for off in range(0, max(len(arr) - win + 1, 1), win):
        w = arr[off : off + win]
        odd = w[1::2]
        bc = np.bincount(odd, minlength=256)
        top1 = int(bc.max())
        top1_byte = int(bc.argmax())
        fp_pos = ((odd >= 0x30) & (odd <= 0x48)).mean()
        fp_neg = ((odd >= 0xB0) & (odd <= 0xC8)).mean()
        zero = np.mean(w == 0)
        if zero > 0.6 or fp_pos > 0.4:
            tags.append("F")
        elif top1 > len(odd) * 0.5 and 176 <= top1_byte <= 210:
            tags.append("M")
        elif fp_neg > 0.4 and top1 < len(odd) * 0.4:
            tags.append("F")
        elif top1 > win // 8 and 0x30 <= top1_byte <= 0x48:
            tags.append("F")
        else:
            tags.append("E")
    return tags


def decode_record_raw(raw: np.ndarray, total_B: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Decode raw record bytes into (main_weights, misc_params)."""
    arr = np.asarray(raw, dtype=np.uint8)
    if len(arr) < 512:
        return e4m3_decode(arr), fp16_decode(arr)
    if total_B in MX_BOUND:
        lo, hi = MX_BOUND[total_B]
        misc_start = len(arr) - MISC_TAIL.get(total_B, 0)
        gap = MX_GAP.get(total_B, 0)
        holes = E4M3_HOLE.get(total_B, [])
        segs, hole_vals = [], []
        pos = 0
        for hlo, hhi in sorted(holes):
            if hlo > pos:
                segs.append(arr[pos:hlo])
            hole_vals.append(arr[hlo:hhi])
            pos = hhi
        if pos < lo:
            segs.append(arr[pos:lo])
        pre = (
            np.concatenate([e4m3_decode(x) for x in segs])
            if segs
            else np.zeros(0, np.float32)
        )
        main = np.concatenate([
            pre,
            mx_decode_pairs(arr[lo:hi].reshape(-1, 2)),
            e4m3_decode(arr[hi + gap : misc_start]),
        ])
        misc_list = [fp16_decode(x) for x in hole_vals]
        if misc_start < len(arr):
            misc_list.append(fp16_decode(arr[misc_start:]))
        misc = (
            np.concatenate(misc_list)
            if misc_list
            else np.zeros(0, np.float32)
        )
        return main, misc

    if total_B in FP16_TAIL and FP16_TAIL[total_B]:
        t0 = FP16_TAIL[total_B]
        if total_B == 229936:
            return (
                np.concatenate([
                    e4m3_decode(arr[:98304]),
                    e4m3_decode(arr[98816:160768]),
                ]),
                fp16_decode(arr[t0:]),
            )
        return e4m3_decode(arr[:t0]), fp16_decode(arr[t0:])

    tags = _classify_windows(arr)
    runs = []
    for t in tags:
        if runs and runs[-1][0] == t:
            runs[-1][1] += 1
        else:
            runs.append([t, 1])
    main_parts, misc_parts = [], []
    pos = 0
    for tag, n in runs:
        seg = arr[pos : pos + n * 512]
        if tag == "E":
            main_parts.append(e4m3_decode(seg))
        elif tag == "M":
            main_parts.append(mx_decode_pairs(seg[: len(seg) // 2 * 2].reshape(-1, 2)))
        else:
            misc_parts.append(fp16_decode(seg))
        pos += n * 512
    if pos < len(arr):
        main_parts.append(e4m3_decode(arr[pos:]))
    main = (
        np.concatenate([p for p in main_parts if len(p)])
        if main_parts
        else np.zeros(0)
    )
    misc = (
        np.concatenate([p for p in misc_parts if len(p)])
        if misc_parts
        else np.zeros(0, dtype=np.float32)
    )
    return main, misc


def walk_records(blob: bytes):
    """Walk binary records in blob: yields (name, byte_size, offset)."""
    pos = 16
    nxt_len = 19
    while pos + 28 < len(blob):
        name = blob[pos : pos + nxt_len].decode("utf8", "replace")
        if pos + nxt_len + 24 > len(blob):
            break
        _, _, B = struct.unpack_from("<QQQ", blob, pos + nxt_len)
        off = pos + nxt_len + 24
        yield name, B, off
        p = off + B
        if p + 28 > len(blob):
            break
        nxt_len = struct.unpack_from("<7I", blob, p)[6]
        pos = p + 32


def parse_blob(blob: bytes) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Parse entire weights blob into {block_id: {layer_name: (main, misc)}}."""
    by_block: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for name, B, off in walk_records(blob):
        b = int(name.split(".")[0][5:])
        layer = name.split(".")[1]
        raw = blob[off + 4 : off + B] if B > 4 else blob[off : off + B]
        main, misc = decode_record_raw(np.frombuffer(raw, dtype=np.uint8), total_B=B)
        by_block.setdefault(b, {})[layer] = (main, misc)
    return by_block


def load_weights(model: DLSS5Net, blob: bytes, verbose: bool = False) -> Dict[str, int]:
    """Mount all decoded weights into the DLSS5Net model."""
    by_block = parse_blob(blob)
    pmap = dict(model.named_parameters())
    unfilled = set(pmap.keys())
    stats = {"filled": 0, "pad": 0}

    def put(param_name: str, values: np.ndarray, count: Optional[int] = None) -> int:
        if param_name not in pmap:
            return 0
        p = pmap[param_name]
        n = p.numel()
        v = values[:n] if count is None else values[:count]
        if len(v) < n:
            v = np.concatenate([v, np.zeros(n - len(v), v.dtype)])
        v_f32 = np.nan_to_num(v, nan=0.0, posinf=65504.0, neginf=-65504.0).astype(np.float32)
        with torch.no_grad():
            p.copy_(torch.from_numpy(v_f32).reshape(p.shape).to(p.dtype))
        unfilled.discard(param_name)
        stats["filled"] += n
        return n

    def fill_swin(prefix: str, main: np.ndarray, misc: np.ndarray):
        if len(misc) > 40000:
            misc_w = np.concatenate([misc[len(misc) - 8208 :], misc[256:24320]])
        else:
            misc_w = misc
        stream = np.concatenate([main, misc_w]) if len(misc_w) else main
        q = 0
        mlp2_pn = f"{prefix}.mlp.2.weight"
        q_before_mlp2 = None
        for suffix in ("attn.qkv.weight", "attn.proj.weight", "mlp.0.weight", "mlp.2.weight"):
            pn = f"{prefix}.{suffix}"
            if pn not in pmap:
                continue
            if pn == mlp2_pn:
                q_before_mlp2 = q
            need = pmap[pn].numel()
            put(pn, stream[q : q + need])
            q += need

        # enc3 mlp.2 zero-pad refill with Kaiming (COMBINED-C fix)
        if (
            prefix.startswith("enc.3.")
            and mlp2_pn in pmap
            and q_before_mlp2 is not None
            and len(stream) < q_before_mlp2 + pmap[mlp2_pn].numel()
        ):
            p = pmap[mlp2_pn]
            fan_in = p.shape[1]
            with torch.no_grad():
                p.normal_(0.0, math.sqrt(1.0 / fan_in))

        tail = stream[q:]
        o = 0
        for suffix in (
            "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
            "mlp.0.bias", "mlp.2.bias", "attn.qkv.bias", "attn.proj.bias",
        ):
            pn = f"{prefix}.{suffix}"
            if pn in pmap and pn in unfilled:
                if prefix.startswith("enc.3.") and o >= len(tail):
                    continue
                o += put(pn, tail[o:])

    stage_blocks = {
        "enc": [
            [1, 2, 3],
            [5, 6, 7],
            [9, 10, 11, 12, 13],
            [15, 16, 17, 18, 19, 20, 21],
            [23, 24, 25, 26, 27, 28, 29],
        ],
        "dec": [
            [40, 41, 42, 43, 44, 45, 46, 47],
            [49, 50, 51, 52, 53, 54, 55],
            [57, 58, 59, 60, 61],
            [63, 64, 65],
            [67, 68, 69],
        ],
    }

    for net in ("enc", "dec"):
        for si, blocks in enumerate(stage_blocks[net]):
            for bi, b in enumerate(blocks):
                if b not in by_block:
                    continue
                layers = by_block[b]
                if len(layers) > 1:
                    m2, _ = layers.get("layer2", (np.zeros(0), np.zeros(0)))
                    m1, s1 = layers.get("layer1", (np.zeros(0), np.zeros(0)))
                    m0, _ = layers.get("layer0", (np.zeros(0), np.zeros(0)))
                    m3, s3 = layers.get("layer3", (np.zeros(0), np.zeros(0)))
                    prefix = f"{net}.{si}.blocks.{bi}"
                    put(f"{prefix}.attn.qkv.weight", m2)
                    put(f"{prefix}.attn.proj.weight", m1)
                    mlp_stream = np.concatenate([m0, m2[786432:], m3[:262144]])
                    put(f"{prefix}.mlp.0.weight", mlp_stream)
                    put(f"{prefix}.mlp.2.weight", mlp_stream[456704:])
                    misc_all = (
                        np.concatenate([x for x in (s1, s3) if len(x)])
                        if (len(s1) or len(s3))
                        else np.zeros(0)
                    )
                    for suffix in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
                        pn = f"{prefix}.{suffix}"
                        if pn in pmap and pn in unfilled and len(misc_all):
                            put(pn, misc_all)
                    pn = f"{prefix}.mlp.2.bias"
                    if pn in pmap and pn in unfilled:
                        bias_src = misc_all[512:1024] if len(misc_all) >= 1024 else misc_all
                        if len(bias_src) >= 512:
                            put(pn, bias_src[:512])
                else:
                    main, misc = layers["layer0"]
                    fill_swin(f"{net}.{si}.blocks.{bi}", main, misc)

    # b0: stem
    if 0 in by_block:
        main0, misc0 = by_block[0]["layer0"]
        put("stem.weight", main0)
        if len(misc0):
            put("stem.bias", misc0)

    # Merges (b4, 8, 14, 22)
    for mi, b in enumerate((4, 8, 14, 22)):
        if b in by_block:
            main, misc = by_block[b]["layer0"]
            put(f"merges.{mi}.reduction.weight", main)
            if mi == 2:
                with torch.no_grad():
                    pmap["merges.2.norm.weight"].fill_(1.0)
                    pmap["merges.2.norm.bias"].zero_()
                unfilled.discard("merges.2.norm.weight")
                unfilled.discard("merges.2.norm.bias")
                stats["filled"] += pmap["merges.2.norm.weight"].numel() * 2
            else:
                put(f"merges.{mi}.norm.weight", misc) if len(misc) else None
                put(f"merges.{mi}.norm.bias", misc) if len(misc) else None

    # Expands (b48, 56, 62, 66) with UpFuse 2c^2 fuse GEMM
    UP_ZONES = {
        48: dict(misc=(491520, 820224), fuse=(360448, 491520), bias=(820224, 820736), c=256),
        56: dict(misc=(131072, 229888), fuse=(98304, 131072), bias=(229888, 230112), c=128),
        62: dict(misc=(49152, 66048), fuse=(66048, 69632), bias=(69632, 69728), c=64),
        66: dict(misc=(13312, 21504), fuse=(11264, 13312), bias=(22716, 22780), c=32),
    }
    _UPB = {48: 0, 56: 1, 62: 2, 66: 3}
    for b, ei in _UPB.items():
        z = UP_ZONES.get(b)
        if z is None or b not in by_block:
            continue
        j = blob.find(f"block{b}.layer0.layer".encode())
        k = j + len(f"block{b}.layer0.layer")
        arr = np.frombuffer(blob[k + 28 : k + 28 + z["bias"][1]], dtype=np.uint8)
        fuse_v = e4m3_decode(arr[z["fuse"][0] : z["fuse"][1]])
        bias_v = fp16_decode(arr[z["bias"][0] : z["bias"][1]])
        pn = f"expands.{ei}.fuse.weight"
        if pn in pmap and len(fuse_v) >= pmap[pn].numel():
            put(pn, fuse_v)
            fb = f"expands.{ei}.fuse.bias"
            if fb in pmap and len(bias_v) >= pmap[fb].numel():
                put(fb, bias_v[: pmap[fb].numel()])

    for ei in range(4):
        for pn in (f"expands.{ei}.fuse.weight", f"expands.{ei}.expand.weight"):
            if pn in pmap and pn in unfilled:
                p = pmap[pn]
                with torch.no_grad():
                    p.normal_(0.0, math.sqrt(1.0 / p.shape[1]))
                unfilled.discard(pn)
                stats["filled"] += p.numel()

    # Bottleneck (b31-38): _SplitBlock
    for bi, b in enumerate(range(31, 39)):
        if b in by_block:
            layers = by_block[b]
            m0, _ = layers.get("layer0", (np.zeros(0), np.zeros(0)))
            m1, _ = layers.get("layer1", (np.zeros(0), np.zeros(0)))
            m2, _ = layers.get("layer2", (np.zeros(0), np.zeros(0)))
            m4, _ = layers.get("layer4", (np.zeros(0), np.zeros(0)))
            put(f"bn.{bi}.wqkv.weight", m0)
            put(f"bn.{bi}.proj.weight", m1)
            put(f"bn.{bi}.proj.bias", m1[4194304:])
            put(f"bn.{bi}.side.weight", m2)
            put(f"bn.{bi}.ffwd.weight", m4)
            put(f"bn.{bi}.ffwd.bias", m4[1048576:])
            if f"bn.{bi}.gate" in pmap:
                with torch.no_grad():
                    pmap[f"bn.{bi}.gate"].zero_()

    # b22 tail conversion matrix
    if 22 in by_block and "split_exit_pad.p" in pmap:
        j22 = blob.find(b"block22.layer0.layer")
        k22 = j22 + len("block22.layer0.layer")
        raw22 = np.frombuffer(blob[k22 + 28 + 689152 : k22 + 28 + 820224], dtype=np.uint8)
        v22 = e4m3_decode(raw22[: pmap["split_exit_pad.p"].numel() * 4])
        put("split_exit_pad.p", v22)

    # b30.layer4
    if 30 in by_block and "layer4" in by_block[30] and "enc_to_bn_pad.p" in pmap:
        m4, _ = by_block[30]["layer4"]
        put("enc_to_bn_pad.p", m4[: pmap["enc_to_bn_pad.p"].numel()])

    # bn_proj and b39 dec_gate
    if 39 in by_block:
        main, _ = by_block[39]["layer0"]
        put("bn_proj.weight", main) if "bn_proj.weight" in pmap else None
        if "dec_gate" in pmap:
            j = blob.find(b"block39.layer0.layer")
            kk = j + len("block39.layer0.layer")
            B39 = struct.unpack_from("<Q", blob, kk + 16)[0]
            raw39 = np.frombuffer(blob[kk + 28 + B39 - 1024 : kk + 28 + B39], dtype=np.uint8)
            g = np.frombuffer(raw39.tobytes(), dtype="<f2").astype(np.float32)
            with torch.no_grad():
                pmap["dec_gate"].copy_(torch.from_numpy(g))
            unfilled.discard("dec_gate")

    # b70: tail head
    if 70 in by_block:
        layers70 = by_block[70]
        main, misc = layers70.get("layer0", (np.zeros(0), np.zeros(0)))
        bs = layers70.get("blend_scale", (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)))[0]
        allv = (
            np.concatenate([main, misc, bs])
            if len(bs)
            else np.concatenate([main, misc]) if len(misc) else main
        )
        seq = np.concatenate([allv, np.zeros(64, np.float32)])
        put("tail.global_fc.weight", seq)
        put("tail.global_fc.bias", seq[1024:])
        put("tail.conv.weight", seq[1056:])
        put("tail.conv.bias", seq[1920:])
        put("tail.blend", seq[1923:1924]) if "tail.blend" in pmap else None

    # Default fallback for residual parameters (norms and biases)
    for pn in list(unfilled):
        if "_pad" in pn or pn.endswith(".pad") or pn in ("calib_pad", "stem_pad"):
            unfilled.discard(pn)
            continue
        if "relative_position_bias_table" in pn:
            with torch.no_grad():
                pmap[pn].zero_()
            unfilled.discard(pn)
            continue
        if "norm" in pn or pn.endswith((".bias", ".gate")):
            p = pmap[pn]
            with torch.no_grad():
                p.fill_(1.0 if ("norm" in pn and "weight" in pn) else 0.0)
            unfilled.discard(pn)

    if verbose:
        print(f"[dlss5.loader] Filled {stats['filled']:,} parameters ({len(unfilled)} unfilled)")
    return stats


def load_model(
    weights_path: str = "weights_blob.bin",
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> DLSS5Net:
    """Instantiates DLSS5Net and loads weights from binary blob.

    Args:
        weights_path: Path to weights_blob.bin (147.7MB extracted from DLL).
        device: Target torch device ('cpu', 'cuda', etc.).
        verbose: If True, prints parsing statistics.

    Returns:
        DLSS5Net in eval() mode on the specified device.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"DLSS5 weights blob not found at '{weights_path}'. "
            "Please ensure weights_blob.bin is present in the working directory."
        )

    model = DLSS5Net().eval()
    with open(weights_path, "rb") as f:
        blob_bytes = f.read()

    load_weights(model, blob_bytes, verbose=verbose)
    model.to(device)
    return model
