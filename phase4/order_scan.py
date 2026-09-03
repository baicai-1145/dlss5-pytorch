"""Phase 6 Task 5 — b23-29 fill-order enumeration.

Background: phase4/semantic_fill.py fills c=512 SwinBlocks (b23-29) by a
specific sub-record → tensor mapping that may be ambiguous. This script tests
12 structurally plausible candidates and reports (enc3_std, bn_avg_std,
tail_std, corr(tail, simple_delta)) for each.

Memory discipline (16GB):
  - One model lives at a time. Built, filled, forward'd, freed inside the
    loop body so peak RSS stays bounded.
  - Official delta is loaded once as fp16 for the 3 test crops only.
  - Capture frames loaded once as fp16, kept in shared memory.

Judge = simple (after - before) per the supervisor (NOT the AP1^-1 npz).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "phase3"))

from phase3.dlss5.calib_model import DLSS5NetCalib
from phase4.semantic_fill import load_all, fill_model

CAP_DIR = os.path.join(HERE, "..", ".tmp", "cap2_live")
W, H = 1920, 1050
SIZE = 96
BLOB = os.path.join(HERE, "..", "weights_blob.bin")

CROPS = [(400, 700), (300, 1100), (600, 500)]


def load_rgb(name: str) -> np.ndarray:
    u = np.fromfile(os.path.join(CAP_DIR, name), dtype=np.uint32).reshape(H, W)
    return np.stack([((u >> s) & 0x3FF) / 1023.0 for s in (0, 10, 20)], axis=-1)


def load_depth(name: str) -> np.ndarray:
    u = np.fromfile(os.path.join(CAP_DIR, name), dtype=np.uint32).reshape(H, W)
    r11 = (u & 0x7FF).astype(np.uint16)
    e = (r11 >> 6) & 0x1F
    m = r11 & 0x3F
    val = np.where(e == 0, (m / 64.0) * 2.0 ** -14,
                   (1.0 + m / 64.0) * 2.0 ** (e.astype(np.float32) - 15.0))
    return val.astype(np.float32)


def load_motion(name: str) -> np.ndarray:
    m = np.fromfile(os.path.join(CAP_DIR, name), dtype="<f2").reshape(H, W, 2)
    return m.astype(np.float32) * np.array([1920.0, 1050.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Candidate specs
# ---------------------------------------------------------------------------
# Each spec is a dict with keys:
#   proj_src:       "L1" or "L3"             -- which layer supplies proj.weight
#   mlp0_src:       "L0_L2rest" | "L2rest_L3" | "L0" | "L3" | "L1" | "L0_split"
#                   where "X_Y" means concat([X, Y]), "L0" means take L0 alone
#                   and truncate, "L0_split" means L0[:456704] only for mlp0
#   mlp2_src:       (same encoding)
#   bias_src:       "zero" | "L0_first" | "L1_misc" | "L3_misc"
#
# Constraints enforced:
#   - qkv.weight always <- L2[:786432] (the only 786,432-byte E4 chunk)
#   - mlp.0.bias / mlp.2.bias from bias_src if bias_src has values; else zero
# ---------------------------------------------------------------------------

CANDIDATES = [
    # ---- baseline (current semantic_fill mapping) ----
    {"id": "C00", "name": "baseline: proj=L1, mlp0=canon, mlp2=canon (semantic_fill mapping)",
     "proj_src": "L1", "mlp0_src": "canon_mlp0", "mlp2_src": "canon_mlp2",
     "bias_src": "zero"},
    # ---- proj from L3 (mirror of L1) ----
    {"id": "C01", "name": "proj=L3, mlp0=L0_L2rest, mlp2=L1_clean",
     "proj_src": "L3", "mlp0_src": "L0_L2rest", "mlp2_src": "L1_clean",
     "bias_src": "zero"},
    # ---- swap mlp0 and mlp2 sources ----
    {"id": "C02", "name": "mlp0=L2rest_L3_clean, mlp2=L0 (proj=L1)",
     "proj_src": "L1", "mlp0_src": "L2rest_L3_clean", "mlp2_src": "L0",
     "bias_src": "zero"},
    {"id": "C03", "name": "proj=L3, mlp0=L2rest_L1_clean, mlp2=L0",
     "proj_src": "L3", "mlp0_src": "L2rest_L1_clean", "mlp2_src": "L0",
     "bias_src": "zero"},
    # ---- truncate L0 alone for mlp0, the rest for mlp2 ----
    {"id": "C04", "name": "mlp0=L0 (trunc), mlp2=L3_clean+L2rest",
     "proj_src": "L1", "mlp0_src": "L0", "mlp2_src": "L3_clean_L2rest",
     "bias_src": "zero"},
    {"id": "C05", "name": "proj=L3, mlp0=L0 (trunc), mlp2=L1_clean+L2rest",
     "proj_src": "L3", "mlp0_src": "L0", "mlp2_src": "L1_clean_L2rest",
     "bias_src": "zero"},
    # ---- truncate L3 alone for mlp0 ----
    {"id": "C06", "name": "mlp0=L3_clean (trunc, small std), mlp2=L0+L2rest",
     "proj_src": "L1", "mlp0_src": "L3_clean", "mlp2_src": "L0_L2rest",
     "bias_src": "zero"},
    {"id": "C07", "name": "proj=L3, mlp0=L1_clean (trunc, high std), mlp2=L0+L2rest",
     "proj_src": "L3", "mlp0_src": "L1_clean", "mlp2_src": "L0_L2rest",
     "bias_src": "zero"},
    # ---- L0 internal split (mirroring FFN: first half -> mlp.0, rest -> mlp.2) ----
    {"id": "C08", "name": "L0 internal split: mlp0=L0_split, mlp2=L0_rest_L2rest_L3_clean",
     "proj_src": "L1", "mlp0_src": "L0_split", "mlp2_src": "L0_rest_L2rest_L3_clean",
     "bias_src": "zero"},
    {"id": "C09", "name": "L0 internal split + proj=L3: mlp0=L0_split, mlp2=L0_rest_L2rest_L1_clean",
     "proj_src": "L3", "mlp0_src": "L0_split", "mlp2_src": "L0_rest_L2rest_L1_clean",
     "bias_src": "zero"},
    # ---- baseline + bias source variants ----
    {"id": "C10", "name": "baseline + ffn2_bias from L0[:512]",
     "proj_src": "L1", "mlp0_src": "canon_mlp0", "mlp2_src": "canon_mlp2",
     "bias_src": "L0_first"},
    {"id": "C11", "name": "baseline + ffn2_bias from L3 misc (fp16)",
     "proj_src": "L1", "mlp0_src": "canon_mlp0", "mlp2_src": "canon_mlp2",
     "bias_src": "L3_misc"},
]


# ---------------------------------------------------------------------------
# Build concat streams from sub-record names
# ---------------------------------------------------------------------------

# Stream spec encoding: a name like "L0_rest_L2rest_L3" lists sub-parts
# separated by "_" EXCEPT for the well-known tokens L0_rest, L0_split,
# L0_first, L2rest, L1_misc, L3_misc which are composite names. We greedily
# match longest tokens first.
_STREAM_TOKENS = ("L0_first", "L0_split", "L0_rest", "L2rest",
                  "L1_misc", "L3_misc", "L1_clean", "L3_clean",
                  "canon_mlp0", "canon_mlp2",
                  "L0", "L1", "L3")


def _parse_stream_tokens(spec: str) -> list[str]:
    """Greedy longest-token split of a stream spec string.

    Whitespace is stripped; '_' is treated purely as a token separator.
    """
    s = spec.replace(" ", "")
    parts = []
    i = 0
    while i < len(s):
        if s[i] == "_":
            i += 1
            continue
        matched = False
        for tok in _STREAM_TOKENS:
            if s[i:i + len(tok)] == tok:
                parts.append(tok)
                i += len(tok)
                matched = True
                break
        if not matched:
            raise ValueError(f"unknown token at position {i} of spec '{spec}' (remaining='{s[i:]}')")
    return parts


def _stream(spec: str, layers: dict) -> np.ndarray:
    """Resolve a stream spec to a numpy array using the b23-29 layer dict.

    Stream tokens:
      L0, L1, L3                 -- raw main stream of layer0/1/3 (full)
      L1_clean, L3_clean         -- first 262144 vals (skip 508-byte garbage)
      L0_split                   -- L0[:456704]
      L0_rest                    -- L0[456704:]
      L2rest                     -- L2[786432:]   (the MX tail of layer2)
      L1_misc, L3_misc           -- the 256 fp16 misc vals
      L0_first                   -- L0[:512]
      canon_mlp0                 -- concat([L0, L2rest])[:456704]  (baseline mlp.0)
      canon_mlp2                 -- concat([L0, L2rest, L3_clean])[456704:]  (baseline mlp.2)
                                    (this is the canonical semantic_fill mapping)
    """
    tokens = _parse_stream_tokens(spec)
    chunks = []
    for p in tokens:
        if p == "L0":
            chunks.append(layers["layer0"][0])
        elif p == "L1":
            chunks.append(layers["layer1"][0])
        elif p == "L1_clean":
            chunks.append(layers["layer1"][0][:262144])
        elif p == "L3":
            chunks.append(layers["layer3"][0])
        elif p == "L3_clean":
            chunks.append(layers["layer3"][0][:262144])
        elif p == "L0_split":
            chunks.append(layers["layer0"][0][:456704])
        elif p == "L0_rest":
            chunks.append(layers["layer0"][0][456704:])
        elif p == "L2rest":
            chunks.append(layers["layer2"][0][786432:])
        elif p == "L3_misc":
            chunks.append(layers["layer3"][1])
        elif p == "L1_misc":
            chunks.append(layers["layer1"][1])
        elif p == "L0_first":
            chunks.append(layers["layer0"][0][:512])
        elif p == "canon_mlp0":
            # The canonical baseline mlp.0 stream (first 456704 of concat([L0, L2rest]))
            full = np.concatenate([layers["layer0"][0], layers["layer2"][0][786432:]])
            chunks.append(full[:456704])
        elif p == "canon_mlp2":
            # The canonical baseline mlp.2 stream (offset 456704 onwards of concat([L0, L2rest, L3_clean]))
            full = np.concatenate([
                layers["layer0"][0],
                layers["layer2"][0][786432:],
                layers["layer3"][0][:262144],   # L3_clean
            ])
            chunks.append(full[456704:])
        else:
            raise ValueError(f"unknown stream part: {p}")
    return np.concatenate(chunks) if chunks else np.zeros(0)


# ---------------------------------------------------------------------------
# Custom b23-29 fill: replaces the c=512 4-sub-record fill in semantic_fill.
# Other blocks use the canonical semantic_fill path.
# ---------------------------------------------------------------------------

def _do_fill_custom(model, by_block, spec: dict, blob_full: bytes):
    """Call canonical fill_model first, then OVERWRITE only b23-29 enc4 blocks
    (enc.4.blocks.0..6). All other blocks untouched."""
    # First do the canonical fill (so all blocks except enc4 are loaded
    # using the existing mapping). Then overwrite enc4.
    unfilled, stats = fill_model(model, by_block, blob_full=blob_full)

    pmap = dict(model.named_parameters())
    layers_b = [by_block[b] for b in (23, 24, 25, 26, 27, 28, 29)]

    for bi, layers in enumerate(layers_b):
        prefix = f"enc.4.blocks.{bi}"
        # qkv ← L2 E4前段 (FIXED across all candidates — only source of 786432 vals)
        qkv_src = layers["layer2"][0][:786432]
        proj_src = _stream(spec["proj_src"], layers)
        mlp0_src = _stream(spec["mlp0_src"], layers)
        mlp2_src = _stream(spec["mlp2_src"], layers)

        def put(pname, vals):
            p = pmap[pname]
            n = p.numel()
            v = np.asarray(vals, dtype=np.float32)
            if len(v) < n:
                v = np.concatenate([v, np.zeros(n - len(v), v.dtype)])
            with torch.no_grad():
                p.copy_(torch.from_numpy(v[:n].astype(np.float32)).reshape(p.shape).to(p.dtype))

        put(f"{prefix}.attn.qkv.weight", qkv_src)
        put(f"{prefix}.attn.proj.weight", proj_src)
        put(f"{prefix}.mlp.0.weight", mlp0_src)
        put(f"{prefix}.mlp.2.weight", mlp2_src)

        # ffn2 bias (512 vals)
        bias_src_name = spec["bias_src"]
        if bias_src_name == "zero":
            # Explicitly zero the bias (canonical fill overwrote it with misc)
            with torch.no_grad():
                pmap[f"{prefix}.mlp.2.bias"].zero_()
        else:
            bias_src = _stream(bias_src_name, layers)
            put(f"{prefix}.mlp.2.bias", bias_src)

    return unfilled, stats


# ---------------------------------------------------------------------------
# Forward + stats hooks
# ---------------------------------------------------------------------------

STAGE_NAMES = ["stem"] + [f"enc{i}" for i in range(5)] + [f"merge{i}" for i in range(4)] \
              + [f"bn{i}" for i in range(8)] + ["bn_proj"] \
              + [f"dec{i}" for i in range(5)] + [f"expand{i}" for i in range(4)] \
              + ["tail"]


class StageStats:
    __slots__ = ("mean", "std", "min", "max", "shape")

    def __init__(self):
        self.mean = 0.0
        self.std = 0.0
        self.min = 0.0
        self.max = 0.0
        self.shape = ()


def attach_stats_hooks(model):
    stats = {n: StageStats() for n in STAGE_NAMES}

    def make_hook(name):
        def _hook(_mod, _inp, out):
            if out is None:
                return
            t = out if isinstance(out, torch.Tensor) else out[0]
            if not isinstance(t, torch.Tensor):
                return
            with torch.no_grad():
                f = t.detach().float()
                stats[name].mean = float(f.mean().item())
                stats[name].std = float(f.std().item())
                stats[name].min = float(f.amin().item())
                stats[name].max = float(f.amax().item())
                stats[name].shape = tuple(f.shape)
        return _hook

    handles = []
    handles.append(model.stem.register_forward_hook(make_hook("stem")))
    for i, st in enumerate(model.enc):
        handles.append(st.register_forward_hook(make_hook(f"enc{i}")))
    for i, m in enumerate(model.merges):
        handles.append(m.register_forward_hook(make_hook(f"merge{i}")))
    for i, blk in enumerate(model.bn):
        handles.append(blk.register_forward_hook(make_hook(f"bn{i}")))
    handles.append(model.bn_proj.register_forward_hook(make_hook("bn_proj")))
    for i, st in enumerate(model.dec):
        handles.append(st.register_forward_hook(make_hook(f"dec{i}")))
    for i, e in enumerate(model.expands):
        handles.append(e.register_forward_hook(make_hook(f"expand{i}")))
    handles.append(model.tail.register_forward_hook(make_hook("tail")))

    def _cleanup():
        for h in handles:
            h.remove()
    return stats, _cleanup


# ---------------------------------------------------------------------------
# Per-crop runner
# ---------------------------------------------------------------------------

def run_crop(model, color_full, depth_full, mv_full, target_full, y0, x0):
    c = lambda a: a[y0:y0 + SIZE, x0:x0 + SIZE]
    rgb = c(color_full).transpose(2, 0, 1)[None].astype(np.float32)
    d = c(depth_full)[None, None].astype(np.float32)
    v = c(mv_full).transpose(2, 0, 1)[None].astype(np.float32)

    dmed = float(np.median(d))
    dn = np.clip(d / max(dmed, 1e-6), 0, 4) / 4.0

    with torch.no_grad():
        res = model(
            torch.from_numpy(rgb.copy()).float(),
            torch.from_numpy(dn.astype(np.float32)),
            torch.from_numpy((v * 0.02).astype(np.float32)),
            torch.from_numpy(rgb.copy()).float(),
        )[0].numpy()

    off = c(target_full).transpose(2, 0, 1)   # 3,S,S

    cc = float(np.corrcoef(res.ravel(), off.ravel())[0, 1])
    return res, off, cc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== order_scan: b23-29 fill-order enumeration ===\n")
    t_global = time.time()

    # ---- load capture (kept in memory throughout the run) ----
    print("loading cap2_live frame 0 ...")
    bef = load_rgb("model_input_00.raw").astype(np.float32)
    aft = load_rgb("after_00.raw").astype(np.float32)
    dep = load_depth("depth_00.raw").astype(np.float32)
    mv = load_motion("motion_00.raw").astype(np.float32)
    simple_delta_full = (aft - bef).astype(np.float32)
    print(f"  bef std={bef.std():.3f}  aft std={aft.std():.3f}  delta std={simple_delta_full.std():.4f}")

    # ---- load by_block (one-time decode) ----
    print("loading by_block (decode once) ...")
    blob_full = open(BLOB, "rb").read()
    by_block = load_all()
    print(f"  loaded {len(by_block)} blocks\n")

    # ---- target stats ----
    target_std_simple = []
    for y0, x0 in CROPS:
        c = lambda a: a[y0:y0 + SIZE, x0:x0 + SIZE]
        d = c(simple_delta_full)
        target_std_simple.append(float(d.std()))
    target_std_simple_mean = float(np.mean(target_std_simple))
    print(f"  target std (simple_delta, 3 crops mean): {target_std_simple_mean:.4f} (per-crop: {[f'{x:.3f}' for x in target_std_simple]})\n")

    # ---- results table ----
    results = []

    for cand in CANDIDATES:
        cid = cand["id"]
        cname = cand["name"]
        t0 = time.time()

        # Build model in fresh state
        torch.manual_seed(42)
        model = DLSS5NetCalib().eval()

        # Custom fill (canonical first, then enc4 override)
        try:
            unfilled, stats_fill = _do_fill_custom(model, by_block, cand, blob_full)
        except Exception as e:
            print(f"[{cid}] FILL ERROR: {e}")
            del model
            gc.collect()
            continue

        # Attach hooks
        stats, cleanup = attach_stats_hooks(model)

        # Run 3 crops
        corrs = []
        tail_stds = []
        for y0, x0 in CROPS:
            res, off, cc = run_crop(model, bef, dep, mv, simple_delta_full, y0, x0)
            corrs.append(cc)
            tail_stds.append(float(res.std()))

        cleanup()
        del model
        gc.collect()

        # Pull activation stats (last crop's hooks)
        enc3_std = stats["enc3"].std
        bn_stds = [stats[f"bn{i}"].std for i in range(8)]
        bn_avg_std = float(np.mean(bn_stds))
        tail_std = tail_stds[-1]   # use last crop (600,500)
        # Mean across crops for more stable metric
        tail_std_mean = float(np.mean(tail_stds))
        corr_mean = float(np.mean(corrs))

        # Score 1: closer to per-crop target on tail_std AND high corr
        tail_std_err = abs(tail_std_mean - target_std_simple_mean)
        # Score 2: closer to seed=42 baseline 88 on bn_avg (target for bottleneck heat)
        BN_TARGET = 88.0
        bn_avg_err = abs(bn_avg_std - BN_TARGET)
        # Combined score (maximize): lower err on tail AND bn, plus corr
        score = -tail_std_err - 0.5 * bn_avg_err + 0.5 * corr_mean

        dt = time.time() - t0
        print(f"[{cid}] {cname[:55]:55s}  dt={dt:.1f}s")
        print(f"     enc3 std={enc3_std:8.3f}  bn_avg std={bn_avg_std:8.3f}  tail_std={tail_std:6.4f}  "
              f"corrs={[f'{c:+.3f}' for c in corrs]}  mean_corr={corr_mean:+.3f}")
        print(f"     tail_std_err={tail_std_err:.4f}  score={score:+.4f}")

        results.append({
            "id": cid,
            "name": cname,
            "proj_src": cand["proj_src"],
            "mlp0_src": cand["mlp0_src"],
            "mlp2_src": cand["mlp2_src"],
            "bias_src": cand["bias_src"],
            "enc3_std": enc3_std,
            "bn_avg_std": bn_avg_std,
            "bn_avg_err_vs_88": abs(bn_avg_std - 88.0),
            "tail_std": tail_std,
            "tail_std_mean": tail_std_mean,
            "tail_std_err": tail_std_err,
            "corr_mean": corr_mean,
            "corr_per_crop": corrs,
            "tail_std_per_crop": tail_stds,
            "score": score,
            "dt_sec": dt,
        })

    # ---- Sort & report ----
    print("\n=== results sorted by combined score (tail_err + bn_err vs 88, + corr) ===")
    by_sort = sorted(results, key=lambda r: -r["score"])

    print(f"\n  {'id':4s}  {'enc3':>7s}  {'bn':>7s}  {'bn_err':>7s}  {'tail':>7s}  {'corr':>6s}  {'err':>6s}  spec")
    for r in by_sort:
        spec = f"{r['proj_src']:4s} | {r['mlp0_src']:18s} | {r['mlp2_src']:18s} | bias={r['bias_src']:9s}"
        print(f"  {r['id']:4s}  {r['enc3_std']:7.3f}  {r['bn_avg_std']:7.3f}  "
              f"{r['bn_avg_err_vs_88']:7.3f}  {r['tail_std_mean']:7.4f}  {r['corr_mean']:+6.3f}  {r['tail_std_err']:6.4f}  {spec}")

    # Save
    out_json = os.path.join(HERE, ".tmp", "order_scan_results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "target_std_simple_mean": target_std_simple_mean,
            "target_std_per_crop": target_std_simple,
            "results": results,
            "sorted_by_score": [r["id"] for r in by_sort],
        }, f, indent=2)
    print(f"\nresults saved to {out_json}")

    best = by_sort[0]
    baseline = next((r for r in results if r["id"] == "C00"), None)
    print(f"\nbest candidate: {best['id']} ({best['name']})")
    print(f"  enc3_std = {best['enc3_std']:.3f}")
    print(f"  bn_avg_std = {best['bn_avg_std']:.3f}  (target 88, err {best['bn_avg_err_vs_88']:.2f})")
    print(f"  tail_std (per-crop mean) = {best['tail_std_mean']:.4f}  (target {target_std_simple_mean:.4f}, err {best['tail_std_err']:.4f})")
    print(f"  corr_mean = {best['corr_mean']:+.3f}")
    if baseline:
        print(f"\nbaseline (C00):")
        print(f"  bn_avg_std = {baseline['bn_avg_std']:.3f}  err vs 88 = {baseline['bn_avg_err_vs_88']:.2f}")
        print(f"  tail_std = {baseline['tail_std_mean']:.4f}  err = {baseline['tail_std_err']:.4f}")
        print(f"  corr_mean = {baseline['corr_mean']:+.3f}")
        print(f"\nlift over baseline:")
        print(f"  bn_avg_delta = {best['bn_avg_std'] - baseline['bn_avg_std']:+.3f}  (closer to 88 is better)")
        print(f"  bn_avg_err delta = {best['bn_avg_err_vs_88'] - baseline['bn_avg_err_vs_88']:+.3f} (negative = better)")
        print(f"  tail_std delta = {best['tail_std_mean'] - baseline['tail_std_mean']:+.4f}")
        print(f"  tail_std_err delta = {best['tail_std_err'] - baseline['tail_std_err']:+.4f} (negative = better)")
        print(f"  corr delta = {best['corr_mean'] - baseline['corr_mean']:+.4f}")

    print(f"\ntotal dt = {time.time() - t_global:.0f}s")


if __name__ == "__main__":
    main()