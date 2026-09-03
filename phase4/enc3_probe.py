"""Phase 6 Task 6 — enc3 14× jump root cause.

Background:
- Task 5 (b23-29 fill-order sweep) showed enc3 std stays at 21.74 across
  every candidate. The bottleneck is fixed in enc3/b15-21.
- The 14× jump is owned by **merges.2** (merge2 = c=128→256 transition),
  not by the enc3 swin blocks themselves. enc2.std=1.55, merge2.std=21.76
  (the merge output is what enc3 reads as input).
- merges.2.norm.weight is filled from b14 misc's first 512 fp16 vals, which
  are actually MX-scaled magnitudes (mean=-8.78, std=2.21, range [-12.33, 0])
  mis-classified as fp16 misc by semantic_fill.py's FP16_TAIL[229936]=155936
  boundary. Putting those into LN gamma amplifies activations by ~14× at
  the merge output, producing the observed enc3/merge2 std jump.

This script:
1. Static-audits enc3 + b14 + b22 (merge2 + merge3) tensors against priors
   (LN gamma ≈ 1±0.3, Linear std ≈ 1/√fan_in ±50%, FFN bias ≈ 0±0.5).
2. Lists tensors deviating >3σ from prior in `phase4/.tmp/enc3_audit.json`.
3. Runs single-tensor replacement experiments for the most anomalous tensors.
   Each candidate: fresh model, fill canonical, mutate one tensor to prior,
   run 3 crops (seed=42, (400,700)/(300,1100)/(600,500)), record enc3/merge2/
   bn_avg/tail std + corr(tail, simple_delta).
4. If a single fix brings enc3 from 21.7 → <5 with corr unchanged, freezes
   it as `recommended` and writes the fix instruction into
   `phase4/.tmp/enc3_recommended_fix.json`.

Memory discipline: one model lives at a time, fresh per experiment,
freed before next. Official delta loaded once as fp32.
"""
from __future__ import annotations

import gc
import json
import math
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
# 1) Static audit: tensor → (expected prior, observed, deviation σ)
# ---------------------------------------------------------------------------

# Priors for different tensor types
# LN gamma: 1.0 ± 0.3 (after training), std ≈ 0.1-0.3
# LN bias: 0.0 ± 0.5, std ≈ 0.1-0.3
# Linear weight: Kaiming init std = sqrt(2/fan_in). Trained: depends.
# Linear bias: 0.0 ± 0.1 typically
# attn qkv bias (if exists): 0.0 ± 0.1
# rel_bias_table: 0.0 ± 0.05 (initialised small)

PRIOR_LN_GAMMA_MEAN = 1.0
PRIOR_LN_GAMMA_STD = 0.3
PRIOR_LN_BIAS_MEAN = 0.0
PRIOR_LN_BIAS_STD = 0.5
PRIOR_LINEAR_BIAS_MEAN = 0.0
PRIOR_LINEAR_BIAS_STD = 0.2
PRIOR_REL_BIAS_MEAN = 0.0
PRIOR_REL_BIAS_STD = 0.05


def audit_tensor(name: str, t: torch.Tensor) -> dict:
    """Compute prior deviation for a tensor."""
    v = t.detach().float().numpy()
    obs_mean = float(v.mean())
    obs_std = float(v.std())
    obs_min = float(v.min())
    obs_max = float(v.max())
    obs_nz = int((v != 0).sum())
    obs_total = v.size

    # Classify
    if "norm" in name and name.endswith(".weight"):
        p_mean = PRIOR_LN_GAMMA_MEAN
        p_std = PRIOR_LN_GAMMA_STD
        kind = "ln_gamma"
    elif "norm" in name and name.endswith(".bias"):
        p_mean = PRIOR_LN_BIAS_MEAN
        p_std = PRIOR_LN_BIAS_STD
        kind = "ln_bias"
    elif name.endswith(".bias"):
        p_mean = PRIOR_LINEAR_BIAS_MEAN
        p_std = PRIOR_LINEAR_BIAS_STD
        kind = "linear_bias"
    elif "relative_position_bias_table" in name:
        p_mean = PRIOR_REL_BIAS_MEAN
        p_std = PRIOR_REL_BIAS_STD
        kind = "rel_bias"
    else:
        # Linear weight — we just record stats
        kind = "linear_weight"
        p_mean = 0.0
        p_std = float('nan')

    # Deviation from prior (mean)
    if not math.isnan(p_std):
        mean_dev_sigma = (obs_mean - p_mean) / p_std if p_std > 0 else abs(obs_mean - p_mean)
    else:
        mean_dev_sigma = 0.0

    return {
        "name": name,
        "kind": kind,
        "shape": list(t.shape),
        "numel": obs_total,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "obs_min": obs_min,
        "obs_max": obs_max,
        "obs_nonzero": obs_nz,
        "obs_zeros_pct": 1.0 - obs_nz / obs_total if obs_total > 0 else 0.0,
        "prior_mean": p_mean,
        "prior_std": p_std,
        "mean_dev_sigma": mean_dev_sigma,
    }


def run_audit(model: DLSS5NetCalib, names: List[str]) -> List[dict]:
    pmap = dict(model.named_parameters())
    results = []
    for n in names:
        if n in pmap:
            results.append(audit_tensor(n, pmap[n]))
    return results


# ---------------------------------------------------------------------------
# 2) Forward runner + stats hooks (reuses order_scan.py patterns)
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


def run_crops(model, bef, dep, mv, target):
    """Run 3 crops, return (corrs, tail_std_mean, enc3_std, merge2_std)."""
    corrs, tail_stds = [], []
    for y0, x0 in CROPS:
        c = lambda a: a[y0:y0 + SIZE, x0:x0 + SIZE]
        rgb = c(bef).transpose(2, 0, 1)[None].astype(np.float32)
        d = c(dep)[None, None].astype(np.float32)
        v = c(mv).transpose(2, 0, 1)[None].astype(np.float32)
        dmed = float(np.median(d))
        dn = np.clip(d / max(dmed, 1e-6), 0, 4) / 4.0
        with torch.no_grad():
            res = model(
                torch.from_numpy(rgb.copy()).float(),
                torch.from_numpy(dn.astype(np.float32)),
                torch.from_numpy((v * 0.02).astype(np.float32)),
                torch.from_numpy(rgb.copy()).float(),
            )[0].numpy()
        off = c(target).transpose(2, 0, 1)
        cc = float(np.corrcoef(res.ravel(), off.ravel())[0, 1])
        corrs.append(cc)
        tail_stds.append(float(res.std()))
    return corrs, tail_stds


# ---------------------------------------------------------------------------
# 3) Replacement experiments
# ---------------------------------------------------------------------------

def run_experiment(name: str, transforms: List[Tuple[str, callable]], bef, dep, mv, target, by_block, blob_full) -> dict:
    """Build model, apply transforms after canonical fill, run 3 crops, return metrics."""
    torch.manual_seed(42)
    model = DLSS5NetCalib().eval()
    fill_model(model, by_block, blob_full=blob_full)
    pmap = dict(model.named_parameters())
    with torch.no_grad():
        for tname, fn in transforms:
            for bi in range(7):
                full = f'enc.3.blocks.{bi}.{tname}'
                if full in pmap:
                    p = pmap[full]
                    fn(p)
            # Also try the parameter path WITHOUT the bi. prefix
            full_alt = tname
            if full_alt in pmap:
                p = pmap[full_alt]
                fn(p)

    stats, cleanup = attach_stats_hooks(model)
    corrs, tail_stds = run_crops(model, bef, dep, mv, target)
    cleanup()
    enc3_std = stats["enc3"].std
    merge2_std = stats["merge2"].std
    bn_avg_std = float(np.mean([stats[f"bn{i}"].std for i in range(8)]))
    tail_std_mean = float(np.mean(tail_stds))
    corr_mean = float(np.mean(corrs))
    del model
    gc.collect()
    return {
        "name": name,
        "enc3_std": enc3_std,
        "merge2_std": merge2_std,
        "bn_avg_std": bn_avg_std,
        "tail_std_mean": tail_std_mean,
        "corr_mean": corr_mean,
        "corr_per_crop": corrs,
        "tail_std_per_crop": tail_stds,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== enc3_probe: b15-21 + merge2 (b14) root cause ===\n")
    t_global = time.time()

    # ---- load capture ----
    print("loading cap2_live frame 0 ...")
    bef = load_rgb("model_input_00.raw").astype(np.float32)
    aft = load_rgb("after_00.raw").astype(np.float32)
    dep = load_depth("depth_00.raw").astype(np.float32)
    mv = load_motion("motion_00.raw").astype(np.float32)
    simple_delta = (aft - bef).astype(np.float32)
    print(f"  bef std={bef.std():.3f}  aft std={aft.std():.3f}  delta std={simple_delta.std():.4f}")

    # ---- load by_block ----
    print("loading by_block ...")
    blob_full = open(BLOB, "rb").read()
    by_block = load_all()

    # ============================================================
    # PART 1: STATIC AUDIT
    # ============================================================
    print("\n=== PART 1: Static audit of enc3 + b14 + b22 tensors ===\n")

    torch.manual_seed(42)
    model = DLSS5NetCalib().eval()
    fill_model(model, by_block, blob_full=blob_full)

    # Audit enc3 + b14 (merge2) + b22 (merge3)
    audit_names = []
    # enc3 (7 blocks)
    for bi in range(7):
        for n in ["norm1.weight", "norm2.weight", "attn.relative_position_bias_table",
                  "attn.qkv.weight", "attn.proj.weight",
                  "mlp.0.weight", "mlp.0.bias", "mlp.2.weight"]:
            audit_names.append(f"enc.3.blocks.{bi}.{n}")
    # b14 (merge2), b22 (merge3), b4 (merge0), b8 (merge1)
    for merge_name, mi in [("merges.0", 0), ("merges.1", 1), ("merges.2", 2), ("merges.3", 3)]:
        for n in ["norm.weight", "norm.bias", "reduction.weight"]:
            audit_names.append(f"{merge_name}.{n}")
    audit = run_audit(model, audit_names)
    del model
    gc.collect()

    # Print worst offenders (|mean dev| > 3 sigma from prior)
    print("=== enc3 + merge tensors deviating >3σ from prior ===\n")
    print(f"{'name':50s}  {'kind':14s}  {'obs_mean':>10s}  {'obs_std':>10s}  {'prior_mean':>10s}  {'dev_σ':>8s}")
    print("-" * 110)
    bad = []
    for a in audit:
        if a["kind"] in ("ln_gamma", "ln_bias", "linear_bias", "rel_bias"):
            if abs(a["mean_dev_sigma"]) > 3.0:
                bad.append(a)
                print(f"{a['name']:50s}  {a['kind']:14s}  {a['obs_mean']:+10.3f}  {a['obs_std']:10.3f}  "
                      f"{a['prior_mean']:+10.3f}  {a['mean_dev_sigma']:+8.2f}")
    print(f"\nTotal bad tensors: {len(bad)}")
    print(f"  by kind: ln_gamma={sum(1 for a in bad if a['kind']=='ln_gamma')}, "
          f"ln_bias={sum(1 for a in bad if a['kind']=='ln_bias')}, "
          f"linear_bias={sum(1 for a in bad if a['kind']=='linear_bias')}, "
          f"rel_bias={sum(1 for a in bad if a['kind']=='rel_bias')}")

    # Save audit
    out_audit = os.path.join(HERE, ".tmp", "enc3_audit.json")
    os.makedirs(os.path.dirname(out_audit), exist_ok=True)
    with open(out_audit, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  audit saved → {out_audit}")

    # ============================================================
    # PART 2: REPLACEMENT EXPERIMENTS
    # ============================================================
    print("\n=== PART 2: Single-tensor replacement experiments ===\n")

    # Baseline (canonical fill, no mutation)
    base = run_experiment("BASELINE (canonical, no mutation)", [], bef, dep, mv, simple_delta, by_block, blob_full)
    print(f"  BASELINE              : enc3={base['enc3_std']:7.3f}  merge2={base['merge2_std']:7.3f}  "
          f"bn_avg={base['bn_avg_std']:7.3f}  tail={base['tail_std_mean']:7.4f}  corr={base['corr_mean']:+.4f}")

    # The most likely candidate: merges.2.norm.weight (LN gamma) and merges.2.norm.bias
    candidates = []

    # 1. Fix merges.2.norm.weight = 1 (LN gamma reset)
    def fix_merges2_norm_weight(p):
        p.fill_(1.0)

    def fix_merges2_norm_bias(p):
        p.zero_()

    def fix_merges2_both(p):
        p.fill_(1.0)

    # For the multi-arg transform, we need to apply differently for weight vs bias
    def _fix_merges2_all(model_pmap, bname):
        """Fix merges.2.norm.weight=1, .bias=0 in-place."""
        if f'merges.2.norm.weight' in model_pmap:
            model_pmap['merges.2.norm.weight'].fill_(1.0)
        if f'merges.2.norm.bias' in model_pmap:
            model_pmap['merges.2.norm.bias'].zero_()

    # Use a custom experiment runner for merges.2 (which is not per-block)
    def run_merge_experiment(name, fix_fn):
        torch.manual_seed(42)
        m = DLSS5NetCalib().eval()
        fill_model(m, by_block, blob_full=blob_full)
        with torch.no_grad():
            fix_fn(dict(m.named_parameters()))
        stats, cleanup = attach_stats_hooks(m)
        corrs, tail_stds = run_crops(m, bef, dep, mv, simple_delta)
        cleanup()
        result = {
            "name": name,
            "enc3_std": stats["enc3"].std,
            "merge2_std": stats["merge2"].std,
            "bn_avg_std": float(np.mean([stats[f"bn{i}"].std for i in range(8)])),
            "tail_std_mean": float(np.mean(tail_stds)),
            "corr_mean": float(np.mean(corrrs := corrs)),
            "corr_per_crop": corrs,
            "tail_std_per_crop": tail_stds,
        }
        del m
        gc.collect()
        return result

    # A. merges.2.norm.weight = 1 (LN gamma reset)
    res = run_merge_experiment("merges.2.norm.weight = 1, bias = 0 (canonical fix)",
                               lambda p: _fix_merges2_all(None, None) if False else None)
    res = run_merge_experiment(
        "merges.2.norm.weight = 1, bias = 0",
        lambda pmap: (pmap['merges.2.norm.weight'].fill_(1.0), pmap['merges.2.norm.bias'].zero_())
    )
    candidates.append(res)
    print(f"  merges.2.norm=1       : enc3={res['enc3_std']:7.3f}  merge2={res['merge2_std']:7.3f}  "
          f"bn_avg={res['bn_avg_std']:7.3f}  tail={res['tail_std_mean']:7.4f}  corr={res['corr_mean']:+.4f}")

    # B. merges.2.norm.weight = 1 only (keep bias from canonical)
    res = run_merge_experiment(
        "merges.2.norm.weight = 1 (bias kept)",
        lambda pmap: pmap['merges.2.norm.weight'].fill_(1.0)
    )
    candidates.append(res)
    print(f"  merges.2.norm.weight=1: enc3={res['enc3_std']:7.3f}  merge2={res['merge2_std']:7.3f}  "
          f"bn_avg={res['bn_avg_std']:7.3f}  tail={res['tail_std_mean']:7.4f}  corr={res['corr_mean']:+.4f}")

    # C. merges.2.norm.weight & bias both = 1
    res = run_merge_experiment(
        "merges.2.norm.weight = bias = 1",
        lambda pmap: (pmap['merges.2.norm.weight'].fill_(1.0), pmap['merges.2.norm.bias'].fill_(1.0))
    )
    candidates.append(res)
    print(f"  merges.2.norm=1,bias=1: enc3={res['enc3_std']:7.3f}  merge2={res['merge2_std']:7.3f}  "
          f"bn_avg={res['bn_avg_std']:7.3f}  tail={res['tail_std_mean']:7.4f}  corr={res['corr_mean']:+.4f}")

    # D. Fix ALL merge norms (merges.{0,1,2,3}.norm.weight = 1, bias = 0)
    res = run_merge_experiment(
        "ALL merges.norm.weight = 1, bias = 0",
        lambda pmap: [pmap[f'merges.{i}.norm.weight'].fill_(1.0) for i in range(4)] +
                     [pmap[f'merges.{i}.norm.bias'].zero_() for i in range(4)]
    )
    candidates.append(res)
    print(f"  ALL merges.norm=1     : enc3={res['enc3_std']:7.3f}  merge2={res['merge2_std']:7.3f}  "
          f"bn_avg={res['bn_avg_std']:7.3f}  tail={res['tail_std_mean']:7.4f}  corr={res['corr_mean']:+.4f}")

    # E. Re-set merges.2.norm from the LAST 512 vals of b14 misc (the small ones)
    def fix_merges2_norm_from_tail(pmap):
        b14_misc = by_block[14]['layer0'][1]
        if len(b14_misc) >= 512:
            v = b14_misc[-512:].astype(np.float32)  # last 512 fp16 vals
            with torch.no_grad():
                pmap['merges.2.norm.weight'].copy_(torch.from_numpy(v))
                pmap['merges.2.norm.bias'].zero_()
        else:
            pmap['merges.2.norm.weight'].fill_(1.0)
            pmap['merges.2.norm.bias'].zero_()
    res = run_merge_experiment(
        "merges.2.norm from b14 misc[-512:] (the small fp16 tail)",
        fix_merges2_norm_from_tail
    )
    candidates.append(res)
    print(f"  merges.2.norm=tail512 : enc3={res['enc3_std']:7.3f}  merge2={res['merge2_std']:7.3f}  "
          f"bn_avg={res['bn_avg_std']:7.3f}  tail={res['tail_std_mean']:7.4f}  corr={res['corr_mean']:+.4f}")

    # F. Single-tensor swap experiments for enc3 (the b15-21 swin blocks)    # Confirm none of these affect enc3 by themselves (already verified, but
    # run a few here for the report table).
    print("\n--- Sanity: per-tensor swap experiments on enc3 (b15-21) ---")
    enc3_experiments = [
        ("enc3 mlp.2.weight -> Kaiming",
         lambda pmap: [pmap[f'enc.3.blocks.{bi}.mlp.2.weight'].copy_(
             torch.randn_like(pmap[f'enc.3.blocks.{bi}.mlp.2.weight']) / np.sqrt(384))
             for bi in range(7)]),
        ("enc3 norm1+norm2.weight = 1",
         lambda pmap: [pmap[f'enc.3.blocks.{bi}.norm1.weight'].fill_(1.0) for bi in range(7)] +
                      [pmap[f'enc.3.blocks.{bi}.norm2.weight'].fill_(1.0) for bi in range(7)]),
        ("enc3 rel_bias -> Kaiming",
         lambda pmap: [pmap[f'enc.3.blocks.{bi}.attn.relative_position_bias_table'].add_(
             torch.randn_like(pmap[f'enc.3.blocks.{bi}.attn.relative_position_bias_table']) * 0.02)
             for bi in range(7)]),
    ]
    for name, fix_fn in enc3_experiments:
        res = run_merge_experiment(name, fix_fn)
        print(f"  {name:40s}: enc3={res['enc3_std']:7.3f}  merge2={res['merge2_std']:7.3f}  "
              f"tail={res['tail_std_mean']:7.4f}  corr={res['corr_mean']:+.4f}")

    # G. COMBINED fix: merge2.norm + enc3 norm + enc3 mlp.2 (the full recipe)
    #    Note: experiment B (merge2 fix + enc3 norm=1 only) makes enc3 WORSE
    #    (1.9 -> 22.9) because enc3's mlp.2.weight is still pathological
    #    (67% zeros, 2 hot rows) — once norm=1 lets signal flow, the broken
    #    mlp amplifies. So the healthy recipe needs all three fixes.
    print("\n--- Combined fix experiments (merge2 + enc3) ---")
    def _fix_merge2(pmap):
        pmap['merges.2.norm.weight'].fill_(1.0)
        pmap['merges.2.norm.bias'].zero_()

    def _fix_enc3_norms(pmap):
        for bi in range(7):
            pmap[f'enc.3.blocks.{bi}.norm1.weight'].fill_(1.0)
            pmap[f'enc.3.blocks.{bi}.norm2.weight'].fill_(1.0)

    def _fix_enc3_mlp2(pmap):
        for bi in range(7):
            p = pmap[f'enc.3.blocks.{bi}.mlp.2.weight']
            p.copy_(torch.randn_like(p) / np.sqrt(384))

    combined = [
        ("COMBINED-A: merge2 fix only", [_fix_merge2]),
        ("COMBINED-B: merge2 + enc3 norm=1", [_fix_merge2, _fix_enc3_norms]),
        ("COMBINED-C: merge2 + enc3 norm=1 + enc3 mlp.2 Kaiming",
         [_fix_merge2, _fix_enc3_norms, _fix_enc3_mlp2]),
        ("COMBINED-D: all merges + enc3 norm=1 + mlp.2 Kaiming",
         [lambda pmap: [pmap[f'merges.{i}.norm.weight'].fill_(1.0) for i in range(4)] and
                       [pmap[f'merges.{i}.norm.bias'].zero_() for i in range(4)],
          _fix_enc3_norms, _fix_enc3_mlp2]),
    ]
    for name, fixes in combined:
        def run(name=name, fixes=fixes):
            torch.manual_seed(42)
            m = DLSS5NetCalib().eval()
            fill_model(m, by_block, blob_full=blob_full)
            pmap = dict(m.named_parameters())
            with torch.no_grad():
                for f in fixes:
                    f(pmap)
            stats, cleanup = attach_stats_hooks(m)
            corrs, tail_stds = run_crops(m, bef, dep, mv, simple_delta)
            cleanup()
            r = {
                "name": name,
                "enc2_std": stats["enc2"].std,
                "enc3_std": stats["enc3"].std,
                "merge2_std": stats["merge2"].std,
                "bn_avg_std": float(np.mean([stats[f"bn{i}"].std for i in range(8)])),
                "dec1_std": stats["dec1"].std,
                "tail_std_mean": float(np.mean(tail_stds)),
                "corr_mean": float(np.mean(corrs)),
                "corr_per_crop": corrs,
                "tail_std_per_crop": tail_stds,
            }
            del m
            gc.collect()
            return r
        res = run()
        candidates.append(res)
        print(f"  {name:52s}: enc2={res['enc2_std']:6.3f}  merge2={res['merge2_std']:6.3f}  "
              f"enc3={res['enc3_std']:6.3f}  bn_avg={res['bn_avg_std']:8.3f}  "
              f"tail={res['tail_std_mean']:.4f}  corr={res['corr_mean']:+.4f}")

    # ============================================================
    # PART 3: SUMMARY
    # ============================================================
    print("\n=== SUMMARY ===\n")
    print(f"  baseline enc3_std     : {base['enc3_std']:.3f}  (14× jump over enc2 std=1.55)")
    print(f"  baseline merge2_std   : {base['merge2_std']:.3f}")
    print()

    # Sort candidates by combined: low enc3, low merge2, high corr
    candidates.sort(key=lambda r: (r["enc3_std"], -r["corr_mean"]))

    print(f"  best enc3 candidates (sorted by enc3_std, then corr):\n")
    print(f"  {'name':50s}  {'enc3':>7s}  {'merge2':>7s}  {'tail':>7s}  {'corr':>6s}")
    for r in candidates:
        print(f"  {r['name'][:50]:50s}  {r['enc3_std']:7.3f}  {r['merge2_std']:7.3f}  "
                  f"{r['tail_std_mean']:7.4f}  {r['corr_mean']:+6.4f}")

    # Recommended fix
    # NOTE: candidates with enc3_std < 0.5 are DEGENERATE — they kill the
    # merge/enc3 signal entirely (e.g. filling norm with tiny 0.003-scale
    # tail values collapses merge2 output). A healthy fix keeps enc3_std in
    # the 1-6 band (enc2 std is 1.55; a mild growth through 7 blocks is
    # expected) and preserves corr.
    healthy = [r for r in candidates if 0.5 <= r["enc3_std"] <= 8.0]
    pool = healthy if healthy else candidates
    best = max(pool, key=lambda r: r["corr_mean"] - 0.1 * abs(r["enc3_std"] - 3.0))
    print(f"\n  (degenerate candidates excluded: {[r['name'][:40] for r in candidates if r not in pool]})")
    print(f"\n  RECOMMENDED FIX: {best['name']}")
    print(f"    enc3_std : {base['enc3_std']:.3f} → {best['enc3_std']:.3f}  (Δ = {best['enc3_std']-base['enc3_std']:+.3f})")
    print(f"    merge2_std: {base['merge2_std']:.3f} → {best['merge2_std']:.3f}  (Δ = {best['merge2_std']-base['merge2_std']:+.3f})")
    print(f"    corr      : {base['corr_mean']:+.4f} → {best['corr_mean']:+.4f}  (Δ = {best['corr_mean']-base['corr_mean']:+.4f})")

    # Save recommended fix
    fix_recipe = {
        "root_cause": "b14 misc first 512 fp16 vals (std=2.21, mean=-8.78) are mis-decoded "
                      "as LN gamma for merges.2 (c=128→256 merge). The b14 record's "
                      "FP16_TAIL[229936]=155936 boundary is too tight — bytes 155936-end "
                      "include MX-scale factors that are NOT gamma values.",
        "tensor_to_fix": "merges.2.norm.weight",
        "current_value_summary": {
            "std": 2.21, "mean": -8.78, "range": [-12.33, 0.0],
            "first_10_vals": [-2.41, -6.18, -8.94, -9.35, -1.63, -10.60, -11.24, -10.39, -1.68, -7.75]
        },
        "fix_recipe": best["name"],
        "metrics": {
            "baseline": {"enc3_std": base["enc3_std"], "merge2_std": base["merge2_std"],
                         "tail_std_mean": base["tail_std_mean"], "corr_mean": base["corr_mean"]},
            "fixed": {"enc3_std": best["enc3_std"], "merge2_std": best["merge2_std"],
                      "tail_std_mean": best["tail_std_mean"], "corr_mean": best["corr_mean"]},
        }
    }
    out_fix = os.path.join(HERE, ".tmp", "enc3_recommended_fix.json")
    with open(out_fix, "w") as f:
        json.dump(fix_recipe, f, indent=2)
    print(f"\n  recommended fix saved → {out_fix}")

    print(f"\ntotal dt = {time.time() - t_global:.0f}s")


if __name__ == "__main__":
    main()