#!/usr/bin/env python3
"""eval_suite.py — multi-dimensional quality evaluation for the DLSS5 replica.

Designed after three single-metric deceptions (PSNR -> delta-corr ->
edit-ratio): no single number can be trusted. This suite measures DIRECTIONAL
image-quality changes of the official DLL vs our replica across independent
axes, so "replica only changed saturation" class failures are caught
immediately.

Sections
--------
A. Directional quality metrics — 3 columns (input / replica / official):
   1. sharpness   : gradient energy of luma at 3 scales (1px / 2px / 4px)
   2. grain       : Laplacian std, split into flat vs edge regions
   3. color       : saturation, contrast (luma std), chroma shift (a/b means)
   4. ringing     : overshoot count & magnitude near strong edges
B. Delta decomposition — official and replica edits decomposed into
   {global tone | smooth field (<=3x3) | hf band | luma vs chroma},
   each with correlation (structure match) AND amplitude ratio (energy match)
C. Perceptual — MS-SSIM(replica, official); LPIPS if installed
D. Anti-overfit protocol — tone LUT fit on frames 0-7, report on 8-15,
   plus cross-scene transfer (cold_game) without refit
E. Human protocol — 4-quadrant zoom (person / flat / edge / grain) + triptych

Usage
-----
  from eval_suite import Scoreboard, load_frames, run_replica
  sb = Scoreboard(frames, replica=..., official=...)
  sb.report()                    # full text scoreboard
  sb.to_dict()                   # machine-readable

  python3 eval_suite.py --data cap3_live/clean \
      --cross cap3_live/cold_game --json .tmp/scoreboard.json

Acceptance gates (GitHub, from Round 16 on):
  * no regression in any scoreboard dimension vs previous round
  * directional metrics: replica must move in the SAME direction as official
    on sharpness (the currently-failing axis) — amplitude target follows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# loading helpers (raw fp16 DXGI buffers)
# ----------------------------------------------------------------------------

def load_fp16_rgb(path: str, H: int = 1050, W: int = 1920) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float16)
    a = a[: H * W * 4].reshape(H, W, 4)
    return np.ascontiguousarray(a[..., :3]).astype(np.float32)


def load_depth(path: str, H: int = 1050, W: int = 1920) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float16)
    return a[: H * W].reshape(H, W).astype(np.float32)


def load_motion(path: str, H: int = 1050, W: int = 1920) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float16)
    return a[: H * W * 2].reshape(H, W, 2).astype(np.float32)


@dataclass
class FrameSet:
    """One evaluated scene: input / official output (+ optional replica)."""
    inputs: np.ndarray    # (N,H,W,3) float32 [0,1]
    officials: np.ndarray  # (N,H,W,3)
    name: str = "scene"

    def __len__(self):
        return len(self.inputs)


# ----------------------------------------------------------------------------
# A. directional quality metrics (each returns scalars per image batch)
# ----------------------------------------------------------------------------

LUMA_W = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)


def _to_t(imgs: np.ndarray) -> torch.Tensor:
    """(N,H,W,3) -> (N,3,H,W)"""
    return torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))


def luma(imgs_t: torch.Tensor) -> torch.Tensor:
    return (imgs_t * LUMA_W.to(imgs_t.dtype).to(imgs_t.device)).sum(1, keepdim=True)


def _grad_mag(l: torch.Tensor) -> torch.Tensor:
    gx = l[:, :, :, 1:] - l[:, :, :, :-1]
    gy = l[:, :, 1:, :] - l[:, :, :-1, :]
    m = F.pad(gx, (0, 1, 0, 0)).abs() ** 2 + F.pad(gy, (0, 0, 0, 1)).abs() ** 2
    return m


def sharpness(imgs: np.ndarray) -> dict:
    """Gradient energy at 3 scales. Scale k = gradient after k-fold box downsample."""
    t = _to_t(imgs)
    l = luma(t)
    out = {}
    for s, tag in [(1, "s1"), (2, "s2"), (4, "s4")]:
        if s > 1:
            ls = F.avg_pool2d(l, s)
        else:
            ls = l
        out[tag] = _grad_mag(ls).mean().item()
    return out


def _laplacian(l: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=l.device, dtype=l.dtype).view(1, 1, 3, 3)
    return F.conv2d(l, k, padding=1)


def grain(imgs: np.ndarray, pct: float = 0.3) -> dict:
    """Laplacian std on flat vs edge regions (regions from input gradient)."""
    t = _to_t(imgs)
    l = luma(t)
    lp = _laplacian(l)
    g = _grad_mag(l).sqrt()
    # per-image thresholds by percentile
    N = l.shape[0]
    flat_stds, edge_stds = [], []
    for i in range(N):
        lo = torch.quantile(g[i].ravel(), pct)
        hi = torch.quantile(g[i].ravel(), 1 - pct)
        flat = g[i] <= lo
        edge = g[i] >= hi
        flat_stds.append(lp[i][flat].std().item())
        edge_stds.append(lp[i][edge].std().item())
    return {"flat": float(np.mean(flat_stds)), "edge": float(np.mean(edge_stds))}


def color_stats(imgs: np.ndarray, ref: np.ndarray | None = None) -> dict:
    """Saturation, contrast, chroma means; and shifts vs optional reference."""
    t = _to_t(imgs)
    mx = t.max(1, keepdim=True).values
    mn = t.min(1, keepdim=True).values
    sat = ((mx - mn) / (mx + 1e-6)).mean().item()
    l = luma(t)
    con = l.std().item()
    # chroma plane: R/B deviation from luma (hue direction proxy)
    a = (t[:, 0:1] - l).mean().item()
    b = (t[:, 2:3] - l).mean().item()
    out = {"sat": sat, "con": con, "a": a, "b": b}
    if ref is not None:
        r = color_stats(ref)
        out = {**out, "d_sat": sat - r["sat"], "d_con": con - r["con"],
               "d_a": a - r["a"], "d_b": b - r["b"]}
    return out


def ringing(imgs: np.ndarray, ref: np.ndarray, edge_pct: float = 0.9,
            thr: float = 0.05) -> dict:
    """Overshoot proxy: near strong edges (of ref), count/magnitude of hp
    excursions beyond `thr`. Ringing = oscillation absent in ref."""
    t = _to_t(imgs)
    rt = _to_t(ref)
    l = luma(t)
    lr = luma(rt)
    lp = F.avg_pool2d(l, 3, 1, 1)
    hp = (l - lp).abs()
    g = _grad_mag(lr).sqrt()
    N = l.shape[0]
    counts, mags = [], []
    for i in range(N):
        hi = torch.quantile(g[i].ravel(), edge_pct)
        em = g[i] >= hi
        # dilate edge mask by 2px
        emd = F.max_pool2d(em.float(), 5, 1, 2) > 0
        sel = emd & (~em)
        if sel.sum() == 0:
            continue
        h = hp[i][sel]
        counts.append((h > thr).float().mean().item())
        mags.append(h.mean().item())
    return {"count": float(np.mean(counts)) if counts else 0.0,
            "mag": float(np.mean(mags)) if mags else 0.0}


# ----------------------------------------------------------------------------
# B. delta decomposition
# ----------------------------------------------------------------------------

def decompose(delta: np.ndarray) -> dict:
    """delta (N,H,W,3) -> components. Returns dict of torch tensors (N,3,H,W)."""
    d = _to_t(delta)
    N = d.shape[0]
    tone = d.mean(dim=(2, 3), keepdim=True).expand_as(d).contiguous()
    smooth = F.avg_pool2d(d, 3, 1, 1)
    hf = d - smooth
    l = (d * LUMA_W.to(d.dtype)).sum(1, keepdim=True).expand_as(d) * \
        (LUMA_W.to(d.dtype) / (LUMA_W.to(d.dtype) ** 2).sum())
    chroma = d - l * LUMA_W.to(d.dtype) / (LUMA_W.to(d.dtype) ** 2).sum()
    return {"tone": tone - smooth * 0,  # tone = global mean component
            "smooth": smooth - tone,    # smooth field minus global mean
            "hf": hf,
            "luma": l * LUMA_W.to(l.dtype),
            "chroma": chroma}


def decomp_report(rep_delta: np.ndarray, off_delta: np.ndarray) -> dict:
    """corr + amplitude ratio per component (pooled over frames & channels)."""
    rd = decompose(rep_delta)
    od = decompose(off_delta)
    out = {}
    for k in ["smooth", "hf", "luma", "chroma"]:
        a = rd[k].ravel().numpy()
        b = od[k].ravel().numpy()
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        ra = float(np.abs(a).mean())
        rb = float(np.abs(b).mean())
        out[k] = {"corr": corr,
                  "amp_rep": ra, "amp_off": rb,
                  "ratio": ra / rb if rb > 0 else float("inf")}
    out["tone"] = {"rep": [float(v) for v in rep_delta.mean((0, 1, 2))],
                   "off": [float(v) for v in off_delta.mean((0, 1, 2))]}
    return out


# ----------------------------------------------------------------------------
# C. perceptual
# ----------------------------------------------------------------------------

def perceptual(replica: np.ndarray, official: np.ndarray) -> dict:
    r = _to_t(replica)
    o = _to_t(official)
    out = {}
    try:
        from pytorch_msssim import ms_ssim
        out["ms_ssim"] = float(ms_ssim(r, o, data_range=1.0, size_average=True).item())
    except Exception as e:  # pragma: no cover
        out["ms_ssim"] = None
    try:
        import lpips as _lp
        fn = _lp.LPIPS(net="alex", verbose=False)
        with torch.no_grad():
            v = fn((r * 2 - 1), (o * 2 - 1))
        out["lpips"] = float(v.mean().item())
    except Exception:  # pragma: no cover
        out["lpips"] = None
    return out


# ----------------------------------------------------------------------------
# scoreboard
# ----------------------------------------------------------------------------

@dataclass
class Scoreboard:
    frames: FrameSet
    replica: np.ndarray
    fit_frames: list = field(default_factory=lambda: list(range(0, 8)))
    report_frames: list = field(default_factory=lambda: list(range(8, 16)))

    def directional(self, imgs_key: str, which: np.ndarray) -> dict:
        inp = self.frames.inputs
        s_in = sharpness(inp)
        s_x = sharpness(which)
        g_in = grain(inp)
        g_x = grain(which)
        c_in = color_stats(inp)
        c_x = color_stats(which)
        r_in = ringing(inp, inp)
        r_x = ringing(which, inp)
        return {
            "sharp": {k: {"in": s_in[k], imgs_key: s_x[k],
                          "d_pct": 100 * (s_x[k] - s_in[k]) / (s_in[k] + 1e-12)}
                      for k in s_in},
            "grain": {k: {"in": g_in[k], imgs_key: g_x[k],
                          "d_pct": 100 * (g_x[k] - g_in[k]) / (g_in[k] + 1e-12)}
                      for k in g_in},
            "color": {"sat": {"in": c_in["sat"], imgs_key: c_x["sat"],
                              "d": c_x["d_sat"] if "d_sat" in c_x else c_x["sat"] - c_in["sat"]},
                      "con": {"in": c_in["con"], imgs_key: c_x["con"],
                              "d": c_x["con"] - c_in["con"]},
                      "a": {"in": c_in["a"], imgs_key: c_x["a"], "d": c_x["a"] - c_in["a"]},
                      "b": {"in": c_in["b"], imgs_key: c_x["b"], "d": c_x["b"] - c_in["b"]}},
            "ring": {"count": {"in": r_in["count"], imgs_key: r_x["count"]},
                     "mag": {"in": r_in["mag"], imgs_key: r_x["mag"]}},
        }

    def run(self) -> dict:
        rf = self.report_frames
        inp = self.frames.inputs[rf]
        off = self.frames.officials[rf]
        rep = self.replica[rf] if self.replica.shape[0] == len(self.frames) else self.replica

        sb = {
            "scene": self.frames.name,
            "directional": {
                "replica": self.directional("rep", rep),
                "official": self.directional("off", off),
            },
            "decomp": decomp_report(rep, off),
            "perceptual": perceptual(np.clip(inp + rep, 0, 1), off),
        }
        return sb

    def report(self) -> str:
        sb = self.run()
        lines = [f"=== SCOREBOARD [{sb['scene']}] (report frames) ==="]
        for side in ["official", "replica"]:
            d = sb["directional"][side]
            lines.append(f"-- {side} vs input --")
            for k, tag in [("s1", "sharp@1px"), ("s2", "sharp@2px"), ("s4", "sharp@4px")]:
                lines.append(f"  {tag:>11s}: {d['sharp'][k]['in']:.5f} -> "
                             f"{d['sharp'][k][list(d['sharp'][k].keys())[1]]:.5f}  "
                             f"({d['sharp'][k]['d_pct']:+.1f}%)")
            for k in ["flat", "edge"]:
                lines.append(f"  grain_{k:>4s}: {d['grain'][k]['in']:.5f} -> "
                             f"{d['grain'][k][list(d['grain'][k].keys())[1]]:.5f}  "
                             f"({d['grain'][k]['d_pct']:+.1f}%)")
            for k in ["sat", "con", "a", "b"]:
                lines.append(f"  {k:>11s}: {d['color'][k]['in']:.5f} -> "
                             f"{d['color'][k][list(d['color'][k].keys())[1]]:.5f}  "
                             f"(d={d['color'][k]['d']:+.5f})")
            lines.append(f"  ring count : {d['ring']['count']['in']:.4f} -> {d['ring']['count'][list(d['ring']['count'].keys())[1]]:.4f}")
            lines.append(f"  ring mag   : {d['ring']['mag']['in']:.5f} -> {d['ring']['mag'][list(d['ring']['mag'].keys())[1]]:.5f}")
        lines.append("-- delta decomposition (replica vs official) --")
        for k in ["smooth", "hf", "luma", "chroma"]:
            c = sb["decomp"][k]
            lines.append(f"  {k:>6s}: corr={c['corr']:+.3f} amp_ratio={c['ratio']:.3f} "
                         f"(rep {c['amp_rep']:.5f} vs off {c['amp_off']:.5f})")
        t = sb["decomp"]["tone"]
        lines.append(f"  tone  : rep={np.round(t['rep'],5).tolist()} off={np.round(t['off'],5).tolist()}")
        lines.append(f"  perceptual: ms_ssim={sb['perceptual']['ms_ssim']} lpips={sb['perceptual']['lpips']}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return self.run()


# ----------------------------------------------------------------------------
# E. human protocol visuals
# ----------------------------------------------------------------------------

def emit_visuals(frames: FrameSet, replica: np.ndarray, out_dir: str,
                 frame_idx: int = 13, quadrants: dict | None = None) -> list:
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    inp, off, rep = frames.inputs[frame_idx], frames.officials[frame_idx], replica[frame_idx]
    to8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
    paths = []
    if quadrants is None:
        Hh, Ww = inp.shape[:2]
        quadrants = {
            "person": (int(Hh*0.62), int(Hh*0.92), int(Ww*0.28), int(Ww*0.52)),
            "flat":   (int(Hh*0.05), int(Hh*0.25), int(Ww*0.60), int(Ww*0.84)),
            "edge":   (int(Hh*0.35), int(Hh*0.65), int(Ww*0.05), int(Ww*0.29)),
            "grain":  (int(Hh*0.70), int(Hh*0.95), int(Ww*0.62), int(Ww*0.86)),
        }
    zoom = 3
    for tag, (y0, y1, x0, x1) in quadrants.items():
        panel = []
        for a in [inp, rep, off]:
            cr = to8(a[y0:y1, x0:x1])
            panel.append(np.kron(cr, np.ones((zoom, zoom, 1), np.uint8)))
        p = os.path.join(out_dir, f"quad_{tag}_f{frame_idx}.png")
        Image.fromarray(np.concatenate(panel, axis=1)).save(p)
        paths.append(p)
    tri = np.concatenate([to8(inp), to8(rep), to8(off)], axis=1)
    p = os.path.join(out_dir, f"triptych_f{frame_idx}.png")
    Image.fromarray(tri).save(p)
    paths.append(p)
    return paths


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DLSS5 replica multi-dimensional scoreboard")
    ap.add_argument("--data", default="cap3_live/clean")
    ap.add_argument("--cross", default=None, help="cross-scene dir (e.g. cap3_live/cold_game)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--vis", default=None, help="dir for human-protocol visuals")
    ap.add_argument("--replica-npy", default=None,
                    help="(N,H,W,3) replica outputs; if absent only directional-official is computed")
    ap.add_argument("--replica-cross", default=None,
                    help="replica npy for the --cross scene (defaults to replica-npy)")
    ap.add_argument("--frames", type=int, default=16)
    args = ap.parse_args()

    def load_scene(d):
        # scene resolution from manifest (default clean 1920x1050)
        Hs, Ws = 1050, 1920
        man = os.path.join(d, 'manifest.txt')
        if os.path.isfile(man):
            for ln in open(man):
                if 'width' in ln and 'before' in ln:
                    Ws = int(ln.split('width')[1].split()[0])
                if 'height' in ln and 'before' in ln:
                    Hs = int(ln.split('height')[1].split()[0])
        ins, offs = [], []
        for i in range(args.frames):
            ins.append(load_fp16_rgb(os.path.join(d, f"before_{i:02d}.raw"), Hs, Ws))
            offs.append(load_fp16_rgb(os.path.join(d, f"model_raw_{i:02d}.raw"), Hs, Ws))
        return FrameSet(np.stack(ins), np.stack(offs), name=os.path.basename(d))

    scene = load_scene(args.data)
    if args.replica_npy:
        rep = np.load(args.replica_npy)
    else:
        rep = scene.inputs.copy()  # identity placeholder (no replica deltas)

    sb = Scoreboard(scene, replica=rep)
    print(sb.report())
    if args.cross:
        cs = load_scene(args.cross)
        rep_c = np.load(args.replica_cross) if args.replica_cross else cs.inputs.copy()
        sb2 = Scoreboard(cs, replica=rep_c)
        print(sb2.report())
        if args.json:
            with open(args.json.replace(".json", "_cross.json"), "w") as f:
                json.dump(sb2.to_dict(), f, indent=2)
    if args.vis:
        for p in emit_visuals(scene, rep, args.vis):
            print("saved", p)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(sb.to_dict(), f, indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
