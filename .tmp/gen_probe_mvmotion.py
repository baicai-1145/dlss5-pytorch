#!/usr/bin/env python3
"""gen_probe_mvmotion.py — R30 0x6e-binding probes, PRB1 bin format.

Windows recon: probe_queue/*.bin = COLOR-ONLY frame sequences (RGBA16F,
PRB1 header: <magic 0x31425250><W><H><frameCount>), matching the original
gen_probes.py. Depth/motion come from the game side, so real MV injection
is NOT available in the bin. The probes therefore use TEMPORAL DISPLACEMENT
(pattern jumps between frames while the game's own motion field is ~0 on a
frozen scene) — the discriminator survives:

  frame 0 (reset=1, no history)  : both hypotheses see the pattern only
  frame 1..3 (reset=0)           : pattern has JUMPED to a new position
    H_A (0x6e = current input)  : bicubic samples the CURRENT input →
                                  delta shows ONLY the new position
    H_B (0x6e = prev net output): bicubic samples the PREVIOUS output →
                                  delta carries a GHOST of the old position
                                  (smoothed by the network footprint)

Sets (each its own bin, 4 frames unless noted):
  p3imp — impulse step: 8x8 dot at (cx,cy) on f0, at (cx+64,cy) on f1-3
          THE discriminator: ghost dot at the OLD position in delta = H_B
  p2edge — edge step: vertical edge x=960 on f0, x=1024 on f1-3
          ghost edge at x=960 in delta = H_B; also band-width measure
  p1dc  — 2 frames only: gray 0.25 then 0.75 (DC step; gate/baseline
          calibration, no spatial signature)

Sizes: 4-frame bin = 64.5MB, 2-frame bin = 32.2MB (matches the known
comb/grayramp/impulse/reset size family).

Usage:
  python3 gen_probe_mvmotion.py p3imp  capP3_impulse.bin
  python3 gen_probe_mvmotion.py p2edge capP2_edge.bin
  python3 gen_probe_mvmotion.py p1dc   capP1_dc.bin
"""
import struct
import sys

import numpy as np

W, H = 1920, 1050


def pack(frames, path):
    frames = np.asarray(frames, np.float32)
    assert frames.shape[1:] == (H, W, 4), frames.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", 0x31425250, W, H, len(frames)))
        f.write(frames.astype(np.float16).tobytes())
    print(f"{path}: {len(frames)} frames, {len(frames)*W*H*8/1e6:.1f}MB")


def blank():
    fr = np.zeros((H, W, 4), np.float32)
    fr[..., 3] = 1.0
    return fr


def main():
    setname, path = sys.argv[1], sys.argv[2]
    cx, cy = W // 2, H // 2

    if setname == "p3imp":
        # f0: dot at (cx, cy); f1-3: dot at (cx+64, cy)
        f0 = blank(); f0[cy:cy + 8, cx:cx + 8, :3] = 1.0
        f1 = blank(); f1[cy:cy + 8, cx + 64:cx + 72, :3] = 1.0
        frames = [f0, f1, f1.copy(), f1.copy()]
    elif setname == "p2edge":
        # f0: edge at x=960 (left 0.3 / right 0.7); f1-3: edge at x=1024
        def edge(x0):
            fr = blank()
            fr[:, :x0, :3] = 0.3
            fr[:, x0:, :3] = 0.7
            return fr
        frames = [edge(960), edge(1024), edge(1024), edge(1024)]
    elif setname == "p1dc":
        a = blank(); a[..., :3] = 0.25
        b = blank(); b[..., :3] = 0.75
        frames = [a, b]
    else:
        sys.exit(f"unknown set {setname}")

    pack(frames, path)


if __name__ == "__main__":
    main()
