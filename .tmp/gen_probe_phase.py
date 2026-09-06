#!/usr/bin/env python3
"""gen_probe_phase.py — R41 line-a: comb PHASE FAMILY bin for shift-wiring
phase-response calibration (user generates the official response curve).

PRB1 color-only format (magic 0x31425250, W=1920, H=1050, RGBA16F frames).

Structure (7 frames, single bin):
  f0  BASELINE  : flat 0.5 (reset semantics via the game side; pattern-
                  free anchor for both orientations)
  f1  V delta=2 : vertical 1px stripes, period 8, phase 0.5 (x offset by
                  2px vs the original comb f0 pattern)
  f2  V delta=4 : vertical stripes phase 1.0 == original f0 alignment
  f3  V delta=6 : vertical stripes phase 1.5
  f4  H delta=2 : horizontal 1px stripes, phase 0.5
  f5  H delta=4 : horizontal stripes phase 1.0
  f6  H delta=6 : horizontal stripes phase 1.5
Period fixed at 8px (matches window size — maximally phase-sensitive).
Phase deltas {2,4,6} = 1/4, 1/2, 3/4 of the period; delta=4 is the
phase-inverted comb and doubles as the sign-flip sanity anchor.

Usage: python3 gen_probe_phase.py capP4_phase.bin
"""
import struct, sys
import numpy as np
W, H = 1920, 1050
PERIOD = 8

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

def vstripes(delta):
    fr = blank()
    xs = (np.arange(W)[None, :] + delta) % PERIOD  # (1, W)
    fr[:, :, :3] = np.where(xs[:, :, None] < 4, 0.3, 0.7)
    return fr

def hstripes(delta):
    fr = blank()
    ys = (np.arange(H)[:, None] + delta) % PERIOD
    fr[:, :, :3] = np.where(ys[:, :, None] < 4, 0.3, 0.7)
    return fr

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "capP4_phase.bin"
    frames = [blank(), vstripes(2), vstripes(4), vstripes(6),
                      hstripes(2), hstripes(4), hstripes(6)]
    pack(frames, path)

if __name__ == "__main__":
    main()
