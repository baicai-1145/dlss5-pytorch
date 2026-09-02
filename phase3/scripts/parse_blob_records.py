#!/usr/bin/env python3
"""Parse nvngx_dlssnr weights blob into 153 tensor records (Phase 1 finishing / Phase 4 prep).

Record layout (verified by chained parse, REPORT §3.3 + supervisor byte-evidence):
    name:    N bytes, NO NUL terminator
    u64: A   (payload_off? == B+40)
    u64: A   (same)
    u64: B   (payload byte length)
    meta:    32 bytes (dtype/scale info — opaque, Phase 4)
    payload: B bytes (MXFP8 E4M3 weights, 1 byte/param; bias interleaved)
Next record starts immediately after payload (no alignment padding).
Blob tail: 8 trailing bytes after last record (file_size = end + 8).

Usage: python3 phase3/scripts/parse_blob_records.py  (writes JSON; prints summary)
"""
from __future__ import annotations

import json
import os
import struct
import sys

BLOB = "/root/dlss5-pytorch/weights_blob.bin"


def parse_records(path: str = BLOB, expected: int = 153):
    """Chained parse: records are back-to-back. Layout per record:
        name (no NUL), u64 A, u64 A, u64 B, 32B meta, B-byte payload
    with A == B + 40 always (incl. tiny B=2 scalar records: A=42).
    Return (records, file_size). Each record: name, A, B, meta_off, payload_off.
    """
    size = os.path.getsize(path)
    recs = []
    off = 16
    with open(path, "rb") as f:
        while len(recs) < expected:
            f.seek(off)
            head = f.read(512)
            # find name length: smallest k with u64s (a,a,a-40) at k, k+8, k+16
            # name may itself contain bytes that look like a tiny A (e.g. '*'+\x00...);
            # require the A,A,B pattern AND B==A-40, and k >= 12.
            k = None
            for kk in range(8, 256):
                a = struct.unpack_from("<Q", head, kk)[0]
                b = struct.unpack_from("<Q", head, kk + 16)[0]
                # a is u64 (payload length + 40) — payload length in [0, 2**32)
                if a == struct.unpack_from("<Q", head, kk + 8)[0] and b == a - 40 \
                        and 0 <= b < 2 ** 32 and a < 2 ** 32:
                    # ensure bytes before kk decode to a printable-ish name (no interior NUL)
                    nm = head[:kk]
                    if kk >= 12 and all(32 <= c < 127 for c in nm):
                        k = kk
                        break
            if k is None:
                raise ValueError(f"no record triplet at offset {off}: head={head[:96]!r}")
            name = head[:k].decode("ascii")
            a1, a2, b = struct.unpack_from("<QQQ", head, k)
            meta_off = off + k + 24
            payload_off = meta_off + 32
            recs.append({"name": name, "A": a1, "B": b,
                         "meta_off": meta_off, "payload_off": payload_off, "name_len": k})
            off = payload_off + b
    return recs, size


def main():
    recs, size = parse_records()
    print(f"file {size} B | {len(recs)} records | last end {recs[-1]['payload_off']+recs[-1]['B']} "
          f"(tail {size - (recs[-1]['payload_off']+recs[-1]['B'])} B)")
    s = 0
    for r in recs:
        s += r["B"]
    print(f"Σ payload = {s:,} B (blob={size:,}, non-payload {size-s:,})")
    # group by block index for supervisor cross-check
    from collections import OrderedDict
    blocks = OrderedDict()
    for r in recs:
        bi = r["name"].split(".")[0]
        blocks.setdefault(bi, []).append(r)
    print(f"blocks: {len(blocks)}")
    for bi, rs in blocks.items():
        print(f"  {bi:8s} n={len(rs):2d} ΣB={sum(r['B'] for r in rs):>11,}  " +
              ", ".join(f"{r['name'].split('.')[-1]}={r['B']}" for r in rs))
    # save json
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blob_records.json")
    with open(out, "w") as f:
        json.dump({"file_size": size, "records": recs}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
