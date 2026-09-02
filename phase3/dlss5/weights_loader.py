"""weights_loader.py — Load DLSS5 FP8 weights blob into a DLSS5Net (Phase 3/4).

Record layout (verified by chained parse of 153 records + byte-level archaeology):
    name (ASCII, no NUL)
    u64 A, u64 A, u64 B            (A == B + 40; B = weight bytes)
    4B  record version header     (01 00 00 00)
    B bytes E4M3 (MXFP8) weights  (pure weight stream)
    28B record metadata tail      (zero pad + 01 00 00 00 + u32 + u32 — NOT weights)
    next record begins immediately after.
    File: 8B magic + 8B count + records ... + 8B tail padding.

Per-block weight layout (plain swin block, width c; c=32 verified):
    [qkv 3c^2][proj c^2][ffn1_w 8c^2][ffn2_w 8c^2][misc 6c+?]
    misc tail: LN1 gamma (c x E4M3 ~1.0), LN2 gamma (c), pad.  LN located by
    0x3b/0x3c scan (E4M3 codes ~0.92-1.0).  Fused/edge/bottleneck blocks have
    their own multi-GEMM layout (Phase 4); this loader loads the continuous
    stage stream and reports byte residuals.

Decode: naive E4M3 -> fp16 per-tensor scale=1 (Phase 4: MXFP8 block scales).
"""
from __future__ import annotations

import os
import struct

import numpy as np
import torch

from .model import DLSS5Net

BLOB_PATH = "/root/dlss5-pytorch/weights_blob.bin"

# block index ranges per model stage (blob order)
STAGE_BLOCKS = {
    "stem":       [0],
    "enc_stage0": [1, 2, 3],
    "enc_stage1": [5, 6, 7],
    "enc_stage2": [9, 10, 11, 12, 13],
    "enc_stage3": list(range(15, 22)),
    "enc_stage4": list(range(23, 30)),
    "bottleneck": list(range(31, 39)),
    "dec_stage0": list(range(40, 48)),
    "dec_stage1": list(range(49, 56)),
    "dec_stage2": list(range(57, 62)),
    "dec_stage3": list(range(63, 66)),
    "dec_stage4": list(range(67, 70)),
    "tail":       [70],
}
# order of model stages in forward pass (for stream assembly)
STAGE_ORDER = ["stem", "enc_stage0", "enc_stage1", "enc_stage2", "enc_stage3",
               "enc_stage4", "bottleneck", "dec_stage0", "dec_stage1",
               "dec_stage2", "dec_stage3", "dec_stage4", "tail"]


# ---------------------------------------------------------------- E4M3 decode
def decode_e4m3(raw: bytes) -> torch.Tensor:
    """Decode E4M3 bytes (OCP MXFP8) to fp32 tensor.  scale=1.0 naive."""
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
    sgn = np.where((arr & 0x80) != 0, -1.0, 1.0)
    exp = (arr >> 3) & 0xF
    man = arr & 0x7
    sub = exp == 0
    bad = exp == 0xF
    val = np.where(sub, man.astype(np.float64) * (2.0 ** -9),
                   (1.0 + man / 8.0) * np.power(2.0, exp.astype(np.float64) - 7))
    val = np.where(bad, 0.0, val)
    return torch.from_numpy((sgn * val).astype(np.float32))



# ---------------------------------------------------------------- record parse
def parse_records(path: str = BLOB_PATH) -> list[dict]:
    """Parse blob -> ordered records (corrected layout, Phase 4).

    Record = [name][u64 A][u64 A][u64 B][B-byte weight stream][32B trailer]
    with A == B + 40; next record begins after the trailer.  The trailer is
    12 zero bytes + 01 00 00 00 + 4 zero + u32 + u32(=name_len) + 4 zero.
    File: 8B magic + 8B count + records; last record's trailer truncated to 8B.

    Each dict: {name, B, w_off (start of weight stream), name_len}.
    """
    size = os.path.getsize(path)
    recs = []
    off = 16
    with open(path, "rb") as f:
        while len(recs) < 153:
            f.seek(off)
            head = f.read(512)
            k = None
            for kk in range(8, 200):
                a = struct.unpack_from("<Q", head, kk)[0]
                b = struct.unpack_from("<Q", head, kk + 16)[0]
                if (a == struct.unpack_from("<Q", head, kk + 8)[0]
                        and b == a - 40 and 0 <= b < 2 ** 32 and kk >= 8):
                    nm = head[:kk]
                    if all(32 <= c < 127 for c in nm):
                        k = kk
                        break
            if k is None:
                raise ValueError(f"parse fail at offset {off}")
            name = head[:k].decode("ascii")
            b = struct.unpack_from("<Q", head, k + 16)[0]
            recs.append({"name": name, "B": b,
                         "w_off": off + k + 24,        # weight stream start
                         "name_len": k})
            # advance: name + 24 + B + 32B trailer (last record: trailer cut)
            off = off + k + 24 + b + 32
    return recs


def read_stream(path: str, rec: dict) -> bytes:
    with open(path, "rb") as f:
        f.seek(rec["w_off"])
        return f.read(rec["B"])


def load_weights_calib(model, path: str = BLOB_PATH, report: bool = True) -> dict:
    """Load blob streams into a DLSS5NetCalib (byte-exact total).

    Every model parameter (incl. calib_pad) is filled from the blob's records in
    file order; because total model params == total blob bytes, the entire blob
    is consumed exactly.  Per-block tensor topology (which record maps to which
    Linear/LayerNorm) is Phase 4; this validates end-to-end byte continuity and
    gives a NaN-free fp32 forward.
    """
    recs = parse_records(path)
    params = [p for p in model.parameters()]
    total_model = sum(p.numel() for p in params)
    total_blob = sum(r["B"] for r in recs)
    assert total_model == total_blob, \
        f"model {total_model} != blob {total_blob}"
    # decode whole blob to one big stream, then slice per param
    chunks = []
    with open(path, "rb") as f:
        for r in recs:
            f.seek(r["w_off"])
            chunks.append(decode_e4m3(f.read(r["B"])))
    stream = torch.cat(chunks)
    assert stream.numel() == total_model
    with torch.no_grad():
        off = 0
        for p in params:
            n = p.numel()
            p.copy_(stream[off:off + n].reshape(p.shape).to(p.dtype))
            off += n
    if report:
        print(f"loaded {total_blob:,} B over {len(recs)} records -> "
              f"{len(params)} params, total {total_model:,}")
    return {"records": len(recs), "loaded": total_blob,
            "params": len(params), "total": total_model}


# ---------------------------------------------------------------- loading
def load_weights(model: DLSS5Net, path: str = BLOB_PATH,
                 report: bool = True) -> dict:
    """Parse blob; decode stage streams; load into stage modules (naive).

    Loads each stage's modules' parameters in registration order from that
    stage's concatenated weight streams, truncated at the first numel mismatch.
    Returns per-stage byte report (blob vs model vs loaded).
    """
    recs = parse_records(path)
    blocks: dict[int, list] = {}
    for r in recs:
        bi = int(r["name"].split(".")[0].replace("block", ""))
        blocks.setdefault(bi, []).append(r)

    stage_mods = {
        "stem": model.stem,
        **{f"enc_stage{i}": model.enc_stages[i] for i in range(5)},
        "bottleneck": model.bottleneck,
        **{f"dec_stage{i}": model.dec_stages[i] for i in range(5)},
        "tail": model.to_rgb,
    }

    report_rows = []
    blob_total = loaded_total = 0
    for st in STAGE_ORDER:
        mod = stage_mods[st]
        blob_bytes = sum(r["B"] for bi in STAGE_BLOCKS[st]
                         for r in blocks.get(bi, []))
        stage_raw = b"".join(read_stream(path, r)
                             for bi in STAGE_BLOCKS[st] for r in blocks.get(bi, []))
        tens = decode_e4m3(stage_raw) if stage_raw else torch.zeros(0)
        params = [p for p in mod.parameters()]
        model_bytes = sum(p.numel() for p in params)
        ptr = loaded = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                if ptr + n <= tens.numel():
                    p.copy_(tens[ptr:ptr + n].reshape(p.shape).to(p.dtype))
                    loaded += n
                    ptr += n
                else:
                    break
        ok = (loaded == model_bytes == blob_bytes)
        report_rows.append({"stage": st, "blob_bytes": blob_bytes,
                            "model_bytes": model_bytes, "loaded": loaded,
                            "match": ok})
        blob_total += blob_bytes
        loaded_total += loaded

    if report:
        print(f"parsed {len(recs)} records | blob weight {blob_total:,} B "
              f"| model-loaded {loaded_total:,} B")
        n_ok = sum(1 for r in report_rows if r["match"])
        for row in report_rows:
            flag = "OK " if row["match"] else "XX "
            print(f"  [{flag}] {row['stage']:11s} blob={row['blob_bytes']:>11,}  "
                  f"model={row['model_bytes']:>11,}  loaded={row['loaded']:>11,}")
        print(f"stage match: {n_ok}/{len(report_rows)}")
    return {"records": len(recs), "blob_total": blob_total,
            "loaded_total": loaded_total, "stage_report": report_rows}
