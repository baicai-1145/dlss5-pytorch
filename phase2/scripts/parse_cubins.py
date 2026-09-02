#!/usr/bin/env python3
"""parse_cubins.py — Extract & inventory DLSS5 cubins from split/data.bin (NV fatbin).

Reads the CG2R fatbin container in split/data.bin (17 MB), slices the 15 embedded
ELF cubins, and lists every kernel from .nv.info sections with its .nv.constant0
parameter-section byte size.

Usage:
    python3 phase2/scripts/parse_cubins.py            # inventory kernels
    python3 phase2/scripts/parse_cubins.py --json     # machine-readable dump

Notes:
    - cuobjdump/nvdisasm reject these files ("does not contain device code" /
      "invalid ELF") despite valid ELF headers; this is because the embedded
      cubins are *slim* (metadata only, no .text SASS).  We parse sections with
      pure struct, which works.
    - .nv.constant0.<kernel> = per-kernel __constant__ param area (not weights;
      weights live in the separate weights_blob.bin).  Its size bounds the
      kernel's scalar params, not the GEMM shapes.
"""
from __future__ import annotations

import json
import os
import re
import struct
import sys

DATA_BIN = os.path.join(os.path.dirname(__file__), "..", "..", "split", "data.bin")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "cubins")


def find_elf_offsets(data: bytes):
    """Return offsets of all ELF headers in the fatbin blob."""
    out = []
    start = 0
    while True:
        i = data.find(b"\x7fELF", start)
        if i < 0:
            break
        out.append(i)
        start = i + 1
    return out


def elf_sections(data: bytes, base: int):
    """Parse ELF64 section table; return (name, sh_offset, sh_size) list."""
    e_shoff = struct.unpack_from("<Q", data, base + 0x28)[0]
    e_shentsize = struct.unpack_from("<H", data, base + 0x3A)[0]
    e_shnum = struct.unpack_from("<H", data, base + 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", data, base + 0x3E)[0]
    if not e_shoff or not e_shnum:
        return []
    strtab_sec = e_shoff + e_shstrndx * e_shentsize
    str_off = struct.unpack_from("<Q", data, strtab_sec + 0x18)[0]
    str_size = struct.unpack_from("<Q", data, strtab_sec + 0x20)[0]
    strtab = data[str_off:str_off + str_size]
    secs = []
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        name_off = struct.unpack_from("<I", data, sh)[0]
        nm = strtab[name_off:strtab.find(b"\x00", name_off)].decode("latin1") \
            if name_off < len(strtab) else "?"
        off = struct.unpack_from("<Q", data, sh + 0x18)[0]
        size = struct.unpack_from("<Q", data, sh + 0x20)[0]
        secs.append((nm, off, size))
    return secs


def split_cubins(data: bytes, offsets, outdir):
    """Slice each ELF (by section table extent) to <outdir>/cubin_NN.cubin."""
    os.makedirs(outdir, exist_ok=True)
    files = []
    for i, off in enumerate(offsets):
        secs = elf_sections(data, off)
        end = max((o + s for _, o, s in secs), default=off + 1024)
        end = min(end, offsets[i + 1] if i + 1 < len(offsets) else len(data))
        fn = os.path.join(outdir, f"cubin_{i:02d}.cubin")
        with open(fn, "wb") as f:
            f.write(data[off:end])
        files.append(fn)
    return files


def inventory(fn):
    """Return {kernel_name: const0_size} for one cubin file."""
    data = open(fn, "rb").read()
    secs = elf_sections(data, 0)
    info_names = [nm for nm, _, _ in secs if nm.startswith(".nv.info.")]
    kernels = {}
    for nm, off, size in secs:
        if nm.startswith(".nv.constant0.") and nm != ".nv.constant0":
            kname = nm[len(".nv.constant0."):]
            kernels[kname] = size
    return kernels, len(info_names)


def main():
    as_json = "--json" in sys.argv
    data = open(DATA_BIN, "rb").read()
    offs = find_elf_offsets(data)
    files = split_cubins(data, offs, OUT_DIR)
    report = {}
    for fn in files:
        kernels, ninfo = inventory(fn)
        report[os.path.basename(fn)] = {
            "kernels": kernels,
            "nv_info_sections": ninfo,
        }
        if not as_json:
            print(f"=== {os.path.basename(fn)}: {len(kernels)} const0 kernels ===")
            for k, sz in sorted(kernels.items()):
                print(f"  {sz:5d}B  {k}")
    if as_json:
        print(json.dumps(report, indent=1))
    else:
        print(f"\nextracted {len(files)} cubins -> {OUT_DIR}/")
        allk = [k for v in report.values() for k in v["kernels"]]
        print(f"total kernels: {len(allk)}")


if __name__ == "__main__":
    main()
