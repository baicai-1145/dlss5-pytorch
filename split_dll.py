#!/usr/bin/env python3
"""split_dll.py — Split nvngx_dlssnr.dll into analyzable sections.

Splits the 158 MB DLL into:
- text_section.bin    (.text, ~700 KB,  CPU code for IDA/Ghidra)
- rdata_section.bin   (.rdata, ~210 KB, strings / constants)
- data_section.bin    (.data,  ~17 MB,  cubin + runtime data for cuobjdump)
- weights_blob.bin    (.rsrc  WEIGHTS_HT, ~140 MB, neural network weights)

Usage:
    python split_dll.py
"""
import os
import hashlib
import pefile

DLL = '/Users/baicai1145/Downloads/DLSS5Tool/nvngx_dlssnr.dll'
OUT_DIR = '/Users/baicai1145/Downloads/dlss5-pytorch/split'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pe = pefile.PE(DLL)
    dll_sha = sha256(DLL)
    print(f"DLL: {DLL}")
    print(f"SHA256: {dll_sha}\n")

    # Section extraction
    for s in pe.sections:
        name = s.Name.rstrip(b'\0').decode()
        data = s.get_data()
        out = os.path.join(OUT_DIR, f'{name.strip(".")}.bin')
        with open(out, 'wb') as f:
            f.write(data)
        print(f"  {name:<10}  {len(data):>10,} bytes  -> {out}")

    # Weights resource extraction (PE resource WEIGHTS_HT)
    for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        if entry.id != 10:
            continue
        for name_entry in entry.directory.entries:
            name = name_entry.name if name_entry.name else None
            if name and 'WEIGHTS' in str(name):
                for lang_entry in name_entry.directory.entries:
                    data_rsrc = lang_entry.data
                    data_bytes = pe.get_data(lang_entry.data.struct.OffsetToData,
                                              lang_entry.data.struct.Size)
                    out = os.path.join(OUT_DIR, 'weights_blob.bin')
                    with open(out, 'wb') as f:
                        f.write(data_bytes)
                    print(f"  WEIGHTS_HT  {len(data_bytes):>10,} bytes  -> {out}")
                    print(f"             sha256: {sha256(out)}")

    print(f"\nAll files written to: {OUT_DIR}")
    print(f"Total files ready for upload to cloud (Linux RTX 5060 + Windows RTX 5090):")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f:<30}  {sz:>12,} bytes ({sz/1024/1024:>6.2f} MiB)")

if __name__ == '__main__':
    main()