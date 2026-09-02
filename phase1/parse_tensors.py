#!/usr/bin/env python3
"""parse_tensors.py — DLSS5 weights_blob.bin 记录结构解析器 (Phase 1 收尾).

已验证的 blob 格式 (链式验证 152/153 吻合, 唯一例外是文件尾部截断 8 字节):

    u64 magic   = 0x0000000008cda732
    u64 count   = 19            (语义未知 — 实际记录数是 153, 可能是顶层分组数)
    153 × record:
        cstring  name            "block{N}.layer{M}.layer" | "block70.layer0.blend_scale"
        u64      A               == B + 40   (B字段8B + meta32 32B + payload B)
        u64      A2              == A (重复)
        u64      B               payload 字节数
        byte[32] meta32          标量记录: 结构化(dtype/shape 编码); 大张量: E4M3 权重数据
        byte[B]  payload         权重数据 (E4M3 FP8, 待 Phase 4 精解)

对账: 16 + Σ(len(name)+24+32+B) == filesize + 8  (最后一条记录 B 超 EOF 8 字节,
      即 weights_blob.bin 提取时被截断 8 字节, 或末记录 B 含 8 字节 padding)

用法:
    python3 parse_tensors.py [blob_path] [out_json]
"""
import json
import mmap
import re
import struct
import sys

BLOB = sys.argv[1] if len(sys.argv) > 1 else '/Users/baicai1145/dlss5-pytorch/weights_blob.bin'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/Users/baicai1145/dlss5-pytorch/phase1/tensor_metadata.json'

NAME_RE = re.compile(rb'(block\d+\.layer\d+\.(?:layer|blend_scale))')


def parse(blob_path):
    f = open(blob_path, 'rb')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    size = len(m)

    magic, count = struct.unpack_from('<QQ', m, 0)
    assert magic == 0x08cda732, hex(magic)

    # 收集全部 anchor
    anchors = []
    pos = 16
    while True:
        i = m.find(b'block', pos)
        if i == -1:
            break
        mm = NAME_RE.match(m[i:i + 64])
        if mm:
            anchors.append(i)
            pos = i + len(mm.group(1))
        else:
            pos = i + 5

    records = []
    chain_ok = 0
    for idx, i in enumerate(anchors):
        mm = NAME_RE.match(m[i:i + 64])
        name = mm.group(1).decode()
        e = i + len(mm.group(1))
        A, A2, B = struct.unpack_from('<QQQ', m, e)
        assert A == A2 == B + 40, (name, A, A2, B)
        meta32 = m[e + 24:e + 56]
        payload_off = e + 56
        payload_end = payload_off + B
        nxt = anchors[idx + 1] if idx + 1 < len(anchors) else size
        if payload_end == nxt:
            chain_ok += 1
        records.append({
            'name': name,
            'A': A,
            'nbytes': B,
            'meta32_hex': meta32.hex(),
            'dtype_tag': struct.unpack_from('<I', meta32, 0)[0],
            'payload_off': payload_off,
            'chain_ok': payload_end == nxt,
        })

    total = 16 + sum(len(r['name']) + 24 + 32 + r['nbytes'] for r in records)
    return {
        'magic': hex(magic),
        'header_count_field': count,
        'num_records': len(records),
        'chain_ok': chain_ok,
        'accounting': {
            'computed_total': total,
            'file_size': size,
            'delta': total - size,  # +8 = 末记录被截断 8 字节
        },
        'records': records,
    }


def main():
    result = parse(BLOB)
    with open(OUT, 'w') as f:
        json.dump(result, f, indent=1)
    print(f"magic={result['magic']} count_field={result['header_count_field']} "
          f"records={result['num_records']} chain_ok={result['chain_ok']}")
    print(f"accounting: {result['accounting']}")
    print(f"-> {OUT}")


if __name__ == '__main__':
    main()
