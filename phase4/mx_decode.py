"""DLSS5 blob 最终解码器 (Phase 4).

三段式块布局 (每条记录 payload = [4B 01000000 标签][数据]):
  1. E4M3 权重区   — 纯 E4M3 字节, 零大值 (|v|<16), 直接解码
  2. MX 交错区      — (W:E4M3, S:E8M0) 2B 对, 仅 c=32/64 块存在;
                      S 集中 [196,200] 与 |W| 独立 → per-matrix 量化
                      解码: W * 2^(S - 205)
  3. fp16 尾区      — 大块尾部: ffn2 权重 + bias + LN gamma 全 fp16;
                      开头特征: 16B 零 + gamma 表 (0.86-1.0, XX3b 对)

检测判据 (每 512B 窗):
  - MX 特征:  odd 字节 top1 频次 > 50% (scale 集中)
  - fp16 特征: hi 字节 (odd) ∈ [0x30,0x48] 占比 > 50% 或零字节 > 60%
"""
import struct
import numpy as np

E4M3_BIAS = 205  # S_byte - 205 = log2(scale)


def e4m3_decode(u8: np.ndarray) -> np.ndarray:
    """uint8 → float (E4M3, 0x7F/0xFF NaN→0)."""
    u8 = np.asarray(u8, dtype=np.uint8)
    e = (u8 >> 3) & 0xF
    m = u8 & 7
    sign = np.where(u8 & 0x80, -1.0, 1.0)
    mag = np.where(
        e == 0,
        (m / 16.0) * (2.0 ** -6),
        (1.0 + m / 8.0) * np.power(2.0, e.astype(np.float64) - 7.0),
    )
    mag = np.where((e == 15) & (m == 7), 0.0, mag)
    return sign * mag


def mx_decode(pairs: np.ndarray, bias: int = E4M3_BIAS) -> np.ndarray:
    """(W,S) 交错对 → 真实权重."""
    w = e4m3_decode(pairs[:, 0])
    s = pairs[:, 1].astype(np.float64)
    return w * np.power(2.0, s - bias)


def fp16_decode(raw: np.ndarray) -> np.ndarray:
    v = np.frombuffer(raw[: len(raw) // 2 * 2].tobytes(), dtype='<f2')
    return v.astype(np.float32)


def classify_windows(arr: np.ndarray, win: int = 512):
    """返回每窗标签: 'E4' | 'MX' | 'FP16'."""
    tags = []
    for off in range(0, len(arr) - win + 1, win):
        w = arr[off:off + win]
        odd = w[1::2]
        top1 = np.bincount(odd, minlength=256).max()
        fp16_frac = np.mean(((odd >= 0x30) & (odd <= 0x48)) | ((odd >= 0xB0) & (odd <= 0xC8)))
        zero = np.mean(w == 0)
        top1_byte = int(np.bincount(odd, minlength=256).argmax())
        if zero > 0.6 or fp16_frac > 0.4 or (top1 > win // 4 and 0x30 <= top1_byte <= 0x48):
            tags.append('FP16')
        elif top1 > win // 8:  # 512B→256 odd, >25% 集中且 scale 区
            tags.append('MX')
        else:
            tags.append('E4')
    return tags


def merge_runs(tags):
    runs = []
    for t in tags:
        if runs and runs[-1][0] == t:
            runs[-1][1] += 1
        else:
            runs.append([t, 1])
    return runs


def decode_record(raw: np.ndarray, verbose: bool = False):
    """payload[4:] → (values, zone_report)."""
    arr = np.asarray(raw, dtype=np.uint8)
    if len(arr) < 2048:
        return e4m3_decode(arr), [('SMALL', 0, len(arr))]
    tags = classify_windows(arr)
    runs = merge_runs(tags)
    # 主要 MX/FP16 区定位
    out = []
    report = []
    pos = 0
    for tag, n in runs:
        seg = arr[pos:pos + n * 512]
        if tag == 'MX':
            out.append(mx_decode(seg.reshape(-1, 2)))
        elif tag == 'FP16':
            out.append(fp16_decode(seg))
        else:
            out.append(e4m3_decode(seg))
        report.append((tag, pos, pos + n * 512))
        pos += n * 512
    if pos < len(arr):
        out.append(e4m3_decode(arr[pos:]))
        report.append(('TAIL', pos, len(arr)))
    vals = np.concatenate([o for o in out if len(o)])
    if verbose:
        for tag, a, b in report:
            print(f'  {tag:<5} [{a:>8}:{b:>8}]')
    return vals, report


def walk_records(blob: bytes):
    """走链: [magic16]{[name][u64 A][u64 A][u64 B][B payload][28B term][4B pad]}*153."""
    pos = 16
    nxt_len = 19  # b0 name len
    while pos + 28 < len(blob):
        name = blob[pos:pos + nxt_len].decode('utf8', 'replace')
        if pos + nxt_len + 24 > len(blob):
            break
        A1, A2, B = struct.unpack_from('<QQQ', blob, pos + nxt_len)
        payload_off = pos + nxt_len + 24
        yield name, B, payload_off
        p = payload_off + B
        if p + 28 > len(blob):
            break
        if p + 28 > len(blob):
            break
        nxt_len = struct.unpack_from('<7I', blob, p)[6]
        pos = p + 32


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'weights_blob.bin'
    blob = open(path, 'rb').read()
    for name, B, off in walk_records(blob):
        raw = np.frombuffer(blob[off + 4:off + B], dtype=np.uint8)
        vals, report = decode_record(raw, verbose=(B > 100000))
        std = vals.std()
        flag = ' ←!' if std > 1.0 else ''
        print(f'{name:<28} B={B:>9,} vals={len(vals):>9,} std={std:8.4f}{flag}')
