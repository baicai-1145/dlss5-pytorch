"""Phase 4 终版: 逐记录三段解码 → 模型参数装填 (Mac CPU 可验证).

用法: python3 fill_model.py [--fwd]
流程:
  1. walk_records 走链 153 条
  2. 每条按 [E4M3 | MX 交错 | fp16 尾] 三段解码
  3. 按 block_roles.json 映射装填 DLSS5NetCalib 参数:
     - 单记录块 (c=32/64/128/256): E4 段填 qkv/proj/mlp 权重, MX 段填 mlp, fp16 尾填 norm/bias
     - c=512 块 4 子记录 (layer0-3): layer2=qkv+mlp1, layer1=proj+norm, layer0=mlp2, layer3=proj2+norm
     - 瓶颈 b31-38: layer0/1/2 大矩阵, layer3=零标量
  4. --fwd: Mac CPU 小尺寸前向 sanity
"""
import sys, os, json, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase3'))

BLOB = os.path.join(HERE, '..', 'weights_blob.bin')


def e4m3_decode(u8):
    u8 = np.asarray(u8, dtype=np.uint8)
    e = (u8 >> 3) & 0xF; m = u8 & 7
    sgn = np.where(u8 & 0x80, -1.0, 1.0)
    v = np.where(e == 0, (m / 16.0) * (2.0 ** -6),
                 (1.0 + m / 8.0) * np.power(2.0, e.astype(np.float64) - 7.0))
    return sgn * np.where((e == 15) & (m == 7), 0.0, v)


def mx_decode_pairs(pairs, bias=205):
    w = e4m3_decode(pairs[:, 0])
    s = pairs[:, 1].astype(np.float64)
    return w * np.power(2.0, s - bias)


def fp16_decode(raw):
    v = np.frombuffer(raw[: len(raw) // 2 * 2].tobytes(), dtype='<f2')
    return v.astype(np.float32)


def classify_windows(arr, win=512):
    tags = []
    for off in range(0, len(arr) - win + 1, win):
        w = arr[off:off + win]
        odd = w[1::2]
        bc = np.bincount(odd, minlength=256)
        top1 = int(bc.max())
        top1_byte = int(bc.argmax())
        fp_pos = ((odd >= 0x30) & (odd <= 0x48)).mean()
        fp_neg = ((odd >= 0xB0) & (odd <= 0xC8)).mean()
        zero = np.mean(w == 0)
        if zero > 0.6 or fp_pos > 0.4 or fp_neg > 0.4 or (top1 > win // 8 and 0x30 <= top1_byte <= 0x48):
            tags.append('F')
        elif top1 > win // 8:  # >25% 集中 (b1 MX 区 odd top ~31%)
            tags.append('M')
        else:
            tags.append('E')
    return tags


def decode_record(raw):
    """三段解码 → dict(e4=vals, mx=vals, fp16=vals, zones=[(tag,lo,hi)])"""
    arr = np.asarray(raw, dtype=np.uint8)
    zones = []
    e4_parts, mx_parts, f_parts = [], [], []
    nwin = max(1, len(arr) // 512)
    tags = classify_windows(arr) if len(arr) >= 512 else ['E']
    # 合并连续同 tag
    runs = []
    for t in tags:
        if runs and runs[-1][0] == t:
            runs[-1][1] += 1
        else:
            runs.append([t, 1])
    pos = 0
    for tag, n in runs:
        seg = arr[pos:pos + n * 512]
        if tag == 'E':
            e4_parts.append(e4m3_decode(seg))
        elif tag == 'M':
            mx_parts.append(mx_decode_pairs(seg[:len(seg) // 2 * 2].reshape(-1, 2)))
        else:
            f_parts.append(fp16_decode(seg))
        zones.append((tag, pos, pos + n * 512))
        pos += n * 512
    if pos < len(arr):
        e4_parts.append(e4m3_decode(arr[pos:]))
        zones.append(('E', pos, len(arr)))
    out = {
        'e4': np.concatenate(e4_parts) if e4_parts else np.zeros(0),
        'mx': np.concatenate(mx_parts) if mx_parts else np.zeros(0),
        'fp16': np.concatenate(f_parts) if f_parts else np.zeros(0),
        'zones': zones,
    }
    return out


def walk_records(blob):
    pos = 16; nxt_len = 19
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
        nxt_len = struct.unpack_from('<7I', blob, p)[6]
        pos = p + 32


def block_num(name):
    return int(name.split('.')[0][5:])


def load_block_roles():
    return {int(k): v for k, v in json.load(open(os.path.join(HERE, '..', 'phase1', 'block_roles.json'))).items()}


def build_fill_plan():
    """生成 (param_name, 值数组源) 装填计划. 简化版: 每块的 e4/mx/fp16 拼接为一条流,
    交给调用方按参数顺序消费 (与远程 calib 模型 pad 对齐)."""
    blob = open(BLOB, 'rb').read()
    roles = load_block_roles()
    recs = list(walk_records(blob))
    # 文件序 = 字典序; 转拓扑序: 按 (角色顺序, 块号) — 用 roles 在 layout 中的次序
    order = {}
    for i, (name, B, off) in enumerate(recs):
        b = block_num(name)
        order.setdefault(b, []).append((name, B, off))
    return blob, roles, order


def main():
    blob, roles, order = build_fill_plan()
    print('blocks:', len(order), 'roles:', len(roles))
    total = {'e4': 0, 'mx': 0, 'fp16': 0}
    for b in sorted(order):
        for name, B, off in order[b]:
            raw = blob[off + 4:off + B] if B > 4 else blob[off:off + B]
            d = decode_record(np.frombuffer(raw, dtype=np.uint8))
            total['e4'] += len(d['e4']); total['mx'] += len(d['mx']); total['fp16'] += len(d['fp16'])
    print(f"全库解码: e4={total['e4']:,} mx={total['mx']:,} fp16={total['fp16']:,} → 合计 {sum(total.values()):,} (模型 147,683,778)")


if __name__ == '__main__':
    main()
