"""Phase 5 语义装填器: 三段解码 + 记录→参数逐块装填 + 前向验证.

策略 (SwinBlock 单记录块, c=32/64/128/256):
  记录流 = [E4M3 区][MX 区][E4M3 区][fp16 misc]
  → qkv+proj (E4 前段) + mlp0/mlp2 (E4 中后段+MX) + norm/bias (misc fp16)
c=512 SplitSwin 块 (4 子记录) 与瓶颈 (4 子记录) 按专属布局.
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
    if len(raw) < 2:
        return np.zeros(0, dtype=np.float32)
    v = np.frombuffer(raw[: len(raw) // 2 * 2].tobytes(), dtype='<f2')
    return v.astype(np.float32)


def classify_windows(arr, win=512):
    tags = []
    for off in range(0, max(len(arr) - win + 1, 1), win):
        w = arr[off:off + win]
        odd = w[1::2]
        bc = np.bincount(odd, minlength=256)
        top1 = int(bc.max()); top1_byte = int(bc.argmax())
        fp_pos = ((odd >= 0x30) & (odd <= 0x48)).mean()
        fp_neg = ((odd >= 0xB0) & (odd <= 0xC8)).mean()
        zero = np.mean(w == 0)
        if zero > 0.6 or fp_pos > 0.4:
            tags.append('F')
        elif top1 > len(odd) * 0.5 and 176 <= top1_byte <= 210:
            tags.append('M')   # MX: odd top1 >50% 且 scale 区
        elif fp_neg > 0.4 and top1 < len(odd) * 0.4:
            tags.append('F')   # fp16 负小值但无强集中
        elif top1 > win // 8 and 0x30 <= top1_byte <= 0x48:
            tags.append('F')
        else:
            tags.append('E')
    return tags


# 已测绘硬边界 (本地 Phase 4 实测): {块字节长 → (MX 区起, MX 区止)}
MX_BOUND = {20672: (11264, 19456), 61760: (40960, 57344)}
# fp16 尾区起点 (大块): {字节长 → 尾区起}
FP16_TAIL = {197184: 98304, 689232: 360448, 229936: 147968+7968, 524288: None}


def decode_record_raw(raw, total_B=None):
    """三段解码: 返回 (main, misc). MX/fp16 边界优先用硬编码表."""
    arr = np.asarray(raw, dtype=np.uint8)
    if len(arr) < 512:
        return e4m3_decode(arr), fp16_decode(arr)
    if total_B in MX_BOUND:
        lo, hi = MX_BOUND[total_B]
        main = np.concatenate([e4m3_decode(arr[:lo]),
                               mx_decode_pairs(arr[lo:hi].reshape(-1, 2)),
                               e4m3_decode(arr[hi:])])
        return main, np.zeros(0, dtype=np.float32)
    if total_B in FP16_TAIL and FP16_TAIL[total_B]:
        t0 = FP16_TAIL[total_B]
        return e4m3_decode(arr[:t0]), fp16_decode(arr[t0:])
    tags = classify_windows(arr)
    runs = []
    for t in tags:
        if runs and runs[-1][0] == t:
            runs[-1][1] += 1
        else:
            runs.append([t, 1])
    main_parts, misc_parts = [], []
    pos = 0
    for tag, n in runs:
        seg = arr[pos:pos + n * 512]
        if tag == 'E':
            main_parts.append(e4m3_decode(seg))
        elif tag == 'M':
            main_parts.append(mx_decode_pairs(seg[:len(seg) // 2 * 2].reshape(-1, 2)))
        else:
            misc_parts.append(fp16_decode(seg))
        pos += n * 512
    if pos < len(arr):
        main_parts.append(e4m3_decode(arr[pos:]))
    main = np.concatenate([p for p in main_parts if len(p)]) if main_parts else np.zeros(0)
    misc = np.concatenate([p for p in misc_parts if len(p)]) if misc_parts else np.zeros(0, dtype=np.float32)
    return main, misc


def walk_records(blob):
    pos = 16; nxt_len = 19
    while pos + 28 < len(blob):
        name = blob[pos:pos + nxt_len].decode('utf8', 'replace')
        if pos + nxt_len + 24 > len(blob):
            break
        A1, A2, B = struct.unpack_from('<QQQ', blob, pos + nxt_len)
        off = pos + nxt_len + 24
        yield name, B, off
        p = off + B
        if p + 28 > len(blob):
            break
        nxt_len = struct.unpack_from('<7I', blob, p)[6]
        pos = p + 32


def load_all():
    blob = open(BLOB, 'rb').read()
    by_block = {}
    for name, B, off in walk_records(blob):
        b = int(name.split('.')[0][5:])
        layer = name.split('.')[1]  # layer0/2/3 或 blend_scale
        raw = blob[off + 4:off + B] if B > 4 else blob[off:off + B]
        main, misc = decode_record_raw(np.frombuffer(raw, dtype=np.uint8), total_B=B)
        by_block.setdefault(b, {})[layer] = (main, misc)
    return by_block


def fill_model(model, by_block, verbose=False):
    """按块角色把解码流装进 model 参数. 返回未填充参数统计."""
    import torch
    pmap = dict(model.named_parameters())
    unfilled = set(pmap)
    stats = {'filled': 0, 'pad': 0}

    def put(pname, vals, count=None):
        """顺序消费 vals 填 pname."""
        p = pmap[pname]
        n = p.numel()
        v = vals[:n] if count is None else vals[:count]
        if len(v) < n:
            v = np.concatenate([v, np.zeros(n - len(v), v.dtype)])
        with torch.no_grad():
            p.copy_(torch.from_numpy(v.astype(np.float32)).reshape(p.shape).to(p.dtype))
        unfilled.discard(pname)
        stats['filled'] += n
        return n

    roles = {int(k): v for k, v in json.load(
        open(os.path.join(HERE, '..', 'phase1', 'block_roles.json'))).items()}

    # —— SwinBlock 单记录块: enc0(b1-3), merge(b4,8,14), enc1(b5-7), enc2(b9-13),
    #    enc3(b15-21), split_entry(b22), dec 各层, up(b48,56,62,66), tail(b70) ——
    # 简化通用序: 每块 main 流 = [qkv | proj | mlp0 | mlp2] 权重按序, misc = norm/bias
    def fill_swin(prefix, main, misc):
        q = 0
        for suffix, shape in (
            ('attn.qkv.weight', None), ('attn.proj.weight', None),
            ('mlp.0.weight', None), ('mlp.2.weight', None),
        ):
            pn = f'{prefix}.{suffix}'
            if pn in pmap:
                q += put(pn, main[q:])
        # norms/biases 从 misc (gamma 段 [112:176]-类) + 剩余 main
        tail = np.concatenate([main[q:], misc]) if len(misc) else main[q:]
        o = 0
        for suffix in ('norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias',
                       'mlp.0.bias', 'mlp.2.bias', 'attn.qkv.bias', 'attn.proj.bias'):
            pn = f'{prefix}.{suffix}'
            if pn in pmap and pn in unfilled:
                o += put(pn, tail[o:])

    # enc/dec SwinStage 内块序
    stage_blocks = {
        'enc': [[1, 2, 3], [5, 6, 7], [9, 10, 11, 12, 13], [15, 16, 17, 18, 19, 20, 21], [23, 24, 25, 26, 27, 28, 29]],
        'dec': [[40, 41, 42, 43, 44, 45, 46, 47], [49, 50, 51, 52, 53, 54, 55], [57, 58, 59, 60, 61], [63, 64, 65], [67, 68, 69]],
    }
    # 注意 c=512 enc4/dec4 块是多子记录 (layer0-3) — 这里用 layer2 主流+layer1/0/3 补
    for net in ('enc', 'dec'):
        for si, blocks in enumerate(stage_blocks[net]):
            for bi, b in enumerate(blocks):
                if b not in by_block:
                    continue
                layers = by_block[b]
                if len(layers) > 1:  # SplitSwin 4 子记录
                    # layer2: qkv(E4前段)+mlp1(MX), layer1: proj+norm, layer0: mlp2, layer3: 分支
                    m2, _ = layers.get('layer2', (np.zeros(0), np.zeros(0)))
                    m1, s1 = layers.get('layer1', (np.zeros(0), np.zeros(0)))
                    m0, _ = layers.get('layer0', (np.zeros(0), np.zeros(0)))
                    m3, s3 = layers.get('layer3', (np.zeros(0), np.zeros(0)))
                    prefix = f'{net}.{si}.blocks.{bi}'
                    q = 0
                    for pn_suffix, src in (('attn.qkv.weight', m2), ('attn.proj.weight', m1),
                                           ('mlp.0.weight', m2), ('mlp.2.weight', m0)):
                        pn = f'{prefix}.{pn_suffix}'
                        if pn in pmap and pn in unfilled:
                            q2 = put(pn, src) if src is not None else 0
                    misc_all = np.concatenate([x for x in (s1, s3) if len(x)]) if (len(s1) or len(s3)) else np.zeros(0)
                    for suffix in ('norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias'):
                        pn = f'{prefix}.{suffix}'
                        if pn in pmap and pn in unfilled and len(misc_all):
                            put(pn, misc_all)
                else:
                    main, misc = layers['layer0']
                    fill_swin(f'{net}.{si}.blocks.{bi}', main, misc)

    # —— stem (b0): conv3x3 32×9×3×3=2592 → main 前 2592 ——
    if 0 in by_block:
        main0, misc0 = by_block[0]['layer0']
        put('stem.weight', main0)
        if len(misc0): put('stem.bias', misc0)
    # —— merges (b4,8,14,22): norm + reduction conv ——
    for mi, b in enumerate((4, 8, 14, 22)):
        if b in by_block:
            main, misc = by_block[b]['layer0']
            put(f'merges.{mi}.reduction.weight', main)
            put(f'merges.{mi}.norm.weight', misc) if len(misc) else None
            put(f'merges.{mi}.norm.bias', misc) if len(misc) else None
    # —— expands (b48,56,62,66) ——
    for ei, b in enumerate((48, 56, 62, 66)):
        if b in by_block:
            main, misc = by_block[b]['layer0']
            put(f'expands.{ei}.expand.weight', main) if f'expands.{ei}.expand.weight' in pmap else None
            for pn_tail in ('norm.weight','norm.bias'):
                pn=f'expands.{ei}.{pn_tail}'
                if pn in pmap and len(misc): put(pn, misc)
    # —— bottleneck (b31-38): _SplitBlock wqkv/proj/side/ffwd ← layer0/1/2 ——
    for bi, b in enumerate(range(31, 39)):
        if b in by_block:
            layers = by_block[b]
            m0,_ = layers.get('layer0',(np.zeros(0),np.zeros(0)))
            m1,_ = layers.get('layer1',(np.zeros(0),np.zeros(0)))
            m2,_ = layers.get('layer2',(np.zeros(0),np.zeros(0)))
            for suf,src in (('wqkv.weight',m0),('proj.weight',m1),('side.weight',m2),('ffwd.weight',m2)):
                pn=f'bn.{bi}.{suf}'
                if pn in pmap and pn in unfilled: put(pn, src)
    # —— bn_proj / tail (b39? b70) ——
    if 39 in by_block:
        main, misc = by_block[39]['layer0']
        put('bn_proj.weight', main) if 'bn_proj.weight' in pmap else None
    if 70 in by_block:
        layers70 = by_block[70]
        main, misc = layers70.get('layer0',(np.zeros(0),np.zeros(0)))
        bs = layers70.get('blend_scale',(np.zeros(0,dtype=np.float32),np.zeros(0,dtype=np.float32)))[0]
        allv = np.concatenate([main, misc, bs]) if len(bs) else np.concatenate([main,misc]) if len(misc) else main
        seq = np.concatenate([allv, np.zeros(64, np.float32)])
        put('tail.global_fc.weight', seq)            # 1024
        put('tail.global_fc.bias', seq[1024:])       # 32
        put('tail.conv.weight', seq[1056:])          # 864
        put('tail.conv.bias', seq[1920:])            # 3
        put('tail.blend', seq[1923:1924]) if 'tail.blend' in pmap else None

    # —— 兜底: 剩余 norm/bias/gate 用 LN 默认值 (gamma=1, beta/bias=0) ——
    import torch as _t
    for pn in list(unfilled):
        if '_pad' in pn or pn.endswith('.pad') or pn in ('calib_pad', 'stem_pad'):
            unfilled.discard(pn)
            continue
        if 'relative_position_bias_table' in pn:
            with _t.no_grad(): pmap[pn].zero_()
            unfilled.discard(pn); continue
        if 'norm' in pn or pn.endswith(('.bias', '.gate')):
            p = pmap[pn]
            with _t.no_grad():
                p.fill_(1.0 if ('norm' in pn and 'weight' in pn) else 0.0)
            unfilled.discard(pn)
    return unfilled, stats


def main():
    import torch
    from dlss5.calib_model import DLSS5NetCalib
    by_block = load_all()
    print(f'loaded {len(by_block)} blocks')
    m = DLSS5NetCalib().eval()
    unfilled, stats = fill_model(m, by_block)
    print(f"filled params: {stats['filled']:,}; unfilled tensors: {len(unfilled)}/{len(list(m.named_parameters()))}")
    # 前向 (CPU 小尺寸)
    with torch.no_grad():
        out = m(torch.rand(1, 3, 64, 64), torch.rand(1, 1, 64, 64),
                torch.rand(1, 2, 64, 64), torch.rand(1, 3, 64, 64))
    print(f'CPU fwd: {tuple(out.shape)} NaN={torch.isnan(out).sum().item()} absmean={out.abs().mean():.4f}')
    stats={}
    def hook(nm):
        def h(mod,i,o):
            a=o.detach().float()
            stats[nm]=(a.abs().mean().item(),a.abs().max().item(),torch.isnan(a).sum().item())
        return h
    m.stem.register_forward_hook(hook('stem'))
    for i,st in enumerate(m.enc): st.register_forward_hook(hook(f'enc{i}'))
    for i,blk in enumerate(m.bn): blk.register_forward_hook(hook(f'bn{i}'))
    m.bn_proj.register_forward_hook(hook('bn_proj'))
    for i in range(5): m.dec[i].register_forward_hook(hook(f'dec{i}'))
    m.tail.register_forward_hook(hook('tail'))
    with torch.no_grad():
        m(torch.rand(1,3,64,64),torch.rand(1,1,64,64),torch.rand(1,2,64,64),torch.rand(1,3,64,64))
    print('stage absmean/absmax/NaN:')
    for k,(mn,mx,nn) in stats.items():
        flag='' if (1e-3<mn<1e4 and nn==0) else '  <<< CHECK'
        print(f'  {k:9s} {mn:12.4e} {mx:12.4e} NaN={nn}{flag}')


if __name__ == '__main__':
    main()
