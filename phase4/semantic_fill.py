"""Phase 5 语义装填器: 三段解码 + 记录→参数逐块装填 + 前向验证.

策略 (SwinBlock 单记录块, c=32/64/128/256):
  记录流 = [E4M3 区][MX 区][E4M3 区][fp16 misc]
  → qkv+proj (E4 前段) + mlp0/mlp2 (E4 中后段+MX) + norm/bias (misc fp16)
c=512 SplitSwin 块 (4 子记录) 与瓶颈 (4 子记录) 按专属布局.
"""
import sys, os, json, struct, math
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


def mx_decode_pairs(pairs, bias=None):
    """MX 解码: v = W × 2^(S - median(S) - 8) — per-matrix 自适应.
    median(S) = 该矩阵量化零点 (c32≈198, c512: 201-212 每块不同);
    -8 归一 E4M3 满幅到权重典型幅度. c32 块等价于旧 bias≈205."""
    w = e4m3_decode(pairs[:, 0])
    s = pairs[:, 1].astype(np.float64)
    return w * np.power(2.0, s - np.median(s) - 8.0)


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
# c32: [E4 0:11264][MX 11264:19456][E4 19456:20480][misc fp16 20480:]; c64 同构尾 320B+
# merge b4: MX [11264:19712] 尾E4 [19712:20480] misc 2208B; b8: MX [40960:57600] 尾E4 [57600:61440] misc 8464B
MX_BOUND = {20672: (11264, 19456), 61760: (40960, 57344),
            22720: (11264, 19712), 69936: (40960, 57600)}
MISC_TAIL = {20672: 192, 61760: 320, 22720: 2208, 69936: 8464}
MX_GAP = {20672: 112, 61760: 1024, 22720: 0, 69936: 0}
E4M3_HOLE = {20672: [(8192, 8448)], 61760: [(28672, 28928)],
             # merge 块 (B 独有): 同构布局
             22720: [(8192, 8448)], 69936: [(28672, 28928)]}
# c=512 SplitSwin layer2 (917,568B): MX 尾区 [786432:917560]
MX_BOUND[917568] = (786432, 917504)  # 尾 60B 是杂讯, 65536 对精确
# fp16 尾区起点 (大块): {字节长 → 尾区起}
FP16_TAIL = {197184: 98304, 689232: 360448, 229936: 147968+7968, 524288: None}

# Phase 6.5 (Task 7a) — b14 (merges.2) misc 修复:
# 旧边界 FP16_TAIL[229936]=155936 把 b14 尾部的 MX scale 因子表 (前 512 个
# fp16 值, mean=-8.78, std=2.21) 误当成 LN gamma 装进 merges.2.norm.{weight,bias},
# 造成 enc3 段 14× 幅度跳变 (enc2 std 1.55 → enc3 21.7)。
# 证据链: phase4/enc3_probe.py + ENC3_PROBE_REPORT.md。
# 修复: b14 的 misc 起点后移到 MX scale 表之后 (155936 + 1024 字节 = 512 fp16 值),
# 并在装填侧对 merges.2.norm 用健康带 1.0 兑底 (blob 里未找到该 merge 的
# 真实 gamma 区段, 兑底值与 LN canonical init 一致)。
B14_MXSCALE_BYTES = 1024   # 512 fp16 vals = MX scale 因子表 (跳过, 不进 misc)
FP16_TAIL[229936] = 155936 + B14_MXSCALE_BYTES


def decode_record_raw(raw, total_B=None):
    """三段解码: 返回 (main, misc). MX/fp16 边界优先用硬编码表."""
    arr = np.asarray(raw, dtype=np.uint8)
    if len(arr) < 512:
        return e4m3_decode(arr), fp16_decode(arr)
    if total_B in MX_BOUND:
        lo, hi = MX_BOUND[total_B]
        misc_start = len(arr) - MISC_TAIL.get(total_B, 0)
        gap = MX_GAP.get(total_B, 0)   # MX→E4 过渡带 (混合字节, 跳过)
        holes = E4M3_HOLE.get(total_B, [])
        # E4 前段 (剥内嵌 fp16 段)
        segs, hole_vals = [], []
        pos = 0
        for hlo, hhi in sorted(holes):
            if hlo > pos: segs.append(arr[pos:hlo])
            hole_vals.append(arr[hlo:hhi]); pos = hhi
        if pos < lo: segs.append(arr[pos:lo])
        pre = np.concatenate([e4m3_decode(x) for x in segs]) if segs else np.zeros(0, np.float32)
        main = np.concatenate([pre,
                               mx_decode_pairs(arr[lo:hi].reshape(-1, 2)),
                               e4m3_decode(arr[hi + gap:misc_start])])
        misc = (np.concatenate([fp16_decode(x) for x in hole_vals] +
                                ([fp16_decode(arr[misc_start:])] if misc_start < len(arr) else []))
                if (hole_vals or misc_start < len(arr)) else np.zeros(0, np.float32))
        return main, misc
    if total_B in FP16_TAIL and FP16_TAIL[total_B]:
        t0 = FP16_TAIL[total_B]
        if total_B == 229936:  # b14/merge2: E4[0:98304]+hole+MX 混布; reduction 只用前 131072
            return (np.concatenate([e4m3_decode(arr[:98304]),
                                    e4m3_decode(arr[98816:160768])]),
                    fp16_decode(arr[t0:]))
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


def fill_model(model, by_block, verbose=False, blob_full=None):
    if blob_full is None:
        blob_full = open(BLOB, 'rb').read()
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
        # 权重流 = main (E4M3+MX 区); misc (fp16) = norm/bias 专用, 不混入权重
        # c128/256 大块: main 只有 qkv+proj+ffn1, ffn2 在 fp16 misc → 需拼接但只到 mlp2 为止
        # 用 misc 作为 mlp2 的补充源: stream = main + misc(权重部分)
        # misc 布局 (b9): [16B零][gamma 128][小bias 24k][大值区 ffn2 权重] — 权重在大值区!
        # misc = fp16 结构区: 小bias + gamma + 大负值区(疑似 per-tensor scaled ffn2)
        # 权重流 = main; mlp2 若 main 不够, 用 misc 归一化段补 (std→0.02)
        if len(misc) > 40000:
            # c128 misc (字节): [0:512]gamma | [512:49152]小值 | [49152:82432]大值区 | [82432:] U[0,1]尾
            # ffn2 权重流 = U[0,1] 尾 + 小值区 (跳过大值区)
            misc_w = np.concatenate([misc[len(misc)-8208:], misc[256:24320]])
        else:
            misc_w = misc
        stream = np.concatenate([main, misc_w]) if len(misc_w) else main
        q = 0
        mlp2_pn = f'{prefix}.mlp.2.weight'
        q_before_mlp2 = None
        for suffix, shape in (
            ('attn.qkv.weight', None), ('attn.proj.weight', None),
            ('mlp.0.weight', None), ('mlp.2.weight', None),
        ):
            pn = f'{prefix}.{suffix}'
            if pn in pmap:
                if pn == mlp2_pn:
                    q_before_mlp2 = q
                q += put(pn, stream[q:])
        # Phase 6.5 (Task 7a): c=256 块 (enc3 b15-21) 的 stream 只有 392,720 vals,
        # 而 mlp.0+mlp.2 共需 196,608 — mlp.2 会拿到 66k 零填充 (病态:
        # 67% 零 + 少量 hot row, enc3_probe.py 实测)。此处仅对 enc3 检测
        # mlp.2 是否实际拿到全量数据; 若没拿到, 用 Kaiming (fan_in=hidden)
        # 兑底, 保证 FFN 第二层幅度健康。其他 stage 行为不变 (最小改动)。
        if (prefix.startswith('enc.3.') and mlp2_pn in pmap
                and q_before_mlp2 is not None
                and len(stream) < q_before_mlp2 + pmap[mlp2_pn].numel()):
            p = pmap[mlp2_pn]
            fan_in = p.shape[1]
            with torch.no_grad():
                p.normal_(0.0, math.sqrt(1.0 / fan_in))
        tail = stream[q:]
        o = 0
        for suffix in ('norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias',
                       'mlp.0.bias', 'mlp.2.bias', 'attn.qkv.bias', 'attn.proj.bias'):
            pn = f'{prefix}.{suffix}'
            if pn in pmap and pn in unfilled:
                # Phase 6.5 (Task 7a): enc3 (b15-21) 的 stream 在权重段就耗尽,
                # tail 为空 — 旧行为把 norm gamma 零填充并把块变成近恒等映射
                # (enc3_probe.py 实测)。此处仅对 enc3 跳过零装填, 留在 unfilled
                # 集合里让末尾兑底逻辑填 LN canonical 值 (gamma=1, beta=0)。
                if prefix.startswith('enc.3.') and o >= len(tail):
                    continue
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
                    # layer2 = [qkv E4 786432][MX 65536 值]; layer0 = mlp E4 524288
                    put(f'{prefix}.attn.qkv.weight', m2)          # 前 786432 = 3×512² 精确, put 只取所需
                    put(f'{prefix}.attn.proj.weight', m1)         # 262144 = 512²
                    # 数据顺序对齐模型参数序: qkv←layer2前段, proj←layer1, mlp←layer0+layer2MX段+layer3
                    mlp_stream = np.concatenate([m0, m2[786432:], m3[:262144]])
                    put(f'{prefix}.mlp.0.weight', mlp_stream)
                    put(f'{prefix}.mlp.2.weight', mlp_stream[456704:])
                    put(f'{prefix}.mlp.0.bias', m1[262144:] if len(m1)>262144 else np.zeros(0)) if False else None
                    misc_all = np.concatenate([x for x in (s1, s3) if len(x)]) if (len(s1) or len(s3)) else np.zeros(0)
                    for suffix in ('norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias'):
                        pn = f'{prefix}.{suffix}'
                        if pn in pmap and pn in unfilled and len(misc_all):
                            put(pn, misc_all)   # 原重叠消费: 4 个 norm 共用 misc 前 512 elem
                    # mlp.2.bias (仅当 ffn2_bias=1, 如 dec0 搜索结果): 从 misc 后面 / m3 剩余取
                    pn = f'{prefix}.mlp.2.bias'
                    if pn in pmap and pn in unfilled:
                        bias_src = misc_all[512:1024] if len(misc_all) >= 1024 else misc_all
                        if len(bias_src) >= 512:
                            put(pn, bias_src[:512])
                        # else: 留默认零 (Kaiming init bias=0), 与兜底分支一致
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
            # Phase 6.5 (Task 7a): merges.2 的 misc 现在已剥去 MX scale 表
            # (见 B14_MXSCALE_BYTES), 但 blob 里仍未定位到真实 LN gamma 区段,
            # 故按 enc3_probe.py 结论用健康带 1.0 兑底 (gamma=1, beta=0),
            # 避免任何异常幅度进入 LN affine。
            if mi == 2:
                with torch.no_grad():
                    pmap[f'merges.2.norm.weight'].fill_(1.0)
                    pmap[f'merges.2.norm.bias'].zero_()
                unfilled.discard(f'merges.2.norm.weight')
                unfilled.discard(f'merges.2.norm.bias')
                stats['filled'] += pmap['merges.2.norm.weight'].numel() * 2
            else:
                put(f'merges.{mi}.norm.weight', misc) if len(misc) else None
                put(f'merges.{mi}.norm.bias', misc) if len(misc) else None
    # —— expands (b48,56,62,66) — Phase 6.6 重测繪 (up = swin + 2c^2 fuse + γc bias):
    #    [swin payload 含自身 fp16 misc][fuse E4 2c^2][fuse bias fp16][junk]
    #    b48: swin E4[0:360448] misc fp16[491520:820224] fuse E4[360448:491520]
    #         (E4 std=0.0418 健康) fuse bias[820224:820736] 256 vals (mean +0.78)
    #    b56: swin E4[0:98304] fuse E4[98304:131072] (std=0.0540)
    #         misc fp16[131072:229888] fuse bias[229888:230112] 112 vals
    #    b62: swin E4[0:40960] fuse E4[66048:69632] 仅 3584/8192 vals (不完整)
    #         misc fp16[49152:66048] fuse bias[69632:69728] 48 vals (≈零)
    #    b66: swin E4[0:10240] fuse E4[11264:13312] (std=0.177) 2048 vals
    #         misc fp16[13312:21504] fuse bias[22716:22780] 32 vals (mean +0.72)
    #    fuse 输入序 [up(c) | skip(c)] — 合并序镜像 (b22 尾矩阵 256→512 同构)
    # ——
    UP_ZONES = {
        48: dict(misc=(491520, 820224), fuse=(360448, 491520), bias=(820224, 820736), c=256),
        56: dict(misc=(131072, 229888), fuse=(98304, 131072), bias=(229888, 230112), c=128),
        62: dict(misc=(49152, 66048), fuse=(66048, 69632), bias=(69632, 69728), c=64),
        66: dict(misc=(13312, 21504), fuse=(11264, 13312), bias=(22716, 22780), c=32),
    }
    _UPB = {48: 0, 56: 1, 62: 2, 66: 3}
    for b, ei in _UPB.items():
        z = UP_ZONES.get(b)
        if z is None or b not in by_block:
            continue
        c = z['c']
        # fuse weights: 直接从 blob 原始字节切 E4M3 区 (绕过 load_all 的通用解码)
        if blob_full is not None:
            j = blob_full.find(f'block{b}.layer0.layer'.encode())
            k = j + len(f'block{b}.layer0.layer')
            arr = np.frombuffer(blob_full[k+28:k+28+z['bias'][1]], dtype=np.uint8)
        else:
            arr = np.zeros(z['bias'][1], dtype=np.uint8)
        fuse_v = e4m3_decode(arr[z['fuse'][0]:z['fuse'][1]])
        bias_v = fp16_decode(arr[z['bias'][0]:z['bias'][1]])
        pn = f'expands.{ei}.fuse.weight'
        if pn in pmap and len(fuse_v) >= pmap[pn].numel():
            put(pn, fuse_v)   # 完整 (b48/b56/b66); 不完整 (b62) 留给 Kaiming 兑底
            fb = f'expands.{ei}.fuse.bias'
            if fb in pmap:
                if len(bias_v) >= pmap[fb].numel():
                    put(fb, bias_v[:pmap[fb].numel()])
                # else: 留给兑底 (bias=0, LN canonical 一致)
        # norm γ/β: blob 中无法定位 expand LN(2c→) 的真实源区 — misc 头部是
        # up 记录内部 swin 块自己的 γ1/β1 (b48 [0:256]=+0.915 与 b15 γ1=+0.884
        # 对齐), 且 expand LN 维度 (512/256/128/64) 与头部段不匹配。
        # 按项目先例 (b14 merges.2 / enc3, 见 Task 7a): canonical γ=1, β=0 兑底
        # (留在 unfilled, 末尾兑底逻辑处理) — 不猜测性装填。
    # fuse.weight 不足段 (b62 3584/8192) / expand GEMM (blob 无对应字节) →
    # 结构占位 Kaiming 兑底, 真实信号由 fuse 携带
    for ei in range(4):
        for pn, pin in ((f'expands.{ei}.fuse.weight', 1),
                        (f'expands.{ei}.expand.weight', 1)):
            if pn in pmap and pn in unfilled:
                p = pmap[pn]
                with torch.no_grad():
                    p.normal_(0.0, math.sqrt(1.0 / p.shape[1]))
                unfilled.discard(pn); stats['filled'] += p.numel()
    # —— bottleneck (b31-38): _SplitBlock wqkv/proj/side/ffwd ← layer0/1/2/4 (layer4=ffwd+bias!) ——
    for bi, b in enumerate(range(31, 39)):
        if b in by_block:
            layers = by_block[b]
            m0,_ = layers.get('layer0',(np.zeros(0),np.zeros(0)))
            m1,_ = layers.get('layer1',(np.zeros(0),np.zeros(0)))
            m2,_ = layers.get('layer2',(np.zeros(0),np.zeros(0)))
            m4,_ = layers.get('layer4',(np.zeros(0),np.zeros(0)))  # 1050624 = ffwd 2048×512 + bias 2048
            put(f'bn.{bi}.wqkv.weight', m0)          # 4194304 + qkv_pad16
            put(f'bn.{bi}.proj.weight', m1)           # 4194304
            put(f'bn.{bi}.proj.bias', m1[4194304:])   # 2048
            put(f'bn.{bi}.side.weight', m2)           # 3145728 + side_pad128
            put(f'bn.{bi}.ffwd.weight', m4)           # 1048576
            put(f'bn.{bi}.ffwd.bias', m4[1048576:])   # 2048
            if f'bn.{bi}.gate' in pmap:
                with torch.no_grad(): pmap[f'bn.{bi}.gate'].zero_()
    # —— b22 尾部 256→512 转换矩阵: 挂载 split_exit_pad (原始 E[689152:820224] E4M3) ——
    if 22 in by_block and 'split_exit_pad.p' in pmap:
        import struct as _st2
        j22 = blob_full.find(b'block22.layer0.layer'); k22 = j22 + len('block22.layer0.layer')
        raw22 = np.frombuffer(blob_full[k22+28+689152 : k22+28+820224], dtype=np.uint8)
        v22 = e4m3_decode(raw22[:pmap['split_exit_pad.p'].numel()*4])
        put('split_exit_pad.p', v22)

    # —— b30.layer4 (enc→bn 512→1024 proj): 旁路挂载 enc_to_bn_pad ——
    if 30 in by_block and 'layer4' in by_block[30] and 'enc_to_bn_pad.p' in pmap:
        m4, _ = by_block[30]['layer4']
        put('enc_to_bn_pad.p', m4[:pmap['enc_to_bn_pad.p'].numel()])

    # —— bn_proj / tail (b39? b70) ——
    if 39 in by_block:
        main, misc = by_block[39]['layer0']
        put('bn_proj.weight', main) if 'bn_proj.weight' in pmap else None
        # b39.layer0 = 2×(512,512,1,1) conv + 1024B fp16 gate (尾 512 值 U[0.2,0.8])
        if 'dec_gate' in pmap:
            import struct as _st
            j=blob_full.find(b'block39.layer0.layer'); kk=j+len('block39.layer0.layer')
            B39=_st.unpack_from('<Q',blob_full,kk+16)[0]
            raw39=np.frombuffer(blob_full[kk+28+B39-1024:kk+28+B39],dtype=np.uint8)
            g=np.frombuffer(raw39.tobytes(),dtype='<f2').astype(np.float32)
            with torch.no_grad(): pmap['dec_gate'].copy_(torch.from_numpy(g))
            unfilled.discard('dec_gate')
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
    unfilled, stats = fill_model(m, by_block, blob_full=open(BLOB,'rb').read())
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
