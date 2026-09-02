"""Task C — gate/残差缩放字节定位.

bn chain activation growth 17→589 (8 bottleneck blocks ×~1.5 each).
Decoded: 145,714,050 values vs model 147,683,778 params (diff = 1,969,728 absorbed by calib_pad).
Model has fake params (_pad, gate_pad, etc.).

Tasks:
1. Parse all model parameter names
2. Classify into: (a) real weights (b) pad fake params (c) gate/scalar params
3. Compute per-block gap = blob_budget - decoded_real_weights
4. The gap bytes are where per-block gate/scale could live
"""
import sys, os, json, re, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase3'))
from dlss5.calib_model import DLSS5NetCalib

sys.path.insert(0, os.path.join(HERE, '..', 'phase3', 'dlss5'))
from blob_budget import STAGE_TARGET, STAGE_BLOCKS, BLOCK_B

OUT = os.path.join(HERE, 'GATE_BYTES.md')


def classify_param(name):
    """Return one of: 'real_w', 'real_b', 'pad', 'gate', 'ln_w', 'ln_b', 'mlp_b', 'qkv_b', 'rel_b', 'other'."""
    nl = name.lower()
    if 'pad' in nl and (name.endswith('.pad') or name.endswith('_pad') or name == 'calib_pad' or name == 'stem_pad'):
        return 'pad'
    if 'gate' in nl or 'blend' in nl:
        return 'gate'
    if 'relative_position_bias_table' in name:
        return 'rel_b'
    if 'norm' in nl and 'weight' in name:
        return 'ln_w'
    if 'norm' in nl and 'bias' in name:
        return 'ln_b'
    if 'mlp' in nl and 'bias' in name:
        return 'mlp_b'
    if 'qkv' in nl and 'bias' in name:
        return 'qkv_b'
    if 'proj' in nl and 'bias' in name:
        return 'qkv_b'
    if 'bias' in name:
        return 'real_b'
    if 'weight' in name:
        return 'real_w'
    return 'other'


def main():
    m = DLSS5NetCalib()
    n_total = sum(p.numel() for p in m.parameters())

    cat_count = collections.defaultdict(lambda: {'n': 0, 'numel': 0, 'names': []})
    for n, p in m.named_parameters():
        cat = classify_param(n)
        cat_count[cat]['n'] += 1
        cat_count[cat]['numel'] += p.numel()
        cat_count[cat]['names'].append((n, p.numel()))

    # Per-stage summary using STAGE_BLOCKS
    roles = json.load(open(os.path.join(HERE, 'block_roles.json')))
    stage_agg = {}
    for stage, blocks in STAGE_BLOCKS.items():
        target = sum(BLOCK_B[b] for b in blocks)
        if stage == 'stem':
            params = sum(p.numel() for n, p in m.named_parameters() if n.startswith('stem.'))
        elif stage == 'tail':
            params = sum(p.numel() for n, p in m.named_parameters() if n.startswith('tail.'))
        elif stage.startswith('enc'):
            i = int(stage[3])
            params = sum(p.numel() for n, p in m.named_parameters() if n.startswith(f'enc.{i}.'))
            if i > 0:
                params += sum(p.numel() for n, p in m.named_parameters() if n.startswith(f'merges.{i-1}.'))
        elif stage.startswith('dec'):
            i = int(stage[3])
            params = sum(p.numel() for n, p in m.named_parameters() if n.startswith(f'dec.{i}.'))
            if i > 0:
                params += sum(p.numel() for n, p in m.named_parameters() if n.startswith(f'expands.{i-1}.'))
        elif stage == 'bn':
            params = sum(p.numel() for n, p in m.named_parameters() if n.startswith('bn.'))
        diff = target - params
        stage_agg[stage] = (params, target, diff)

    # Per-block budget vs real_weights
    block_real = collections.defaultdict(int)
    block_pad = collections.defaultdict(int)
    block_gate = collections.defaultdict(int)
    for n, p in m.named_parameters():
        cat = classify_param(n)
        block_id = None
        parts = n.split('.')
        if n.startswith('stem.'):
            block_id = 0
        elif n.startswith('enc.') and 'blocks' in n:
            stage_idx = int(parts[1])
            block_in_stage = int(parts[3])
            if stage_idx == 0: block_id = block_in_stage + 1
            elif stage_idx == 1: block_id = block_in_stage + 5
            elif stage_idx == 2: block_id = block_in_stage + 9
            elif stage_idx == 3: block_id = block_in_stage + 15
            elif stage_idx == 4: block_id = block_in_stage + 23
        elif n.startswith('dec.') and 'blocks' in n:
            stage_idx = int(parts[1])
            block_in_stage = int(parts[3])
            if stage_idx == 0: block_id = block_in_stage + 40
            elif stage_idx == 1: block_id = block_in_stage + 49
            elif stage_idx == 2: block_id = block_in_stage + 57
            elif stage_idx == 3: block_id = block_in_stage + 63
            elif stage_idx == 4: block_id = block_in_stage + 67
        elif n.startswith('bn.') and parts[1].isdigit():
            block_id = 31 + int(parts[1])
        elif n.startswith('bn_proj'):
            block_id = 39
        elif n.startswith('merges.'):
            merge_idx = int(parts[1])
            block_id = [4, 8, 14, 22][merge_idx]
        elif n.startswith('expands.'):
            expand_idx = int(parts[1])
            block_id = [48, 56, 62, 66][expand_idx]
        elif n.startswith('tail.'):
            block_id = 70

        if block_id is None:
            continue
        if cat == 'pad':
            block_pad[block_id] += p.numel()
        elif cat == 'gate':
            block_gate[block_id] += p.numel()
        else:
            block_real[block_id] += p.numel()

    # Build markdown
    lines = []
    lines.append('# Gate / 残差缩放字节定位\n')
    lines.append('> Phase 1 收尾 — Task C\n')
    lines.append('## 0. 假设 & 数据\n')
    lines.append('- 模型 bn 链激活渐增 **17 → 589** (8 个瓶颈块 ×~1.5 倍增), 疑似缺 per-block gate/scale')
    lines.append('- 全库解码消费 **145,714,050** 值 vs 模型 **147,683,778** 参数 (差 ≈ 1.97M 被 calib_pad 吸收)')
    lines.append(f'- 模型 `calib_pad` = **3,637,160** 占 99% pad, fake param 多用于吸收 stage 字节残差\n')

    lines.append('## 1. 模型参数分类\n')
    lines.append('| 类别 | tensors | numel | 含义 |')
    lines.append('|---|---|---|---|')
    cat_desc = {
        'real_w': '**真实权重** (qkv/proj/mlp 主权重)',
        'real_b': '**真实 bias** (mlp/stem/bn_proj 等非 LN bias)',
        'ln_w': '**LN gamma** (norm.weight)',
        'ln_b': '**LN beta** (norm.bias)',
        'mlp_b': '**MLP 内部 bias** (mlp.0.bias / mlp.2.bias)',
        'qkv_b': '**QKV / proj bias**',
        'rel_b': '**Relative position bias table**',
        'pad': '**pad 假参数** (calib_pad/stem_pad/gate_pad/qkv_pad/side_pad/tail._pad)',
        'gate': '**gate/scalar 参数** (bn.gate / tail.blend)',
        'other': '其他',
    }
    for cat in sorted(cat_count.keys()):
        cd = cat_count[cat]
        lines.append(f'| `{cat}` | {cd["n"]} | {cd["numel"]:,} | {cat_desc.get(cat, "?")} |')
    lines.append(f'| **TOTAL** | {sum(cd["n"] for cd in cat_count.values())} | {n_total:,} | |')

    lines.append('\n## 2. 关键统计\n')
    lines.append(f'- **(a) 真实权重** (qkv/proj/mlp + LN + biases + rel_bias): **{cat_count["real_w"]["numel"] + cat_count["ln_w"]["numel"] + cat_count["ln_b"]["numel"] + cat_count["mlp_b"]["numel"] + cat_count["real_b"]["numel"] + cat_count["qkv_b"]["numel"] + cat_count["rel_b"]["numel"]:,}** params')
    lines.append(f'- **(b) pad 假参数**: **{cat_count["pad"]["numel"]:,}** params (含 calib_pad 3,637,160 占 99%)')
    lines.append(f'- **(c) gate/scalar**: **{cat_count["gate"]["numel"]:,}** params (**太少** — 仅 8 个 bn.gate + 1 个 tail.blend = 9 bytes!)')
    lines.append(f'- 总模型: **{n_total:,}** (= 147,683,778)')
    lines.append(f'- blob 总: 147,695,410 bytes (含 8B header + 32B pad + records)\n')

    lines.append('## 3. Per-stage 字节缺口 (blob_budget − model_real_params)\n')
    lines.append('| stage | blocks | model | blob_B | **diff** | 解读 |')
    lines.append('|---|---|---|---|---|---|')
    for stage, blocks in STAGE_BLOCKS.items():
        params, target, diff = stage_agg[stage]
        if diff > 1000:
            interp = '**model 缺参数** (gate/scale 候选字节预算)'
        elif diff < -1000:
            interp = '**model 偏多** (额外的 bias / 配置参数)'
        else:
            interp = '匹配'
        lines.append(f'| `{stage}` | {len(blocks)} | {params:,} | {target:,} | **{diff:+,}** | {interp} |')

    lines.append('\n## 4. Per-block 详细缺口表\n')
    lines.append('| block | role | blob_B | real_w+b | pad | gate | **gap** | 候选 |')
    lines.append('|---|---|---|---|---|---|---|---|')
    gate_hypotheses = {
        'split_entry_256to512': '256→512 down-proj (~131K)',
        'merge_32to64': '32→64 down-proj + LN/bias',
        'merge_64to128': '64→128 down-proj',
        'merge_128to256': '128→256 down-proj',
        'up_64to32': '64→32 up-proj',
        'up_128to64': '128→64 up-proj',
        'up_256to128': '256→128 up-proj',
        'up_512to256': '512→256 up-proj',
        'enc_stage4_exit': '**c=512 stage 出口** (extra layer4)',
        'dec_stage4_entry': '**c=512 stage 入口** (extra proj)',
        'tail_out': '**tail head** (final residual head scales)',
        'stem': '**stem 残差** (MX scale + LN)',
        'enc_stage2_128ch': 'c=128 swin (fp16 tail + LN)',
        'enc_stage3_256ch': 'c=256 swin (fp16 tail + LN)',
        'enc_stage4_512ch': 'c=512 swin (extra layer)',
        'bottleneck_split_swin': '**bn per-block gate/scale** (主要候选)',
        'dec_stage3_256ch': 'c=256 decoder swin',
        'dec_stage2_128ch': 'c=128 decoder swin',
        'dec_stage1_64ch': 'c=64 decoder swin',
        'dec_stage0_32ch': 'c=32 decoder swin',
    }
    for b in sorted(BLOCK_B.keys()):
        B = BLOCK_B[b]
        real = block_real.get(b, 0)
        pad = block_pad.get(b, 0)
        gate = block_gate.get(b, 0)
        gap = B - real - pad - gate
        role = roles.get(str(b), '?')
        if abs(gap) > 100:
            flag = ' ⚠' if gap > 0 else ' ← overshoot'
            hyp = gate_hypotheses.get(role, '')
            lines.append(f'| b{b} | `{role}` | {B:,} | {real:,} | {pad} | {gate} | **{gap:+,}**{flag} | {hyp} |')
        else:
            lines.append(f'| b{b} | `{role}` | {B:,} | {real:,} | {pad} | {gate} | {gap:+,} | |')

    lines.append('\n## 5. 主要 gate/scale 字节候选 (按 gap 大小)\n')
    lines.append('| block | role | gap | 候选 gate/scale 类型 |')
    lines.append('|---|---|---|---|')
    block_gaps = []
    for b in sorted(BLOCK_B.keys()):
        B = BLOCK_B[b]
        real = block_real.get(b, 0)
        pad = block_pad.get(b, 0)
        gate = block_gate.get(b, 0)
        gap = B - real - pad - gate
        block_gaps.append((b, gap, BLOCK_B[b], roles.get(str(b), '?')))
    block_gaps.sort(key=lambda x: -abs(x[1]))
    for b, gap, B, role in block_gaps:
        if abs(gap) > 1000:
            hyp = gate_hypotheses.get(role, 'per-block gate/scale')
            lines.append(f'| b{b} | `{role}` | **{gap:+,}** | {hyp} |')

    lines.append('\n## 6. bn 链 per-block gate 字节预算分析\n')
    lines.append('瓶颈块 (b31-38) 8 个, 每块 12,587,154B, 总 ~100.7MB\n')
    lines.append('激活渐增 17→589 ≈ 每块 ×1.5, 假设需要 per-block gate/scale\n')
    lines.append('')
    lines.append('当前模型内 bn 块已建模的 fake param:')
    lines.append('| tensor | shape | numel | per block × 8 |')
    lines.append('|---|---|---|---|')
    lines.append('| `bn.{{i}}.qkv_pad` | (16,) | 16 | 128 |')
    lines.append('| `bn.{{i}}.side_pad` | (128,) | 128 | 1024 |')
    lines.append('| `bn.{{i}}.gate_pad` | (1,) | 1 | 8 |')
    lines.append('| `bn.{{i}}.gate` | (1,) | 1 | 8 |')
    lines.append('| `bn.{{i}}.pad` | (0,) | 0 | 0 |')
    lines.append('| **合计 fake per block** | | **146** | **1168** |')
    lines.append('')
    lines.append(f'**每 bn 块 fake param 总额: 146 bytes** — 太少, 真实 gate/scale 字节远超此数')
    lines.append('')
    lines.append('如果每 bn 块需要 4KB (类似 ConvNeXt LayerScale):')
    lines.append('- 8 blocks × 4,096 bytes = **32,768 bytes**')
    lines.append(f'- 但当前 fake param 仅 1,168 bytes — **缺口 ≈ 31,600 bytes**')
    lines.append(f'- 这些缺口必须藏在 `qkv_pad`/`side_pad` 重新利用, 或在 SplitBlock 内部增加新 param')
    lines.append('')
    lines.append(f'### b30 enc_stage4_exit 特殊关注\n')
    lines.append(f'- b30 = 2,492,496B (5 子记录 vs b23-29 的 4 子记录)')
    lines.append(f'- **模型内没有 b30 对应** (enc.4 只有 blocks 0-6 = b23-29)')
    lines.append(f'- 这 2.49MB 完全未被装填 = b30 layer4 (524,304B) + 真实内容未建模')
    lines.append(f'- **b30 很可能是 c=512 出口 down-proj / final LN 字节预算** = per-stage gate 候选\n')

    lines.append('## 7. 结论\n')
    lines.append('### 7.1 主要候选 gate/scale 字节位置\n')
    lines.append('| 字节预算位置 | gap | 用途假设 |')
    lines.append('|---|---|---|')
    lines.append('| **b30 enc_stage4_exit** | 2,492,496B | **c=512 出口 down-proj + per-stage scale** (主候选!) |')
    lines.append('| **b22 split_entry_256to512** | 293,952B | **256→512 down-proj + per-block scale** (Task A 重点) |')
    lines.append('| **b48 up_512to256** | 295,472B | **512→256 up-proj + per-block scale** |')
    lines.append('| **b39 dec_stage4_entry** | 262,656B | **c=512 入口 proj** |')
    lines.append('| **b14 merge_128to256** | 97,840B | 128→256 down-proj + LN/bias |')
    lines.append('| **b56 up_256to128** | 98,592B | 256→128 up-proj + LN/bias |')
    lines.append('| **b62 up_128to64** | 37,024B | 128→64 up-proj + LN/bias |')
    lines.append('| **b8 merge_64to128** | 36,656B | 64→128 down-proj + LN/bias |')
    lines.append('| **b70 tail_out** | 19,887B | tail head 残差 (global_fc + conv bias + blend_scale) |')
    lines.append('| **b0 stem** | 19,072B | stem 主权重 (conv3x3+norm+pre-LN) |')
    lines.append('| **b66 up_64to32** | 14,464B | 64→32 up-proj + LN/bias |')
    lines.append('| **b4 merge_32to64** | 14,272B | 32→64 down-proj + LN/bias |')
    lines.append('')
    lines.append('### 7.2 bn 链 (b31-38) 缺口 = 1,168 bytes (fake param) — 远低于典型 ConvNeXt LayerScale 需求\n')
    lines.append('- 现 fake param: qkv_pad 16 + side_pad 128 + gate_pad 1 + gate 1 = **146 bytes/block**')
    lines.append('- 真实 gate/scale 字节大概率藏在 **SplitBlock 内部未建模的 fake param 区**,')
    lines.append('  或在 b31-38 的 fp16 misc 字节里 (c=512 split-swin 的 fp16 LN tail)')
    lines.append('')
    lines.append('### 7.3 下一步\n')
    lines.append('1. 在 `_SplitBlock` 内增加 per-block gate 参数 (e.g., `bn.{{i}}.scale` shape=(2048,) fp16, =4KB)')
    lines.append('2. 验证 b30 enc_stage4_exit 是否包含 524,304B down-proj (c=512→c=512 或 c=512→c=2048 wide)')
    lines.append('3. 装填 b22 的 131,072B fp16 区 (Task A) 为 256→512 down-proj weights')
    lines.append('4. 装填 b48 的 295,472B = 512×256=131,072B up-proj + 余 164,400B LN/bias')
    lines.append('5. 重测 GPU 前向, 验证 bn 链激活不再渐增 (17→589 → 应平稳 O(1-10))\n')

    out_md = '\n'.join(lines) + '\n'
    open(OUT, 'w').write(out_md)
    print(f'\nWrote {OUT}')

    # Quick summary
    print(f'\n=== Summary ===')
    print(f'(a) Real weights: {cat_count["real_w"]["numel"] + cat_count["ln_w"]["numel"] + cat_count["ln_b"]["numel"] + cat_count["mlp_b"]["numel"] + cat_count["real_b"]["numel"] + cat_count["qkv_b"]["numel"] + cat_count["rel_b"]["numel"]:,}')
    print(f'(b) Pad fake params: {cat_count["pad"]["numel"]:,}')
    print(f'(c) Gate scalars: {cat_count["gate"]["numel"]:,}')
    print(f'\nLargest gap blocks (gate/scale byte budgets):')
    for b, gap, B, role in block_gaps[:8]:
        if abs(gap) > 100:
            print(f'  b{b} ({role}): {gap:+,} bytes')


if __name__ == '__main__':
    main()