# Gate / 残差缩放字节定位

> Phase 1 收尾 — Task C

## 0. 假设 & 数据

- 模型 bn 链激活渐增 **17 → 589** (8 个瓶颈块 ×~1.5 倍增), 疑似缺 per-block gate/scale
- 全库解码消费 **145,714,050** 值 vs 模型 **147,683,778** 参数 (差 ≈ 1.97M 被 calib_pad 吸收)
- 模型 `calib_pad` = **3,637,160** 占 99% pad, fake param 多用于吸收 stage 字节残差

## 1. 模型参数分类

| 类别 | tensors | numel | 含义 |
|---|---|---|---|
| `gate` | 9 | 9 | **gate/scalar 参数** (bn.gate / tail.blend) |
| `ln_b` | 70 | 21,184 | **LN beta** (norm.bias) |
| `ln_w` | 110 | 29,120 | **LN gamma** (norm.weight) |
| `mlp_b` | 51 | 23,056 | **MLP 内部 bias** (mlp.0.bias / mlp.2.bias) |
| `pad` | 35 | 3,677,278 | **pad 假参数** (calib_pad/stem_pad/gate_pad/qkv_pad/side_pad/tail._pad) |
| `qkv_b` | 9 | 16,896 | **QKV / proj bias** |
| `real_b` | 11 | 16,451 | **真实 bias** (mlp/stem/bn_proj 等非 LN bias) |
| `real_w` | 248 | 143,811,584 | **真实权重** (qkv/proj/mlp 主权重) |
| `rel_b` | 39 | 88,200 | **Relative position bias table** |
| **TOTAL** | 582 | 147,683,778 | |

## 2. 关键统计

- **(a) 真实权重** (qkv/proj/mlp + LN + biases + rel_bias): **144,006,491** params
- **(b) pad 假参数**: **3,677,278** params (含 calib_pad 3,637,160 占 99%)
- **(c) gate/scalar**: **9** params (**太少** — 仅 8 个 bn.gate + 1 个 tail.blend = 9 bytes!)
- 总模型: **147,683,778** (= 147,683,778)
- blob 总: 147,695,410 bytes (含 8B header + 32B pad + records)

## 3. Per-stage 字节缺口 (blob_budget − model_real_params)

| stage | blocks | model | blob_B | **diff** | 解读 |
|---|---|---|---|---|---|
| `stem` | 1 | 2,624 | 21,696 | **+19,072** | **model 缺参数** (gate/scale 候选字节预算) |
| `enc0` | 4 | 62,016 | 84,736 | **+22,720** | **model 缺参数** (gate/scale 候选字节预算) |
| `enc1` | 4 | 194,013 | 255,216 | **+61,203** | **model 缺参数** (gate/scale 候选字节预算) |
| `enc2` | 6 | 1,018,900 | 1,215,856 | **+196,956** | **model 缺参数** (gate/scale 候选字节预算) |
| `enc3` | 8 | 4,960,227 | 5,644,912 | **+684,685** | **model 缺参数** (gate/scale 候选字节预算) |
| `enc4` | 8 | 14,303,344 | 16,269,840 | **+1,966,496** | **model 缺参数** (gate/scale 候选字节预算) |
| `bn` | 9 | 100,697,232 | 101,222,544 | **+525,312** | **model 缺参数** (gate/scale 候选字节预算) |
| `dec0` | 9 | 15,745,152 | 16,566,320 | **+821,168** | **model 缺参数** (gate/scale 候选字节预算) |
| `dec1` | 8 | 5,353,443 | 5,054,800 | **-298,643** | **model 偏多** (额外的 bias / 配置参数) |
| `dec2` | 6 | 1,117,204 | 1,055,968 | **-61,236** | **model 偏多** (额外的 bias / 配置参数) |
| `dec3` | 4 | 218,589 | 208,064 | **-10,525** | **model 偏多** (额外的 bias / 配置参数) |
| `dec4` | 3 | 70,336 | 62,016 | **-8,320** | **model 偏多** (额外的 bias / 配置参数) |
| `tail` | 1 | 21,810 | 21,810 | **+0** | 匹配 |

## 4. Per-block 详细缺口表

| block | role | blob_B | real_w+b | pad | gate | **gap** | 候选 |
|---|---|---|---|---|---|---|---|
| b0 | `stem` | 21,696 | 2,624 | 0 | 0 | **+19,072** ⚠ | **stem 残差** (MX scale + LN) |
| b1 | `enc_stage0_32ch` | 20,672 | 20,672 | 0 | 0 | +0 | |
| b2 | `enc_stage0_32ch` | 20,672 | 20,672 | 0 | 0 | +0 | |
| b3 | `enc_stage0_32ch` | 20,672 | 20,672 | 0 | 0 | +0 | |
| b4 | `merge_32to64` | 22,720 | 8,448 | 0 | 0 | **+14,272** ⚠ | 32→64 down-proj + LN/bias |
| b5 | `enc_stage1_64ch` | 61,760 | 61,855 | 0 | 0 | -95 | |
| b6 | `enc_stage1_64ch` | 61,760 | 61,855 | 0 | 0 | -95 | |
| b7 | `enc_stage1_64ch` | 61,760 | 61,855 | 0 | 0 | -95 | |
| b8 | `merge_64to128` | 69,936 | 33,280 | 0 | 0 | **+36,656** ⚠ | 64→128 down-proj |
| b9 | `enc_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b10 | `enc_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b11 | `enc_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b12 | `enc_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b13 | `enc_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b14 | `merge_128to256` | 229,936 | 132,096 | 0 | 0 | **+97,840** ⚠ | 128→256 down-proj |
| b15 | `enc_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 swin (fp16 tail + LN) |
| b16 | `enc_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 swin (fp16 tail + LN) |
| b17 | `enc_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 swin (fp16 tail + LN) |
| b18 | `enc_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 swin (fp16 tail + LN) |
| b19 | `enc_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 swin (fp16 tail + LN) |
| b20 | `enc_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 swin (fp16 tail + LN) |
| b21 | `enc_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 swin (fp16 tail + LN) |
| b22 | `split_entry_256to512` | 820,288 | 526,336 | 0 | 0 | **+293,952** ⚠ | 256→512 down-proj (~131K) |
| b23 | `enc_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b24 | `enc_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b25 | `enc_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b26 | `enc_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b27 | `enc_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b28 | `enc_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b29 | `enc_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b30 | `enc_stage4_exit` | 2,492,496 | 0 | 0 | 0 | **+2,492,496** ⚠ | **c=512 stage 出口** (extra layer4) |
| b31 | `bottleneck_split_swin` | 12,587,154 | 12,587,008 | 145 | 1 | +0 | |
| b32 | `bottleneck_split_swin` | 12,587,154 | 12,587,008 | 145 | 1 | +0 | |
| b33 | `bottleneck_split_swin` | 12,587,154 | 12,587,008 | 145 | 1 | +0 | |
| b34 | `bottleneck_split_swin` | 12,587,154 | 12,587,008 | 145 | 1 | +0 | |
| b35 | `bottleneck_split_swin` | 12,587,154 | 12,587,008 | 145 | 1 | +0 | |
| b36 | `bottleneck_split_swin` | 12,587,154 | 12,587,008 | 145 | 1 | +0 | |
| b37 | `bottleneck_split_swin` | 12,587,154 | 12,587,008 | 145 | 1 | +0 | |
| b38 | `bottleneck_split_swin` | 12,587,154 | 12,587,008 | 145 | 1 | +0 | |
| b39 | `dec_stage4_entry` | 525,312 | 262,656 | 0 | 0 | **+262,656** ⚠ | **c=512 stage 入口** (extra proj) |
| b40 | `dec_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b41 | `dec_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b42 | `dec_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b43 | `dec_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b44 | `dec_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b45 | `dec_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b46 | `dec_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b47 | `dec_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b48 | `up_512to256` | 820,784 | 525,312 | 0 | 0 | **+295,472** ⚠ | 512→256 up-proj |
| b49 | `dec_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 decoder swin |
| b50 | `dec_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 decoder swin |
| b51 | `dec_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 decoder swin |
| b52 | `dec_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 decoder swin |
| b53 | `dec_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 decoder swin |
| b54 | `dec_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 decoder swin |
| b55 | `dec_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | **-501** ← overshoot | c=256 decoder swin |
| b56 | `up_256to128` | 230,176 | 131,584 | 0 | 0 | **+98,592** ⚠ | 256→128 up-proj |
| b57 | `dec_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b58 | `dec_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b59 | `dec_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b60 | `dec_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b61 | `dec_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b62 | `up_128to64` | 70,048 | 33,024 | 0 | 0 | **+37,024** ⚠ | 128→64 up-proj |
| b63 | `dec_stage1_64ch` | 61,760 | 61,855 | 0 | 0 | -95 | |
| b64 | `dec_stage1_64ch` | 61,760 | 61,855 | 0 | 0 | -95 | |
| b65 | `dec_stage1_64ch` | 61,760 | 61,855 | 0 | 0 | -95 | |
| b66 | `up_64to32` | 22,784 | 8,320 | 0 | 0 | **+14,464** ⚠ | 64→32 up-proj |
| b67 | `dec_stage0_32ch` | 20,672 | 20,672 | 0 | 0 | +0 | |
| b68 | `dec_stage0_32ch` | 20,672 | 20,672 | 0 | 0 | +0 | |
| b69 | `dec_stage0_32ch` | 20,672 | 20,672 | 0 | 0 | +0 | |
| b70 | `tail_out` | 21,810 | 1,923 | 19886 | 1 | +0 | |

## 5. 主要 gate/scale 字节候选 (按 gap 大小)

| block | role | gap | 候选 gate/scale 类型 |
|---|---|---|---|
| b30 | `enc_stage4_exit` | **+2,492,496** | **c=512 stage 出口** (extra layer4) |
| b48 | `up_512to256` | **+295,472** | 512→256 up-proj |
| b22 | `split_entry_256to512` | **+293,952** | 256→512 down-proj (~131K) |
| b39 | `dec_stage4_entry` | **+262,656** | **c=512 stage 入口** (extra proj) |
| b56 | `up_256to128` | **+98,592** | 256→128 up-proj |
| b14 | `merge_128to256` | **+97,840** | 128→256 down-proj |
| b62 | `up_128to64` | **+37,024** | 128→64 up-proj |
| b8 | `merge_64to128` | **+36,656** | 64→128 down-proj |
| b0 | `stem` | **+19,072** | **stem 残差** (MX scale + LN) |
| b66 | `up_64to32` | **+14,464** | 64→32 up-proj |
| b4 | `merge_32to64` | **+14,272** | 32→64 down-proj + LN/bias |

## 6. bn 链 per-block gate 字节预算分析

瓶颈块 (b31-38) 8 个, 每块 12,587,154B, 总 ~100.7MB

激活渐增 17→589 ≈ 每块 ×1.5, 假设需要 per-block gate/scale


当前模型内 bn 块已建模的 fake param:
| tensor | shape | numel | per block × 8 |
|---|---|---|---|
| `bn.{{i}}.qkv_pad` | (16,) | 16 | 128 |
| `bn.{{i}}.side_pad` | (128,) | 128 | 1024 |
| `bn.{{i}}.gate_pad` | (1,) | 1 | 8 |
| `bn.{{i}}.gate` | (1,) | 1 | 8 |
| `bn.{{i}}.pad` | (0,) | 0 | 0 |
| **合计 fake per block** | | **146** | **1168** |

**每 bn 块 fake param 总额: 146 bytes** — 太少, 真实 gate/scale 字节远超此数

如果每 bn 块需要 4KB (类似 ConvNeXt LayerScale):
- 8 blocks × 4,096 bytes = **32,768 bytes**
- 但当前 fake param 仅 1,168 bytes — **缺口 ≈ 31,600 bytes**
- 这些缺口必须藏在 `qkv_pad`/`side_pad` 重新利用, 或在 SplitBlock 内部增加新 param

### b30 enc_stage4_exit 特殊关注

- b30 = 2,492,496B (5 子记录 vs b23-29 的 4 子记录)
- **模型内没有 b30 对应** (enc.4 只有 blocks 0-6 = b23-29)
- 这 2.49MB 完全未被装填 = b30 layer4 (524,304B) + 真实内容未建模
- **b30 很可能是 c=512 出口 down-proj / final LN 字节预算** = per-stage gate 候选

## 7. 结论

### 7.1 主要候选 gate/scale 字节位置

| 字节预算位置 | gap | 用途假设 |
|---|---|---|
| **b30 enc_stage4_exit** | 2,492,496B | **c=512 出口 down-proj + per-stage scale** (主候选!) |
| **b22 split_entry_256to512** | 293,952B | **256→512 down-proj + per-block scale** (Task A 重点) |
| **b48 up_512to256** | 295,472B | **512→256 up-proj + per-block scale** |
| **b39 dec_stage4_entry** | 262,656B | **c=512 入口 proj** |
| **b14 merge_128to256** | 97,840B | 128→256 down-proj + LN/bias |
| **b56 up_256to128** | 98,592B | 256→128 up-proj + LN/bias |
| **b62 up_128to64** | 37,024B | 128→64 up-proj + LN/bias |
| **b8 merge_64to128** | 36,656B | 64→128 down-proj + LN/bias |
| **b70 tail_out** | 19,887B | tail head 残差 (global_fc + conv bias + blend_scale) |
| **b0 stem** | 19,072B | stem 主权重 (conv3x3+norm+pre-LN) |
| **b66 up_64to32** | 14,464B | 64→32 up-proj + LN/bias |
| **b4 merge_32to64** | 14,272B | 32→64 down-proj + LN/bias |

### 7.2 bn 链 (b31-38) 缺口 = 1,168 bytes (fake param) — 远低于典型 ConvNeXt LayerScale 需求

- 现 fake param: qkv_pad 16 + side_pad 128 + gate_pad 1 + gate 1 = **146 bytes/block**
- 真实 gate/scale 字节大概率藏在 **SplitBlock 内部未建模的 fake param 区**,
  或在 b31-38 的 fp16 misc 字节里 (c=512 split-swin 的 fp16 LN tail)

### 7.3 下一步

1. 在 `_SplitBlock` 内增加 per-block gate 参数 (e.g., `bn.{{i}}.scale` shape=(2048,) fp16, =4KB)
2. 验证 b30 enc_stage4_exit 是否包含 524,304B down-proj (c=512→c=512 或 c=512→c=2048 wide)
3. 装填 b22 的 131,072B fp16 区 (Task A) 为 256→512 down-proj weights
4. 装填 b48 的 295,472B = 512×256=131,072B up-proj + 余 164,400B LN/bias
5. 重测 GPU 前向, 验证 bn 链激活不再渐增 (17→589 → 应平稳 O(1-10))

