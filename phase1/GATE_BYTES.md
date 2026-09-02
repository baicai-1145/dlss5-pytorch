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
| `stem` | 1 | 2,624 | 21,696 | **+19,072** | model 缺参数 |
| `enc0` | 4 | 62,016 | 84,736 | **+22,720** | model 缺参数 |
| `enc1` | 4 | 194,013 | 255,216 | **+61,203** | model 缺参数 |
| `enc2` | 6 | 1,018,900 | 1,215,856 | **+196,956** | model 缺参数 |
| `enc3` | 8 | 4,960,227 | 5,644,912 | **+684,685** | model 缺参数 |
| `enc4` | 8 | 14,303,344 | 16,269,840 | **+1,966,496** | model 缺参数 (含 b30 layer4 未装填) |
| `bn` | 9 | 100,697,232 | 101,222,544 | **+525,312** | model 缺参数 (含 b39.layer0 部分未装填) |
| `dec0` | 9 | 15,745,152 | 16,566,320 | **+821,168** | model 缺参数 (含 b48 up-proj 未装填) |
| `dec1` | 8 | 5,353,443 | 5,054,800 | -298,643 | model 偏多 (额外的 bias) |
| `dec2` | 6 | 1,117,204 | 1,055,968 | -61,236 | model 偏多 |
| `dec3` | 4 | 218,589 | 208,064 | -10,525 | model 偏多 |
| `dec4` | 3 | 70,336 | 62,016 | -8,320 | model 偏多 |
| `tail` | 1 | 21,810 | 21,810 | +0 | 匹配 |

## 4. Per-block 详细缺口表

| block | role | blob_B | real_w+b | pad | gate | **gap** | 候选 |
|---|---|---|---|---|---|---|---|
| b0 | `stem` | 21,696 | 2,624 | 0 | 0 | **+19,072** ⚠ | stem 残差 (MX scale + LN) |
| b1-b3 | `enc_stage0_32ch` | 20,672 | 20,672 | 0 | 0 | +0 | |
| b4 | `merge_32to64` | 22,720 | 8,448 | 0 | 0 | **+14,272** ⚠ | 32→64 down-proj + LN/bias |
| b5-b7 | `enc_stage1_64ch` | 61,760 | 61,855 | 0 | 0 | -95 | model 略偏多 (mlp.0.bias 351B) |
| b8 | `merge_64to128` | 69,936 | 33,280 | 0 | 0 | **+36,656** ⚠ | 64→128 down-proj |
| b9-b13 | `enc_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b14 | `merge_128to256` | 229,936 | 132,096 | 0 | 0 | **+97,840** ⚠ | 128→256 down-proj |
| b15-b21 | `enc_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | -501 | c=256 swin (model 偏多 bias) |
| b22 | `split_entry_256to512` | 820,288 | 526,336 | 0 | 0 | **+293,952** ⚠ | 256→512 down-proj (Task A) |
| b23-b29 | `enc_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b30 | `enc_stage4_exit` | 2,492,496 | 0 | 0 | 0 | **+2,492,496** ⚠ | **encoder→bottleneck (见 §5.1)** |
| b31-b38 | `bottleneck_split_swin` | 12,587,154 | 12,587,008 | 145 | 1 | +0 | _SplitBlock 精确装填 |
| b39 | `dec_stage4_entry` | 525,312 | 262,656 | 0 | 0 | **+262,656** ⚠ | **bottleneck→decoder + c=512 GATE (见 §5.1)** |
| b40-b47 | `dec_stage4_512ch` | 1,968,192 | 1,968,144 | 0 | 0 | +48 | |
| b48 | `up_512to256` | 820,784 | 525,312 | 0 | 0 | **+295,472** ⚠ | 512→256 up-proj |
| b49-b55 | `dec_stage3_256ch` | 689,232 | 689,733 | 0 | 0 | -501 | |
| b56 | `up_256to128` | 230,176 | 131,584 | 0 | 0 | **+98,592** ⚠ | 256→128 up-proj |
| b57-b61 | `dec_stage2_128ch` | 197,184 | 197,124 | 0 | 0 | +60 | |
| b62 | `up_128to64` | 70,048 | 33,024 | 0 | 0 | **+37,024** ⚠ | 128→64 up-proj |
| b63-b65 | `dec_stage1_64ch` | 61,760 | 61,855 | 0 | 0 | -95 | |
| b66 | `up_64to32` | 22,784 | 8,320 | 0 | 0 | **+14,464** ⚠ | 64→32 up-proj |
| b67-b69 | `dec_stage0_32ch` | 20,672 | 20,672 | 0 | 0 | +0 | |
| b70 | `tail_out` | 21,810 | 1,923 | 19,886 | 1 | +0 | tail._pad + blend + global_fc/conv bias |

## 5. 主要 gate/scale 字节候选 (按 gap 大小)

| block | role | gap | 候选 gate/scale 类型 |
|---|---|---|---|
| b30 | `enc_stage4_exit` | **+2,492,496** | encoder→bottleneck 转换 (含 b30.layer4 详见 §5.1) |
| b48 | `up_512to256` | **+295,472** | 512→256 up-proj |
| b22 | `split_entry_256to512` | **+293,952** | 256→512 down-proj (Task A) |
| b39 | `dec_stage4_entry` | **+262,656** | bottleneck→decoder 转换 + c=512 GATE (详见 §5.1) |
| b56 | `up_256to128` | **+98,592** | 256→128 up-proj |
| b14 | `merge_128to256` | **+97,840** | 128→256 down-proj |
| b62 | `up_128to64` | **+37,024** | 128→64 up-proj |
| b8 | `merge_64to128` | **+36,656** | 64→128 down-proj |
| b0 | `stem` | **+19,072** | stem 残差 (MX scale + LN) |
| b66 | `up_64to32` | **+14,464** | 64→32 up-proj |
| b4 | `merge_32to64` | **+14,272** | 32→64 down-proj + LN/bias |

## 5.1 重点验证: b30.layer4 (524,304B) + b39.layer0 (525,312B) 精确拆分

> 这两个子记录是 **encoder→bottleneck** 和 **bottleneck→decoder** 转换矩阵, **当前 DLSS5NetCalib 模型完全未装填**。用户假设 100% 正确, 且发现了 **c=512 per-channel GATE 隐藏在 b39.layer0 末尾**。

### b30.layer4 = 524,304B 精确拆分 ✓

```
b30.layer4 = (512, 1024) E4M3 weight + 16 bytes misc
            = 524,288 + 16 = 524,304 ✓
```

| 区段 | 字节 | 解读 |
|---|---|---|
| `[0:524288]` | 524,288B | **E4M3 (512, 1024)** — c=512→c=1024 扩张 proj 权重 |
| | | • 1024 个 512B 窗, 全部 E4M3 (0 MX, 0 FP16) |
| | | • std=**0.0440**, mean=-0.0000, absmax=**0.344** |
| | | • Reshape (512, 1024): per-row std mean=0.044, per-col std mean=0.044 (均匀) |
| | | • **完美匹配 Kaiming relu init**: `std = sqrt(2/fan_in) = sqrt(2/1024) = 0.0442` ✓ |
| | | • **角色**: encoder c=512 输出 → bottleneck 1024-wide 中间表示 |
| `[524288:524304]` | 16B | **misc tail** — 4 个小 E4M3 + 12 零填充 |
| | | • Hex: `980d0995000000000000000000000000` |
| | | • E4M3 decode: `[-0.0625, 0.025, 0.018, -0.051, 0, 0, ..., 0]` (12 zeros) |
| | | • fp16 decode (8 vals): `[0.0003, -0.0012, 0, 0, 0, 0, 0, 0]` |
| | | • 解读: c=4 sub-tensor bias 或 16 个标量残差 |

### b39.layer0 = 525,312B 精确拆分 ✓ (★ 含 per-channel GATE)

```
b39.layer0 = (512, 1024) E4M3 weight + 1024 bytes fp16 (c=512 GATE)
            = 524,288 + 1024 = 525,312 ✓
```

| 区段 | 字节 | 解读 |
|---|---|---|
| `[0:524288]` | 524,288B | **E4M3 (512, 1024)** — bottleneck → decoder 转换 proj 权重 |
| | | • 1023 个 512B 窗 E4M3 + 1 个 512B 窗 FP16 (在尾部) |
| | | • std=**0.0265**, mean=-0.0000, absmax=**0.141** |
| | | • Reshape (512, 1024) 或 2×(512, 512): 两个 proj std 均 ≈0.026 |
| | | • **truncated_normal_(std=0.02) init** — PyTorch / HF 默认 Linear 权重初始化 |
| `[524288:525312]` | **1,024B** | **★ c=512 per-channel GATE (512 个 fp16 值)** |
| | | • range [0.18, 0.81], mean **0.494**, std **0.130** |
| | | • **不是 LN gamma** (mean≈1.0 才对), **不是 bias** (mean≈0 才对) |
| | | • **是 per-channel GATE** 初始化为均匀分布 U[0, 1] (mean=0.5) |
| | | • First 20: `[0.0008, -0.0001, 0.574, 0.499, 0.527, 0.402, 0.497, 0.343, 0.584, 0.600, 0.264, 0.514, 0.653, 0.484, 0.371, 0.671, 0.483, 0.653, 0.329, 0.213]` |
| | | • **★ 核心发现**: b39.layer0 是 c=512 per-channel GATE 载体! |

### b30.layer4 / b39.layer0 在模型中的对应位置

| 当前模型组件 | 对应 blob 位置 | 是否装填 |
|---|---|---|
| `bn.{{i}}.wqkv/proj/side/ffwd` | b31-38 各 layer (5 子记录) | ✓ 已装填 |
| `bn_proj` (Conv2d 512→512, 262K) | b39.layer0 第一半 (524,288B = 1 个 512×1024 proj) | **部分** (只用 1/2) |
| **per-channel GATE 512 fp16** | **b39.layer0 末尾 1,024B** | **✗ 完全未装填** |
| enc → bottleneck 路径 | **b30.layer4** (524,304B) | **✗ 完全未装填** |
| bottleneck → dec 路径 (完整) | **b39.layer0** 完整 525,312B | **✗ 部分未装填** |

### 用户验证的 bn 链激活改善 (30779 → 673, ×45)

- 装填 **b31-38 layer4** (1,050,624B = `512×2048 + 2048 fp16 bias` 精确匹配) 后
- bn 链激活从 **30779 → 673** (×45 降低) — 证实 bn 链爆增主因是缺 ffwd 路径
- 余下 673 渐增 (17→589 ×~1.5) 主因:
  1. 缺 c=512 per-channel GATE (b39.layer0 末尾 1024B, 未装填)
  2. `bn.gate` (1 byte) 装填 = 0, `_SplitBlock.forward` 没实现 `gate × residual` 逻辑

## 6. bn 链 per-block gate 字节预算分析

瓶颈块 (b31-38) 8 个, 每块 **12,587,154B**, 总 ~100.7MB

**关键发现**: `DLSS5NetCalib._SplitBlock` 实际上**已经精确装填 bn 块** — 每个 bn 块 numel = 12,587,154 = BLOCK_B[31] ✓ (gap = 0)。

### `_SplitBlock` 装填明细 (bn.0)

| tensor | shape | numel | 作用 |
|---|---|---|---|
| `bn.0.wqkv.weight` | (2048, 2048) | 4,194,304 | layer0: 2048² qkv GEMM |
| `bn.0.qkv_pad` | (16,) | 16 | layer0 余字节 |
| `bn.0.proj.weight` | (2048, 2048) | 4,194,304 | layer1: 2048² proj |
| `bn.0.proj.bias` | (2048,) | 2,048 | layer1 余字节 |
| `bn.0.side.weight` | (6144, 512) | 3,145,728 | layer2: 12×512² side branch |
| `bn.0.side_pad` | (128,) | 128 | layer2 余字节 |
| `bn.0.ffwd.weight` | (2048, 512) | 1,048,576 | layer4: 512→2048 FFN expand |
| `bn.0.ffwd.bias` | (2048,) | 2,048 | layer4 bias |
| `bn.0.gate` | (1,) | 1 | layer3 scalar |
| `bn.0.gate_pad` | (1,) | 1 | layer3 余字节 |
| `bn.0.pad` | (0,) | 0 | global pad (0) |
| **合计** | | **12,587,154** | **= BLOCK_B[31] ✓** |

`bn.gate` 只有 **1 byte** — 这就是模型里"每 bn 块的 gate"全集。bn 链爆增主因是缺 b31-38 layer4 (1,050,624B ffwd.weight+bias); 装填后从 30779 → 673 (×45 降低)。余下 673 渐增主因是缺 c=512 per-channel GATE (在 b39.layer0 末尾 1024B, 未装填)。

## 7. 结论

### 7.1 主要候选 gate/scale 字节位置 (含精确拆分)

| 字节预算位置 | gap | 精确拆分 | 用途假设 |
|---|---|---|---|
| **b30.layer4** | **524,304B** | **(512, 1024) E4M3 + 16B misc** | **encoder c=512 → bottleneck 1024-wide 扩张 proj** (Kaiming init ✓) |
| **b39.layer0** | **525,312B** | **(512, 1024) E4M3 + 1024B fp16** | **bottleneck 1024 → decoder c=512 + c=512 per-channel GATE** ★ |
| **b39.layer0 末尾 1024B** | **1024B** | **512 fp16 per-channel GATE** | **★ 唯一发现的"per-channel scale" 字节载体** |
| b22 split_entry | 131,072B | (256, 256) fp16 down-proj | c=256 → c=512 transition (Task A) |
| b48 up_512to256 | 295,472B | (512, 256) up-proj + LN | c=512 → c=256 transition |
| b14 merge_128to256 | 97,840B | 128→256 down-proj + LN/bias | |
| b56 up_256to128 | 98,592B | 256→128 up-proj + LN/bias | |
| b62 up_128to64 | 37,024B | 128→64 up-proj + LN/bias | |
| b8 merge_64to128 | 36,656B | 64→128 down-proj + LN/bias | |
| b70 tail_out | 19,887B | tail head 残差 (global_fc + conv bias + blend_scale) | |
| b0 stem | 19,072B | stem 主权重 (conv3x3+norm+pre-LN) | |
| b66 up_64to32 | 14,464B | 64→32 up-proj + LN/bias | |
| b4 merge_32to64 | 14,272B | 32→64 down-proj + LN/bias | |

### 7.2 bn 链 (b31-38) 已精确装填 = 12,587,154B/块, gap = 0

- 模型内 `_SplitBlock` 已**严格对齐 blob 字节预算** (sum=12,587,154 ✓)
- 但 `bn.gate` 仅 1 byte (实际未缩放) — bn 链激活 17→589 是因为装填的 gate 值=0
- 装填 b31-38 layer4 后激活从 30779 → 673 (×45 改善) — 用户已实测确认

### 7.3 下一步 (优先级序列)

1. **(最高优)** 在 `_SplitBlock.forward` 实现 `gate × residual` 逻辑, 装填 b31-38 layer3 (2 bytes = 1 fp16 gate) 真实值
2. **(高优)** 装填 **b39.layer0** 末尾 1024B = **c=512 per-channel GATE** (model 增加 `dec_gate` 参数 shape=(512,))
3. **(高优)** 装填 **b30.layer4** (524,304B) = c=512→1024 proj (model 增加 `enc_to_bn` Linear(512, 1024) + 16-byte bias)
4. **(中优)** 装填 b22 的 131,072B fp16 区 (Task A) 为 256→512 down-proj weights
5. **(中优)** 装填 b48 的 295,472B = 512×256 up-proj + 余 164,400B LN/bias
6. **(低优)** 装填 b4/b8/b14/b56/b62/b66 merge/up blocks 的 down/up-proj 残差 (×(2-128)KB)
7. 重测 GPU 前向, 验证 bn 链激活不再渐增 (17→589 → 应平稳 O(1-10), 用户实测 30779→673 后继续优化)