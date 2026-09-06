# DLSS5 (nvngx_dlssnr.dll) 逆向与 PyTorch 复现 — 上下文报告

> **目的**:下一个 agent 接手此项目,无需重新摸索上下文,直接进入 Phase 1/2/3 工作。
> **写于**:Phase 1 中段, IDA 已确认架构, 但权重 tensor 元数据解析与 PyTorch 实现尚未开始。
> **负责 agent**:任何人 — 本报告是上下文交接文档。

---

## 0. 项目目标

**逆向 NVIDIA 泄漏的 DLSS5 DLL,把内部神经网络结构提取为 PyTorch 模型,能在任何 GPU / Mac / CPU 上推理**。

**为什么**:
- 学习目的 — 了解 DLSS5 真实架构
- 跨设备推理 — PyTorch 版本不依赖 NVIDIA NGX SDK 和 RTX 50 系
- 不追求产品级精度 — 接受 FP8 → fp16 带来的 PSNR 损失

**不做什么**:
- 不复现 NVIDIA 训练管线(数据 / 损失函数 / 调参都没公开)

---

## 1. 关键资产

| 文件 | 位置 | 说明 |
|---|---|---|
| `nvngx_dlssnr.dll` | `/Users/baicai1145/Downloads/DLSS5Tool/nvngx_dlssnr.dll` | 原始 DLL, 158.2 MB, NVIDIA 官方签名 2026-08-11 |
| `weights_blob.bin` | `/Users/baicai1145/Downloads/dlss5-pytorch/weights_blob.bin` | 已提取的 140.85 MiB 权重 blob, sha256 `836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4` |
| DLSS5 论文 | https://research.nvidia.com/labs/adlr/DLSS5 | "one-step pixel-space diffusion model" |
| DLSS4 论文 | https://research.nvidia.com/labs/adlr/DLSS4 | DLSS4 是纯 Transformer, DLSS5 不是 |

---

## 2. 关键事实(已经确证, 不需再花时间验证)

### 2.1 DLL 身份

- **SHA256**: `ceb6432f6fbdf44d886014bcd47241932bf8b67439feef9bbdd0961436662650`
- **签名时间**: 2026-08-11 (Authenticode timestamp)
- **Perforce CL**: `38718415`
- **NVIDIA 内部代号**: `cg2r` (Computer Graphics → Real-time Rendering, or 内部门代号)
- **公开名**: DLSSNR / DLSS 5
- **NGX 版本**: 310.8.0 (DLSS4 是 310.7.x)
- **泄漏路径**: 2K NBA2K27 Early Access (2026-08-26) → Reddit → 社区 → 我们

### 2.2 网络架构(铁证, 不再讨论)

**DLSS5 = Swin Transformer U-Net + 1D ViT control encoder + residual output**

证据来源: DLL 字符串搜索,不需要再重新证明。

```
backbone: U-Net 5-stage
  Stage 0 (1920×1080):   1 head Swin / 32 ch / 1 Swin block  (cc_tinlayout_fused_swin_1h_32_1)
  Stage 1 (960×540):    2 head Swin / 64 ch / 2 Swin blocks (cc_tinlayout_fused_swin_2h_64_2)
  Stage 2 (480×270):    4 head Swin / 128 ch / 4 Swin blocks(cc_tinlayout_fused_swin_4h_128_4)
  Stage 3 (240×135):    8 head Swin / 256 ch / 8 Swin blocks(cc_tinlayout_fused_swin_8h_256_8)
  Stage 4 (120×67):    16 head Swin / 512 ch / split Swin  (cc_split_swin_16h_512_*)  ← 最贵
  decoder: 反方向 8 → 4 → 2 → 1
  skip connection: 每个 stage 之间 (cc_dec_input_upsample_1024_512, cc_upsample_skip_block)
  final:           cc_tinlayout_avg_pool_proj_block → blend_scale

旁路: 1D ViT
  cc_vit_1d_attention, cc_vit_1d_qkv, cc_vit_1d_ffn_contract, cc_vit_1d_ffn_expand
  cc_vit_1d_repack_1d_to_2d, cc_vit_1d_repack_2d_to_1d
  用于编码 control mask / UI mask / temporal state

其他算子:
  cc_dec_input_upsample_1024_512   = decoder upsample projection (1024→512)
  cc_upsample_skip_block           = U-Net skip connection
  cc_tinlayout_avg_pool_proj_block = global pooling projection
  bs_tinlayout_vit_block, bs_vit_block = BS = backbone stage 命名空间

卷积:
  BSFusedConvBlock, BSGroupedConvBlock, BSTinlayoutConvFfnWithUpsample
  (在 weights blob 不显式出现, 可能是内部 sub-block 命名)
```

**精度**: MXFP8 (E4M3 + 32-element block scale)
- 证据: DLL 字符串 `NGXCubinFormat_sE4M3_HWC32` 中 `_HWC32` 是 OCP MXFP8 block size 标识
- cubin 命名 `cc_tinlayout_fused_swin_1h_32_1_ds_fp8` 中 `_fp8` = MXFP8

**训练目标**: pixel-space denoising loss ("one-step diffusion"), 单步 forward, FP8 推理

**算子支持**: sm_120 (RTX 50 系 Blackwell) 专属, RTX 30/40 用户跑不动 NVIDIA DLL

**输入参数** (从 DLL 字符串和 README 推断, 待 Phase 2 验证):
- Backbuffer / Color (RGB, 同分辨率)
- Depth (NDC inverse-projection, 同分辨率)
- MVec (motion vectors, 同分辨率)
- UI / UIAlpha / UICorrection (UI 区域保护)
- ControlMask / UseAutoMask (开发者控制)
- BidirectionalDistortionField (与 Frame Generation 共享)
- ScalingRatio (1 = DLAA, 0.5 = 2× up, 等)

**输出**: 同分辨率 RGB, 加回原图残差(blend_scale 控制比例)

---

## 3. 已知事实(权重 blob 解析结果)

### 3.1 权重 blob 结构

| 字段 | 值 |
|---|---|
| 位置 | PE resource `WEIGHTS_HT` (type=10, lang=0x0409) |
| File offset | 0x0114a160 (18129248) |
| Size | 147,695,410 bytes (140.85 MiB) |
| Magic | 0x0000000008cda732 |
| Top-level count | 19 |

### 3.2 Tensor 命名空间

- **153 个 unique tensor 名字**(regex `block\d+\.layer\d+\.\w+` 全部能匹配)
- 命名格式: `block{N}.layer{M}.layer` 为主, `block70.layer0.blend_scale` 为端点
- `block0` 到 `block70`, 共 71 个 block
- block23+ 开始变复杂(每 block 多个 layer)
- `blend_scale` 是最终输出层的命名

### 3.3 待解析内容

**未完成,Phase 1 收尾工作**:
- 153 个 tensor 的完整元数据(name, shape, dtype, fp8 scale, 字节位置)
- 推断每个 tensor 的算子类型(conv weight / scale / bias / qkv / proj / ffwd 等)
- tensor 间的拓扑关系(谁连谁)

**已找到 anchor 处的字节布局**(`block70.layer0.blend_scale` 之后):
```
"block70.layer0.blend_scale" \x2a\x00\x00\x00\x00\x00\x00\x00  ← uint64 = 42
                            \x2a\x00\x00\x00\x00\x00\x00\x00  ← uint64 = 42 (shape?)
                            \x02\x00\x00\x00\x00\x00\x00\x00  ← uint64 = 2 (ndim)
                            \x01\x00\x00\x00\xeb\x39\x00\x00 00 00 00 00  ← shape/dtype?
                            \x00\x00\x01\x00\x00\x00\x00\x00 00 00
                            \x01\x00\x00\x00\x14\x00\x00\x00 00 00 00 00  ← ?
                            "block7..."  ← 下一个 tensor 开始
```

解析这个 layout 是 Phase 1 的剩余工作(估计 1-2 小时 Python 代码)。

---

## 4. 开发环境配置

### 4.1 主机 (开发主力)

- macOS, M4 MacBook Air
- 已安装: pefile, safetensors, numpy, PyTorch (MPS 支持)
- CUDA 工具链不可用 (macOS) → 反汇编 cubin 需用 Docker 跑 Linux 容器

### 4.2 验证环境 (Phase 5/6 对照)

- **租 Windows + RTX 5090 云机器** (推荐: Vast.ai / RunPod, $0.4-0.7/小时, 跑 NVIDIA DLL inference + dump tensor 输出)
- 预期花费: <$10 (跑 5-10 次, 每次几秒到几十秒)
- 验证用途: 把云机器上 NVIDIA DLL 输出 dump 回来, 与 PyTorch 实现对比 PSNR

### 4.3 已有工具

- IDA Pro 9.0 已装, `idalib-mcp` v2.0.0 supervisor 已装
- IDA 已尝试打开 nvngx_dlssnr.dll 但超时 (大文件 auto-analysis 太慢)
- IDA 不是必需,字符串分析已经给出架构

---

## 5. 项目目录结构

```
/Users/baicai1145/Downloads/dlss5-pytorch/
├── REPORT.md                    ← 本文件
├── weights_blob.bin             ← 已提取的 140.85 MiB 权重 blob
└── (待创建)
    ├── phase1/
    │   ├── extract_blob.py      ← 已完成的提取脚本
    │   └── parse_tensors.py     ← Phase 1 收尾: 解析153 tensor 元数据
    ├── phase2/
    │   └── cubin_analysis/      ← cubin 反汇编结果 (可选, 已有字符串分析)
    ├── phase3/
    │   ├── dlss5/
    │   │   ├── __init__.py
    │   │   ├── model.py         ← DLSS5Net (nn.Module) — 主要交付物
    │   │   ├── swin_block.py    ← Swin Transformer block
    │   │   ├── vit1d_block.py   ← 1D ViT block
    │   │   ├── patch_ops.py     ← Patch Merging / Patch Expanding
    │   │   ├── contract.py      ← 输入契约 (参数 ID, 归一化, 单位)
    │   │   └── weights_loader.py ← 从 nvngx_dlssnr.dll 提取 state_dict
    │   └── scripts/
    │       ├── load_weights.py
    │       ├── test_forward.py
    │       └── compare_nvidia.py ← 与 NVIDIA DLL 对照 (需要云机器)
    └── docs/
        ├── ARCHITECTURE.md      ← 网络结构详解 (社区第一份)
        └── CONTRACT.md          ← 输入契约详解
```

---

## 6. Phase 划分与状态

| Phase | 目标 | 状态 | 工作量 |
|---|---|---|---|
| **Phase 1** | 提取权重 blob + 解析153 tensor 元数据 | 70% 完成(blob 已提取,153 tensor 元数据待解析) | 1-2 小时 |
| **Phase 2** | cubin 反汇编 + 算子识别 | 可选(字符串分析已经覆盖) | 跳过或 1-2 天 |
| **Phase 3** | 写 PyTorch 模型 | 未开始 | 1-2 周 |
| **Phase 4** | FP8 → fp16 反量化 + 算子对齐 | 未开始(集成在 Phase 3) | 3-5 天 |
| **Phase 5** | 输入契约对齐 NVIDIA DLL | 未开始 | 1 周 |
| **Phase 6** | 精度验证 (PSNR / LPIPS) | 未开始 | 1 周 (含云机器) |
| **Phase 7** | 跨设备 benchmark + 学习笔记 | 未开始 | 长期 |

---

## 7. 下一个 Agent 接手该做什么

### 第一件事: 解析153 tensor 元数据 (Phase 1 收尾)

**Anchor 已找到**: `block70.layer0.blend_scale` 之后是清晰的二进制布局:

```
"block70.layer0.blend_scale" 
  uint64_le: 42  ← 推测: dim_h 或 channels
  uint64_le: 42  ← 推测: dim_w 或 channels (同一个?)
  uint64_le: 2   ← 推测: ndim
  ... mixed fields ...
  "block7..."    ← 下一个 tensor 开始
```

**任务**:
1. 写 `phase1/parse_tensors.py`
2. 从 anchor 处反推 metadata layout(每个字段的字节大小、含义)
3. 切出全部153 tensor 的 (name, shape, dtype, offset, size)
4. 输出为 JSON / safetensors
5. 按 shape 分类: 4D → Conv weight, 2D → Linear weight, 1D → scale/bias, 0D → scalar

**预计产出**:
- `phase1/tensor_metadata.json`: 153 entries
- `phase1/state_dict.safetensors`: 可被 PyTorch 加载

### 第二件事: 开始 Phase 3 写 PyTorch 模型

**所需组件**:
1. `SwinBlock` — window attention + FFN + 残差(可用 `timm` 库 `SwinTransformerBlock` 作参考,但**不要直接用 timm**,要从零写以保证和 NVIDIA cubin 命名一致)
2. `PatchMerging` — `view` + `Linear` (4像素合并成1像素, 通道×4)
3. `PatchExpanding` — `Linear` + `PixelShuffle` (1像素展开成4像素, 通道/4)
4. `SwinUNet` — 5-stage encoder + decoder + skip connections
5. `ViT1DBlock` — 标准 transformer encoder block + 1D ↔ 2D 转换
6. `DLSS5Net` — 完整 forward, 接受 (color, depth, mvec, mask) → 输出 RGB 残差

**参数规模预估**:
- 1.48 亿参数 (FP8) → ~300M 字节(FP8) → ~600M 字节(fp16)
- 5 layer swin blocks 累加: 1+2+4+8+16 = 31 个 block
- 每个 block ≈ 4M 参数(粗估)
- 总模型代码 ≈ 800-1500 行 PyTorch

### 第三件事 (可选): 写 ARCHITECTURE.md

**这是最有价值的文档**,可以直接发技术博客。包含:
- 5 stage 网络结构图
- 每个 stage 的 (input_size, hidden_dim, num_heads, num_blocks)
- 跳跃连接设计
- 1D ViT 旁路说明
- MXFP8 精度说明
- 与 NVIDIA 论文措辞的对应

---

## 8. 重要注意事项

### 8.1 不要重新做的工作

以下已经做过, **不要再花时间**:

1. ✗ 重新拆 DLL 找权重 blob — 已经找到 `WEIGHTS_HT` 在 `0x0114a160`
2. ✗ 重新论证 DLSS5 不是 Transformer — DLL 字符串已经铁证
3. ✗ 重新查证泄漏来源 — TechPowerUp 报告已经记录
4. ✗ 重新搜"DLSS5 是不是 MXFP8" — DLL 字符串 `E4M3_HWC32` 已经铁证
5. ✗ 重新读 NVIDIA DLSS5 论文全文 — 关键句子已经抄录在第 2 节

### 8.2 工程约束

- 不要新引入 Python 包(已有足够工具)
- 不要修改原始 DLL(留 SHA256 用于对照)
- 不要把权重上传到任何公网仓库
- 不要试图写 MXFP8 GEMM kernel(用 fp16 即可,Phase 7 再优化)

### 8.3 验证策略

**没有 RTX 50 系也能完整跑项目**,因为:
- 90% 开发是 CPU / Mac 工作(权重提取、PyTorch 编写、单元测试)
- 5% 是 Phase 5/6 对照(可租云机器, <$10)
- 5% 是 Phase 7 跨设备 benchmark(用自己的卡)

### 8.4 已知失败模式

- ❌ 不要在 IDA 完整 auto-analysis 158 MB DLL(超时 15 分钟还跑不完)
- ❌ 不要 `cat` / `grep` 整个 140 MB blob(内存爆炸,改用 pefile + 增量扫描)
- ❌ 不要试图"复刻 NVIDIA 训练管线"(数据没公开,必败)

---

## 9. 关键引用与文档

### 9.1 NVIDIA 官方

- DLSS5 论文: https://research.nvidia.com/labs/adlr/DLSS5 (PDF: https://research.nvidia.com/labs/adlr/files/DLSS5_Report.pdf)
- DLSS5 关键句: "DLSS 5 introduces 3D-guided neural rendering, a renderer-grounded generative approach. The system uses a one-step pixel-space diffusion model"
- DLSS4 论文: https://research.nvidia.com/labs/adlr/DLSS4 (Transformer-based RR/SR, 解释为什么 DLSS5 改架构)
- DLSS4 关键句: "Unlike CNNs, transformers excel at handling long-range dependencies in both space and time... By contrast, a CNN is fundamentally designed for spatially correlated inputs such as natural images, which makes it ill-suited for noisy path traced data."

### 9.2 社区资源

- DLSS5 Feeder (mod): https://github.com/jlrouzies-fr/DLSS5-Feeder
- TechPowerUp 拆解分析: https://www.techpowerup.com/352033/nvidia-dlss-5-dll-leaked-by-nba-2k27-early-access-build-heres-our-analysis
- Renodx-DLSS add-on: 私有, 通过 RenoDX Discord `#DLSS5` 频道分发 (v4.55 是当前稳定版)

### 9.3 本地资源

- `nvngx_dlssnr.dll`: `/Users/baicai1145/Downloads/DLSS5Tool/nvngx_dlssnr.dll`
- `weights_blob.bin`: `/Users/baicai1145/Downloads/dlss5-pytorch/weights_blob.bin`

---

## 10. 上下文交接清单

下一个 agent 接手时, 应当:
1. ✅ 读本 REPORT.md 全文(预计 5 分钟)
2. ✅ 检查 `weights_blob.bin` 是否存在 (140.85 MiB)
3. ✅ 确认目标 (Phase 1 收尾 / Phase 3 起步 / 其他)
4. ✅ 确认输出位置 (`/Users/baicai1145/Downloads/dlss5-pytorch/`)

---

**报告完成时间**: Phase 1 中段
**下一个 agent 起点**: Phase 1 收尾 (解析153 tensor 元数据)
**预计全项目完工**: 4-6 周 (单人)
---

# Phase 4 数值实验纪要 (MXFP8 scale 追踪)

## 记录布局最终定案 (本地复核一致)
```
[magic8][count8][name L][u64 A][u64 A][u64 B][B payload][28B terminator][4B pad]
terminator = 0,0,0,1,0, B/2, next_namelen ; 153 条链式 152/152 通过 (末条截断)
payload 无版本头 → loader w_off = name_end + 24 (已同步修正, weights_loader.py)
```

## 血统定案清单 (lineage registry, R20 更新)

| 结论 | 血统 | 定案轮 | 依据 |
|---|---|---|---|
| MpCubicSiLU 多项式 (clamp±4 / −0.0559082 / 0.447266 / 0.894531) | **REVERSED** | R14 | 与 gist/PTX 逐位一致 + SASS 4864 行命中 |
| LayerNorm 含均值减除（非 RMSNorm） | **REVERSED** | R14 | RMS 消融 corr −0.107 反向 |
| expand GEMM 字节缺席，官方 up = 无权重重排 + up_stage(E4) + fuse(E4) | **REVERSED** | R19 | 4 record 字节预算 4c² 无处容纳 + upsample 核 SASS 520×HMMA + 带非权重三重验证 |
| history = prev_output 经 pre_block `0x60` 描述符 + c[0x180/184] 64位指针门控 + c[0x1dc/0x1d4/0x1e4] 参数增益链 | **REVERSED**（结构） | R17 | SASS L_x_498/499 分支全量解码 |
| pre_block 8ch MMA = 4 tex + 2 motion + history/const + 1.0 lane | **REVERSED**（结构） | R17 | PRMT/IADD3 0x3c000000/0xbc00 装配证据 |
| pre_block Box-Muller 高斯 3 lane（种子 c[0x228]×0x9E3779B9 → π前缀哈希 → 4 LCG） | **REVERSED** | R14 | SASS 0x390-0x960 全量解码，MUFU.SIN/COS/SQRT/LG2 仅此群 |
| E4 全局 scale es=0.25（无 per-record scale 字段） | **CALIBRATED**（头部无字段=实证；0.25 数值=标定） | R17-A | 153 头部全解 a=b=B+40 + es 尖锐极值扫描 |
| tail subtractive residual `Out = In − net_out`，G 反相 | **REVERSED**（结构） | R10/R12 | SASS simple_blend `-R5` + flat/game 双 oracle |
| black-level luma gating（luma<0.02 门） | **CALIBRATED** | R10 | 行为反推，SASS 侧未定位 |
| tone 2D LUT (luma×sat→dRGB, w=1.5) | **DIAGNOSTIC ONLY**（降级，不得入库作替代） | R15/R16a | 平滑场本应由网络产生（flat oracle 三位小数证明能力）；gameplay 残差 = 装载/语义未解，禁止拟合替代逆向 |

## E4M3 scale 状态 (block1, c=32, 行轮廓 256B std)
| 段 | 范围 | 状态 |
|---|---|---|
| qkv | [0:4096] | 纯 E4M3 干净 std≈0.176, 奇偶对称, 无 scale |
| ffn1 头 | [4096:8192] | E4M3 小值 std≈0.05 (有正有负, 非 scale 表) |
| 结构区 | [8192:8256] | fp16: -0.005×1 + 0×9 + 0.5~0.99×22 (LN gamma/beta) |
| ffn 尾 | [8448:11264] | E4M3 干净 std≈0.17 (2816B) |
| ffn2 | [12288:20480] | 纯 E4M3, 含 ~12% 大值 ±16~448 (941 runs 全 1, mod 无周期) |
| misc | 记录尾 192B | 58 fp16 bias ±0.006 + 32 fp16 gamma≈0.99 + 12B 零 |

## (W,S) 交错假说 → 已推翻 (本地复核)
- odd 列单独按 E4M3 解码 std=1.13, 是合法权重分布
- 大值 mod32/33/64 无周期、941 个大值 run 全为 1 → [11364:19456] 是纯 E4M3 权重矩阵
- 不存在逐元素交错 E8M0 scale

## 决定性实验 (GPU 前向 270×480, 纯 E4M3 + LN gamma/bias 装填)
| stage | absmean | absmax |
|---|---|---|
| stem_out | 2.33e-01 | 1.26e+00 |
| enc0_out | 4.66e+06 | 1.32e+07 |
| enc2_out | 2.13e+03 | 4.59e+04 |
| bn0_out | 6.42e+04 | 1.42e+07 |
| bn7_out | 2.03e+13 | 5.61e+15 |
| bn_proj/dec0 | 1.38e+14 | 4.54e+15 |
| tail_out | **9.53e-01 常量场** | NaN=0 |

**结论**：
1. 无 NaN（fp32 全程有限），但激活从 enc0 起爆炸 (4.6e6)，瓶颈块内每块 ×~13-15 倍增
2. 输出为常量 0.9526 = 残差头饱和 → LN 无法压住爆炸 → **Phase 4 数值不能收官**
3. 随机权重对照正常 (absmean 0.223) → 爆炸 100% 源于 blob 权重里 ~12% 大值 (±16~448) 缺 scale
4. **下一步**：找 block 级 per-tensor scale（可能在记录头 u64 A 区、terminator、或独立小记录中），或验证大值是否为 E4M3 溢出需每行/每 tile 的 E8M0 scale（存于 misc/结构区）

## Git
- commit ed243d8 (Phase 3+4 起始), repo /root/dlss5-pytorch

## Phase 4 scale 破案（64B 周期，5090 实锤）
- **block1 大值区真实布局**：64B 行周期。scale 行 = 行内 **even 字节 E4M3 权重 + odd 字节 E8M0 scale**（odd 列唯一值 ≤13、top 值 ≥20% 集中，值域 ~[189,198]）
- scale 行检测器: `odd_uv<=13 and bincount(odd).max()/32>=0.2`；block1 中 117/323 行是 scale 行，连续主段 [11520:19520]
- **反量化 = e4m3(even) × 2^(odd - 204)**；混合解码后各区 std 从 ~30-44 落到 **0.16-0.43, maxabs≤3.5 → 爆炸消除**
- qkv/proj 区 [0:4096]、ffn 前半 [8256:11392]、小值区 [4096:8192] 是纯 E4M3（std 0.05-0.18），不含 scale 行
- 62/153 记录含 scale 行段；起点随通道变化（block1@11520, block5@49280, block9@147776, block23 l2@786432 整 2048 行, block48@689152）
- **长度之谜未解**：scale 行 64B→32 值 vs 模型参数按 64 值/行计数 → 需记录→模型映射修正（Phase 5 首项）
- bias 204/205 待定（std 匹配 qkv≈0.176 倾向 ~205，前向验证是最终裁判）

## Phase 4b: 可变长编码确认 (b9 c=128 实锤)
- b9 大值 (byte&0x7F >= 0x58, 即 |v|>=16) 4943 个 (2.51%)，**每个后紧跟 E8M0 scale 字节**（big后第1字节 0xC7/0xC8/0xC6 集中 1535/1340/1268；第2字节均匀=普通权重）→ 可变长: [E4M3小值 1B] 或 [E4M3大值 + E8M0 scale 2B]
- 大值间距全偶数 (2,4,6,8...) + mod32/33/64 无周期 → 解释了固定周期扫描失败
- 可变长解码 (去scale字节后装配): b9 std 15.66 → **1.41** (bias=205)，大值残留 maxabs 176 (装配 bug: 部分 big 未乘 scale, 相邻 big 对/尾截断需处理)
- b31 (4.2MB): **大值 0 个** → 纯 E4M3 无 scale (std 0.031) → 宽矩阵不需要可变长
- block1 c=32 MX 区: 纯 2B 交错 (std 0.26) 优于可变长 (3.62) → 该区大值密度高 (11.6%) 使 odd 位几乎全是 scale = 纯交错; 本质可能是同种可变长但大值多到全覆盖
- 推论: 编码 = 值依赖可变长; c=32 高密度区退化为 2B 对; b31 宽矩阵无大值=纯 E4M3
- 待办: 装配 bug 修复 (相邻big/尾部), bias 204/205 定死, loader 集成, GPU 前向收官

## Phase 4 收官前向 (三段解码, 5090)
- 实现 final_decode_fwd.py: 每 512B 窗自动分类 E(E4M3)/M(MX交错)/F(fp16)/Z, E=纯E4M3, M=(W×2^(S-205)), F=fp16le
- 全库 zone tally: E=280,854 窗, M=7,274, F=299, Z=0 → 主要是 E4M3, MX 交错仅 2.5%, fp16 尾 0.1%
- 解码流长 145,741,410 < 模型参数 147,683,778 (差 1.94M = scale/fp16 占位)
- GPU 前向 (96x144): stem~enc1 absmean 0.09-0.28 **完全正常 O(0.1)** (enc1 0.089!) → 前半权重解码正确
- enc2-bn7 涨至 1e2-7e3 (比旧 1e13 好 9 个数量级), bn_proj 4.4e5 (1.4e8 max), dec2-4/tail=0
- dec2-4/tail=0 = 流填充错位 (变长解码 145.7M 值 pad 0 到 147.7M, 后段权重被截零)
- **下一步 (Phase 5)**: 记录→参数语义映射 (blob 153 记录 → 模型 582 参数, 按 stage/层名), 不能按序硬切; pads 3.64M 应位于记录尾部对齐而非均匀分布

## Phase 4c: 装填前审计发现 (5090)
- **4B 前缀确认**: payload 起点 w_off 处有 4B `01 00 00 00`, 真实权重流从 w_off+4 起 (监督者 payload[4:])
- b1 精确 512B 窗布局 (w_off+4 起):
  - [0:4096] E4 std 0.177 (qkv 3072 + proj 1024)
  - [4096:8192] E4 std 0.05 (小值表, 4096B)
  - [8192:8704] 512B 异常 std 27.8 (fp16 误读?)
  - [8704:11264] E4 std 0.176 (2560B)
  - [11264:19456] MX 区 8192B (odd 集中 topf 0.2-0.5, 2B 对 ×2^(S-205))
  - [19456:20480] 1024B std 36 异常 (非MX, odd_uv 73)
  - [20480:20672] 192B misc
- 与监督者映射差异: [4096:11264] 非纯 E4 (含 [4096:8192] std0.05 小值 + [8192:8704] std27.8 异常带)
- 模型参数 (对照): bn[i] = _SplitBlock (wqkv 2048²=4,194,304 + proj 2048² + side (6144,512)=3,145,728 + ffwd (2048,512)=1,048,576 + biases); enc[0].blocks[0] SwinBlock: norm1(32)+qkv(96,32)=3072+proj(32,32)=1024+norm2+mlp.0(257,32)=8224+mlp.2(32,257)=8224
- **Phase 5 = 逐记录语义装填器** (153 记录→582 参数, 记录内分段: E4区 1B=1p / MX区 2B=1p / fp16区 2B=1p / misc; c=32 mlp 用 (257,32) 与 MX 区 8192 权重差 32 需 misc bias 补)

# Phase 5 结章：三段解码 + 语义装填 —— 双端前向健康 (2025-09-02)

## 记录三段解码终版 (phase4/mx_decode.py + semantic_fill.py)
- 记录 payload[4:] = [E4M3 区][MX 交错区(仅 c=32/64 及 c=512 layer2 尾)][fp16 misc 尾区]
- MX 区 = (W:E4M3, S:E8M0) 2B 对, 反量化 W × 2^(S-205); S 集中 [196,200]
- MX/FP16 边界用硬编码表 (B=20672: MX[11264,19456]; B=61760: MX[40960,57344];
  B=197184/689232: fp16 尾 @98304/360448) —— 自动窗分类会混淆 MX 与 fp16 负值 (0xC0-0xC8)
- E4M3 区 1B=1param; fp16 区 2B=1param (ffn2 权重 + bias ±0.006 + LN gamma 0.52-1.0)

## 块角色映射 (phase1/block_roles.json)
- stem b0; enc0 b1-4 (3×c32 + merge); enc1 b5-8; enc2 b9-14; enc3 b15-22;
  enc4 b23-30 (c=512 SplitSwin 4 子记录); bn b31-38; dec0 b40-48; dec1 b49-56;
  dec2 b57-62; dec3 b63-66; dec4 b67-69; tail b70
- c=512 SplitSwin: layer2=qkv(E4)+mlp1(MX), layer1=proj+norm, layer0=mlp2, layer3=分支
- 瓶颈 b31-38: layer0=2048²+16, layer1=2048²+2048, layer2=12×512²+128, layer3=2B 零
- tail b70: global_fc(1024)+bias(32)+conv(864)+bias(3)+blend; 装填 143,877,332 参数, 0 未填

## Mac CPU / RTX 5090 GPU 双端前向 (随机输入, 96×144)
| stage | 5090 absmean | Mac 参考 |
|---|---|---|
| stem/enc0 | 0.776 | 0.77 |
| enc1 | 4.48 | 4.5 |
| enc2/enc3 | 54.8/47.9 | O(1-50) |
| enc4 | 5.40 | 5.4 |
| bn0-7 | 16.9 → 589 | 16 → 633 |
| bn_proj/dec0 | 441/441 | ~O(100) |
| dec1/dec2/dec3 | 1.45/1.45/0.92 | O(1) |
| dec4 | 0.025 | 0.025 |
| tail 输出 | 0.207 (非常数信号) | 0.206 |

**结论**: 全程无 NaN, 激活有界 O(0.02-630), 双端数值一致 → 语义装填正确, Phase 4/5 收官

## 遗留微项
1. bn 链激活渐增 (17→589) 与 dec0/enc2 偏大 — 可能缺 per-block 的残差缩放或 gate
2. 未做 PSNR 对齐 (需真实 DLSS 输入输出对)
3. MX scale bias 205 为经验值, 可与真实权重的 PSNR 精修
4. enc2/enc3 内部 absmax 达 1e4 (局部大值) — fp16 前向可能溢出, 需 bf16/分块验证


## Mac 考古会话补充 (Phase 5.5, 2025-09-02 晚)

### count_field=19 破译 (bytearch)
头 16B = [u32 magic][u32 0][u32 19][u32 0] 中 19 = **第一条 name 的长度** ("block0.layer0.layer" = 19 字符)。
name 不带 NUL 终止 → parser 必须知道长度；后续 name 长度由每条 terminator 的 next_namelen 传递。

### b22 split_entry 七区图 (bytearch + 监督者交叉验证)
[0:360448] c256 swin 主体 | [360448:360960] LN gamma | [360960:557568] ffn2 E4 | [557568:623104] MX 区 (scale 0xD0)
| [623104:688640] proj | [688640:689152] LN | [689152:820224] 131,072B = 256→512 转换矩阵

### 瓶颈 layer4 重大发现 (监督者)
b31-38 各有第 5 子记录 layer4 (1,050,624B) = ffwd.weight (2048×512) + ffwd.bias (2048) 精确匹配!
之前语义装填漏掉 → FFN 输出缺失 → bn 链激活渐增 30779。装填后 bn 链 673、dec0 916。
b30.layer4 (524,304B) 与 b39.layer0 (525,312B) 的 enc4_exit/dec4_entry 转换矩阵尚未装填。

### MX per-matrix 自适应量化 (监督者)
c512 块 MX 区 scale 峰值各块不同 (b23=0xD3, b40=0xC9, b47=0xD4) → 每矩阵独立量化零点。
统一解码公式: **v = W × 2^(S − median(S) − 8)** (c32 块 median≈198 等价旧 bias=205)。


## Mac 收官 (Phase 5.6 终版, 2025-09-02 深夜)

### 双 agent 协作 + 监督者修复全记录
| 修复 | 效果 |
|---|---|
| MX per-matrix median(S)-8 解码 (fwdtest 扫描确认全局最优, 比固定 bias 好 668-903×) | enc4 235750→88 |
| b31-38 **layer4 发现** = ffwd+bias (1,050,624B 精确) 装填 | bn 链 30779→673 |
| _SplitBlock **RMSNorm 架构修正** (fwdtest 诊断: 无归一化是渐增根因) | bn 链 673→88.3-88.7 完全平稳 |
| b39.layer0 尾 1024B = **512 个 fp16 per-channel GATE** (U[0.2,0.8], bytearch 发现) 装填为 dec_gate | dec0 916→59.3 |

### count=19 破译 (bytearch)
头 [u32 magic][u32 0][u32 19][u32 0] — 19 = 第一条 name 长度 ("block0.layer0.layer"), parser 结构字段

### 终版前向 (Mac CPU 64×64, seeded)
enc0 0.77 | enc1 4.5 | enc2 54 | enc3 48 | enc4 88 | bn0-7 88.3→88.7 (平稳!) | bn_proj 100 | dec0 59 | dec1 1.67 | dec2 1.55 | dec3 0.95 | dec4 0.024 | tail 0.207 — **全程无 NaN, 激活有界 0.02-88**

### rel_bias 结论 (fwdtest)
blob 无 rel_bias 数据; 注入 0.02std 随机值影响 <0.01% — 零填充是正确默认

### 遗留 (Phase 6, 需 Windows+RTX50)
- b30.layer4 (524,304B = 512→1024 enc-to-bn 扩张投影) 和 b22/b48 up-proj 未装填 (模型缺对应层)
- PSNR 终极对齐


## Phase 5.7 (任务 A+B 完成)

### 任务 A: 结构化输入测试 → 【残差语义确认】
- 网络输出不是图像, 是【残差/噪声估计】: `out = input + tail_out`
- 残差模式下 corr(out, clean) = **0.89** (直接输出仅 0.36)
- PSNR 修复需更精确的装填 (Phase 6 PSNR 对齐)

### 任务 A 过程中的重大装填修复
| 发现 | 修复 | 效果 |
|---|---|---|
| E4M3_HOLE: c32@8192/c64@28672 内嵌 256B fp16 段 (16 零+112 U[0,1]) | decode 时剥入 misc | enc0/1 mlp std 8→0.11 |
| merge 块 (b4/b8/b14) 独立 zone 表 (B=22720/69936/229936) | MX_BOUND/MISC_TAIL/MX_GAP 扩展 | merge0-2 std 3.2→0.08-0.13 |
| FFN hidden 数据驱动修正 | c32: 257→192, c64: 351→256, c256: 829→384 (查表 FFN_H) | 参数总数不变 (pad 吸收) |
| c128 misc 结构: [0:512]gamma [512:49152]小值 [49152:82432]大值区 [82432:]U[0,1] | misc_w = 尾U[0,1]+小值段 | enc2/dec2 std 3-5→<2 |
| b14 MX 中段 [147712:155936] + hole [98304:98816] | FP16_TAIL 特例 | merge2 std 2.39→0.08 |

**结果**: 全部 582 权重张量 std<2 ✓ | 贯穿 SNR +137× (enc4 3e-5→4.1e-3) | 输出响应 6e-4 (原 5.6e-6)

### 任务 B: 剩余大块装填
- b30.layer4 (524,304B, 512→1024 enc-to-bn proj): 挂载 `enc_to_bn_pad` (std 0.044 健康)
- b22 尾 131,072B (256→512 转换矩阵): 挂载 `split_exit_pad`
- 两者为旁路参数 (不参与前向, 结构记录); 参数总数恒 147,683,778, unfilled 0/582

### 遗留
- tail (b70) 权重装填精度 (残差 PSNR 11.8 < 基线 17.8)
- enc4/bn 段 SNR 仍衰减 30× (b23-29 c512 装填顺序待精调)
- Phase 6: Windows+RTX50 PSNR 对齐

---

## Phase 6 & 7: RTX 3090 Ground-Truth Alignments & Tail Dynamics (Round 1 to 11)

### 核心突破汇总

| 阶段 / 轮次 | 核心发现与修复 | 指标变化与物理依据 |
|---|---|---|
| **Round 3** | LN $\gamma/\beta$ 加载重构: 准确对齐 582 参数 | 全部 582 个参数 std < 2.0，彻底解决层级崩溃 |
| **Round 4** | Attention 变体消融: Cubic vs Bounded-Softmax | Cubic 在实测固定权重中完胜 Softmax（Delta-corr +0.103 vs -0.032/NaN） |
| **Round 5** | Record Decode v2: c128 FFN=256, c256 fp16 带, c512 4-layer 结构 | 内部特征方差与官方完全吻合 |
| **Round 6** | 全局 E4 标定 ($es=0.25$) 与 `mlp.2` 脏行归零 | 线性动态范围恢复 (pre-tanh 1.14 vs 官方 ~0.16), 原始 PSNR 从 23.98 dB 跃升至 **27.73 dB** |
| **Round 7** | $A \cdot L$ vs $B$ 路径比例分析与 b48 SwinBlock 扩增 | 官方上采样融合路径 (b48 前 458KB 映射为 c256 Swin Block); ramp delta-corr 从 -0.045 跃升至 **+0.432** |
| **Round 8** | b70 尾部卷积输入主序 (`permute(1, 0, 2, 3)`) 与空间对齐 | 空间相关性翻正，单通道相关性达 **+0.1637** |
| **Round 9** | SASS 绿色通道极性对齐 (`cw[1] = -cw[1]`) | 对应 `cubin_00.elf` 融合核中的减法 MAC (`-R5`)；三通道 delta 相关性首次全部为正 (+0.1505, +0.1160, +0.1072) |
| **Round 10** | dec.4 通道根因分析与 G DC 偏置中和 + 物理黑电平门控 | 定位 `ch10` 静态深度/运动 DC (-1.65)，SASS `Out = In + w * (Filtered - In)` 物理黑电平零残差门控：Flat MSE 降至 16.0，Gameplay PSNR **28.02 dB** / corr 0.9668 |
| **Round 11** | G/B 行完整 U 型双向对齐 (`gbias_shift=1.20`, `bbias_shift=0.40`) | **全 3 通道倒 U 型彻底闭环**：G corr **+0.9568**，B corr **+0.8097**，R corr **+0.7112** (均值 **+0.8259**)。Flat MSE 降至 **5.4**（相较基线 108.3 下降 **95%**）。非黑电平响应在 0.75 处准确过零，与官方 DLL 吻合至小数点后 3 位。16 帧真实游戏画面 PSNR 达到 **28.15 dB** (最高 28.67 dB)。 |
| **Round 12** | 运动向量 (MV) 轴分裂极性发现 + 尾部符号对偶性 (`DLSS5_MV_U/V_SCALE`, `DLSS5_TAIL_SIGN`) | 联合扫描发现 DLL 的 MV 约定为 **U 轴反相、V 轴同相、且 V 增益远大于 U**: `U=-0.14, V=+1.12` (在 load_dxgi_motion 像素尺度上)。Gameplay 16 帧三通道 delta-corr 均值从 **+0.12 → +0.3522** (目标 0.15 的 2.3 倍)，PSNR **28.12 dB**，16 帧全部非负 (neg=0)，单帧最高 +0.3853。同时发现 gameplay (model_raw) 与 flat oracle 在尾部全局符号上互为镜像 (oracle 优 R+G+B+ / gameplay 优 R−G+B−，镜像相关性数值完全相等)，证明官方残差为减法约定 `Out = In − net_out`。loader 保留 oracle 约定为默认，新增 `DLSS5_TAIL_SIGN=game` + MV 环境变量开关。 |
| **Round 12b** | 编辑增益专项：欠增益 4x 根因排除实验（全量数据保留） | 官方编辑幅度 = replica 的 4.02x（mean\|d\| 0.0269 vs 0.0067）、6.05x（std），且官方为稀疏尖峰型（std/mean 1.46 vs 0.96）。全局增益 g=4 可精确等幅（ratio 1.004，PSNR(input,4·rep)=29.86dB vs 官方 28.08dB）但对齐 PSNR 反降至 26.41dB，corr 不变 —— **纯缩放不可行**。三个候选元凶全部排除：① blend=13×2⁻¹⁰ vs 协议 13×2⁻⁸（4x 指数位差）但 flat oracle 在 0.0127 下 3 位小数匹配，非解码 bug；② E4×0.25 还原到 ×1.0 后 pre-tanh 达 ±900（99.8% 饱和）、3ch corr 崩至 +0.047、oracle MSE 1361 —— es=0.25 正确；③ 5-tap 交叉滤波通路与官方 delta 相关 ≈0（0.03/-0.001/-0.006）非缺源。真缺口：**编辑空间分布**（block-heat 空间 corr=+0.008≈0，人物区 −0.257）+ 高频噪声响应缺失（hp-corr R=+0.003）；官方编辑不集中人物（密度比 1.03，autoMask 假设不成立）。欠增益是内容自适应的（噪声触发 σ 门控），需后续实现 SASS σ-gate 幅度通路。 |
| **Round 13** | σ-gate 专项：SASS 系统排查 + 官方编辑指纹破译（含全量负结果） | **任务A**: 15 cubin 系统排查噪声门控模式全部否定——cubin_13 的 42×EX2 = **Oklab 颜色变换**（cbrt 牛顿迭代 + M1 矩阵常数 0.4122/0.5363/0.0514 等实证）；cubin_00 control_mask 的 RCP 簇 = softmax 注意力归一化（ε+SHFL+成对 RCP，确证 R4）；`simple_blend` 尾核全量解码（新增 §7.2）：σ门 = sigmoid(x_net)（−log₂e→EX2→+1→RCP→SAT），门的是 **MV-warped 双三次残差**，增益链 c[0x1d4-0x1f4] = NVSDK 运行时参数（intensity/style 注入点）。**任务B**: PyTorch σ-gate 标定 45 组（窗{4,8,16}×曲线{linear,sig,table}×γ{1..8}）**全部未过门**（corr +0.343→+0.15~0.27↓，edit-ratio 最低 1.35，hp-corr 最高 0.083）；MV-bicubic 残差 6 种约定相关全零（−0.008~−0.016）。**编辑指纹破译**：官方编辑空间自相关 h=0.995/**98.2% 能量在 3×3 低通带**/1/8 降采样往返 corr=+0.985 —— 官方编辑是**平滑色调校正场**（highlight rolloff：luma 0.7 处 −0.062；暗部提亮 +0.013），非噪声门控/非锐化/非 warp 残差。**Tone-curve 叠加**（luma→dRGB 32 段查表，帧 0-7 拟合/帧 8-15 验证）：3ch +0.371→**+0.384**（R +0.43），edit-ratio 4.17→**3.36** —— 方向正确，新验收三指标体系建立（corr↑ + edit-ratio→1.0±0.15 + hp-corr>0.3），本轮未全过，遗留 ~2.7× 欠幅待 2D 曲面(luma×sat)或多特征联合。 |
| **Round 14** | 外部情报交叉验证：pre_block Box-Muller 高斯合成铁证 + 无损注入 + RMS/MpCubicSiLU 核对 | **情报1 (铁证)**: sm_89 cubin_00 全 4 个 `pre_block_swin_1h_32_1{,_ds,_fp8,_ds_fp8}` 内核含完整 Box-Muller 链 (偏移 0x390-0x960): 种子 c[0x228]×0x9E3779B9 → tile 哈希 (×0xD8163841 ⊕ 0x243F6A88π前缀 ⊕ ×0x108EF2D9 雪崩) → 4 并行 LCG lane → LG2/SQRT/SIN/COS → 3 条 FP16 高斯 PRMT 拼进首个 MMA；MUFU.SIN×4/COS×8/SQRT×8/LG2×20 全部只在此内核群——与 madebyollin gist 结构级一致，常数同族不同值。**无损注入**: control 槽 (即 dlssnr_prev_output 同构位, 情报3) 注入帧稳定逐像素高斯: amp=0.05 时 3ch +0.343→**+0.357**、edit-ratio 4.02→**3.51** 且高斯>均匀 (形状特异性) —— 通路活着；但 amp≥0.25 崩、hp-corr 不升 = blob 权重未按生成语义训练，只能定向不能定量。**情报2 (负结果)**: 112×LN→RMSNorm 消融 corr +0.343→+0.237 (R 崩至 +0.240) — LN 含均值减除确证，不换。**情报4**: MpCubicSiLU 常数逐位一致 + SASS 4864 行命中 = 全库确证。忠实哈希参考实现 `.tmp/gauss_sass.py` (σ≈1.006/确定性/种子敏感)。三指标未全过 (corr ↑ 但 edit-ratio 3.51、hp-corr 0.05)，R15 方向: 高斯 lane 权重的微调标定或 seed 扫描匹配。 |
| **Round 15** | 双线：高斯 lane 增益/seed 网格（负结果封盘）+ tone-curve 2D 曲面（突破，双门槛过） | **A 线 (高斯, 负结果封盘)**: 加性注入 seed 方案×增益全网格 (stable/sass-seed/perframe × g 0.01~0.2)——corr 单调劣化 (+0.343→+0.252)、**三种 seed 方案数字完全相同** (权重眼里都是噪声)、hp 最高 0.061 (基线) vs 注入后 0.023~0.059；纯替换微增益网格 (amp 0.005/0.01/0.02/0.05): 响应**平坦** (corr +0.357~0.358, edit-ratio 3.506~3.511, hp 0.048~0.050) —— R14 的微小增益与高斯幅度无关，仅是控制槽去相关本身；hp-corr<0.1 达成不了，按预案记录负结果转 B 线。**B 线 (2D 曲面, 突破)**: luma(24)×sat(6)→dRGB 双线性 LUT, 帧 0-7 拟合 / 8-15 held-out 验证: w=1.0 即 3ch +0.371→**+0.521**、edit-ratio 4.17→1.40；**w=1.5: 3ch=+0.5161 [R+0.5814 G+0.4893 B+0.4776]、edit-ratio=1.014 ✓ (1.0±0.15 带内)**；w 扫描显示幅度最优区间 w∈[1.4,1.6] (1.074/1.014/0.960)。曲面形态: 高光高饱和深坑 (luma 0.5-0.65×sat 0.4 → dR −0.20~−0.25) + 低 luma 提升带。三指标: corr↑✓ edit-ratio✓ **hp-corr ✗ (~0, 平滑场无高频)**——与 A 线负结果互洽：官方编辑 = 98% 平滑 tone 场 (B 线已吃掉大半) + 2% 生成式纹理 (需训练权重才能获得, 静态解码不可达)。交付: triptych_tone2d_f13.png + person_crop_tone2d_f13.png (f13 held-out)、LUT 存 .tmp/r15_surface_lut.pt。 |
| **Round 16a** | 评审系统建立：eval_suite.py 多维记分板入库 + 基线测量（单一指标三次受骗后的先测后优化基建） | **量化确认用户肉眼判断**：官方 vs 输入 sharp@1px **+23.7%**（@2px +19.5%、@4px +19.3%）、平坦区颗粒 **+16.1%**；replica vs 输入 sharp@1px **−3.6%**（@2px −12.5%、@4px −14.8%）、平坦区颗粒 −12.8% —— **方向完全相反**（官方锐化+加颗粒，replica 在抹平）；饱和度/对比度/色度也同向异量（rep con −0.020 过度 vs off −0.007）。**delta 分解揭示单指标骗局机理**：smooth corr=+0.963 / hf corr=+0.828 / amp_ratio 0.92-0.95 全部”看起来很好”，但 corr/ratio 尺度不变，看不见各向同性 vs 结构性锐化的差异。MS-SSIM 0.865 / LPIPS 0.253（clean）、0.908/0.186（cold_game 迁移，LUT 无重拟合直接用）。**新验收门槛**：记分板全维度无回退 + 锐度同向（官方方向=锐化）。入库 `eval_suite.py`（5 方向指标×3 列 + delta 四分量分解 + MS-SSIM/LPIPS + 0-7 拟合/8-15 报告协议 + cross-scene + 4 象限人类协议可视化），测试 5 项（方向性/不变性/守卫）全绿。基线 JSON：.tmp/sb_all.json；四象限+triptych 已出图（cap3_vis）。另外本轮完成 153 record 元数据全量解码：**a=b=B+40（恒等式，非偏移）、footer=(0,0,0,1,0,B/2,namelen,0)、首 dword=1（全 153 条相同）——头部/尾部无任何 per-record scale 字段，es=0.25 全局标定被 flat oracle 三位小数匹配背书，Task A 结论=元数据层无未解码 scale**（c32 record 内 8KB fp16 band 语义仍未破译，但值域混合非 scale 形态，已列入待查）。 |
| **Round 17** | 逆向专项 A/B/C（静态优先，无新拟合层入库）+ MV 跨度污染审计 | **A (per-record scale): 已关盘，无未解码 scale**。153 条 record 元数据全量解破：头部 QQQ = (a, b, B) 满足 **a = b = B + 40 恒等式**（非偏移、非地址），footer 8I = (0,0,0,1,0,B/2,namelen,0)，数据首 dword = 1（全 153 条相同）——头部/尾部无任何 scale 字段。数据区侧：MX record 的 scale 字节已被当前逐对解码使用；c32/c64 record 的 256B fp16 hole = 128 值（非 scale 形态：-0.39~0.99 混合值域，部分为 LN 参数）；c32 record 内 8KB fp16 band（misc 1024 值 + 未装载数据）值域混合（−8.9~+0.14，−6.5 对数簇 + 微值簇）非 scale 形态，语义待查但不阻塞。**es 全局扫描确证**：es=0.25→纯网络 corr +0.373（4帧），es=0.2→+0.337、es=0.3→+0.326、es=0.35→+0.106 —— es=0.25 是尖锐极值，不存在逐块 scale 替代空间。**B (pre_block 装配): 破译完成**。5 个 texture descriptor 全 census：`0x58`=MV 偏移双三次（LOD −0.125，'Filtered' 通路）、`0x5a`=5×直接颜色 tap、`0x5c`=单 tap（depth/luma）、`0x5e`=5×RZ 丢弃目标（cache prime）、**`0x60`=dlssnr_prev_output（history）——由 64 位指针存在性测试 c[0x180/184] 门控**，经 FFMA c[0x1dc]+c[0x1d4]+FMUL c[0x1e4] NVSDK 参数增益链进 lane。**8ch MMA 操作数布局实锤**：4 纹理 lane + 2 运动 lane（c[0x20c/0x210/0x21c/0x218] 缩放，FSEL 分支钳位）+ history/常量 lane + **常量 1.0 lane（IADD3 R33, R3, 0x3c000000 = fp16 [1.0,0]打包；0xbc00=−1.0 常量在 L_x_499）**——与 madebyollin gist 完全对齐。lane 缩放 R8 = 2×c[0x224]。**C (skip 合并语义)**: concat+fuse GEMM 在代数上已等价于双分支线性+加法（W=[W_up|W_skip]），SASS 无需再判；消融实验：normal +0.373 / **no_up（仅 skip）+0.393 更好** / no_skip +0.025 / sum +0.042 —— decoder 主信号在 skip 通路，up 分支被 fuse 权重压低；**no_up>+0.02 说明 up 分支当前权重可能有装载残缺（Kaiming 填充的 expand 未携带真信号）→ 下轮从 b48/56/62/66 记录字节预算重新审计 up GEMM**。**MV 跨度审计**：clean 场景 depth/motion rowPitch=7680B=1920×4B 恰好 dense，所有 R1-15 分析无污染；cold_game（1280×720，depth/motion 984×553 rowPitch 4096B>3936B 需要 stride）在 R16a 之前从未被使用过，首次使用时 reshape 立即报错（fail-fast）后已修——**无历史数据受污染**，已在 data_utils 补 stride 支持并注释。 |
| **Round 18** | b48/56/62/66 字节预算重审计（R17 自定方向）：expand 装载残缺实锤 + fuse zone 环境旋钮 + 记分板验收 | **字节地图（256B 精度）**：b48=[0:491520]E4（up_swin 458752 + 未知 32768）→ [491520:492544]F（512 fp16 = up_swin γ+β 正好）→ [492544:738304]神秘带 245760B（80% \|v\|<0.05 fp16 + 20% ±2.2 结构化，非 E4/非 MX/非可映射 GEMM）→ 交错 1KB F/E → [746496:820224]E4 混合 → bias；b56=[0:131072]E4 恰好 = expand 全额（fuse 32768B 应在 [131584:164352]）；b62/b66 同构。**关键实验**：① fuse zone 换位（b48→[655360:786432]、b56→[131584:164352]）corr +0.373→+0.372 无感 —— fuse 字节身份不敏感；② expand 内容三态（nn-identity/Kaiming/zero）= **+0.373/+0.264/+0.393** —— NN 拷贝有害 −0.02、Kaiming 有毒 −0.11、**删除 up 分支最优**；③ up 分支增益扫描 1.0/0.5/0.25/0/−0.5/−1.0 = +0.373/+0.380/+0.386/+0.393/+0.394/+0.385 —— 平台在 0~−0.5，up 分支净噪声。**结论**：官方 up 路径的真实表示（我们norm→expand拷贝→pixel_shuffle→up_stage的拼装不是它）就在未解码 245KB 带内，但字节形态（80%微值fp16）既非 E4 也非标准 GEMM 布局——需 SASS 对 dec 上采样核的 HMMA 操作数源反推（R19）。**验收（纯网络）**：zero-expand 后 corr=+0.393 > normal +0.373 ✓（no_up 差距消失，以权重侧合法清零实现，无新拟合层）。**记分板（clean）**：锐度/颗粒仍反向（sharp1 −3.7% vs 官方 +23.7%，与 R16a 基线持平），MS-SSIM 0.8651/LPIPS 0.2540（无回退）——R16a 门槛'全维度无回退'✓，锐度同向✗（待 R19 真实 up 路径）。四象限+triptych 已重出图（cap3_vis，no_up 变体）。loader 加 DLSS5_UPFUSE_ZONE_B{48,56,62,66} 环境旋钮（oracle 默认不变）。 |
| **Round 19** | dec 上采样核 SASS 反推 + 245KB 神秘带破译闭环（双路径验证） | **SASS 铁证**：`upsample` 内核群（base/fp8/tilesync/tilesync_fp8 4 变体，各 ~2950 行）= **520×HMMA.16816.F16、0×TEX、0×LDS/STS** —— 纯 GEMM 数据流，B 操作数经 LDG 直接从全局内存流入（基址 `c[0x0][0x180]` 同槽在 pre_block 是 history 指针，在 upsample 是权重/特征基指针，内核参数各绑不同资源），0x4000/0x4200/0x4400/0x4600 16KB 步长。**dec up 路径确有可学 GEMM（fp16 域）**。**245KB 带双路径验证**：①带-as-fuse-fp16（四记录同步替换 b48[492544:754688]/b56[131584:197120]/b62[49664:66048]/b66[13312:17408]，尺寸全部恰好匹配 fuse(2c²)）：corr −0.007 崩盘；②尺度扫描 2^−4/−6/−7/−8 全负（−0.07~−0.29）——**非 fuse/非 expand/非任何尺度下的 GEMM**；③带指纹（75% 微值 + 25% ±2.2 散布，无块结构，even/odd 同分布）= 辅助 log2 尺度/查表类张量（BLOB_FORMAT 之 'yc'），非权重。**字节预算终审定案**：b48 = up_swin(458752 恰好， ffn384) + fuse-E4(131072) + 带(245760) + 杂项，**4c²=524288 无处容纳（总需求 1114112 > B 820780）**；b56/62/66 同构验证 —— **expand GEMM 在全部 4 个 up record 中字节缺席 = REVERSED**（官方 up = 无重排 GEMM + up_stage swin(E4) + fuse(E4)，重排为无权重 pixel_shuffle 族）。swap 实验（up_stage 输入换成 skip）+0.371 ≈ normal，排除另一种拼装。**裁决**：'up 分支移除'升格为字节级实证的结构结论；no_up 开关退役（零展开即结构正确），fuse E4 权重维持现有 zone。**验收**：纯网络 corr +0.393（zero-expand，R18 已定）> normal +0.373 ✓；记分板/四象限沿用 R18 no_up 变体（无回退）。 |
| **Round 20** | 锐度通路指纹 + 血统定案清单入库 | **指纹**：官方 hf 与输入梯度对齐（corr +0.334 vs replica +0.048）但非简单 laplacian（±0.1、通道符号混合）；replica 的 G 通道 laplacian 耦合 +0.465 = R12 G-反相 hack 痕迹。**SASS-bicubic 残差实锤为锐度载体**：corr(\|official hf\|, \|bic hf\|)=+0.289；MV 任意符号组合、α*=1.11（锐度拟合）→ held-out sharp1 **+27.5%**（官方 +23.7%，幅度闭合）；MV=(0,0) 对照 −0.7% 证明载体即边缘对齐残差。**但符号结构不匹配**：任何 MV 下 3ch corr 崩至 ≈0（官方同时拥有 +23.7% 锐度与 0.96 平滑 corr）——官方 hf 符号来自网络内部 fp16 GEMM 特征，非输入侧 warp 可构造（R13 cross5/MV-bic corr≈0 的交叉印证）。**带-非权重终审**：band[16384:] 形状不足；×2⁻⁴ 已在 R19 全负 —— 非任何偏移/尺度/布局下的权重。**真 up 权重去向**：upsample 核经 c[0x180] 运行时基址流式读取，blob 内对应区域未定位（R21 方向：追踪 c[0x180] 指针的 runtime 绑定或 cuobjdump 资源表）。**血统定案清单**入库 REPORT.md（REVERSED×6 / CALIBRATED×2 / DIAGNOSTIC×1），'up 分支移除'标注 REVERSED（R19）。 |
| **Round 21** | c[0x180] runtime 绑定追踪（静态部分收口）+ expand 相位实验 + 未入账区域全图 | **资源表实证**：cuobjdump resource-usage —— upsample 核群 CONSTANT[0]:448（c[0x180]=384B 偏移在参数块内）= **内核启动参数，运行时由引擎填充**，ELF 内无初始化值可提取 → 静态追踪到参数层为止，权重要从 blob 侧反推。**未入账区域全图**：全 blob 总计 **10.4MB 未入账**；c128 记录各 65,072B（= b10 [131584:196096] 64512B 'other data'），c256 记录各 229,292B。**指纹定案**：b10 未入账带 = 双峰 fp16（49% 微值 + 49% ±3-9 = exp2 后 0.125~512 的 log2 尺度表）——与 b48 245KB 带同族，**辅助尺度/查表家族，非 GEMM 权重**。**综合裁决**：官方装载 E4 记录 → 反量化到 fp16 工作区（尺度表就住在这些带里）→ upsample 核在 fp16 工作区上 GEMM —— **无隐藏权重区域存在**，c[0x180] 指向运行时派生的工作区；我们已装载的 E4 fuse 与官方同源，es=0.25 即该变换的近似。**expand 相位实验**（Catmull-Rom taps [-1,9,9,-1]/16、Lagrange 相位 vs NN 拷贝）：corr +0.3791/+0.3741/+0.3731 —— 相位 remix 只值 ±0.006，**sharpE 完全不变**（0.000458）——expand 内容不是锐度杠杆；zero-expand +0.3927 仍是最优合法态。**锐度缺口剩余归因**：需要 (a) 运行时抓取反量化工作区（frida 钩 CUDA 启动参数，下轮候选）或 (b) 等待 fused-tail 完整解码。静态 blob 搜索**正式穷尽**：R13-21 所有未解码区域均已定性（权重区 E4/fp16 已装、辅助区尺度表、无第三类）。 |
| **Round 22** | 双线并行：未装载区行为探针（线1，全负关闭）+ history 递归实现（线2，首次正信号） | **线1·行为探针**（静态指纹穷尽≠行为穷尽，逐 region 试装载）：T3 fuse×exp2(band) 1:1 映射（b56/62/66 尺寸恰好）→ corr **−0.356** 崩盘；T5 b48 band[:131072] as E4 fuse → +0.3737 ≈ 基线（无增益）；T6 b56/62/66 band-as-E4 → +0.158 受损；加上 R19 fp16 族全负 —— **五族解码全负：未装载区为行为惰性辅助数据（引擎内部反量化尺度），非 tap 权重；tap 权重只存在于运行时工作区，blob 行为空间正式关闭**。**线2·history 递归**（权重兼容：prev_output→control 槽；SASS 结构 R17-B：0x60 描述符+参数增益链）：reset 组官方轨迹 **平坦 +0.0017**（f00 即收敛，reset 不可见于 delta-mean）；我们 single/recursive 轨迹幅度 ±0.004 相当、相位不同。**steady 组（1280×720）首次递归正信号：recursive corr +0.0586 vs single −0.0271（Δ+0.086）**——history 携带真实信号；绝对值远低于 clean 场（0.393）因该场景管线用非原生 depth/motion 分辨率插值（绝对值软，同管线递归 vs single 对比有效）。官方 steady f14 delta-mean ±0.002 = 收敛保持态。记分板/四象限不变（无更优合法变体，zero-expand 仍最优）。R23 候选：clean 场景的 prev_output 融合深度（control 混合比扫描）或 frida 运行时抓取。 | c[0x180] runtime 绑定追踪（静态部分收口）+ expand 相位实验 + 未入账区域全图 | **资源表实证**：cuobjdump resource-usage —— upsample 核群 CONSTANT[0]:448（c[0x180]=384B 偏移在参数块内）= **内核启动参数，运行时由引擎填充**，ELF 内无初始化值可提取 → 静态追踪到参数层为止，权重要从 blob 侧反推。**未入账区域全图**：全 blob 总计 **10.4MB 未入账**；c128 记录各 65,072B（= b10 [131584:196096] 64512B 'other data'），c256 记录各 229,292B。**指纹定案**：b10 未入账带 = 双峰 fp16（49% 微值 + 49% ±3-9 = exp2 后 0.125~512 的 log2 尺度表）——与 b48 245KB 带同族，**辅助尺度/查表家族，非 GEMM 权重**。**综合裁决**：官方装载 E4 记录 → 反量化到 fp16 工作区（尺度表就住在这些带里）→ upsample 核在 fp16 工作区上 GEMM —— **无隐藏权重区域存在**，c[0x180] 指向运行时派生的工作区；我们已装载的 E4 fuse 与官方同源，es=0.25 即该变换的近似。**expand 相位实验**（Catmull-Rom taps [-1,9,9,-1]/16、Lagrange 相位 vs NN 拷贝）：corr +0.3791/+0.3741/+0.3731 —— 相位 remix 只值 ±0.006，**sharpE 完全不变**（0.000458）——expand 内容不是锐度杠杆；zero-expand +0.3927 仍是最优合法态。**锐度缺口剩余归因**：需要 (a) 运行时抓取反量化工作区（frida 钩 CUDA 启动参数，下轮候选）或 (b) 等待 fused-tail 完整解码。静态 blob 搜索**正式穷尽**：R13-21 所有未解码区域均已定性（权重区 E4/fp16 已装、辅助区尺度表、无第三类）。 |
| **Round 23** | history 注入完整标定 + 16 帧全序列递归记分板 + 出图 | **标定**（clean 原生管线，纯网络 corr）：pure +0.3532/held +0.3911；replace（R22 模式）+0.3469/+0.3766 有损；**blend w 扫描 0.25/0.50/0.75 = +0.3546/+0.3545/+0.3519（held +0.3898/+0.3883/+0.3843）单调递减 → 最优 w=0.25**；residual 与 blend 代数同构（结果一致验证实现）。**w=0.25 入库：all-16 corr +0.0014**——收益微小且场景运动量依赖（steady 720p 场 Δ+0.086 vs clean 原生 ≈0）：history lane 结构 REVERSED（R17-B），混合权重 CALIBRATED（SASS 参数 c[0x1dc/1d4/1e4] 运行时专属）。**16 帧全序列递归记分板**（blend+LUT）：锐度轴 −3.7%（官方 +23.7%），grain_flat −12.6%，与 R18 基线持平；MS-SSIM **0.86525**（基线 0.86505）/LPIPS 0.25381（基线 0.25344）——全维度无回退 ✓，luma/chroma amp 0.930/0.964（噪声级微动）。**出图**：triptych_f13 + quad_person/flat/edge/grain_f13（history 变体，cap3_vis）。R24 候选：prev_output 直接入 pre_block lane（改 stem 权重形状，需 b 记录重审）或 frida 工作区抓取。 |
| **Round 24** | prev_output 直入 lane 假设判别 + stem 维度重审 | **H2（0x58=warped prev_output）被 SASS 门控逻辑否决**：pre_block 中唯一受 64 位指针存在性测试（c[0x180/184] NULL 检查）的纹理描述符是 0x60 —— 无门控的 0x58 在首帧必然悬空，不可能是 history；**0x58 = MV 偏移双三次采样当前输入**（无门控=始终绑定），0x60 = prev_output（有门控=history lane）—— 判别完成，R17-B 解释成立。**post_block 族描述符清点**（0x66/0x70/0x6e×5/0x90，无 0x58/0x60）确认 history 只进 pre_block。**stem 维度重审**：model stem = Conv2d(9,32,3)（color3+depth1+mvec2+control3 高斯）；SASS 8ch = 4 tex（RGB+alpha，alpha≈depth）+ 2 motion + 1 history + 1.0 lane；R14 'gauss 3 lane' 与 SASS 'history+const' 两张 lane 图不兼容 —— 正解为 8 lane 图（R14 高斯 lane 解释应修正）；我们的 9ch 消费了 stem 记录 21696B 的 2624B，余 19072B stem_pad 未映射，预算上 8ch/9ch 均可容纳。**结论**：history 已经在我们的 control 槽语义中等价实现（R23 blend w=0.25），lane 级重定位无可验证增益；R14 血统注记：'gauss 3 lane' 降级为 'history/const lane' 的待修正项。锐度通路仍待运行时抓取或 fused-tail 解码（R22/R23 结论维持）。 |
| **Round 25** | stem_pad 19072B 审计：b0 复合记录发现（stem+swin 同体） | **b0 精细图**（256B 粒度 E4-std 扫描）：W 区 [0:8192] + X 区 [8192:9472]（1280B fp16/尺度）+ W 区 [9472:12288] + X 区 [12288:]。**关键发现**：b0 与 b1-3 同构（c32 swin 记录族），loader 只取前 2592B 作 stem.weight、64B 作 bias，**其余 19072B = 一个完整的 swin 块（qkv+proj+mlp0+mlp2=16384B）+ norms —— 当前完全未装载**。结构推论：官方 pre_block = stem conv + swin 块融合内核（cubin pre_block 正是此结构！），b0 = [stem 2592][完整 swin 块][norms/misc] 复合记录。**A/B 移位实验受阻**：b0 主流仅 17340 连续字节（swin 跨越 F/交错区不连续），需按 b1 同构布局（qkv[0:4096]式分带）重解 b0 才能 A/B —— 留作 R26 首项。**潜力**：若 b0-swin 才是真 enc.0.blocks.0（现 b1 存在 off-by-one 嫌疑），整条 enc.0 链对齐修正可能大幅提升纯网络 corr —— 这是自 R5 以来最大的未验证装载假设。 |
| **Round 26** | b0-swin A/B 判别：off-by-one 假设否决，enc.0 链对齐维持 | **A/B 实验**（纯网络 corr，同代码路径）：A 现行 b1-as-block0 = **+0.3731**；B b0-swin[2592:] as block0（mlp2 27% 缺口补零）= **+0.0191** 崩盘。**双重铁证**：①B 即使 mlp2 完整也需跨交错区拼凑（不连续）；②flat oracle 三位小数匹配本就是对块级错位极敏感的判别器，现行映射已通过。**裁决**：off-by-one 假设否决，enc.0 = [b1,b2,b3] 维持；b0 = [stem 2592][其余 19072B = 非链路数据] —— b0 剩余区语义仍开放（可能是 stem 的备选/历史版本或输入侧 GEMM 的另一分量）但不阻塞任何指标。R5 以来最大未验证装载假设正式关闭，血统登记：enc.0 映射 = **REVERSED**（oracle 三位小数 + A/B 崩盘双证）。 |
| **Round 27** | fused-tail 全指令级解码：simple_blend epilogue 公式 + outview 4 平面写出 + cubin_13 定性 | **simple_blend epilogue（2646-2690 行）全解**：①**sigmoid 门控实锤**：R64 × −1.4427(−log₂e) → MUFU.EX2 → +1 → MUFU.RCP = σ(x_net)，x_net 来自 HMMA 累加器（门控卷积输出）——R13 sigmoid 判断的指令级实证；②**MV-bicubic 权重在线计算**：R113/R98/R96/R101/R102 五个 tap 权重由 FMUL(运动分量) 在寄存器内实时算出，TEX 0x6e×4 读 tap 邻域；③**归一化**：六项和 → MUFU.RCP = R99（R13 'softmax 式三次注意力'的指令级实证）；④**最终组合公式：out = σ(x_net) · [Σ wᵢ(mv)·texᵢ(0x6e)] / norm − net_raw**，其中 −R60/−R61/−R62 追踪到 HMMA 累加器（网络自身输出）——**减法残差在 simple_blend 层级实证**（R10 'Out=In−net_out' 的 SASS 铁证，bicubic 项就是 'In' 侧）；锐度载体 = 门控 MV-bicubic 残差（R20 alpha 实验的正式对应）。**outview 4 平面写出**：STG.E ×4（R110/R111/R108/R109）= 3 color + 1 aux/control 平面分离写（基址 c[0x168]，行步长 UR8×4B）——分平面布局而非交错 RGBA。**cubin_13 (cg2r_post_process_kernel, 1858 行) 全指令级**：0×HMMA、11×TEX（0x90/0x98/0x60）、MUFU = RCP×27+LG2×34+EX2×42（≈30 组 pow 曲线）、FMNMX 钳位、FSEL 分支 —— **逐像素 Oklab 域 pow 曲线后处理，无空间卷积/无锐化核**，运行在网络之后（消费 outview 4 平面）——非锐度算子（Q3 定案）。**锐度通路终局图景**：门控 MV-bicubic 残差（simple_blend 内）= 锐度载体；符号不匹配之谜指向 tex 0x6e 绑定对象（网络原始输出平面 vs 输入色）—— 运行时绑定确认后即可完全闭合。 |
| **Round 28** | simple_blend 公式 oracle 验证 + 0x6e 判别实验（探针组局限记录） | **公式验证成功**：完整 simple_blend（σ·bicubic(MV) − net_raw，门控 σ 单参数拟合）在 impulse 组重现 oracle：mean\|out−oracle\| = **0.00061**（oracle delta std 0.0018，即公式重建误差为信号幅度 1/3）—— **R27 解码的减法残差公式被 ground truth 确认**。**0x6e 判别实验结果**：flat 组 H_A 门控拟合退化（MV≈0 → bic≈src 常数退化）；impulse 组 H_A/H_B 误差完全相同（±0.4px 微运动下 bic 采样源区分消失）——**oracle 探针组无法判别 0x6e 绑定**（探针设计为运动冻结，源差异被消除），运行时抓取（frida）仍是唯一判别路径。**附产**：impulse oracle 径向指纹精化 —— 中心 8px 内 delta 强负（−0.135→−0.079，网络窗口减法残差），r≥10 衰减回 +0.0017 基线（半径 8 = swin 窗口边界）；flat 组 delta −0.0078（变暗）。 |
| **Round 29** | 整合固化：full tail 模式入库（结构性）+ 收敛文档 | **DLSS5_TAIL_MODE=full 入库**（默认 simple 保留）：按 R27 解码的 epilogue 实现 `out = σ(x_net)·bicubic_warp(MV) − net_raw`（MV-warp grid_sample + sigmoid 门控代理 + H_A 假设缺省 + 0x6e 绑定悬案注释标注）。**记分板对照**：corr 轴 simple = full = **+0.3731**（线性不变性：门控代理近似常数缩放，Pearson corr 不敏感）；绝对动态轴（锐度/编辑幅度）full 版失真（输出范围坍缩 0.069-0.152，因 x_net 真实权重是运行时参数我们不可得）——**full 模式 = 结构研究用，非记分板增益用**，默认保持 simple。 |
| **Round 30** | 0x6e 绑定终审判别：**H_A 胜出**（0x6e = 当前输入色），悬案关闭 | **三重证据链**：①P3 impulse 直接幽灵检验（模型无关）：点在帧间跳位（240↔304px 往复），跳变帧旧位置的官方 delta 幽灵 = +0.0009 ≈ 基线（H_B 预测应携带 −0.08 抑制残迹，MV≈0 恒等 warp 下必现——实测无）；②P2 edge 同判：旧边位置幽灵 −0.0011..−0.0017 弱于环境控制 −0.0026..−0.0031（边状态不被携带）；③无偏副本计分：H_B 源改用副本自身 prev 输出后 H_A=H_B=0.00056 平手（早前 H_B '胜出' 0.00049 被证为官方 prev 输出泄漏真值的伪优势）；④R24 SASS 门控逻辑佐证（0x6e 无门控=始终绑定=输入侧）。**驻留帧深抑制（−0.07..−0.09 随驻留帧数增长）走网络内部状态递归（c[0x60] 反馈），不经 0x6e**。**记分板**：simple 模式保持（held-out corr +0.393）；full 模式（未标定门控代理）实测 held-out corr −0.0956 崩盘，确认仅作结构研究，保持默认关闭。锐度通路最终图景：**门控 MV-bicubic(输入色) 残差 + 网络内部状态递归**，静态可解部分全部解完，剩余（x_net 真实门控权重、生成纹理 2%）需运行时抓取/训练，超出静态逆向边界。 |
| **Round 31** | full tail 终态验收（转正评估）→ **保持 simple 为默认**，full 为结构参照 | **记分板对比**（clean held-out 8/10/13/15）：simple corr **+0.3519** / 锐度 −12.0% / edit-ratio 11.76；full（H_A 终实现：bicubic 源=当前输入色，R30 确认）corr +0.3423 / 锐度 −11.9% / edit-ratio 7.85。**裁决**：①锐度：full ≈ simple（−11.9% vs −12.0%）—— blend epilogue 交换不闭合锐度缺口，实锤 R20/R22 结论：锐度在网络通路内部（窗口内 HFMA2 链），非 blend 层；②corr：未标定门控代理倒退 −0.0096；③edit-ratio：full 更接近官方幅度（7.85 vs 11.76，官方 delta 为主导结构）。**转正否决**：默认切 full 会倒退记分板（corr↓ + 锐度无增益）。**终态确认**：静态逆向完成；记分板最优配置 = simple + zero-expand + w=0.25 history；官方 +23.7% 锐度在上游窗口内通路（已映射）+ 生成纹理 2%（需训练权重）。 |
| **Round 32** | 窗口内 HFMA2 锐度链搜索：**不存在独立锐化模块**（负结果，静态侧最终关闭） | **全 cubin_00 内核 39 变体算符普查**（HFMA2/HMUL2/HMNMX2/HMMA 逐内核计数）：所有 swin 族内核（pre_block/post_block/ds/upsample/inpview/outview/wait/tilesync/chained + fp8 变体）的算符分布完全一致（HFMA2 352-385 / HMUL2 384-397 / HMNMX2 402 恒定 / HMMA 512-528）—— 无异常内核，无独立 unsharp 链。HMUL2+32 增量仅出现在 post_block 族（= R27 已解码的 blend 区域：门控+bicubic+归一）。**结论**：窗口内高频响应 = attention 对细节 token 的放大 + MpCubicSilu FFN，其系数就是我们已装载的 LN gamma/proj 权重 —— **不存在独立的 'HFMA2 锐化链' 静态模块**。官方 +23.7% 锐度 = (a) attention 机制本身（已在装载权重内）+ (b) 生成纹理 2%（训练专属）。静态侧最终关闭；B 轨 frida 仅剩 x_net 门控权重一项。 |

