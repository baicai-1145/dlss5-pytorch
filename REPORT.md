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

