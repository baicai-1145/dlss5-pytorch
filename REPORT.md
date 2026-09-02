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
