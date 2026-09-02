# DLSS5 权重 blob 格式笔记（Phase 1 → Phase 4 交接）

> 结论先行：**meta32 不是元数据，是权重数据**。每条记录就是 `name + 长度头 + 连续的 E4M3 权重流`，
> 记录自身不携带 shape/dtype —— shape 信息只能来自算子身份推断（Phase 3/4）。

## 1. 记录结构（链式验证 152/153，尾部截断 8B）

```
u64 magic = 0x0000000008cda732
u64 count_field = 19        (≠153，语义未知，疑似顶层分组数)
153 × record:
    cstring name            "block{N}.layer{M}.layer" | "block70.layer0.blend_scale"
    u64 A   = B + 40
    u64 A2  = A             (重复)
    u64 B                 ← payload 字节数
    byte[32+...B] payload ← 权重数据 (E4M3 流), 无独立 meta 段
对账: 16 + Σ(len(name)+24+B) = filesize + 8
```

**证据**（meta32[4:] = E4M3 权重）：
- E4M3 解码后 mean≈-0.02, std≈0.08，与 payload 主体分布一致
- 指数位分布直方图与 payload 同构（峰在 exp 1-3）
- 早期误判来源：blend_scale 等标量记录确实有结构化头部，但那是**标量的值本身**（0x0000 = fp16 零），不是 dtype 编码
- meta32[0:4] 全部 = `01 00 00 00`：每条记录以 4 字节 `1` 开头 —— 版本号/记录头，不属于权重

## 2. block 布局（71 blocks, 153 records，详见 block_map.json）

```
stem b0: 21,696
enc:  b1-3 ×3 @20,672 | b4 merge 22,720 | b5-7 ×3 @61,760 | b8 merge 69,936
      b9-13 ×5 @197,184 | b14 merge 229,936 | b15-21 ×7 @689,232 | b22 entry 820,288
      b23-29 ×7 @1,968,192 (4-layer) | b30 5-layer 2,492,496
bottleneck: b31-38 ×8 @12,587,154  (layer 签名: 4,194,320 / 4,196,352 / 3,145,856 / 2 / 1,050,624)
dec:  b39 exit 525,312 | b40-47 ×8 @1,968,192 | b48 up 820,784 | b49-55 ×7 @689,232
      b56 up 230,176 | b57-61 ×5 @197,184 | b62 up 70,048 | b63-65 ×3 @61,760
      b66 up 22,784 | b67-69 ×3 @20,672 | b70 tail 21,810 (+blend_scale 2B)
```

## 3. 已破译算子（数学精确）

| 签名 | 分解 | 身份 |
|---|---|---|
| 4,196,352 | 2048×2048+2048 | Linear(2048→2048)，瓶颈 GEMM |
| 1,050,624 | 512×2048+2048 | Linear(512→2048)，FFN expand |
| 263,168 | 256×1024+1024 | Linear(256→1024) |
| 525,312 | 512×1024+1024 | Linear(512→1024)，瓶颈出口 proj |
| 2 B ×9 | fp16 0x0000 | 零标量（b31-38 门控? + blend_scale 初值 0）|

## 4. 通道数悖论（未解，Phase 4 攻坚）

- 32ch 模板（`1h_32_1`）对应 block b1-3 每块仅 20,672 B ≈ 4×64² 量级 → 实际通道可能是 **64 而非 32**
- 若 qkv+proj = 4c² 无 bias：c=64 → 16,384 + 192 杂项 ≈ 20,672 ✓
- 但 s4 的 4-layer 分解 (L 256→1024 等) 又支持 c=512 stage 用 c=256 的 qkv？→ 需 cubin 反汇编或试加载验证
- 分支假说：**block 系列可能分属不同分支**（主干 vs ViT 旁路），不能全塞进一条 U-Net

## 4.5 字节账本精确分解（Phase 1 末新增，部分待验证）

**纯 Swin 块（每 stage 每块）**：
| stage | c | 拟合 | FFN ratio |
|---|---|---|---|
| s0 | 32 | **20c²+6c = 20,672 精确** | r=8 (hidden=256) |
| s1 | 64 | **15c²+5c = 61,760 精确** | r=5.5? (非整，存疑) |
| s2 | 128 | 197,184（无整数解） | ~r=4 |
| s3 | 256 | 689,232（无整数解） | ~r=3 |

- FFN ratio 随深度递减（8→…→3），与 `BSTinlayoutConvFfn` 可变 FFN 命名一致
- c=128/256 拟合不出整数系数 → 块内可能有 window-size 相关项（relpos 表）或非 2 的幂 FFN

**边界块**：
- merge（b4/b8/b14/b22）= swin(c) + **2c²−16**（downsample 投影，cubin 后缀 `_ds` 铁证）
- up（b66/b62/b56/b48）= swin(c) + **2c²+γc**（γ=2,1.5,1.75,1.875 → 收敛 2；cubin 后缀 `_upsample` 铁证）

**瓶颈 split-swin（b31-38，每块 12,587,154 B）**：
```
layer0 = 2048×2048 + 16      ← 宽注意力 QKV (16h × 128d = 2048，非普通 32d!)
layer1 = 2048×2048 + 2048    ← 宽注意力 PROJ (Linear 2048→2048)
layer2 = 12×512² + 128       ← 12c² = qkv(3c²)+proj(c²)+ffwd(2·c·4c) 的 c=512 侧支
layer3 = 2 B (fp16 零标量)    ← 门控/开关
layer4 = 512×2048 + 2048     ← ffwd expand 入口 (Linear 512→2048)
```

**辅助 kernel 身份（data.bin 15 个 cubin，arch=sm_120 主 + sm_89 辅）**：
| cubin | 内容 |
|---|---|
| 00 | swin_1h_32_1 全变体（pre/post/ds/upsample/chained/fp8 ×组合）|
| 01 | swin_2h_64_2 同上 |
| 02 | swin_4h_128_4 同上 |
| 03 | swin_8h_256_8 同上 |
| 04 | **split_swin_16h_ffwd_512**（瓶颈专用）|
| 05 | **vit_1d**（attention/qkv/ffn_contract/ffn_expand）|
| 06 | **dec_input_upsample_1024_512**（1024→512 解码器入口上采样）|
| 07-12,14 | sm_89 工具 kernel（font/capture/mv_dilate/clear_view/copy）|
| 13 | cg2r_post_process_kernel |

→ cubin 文件已提取到 `phase2/cubins/`，供远端 `cuobjdump -elf` 解析常量段形状

## 5. Phase 4 建议

1. weights_loader 直接按 §1 结构读流，**不要**假设记录内还有 meta
2. 每 32 权重一组的 MXFP8 scale：若 scale 是 E8M0，应出现在权重流的规律位置——扫描 33 字节周期未发现 → 可能 per-64/per-128 或 scale 表在别处（data.bin 的 cubin 常量区）
3. 加载顺序实验：b31-38 的 layer0-2 签名巨大（4M+），几乎肯定是 split-swin 的 qkv/out 分片
4. 与远程骨架的 nn.Module 树对齐时，用 §3 的 Linear 身份做锚点

## 6. 杂项字节规律（新增，c=256 未解）

各 stage 纯 swin 块主部若取 a·c²，余项：
- c=32: 20c² 余 192 = 64×3
- c=64: 15c² 余 320 = 64×5
- c=128: 12c² 余 576 = 64×9
- c=256: 无整数 a 使余项 = 64×17（689232 = 2⁴×3×83×173，与前三块 2⁶ 整除性不同！）

余项序列 64×(3,5,9) = 64×(2¹+1, 2²+1, 2³+1) → 疑似与 window/头数相关的每级翻倍项。
c=256 块 (swin_8h_256_8, b15-21) 结构特殊，可能为 2-block fused 打包或含大 relpos 表，
需 cuobjdump -elf 解 .nv.info PARAM_CB 才能定案。

## 7. 块内尾部结构（远程实测定稿进行中）

- LN gamma 表：**fp16 编码（每通道 2B）**，不是 E4M3。0x3bXX/0x3cXX ≈ 0.94-1.22（LN gamma 典型值）
  - block1: misc[112:176] = 64×E4M3？→ 远程纠正：实际 gamma 区从 20593 起 128B = 2×32×2B fp16
  - 定位方法：向后扫描连续 fp16≈1.0 的 2 字节小端模式（高字节 0x3b/0x3c）
- misc[0:112]：fp16 全 ~0 → bias 区（值极小或零初始化）
- 记录尾 16B：block1 全零，block2 = `00000000 60280000 14000000 00000000`（0x2860=10336, 20）→ trailer 语义待定
- 块内布局（c=32 工作假设）：
  `[qkv 3c² E4M3][proj c² E4M3][ffn_w1 8c²][ffn_w2 8c²][bias ~112B fp16][LN1 c×2B][LN2 c×2B fp16][pad/trailer 16B]`

## 8. c=32 块内布局（本地最终确认）

```
块字节流 (w_off 起 B=20,672):
  [0 : 3072]      qkv   3c²  E4M3
  [3072 : 4096]   proj  c²   E4M3
  [4096 : 12288]  ffn_w1 8c² E4M3  (c→8c=256)
  [12288 : 20480] ffn_w2 8c² E4M3  (8c→c)
  [20480 : 20592] bias  112B fp16 (值≈0)
  [20592 : 20720] LN gamma 2×c×2B fp16 (值≈1.0, 高字节 0x3b/0x3c)  ← 跨过 B 边界!
  尾部 28B 记录元数据 (01 00 00 00 + u32 + u32)
```
- 注意: gamma 表起于 20592 而块声明 B=20672 → 权重流的最后 ~80B 与 gamma 表重叠区需要切分时小心
- misc 区精确: [20480:20672) 192B 内 = bias 112B fp16 + gamma 起始 64B；gamma 完整表延至下条记录前
- E4M3 大值区 (0x40-0x7f, 8%) 分布在 ffn 段 — MXFP8 块 scale 或行幅值异常，Phase 4 精解


## Phase 4 终版结构 (2025-09-02 深夜会话)

### 记录布局 (153 条链式验证 152/153)
[magic8=0x08cda732][count8=19]{[name L][u64 A=A=B+40][u64 B][B payload][28B term][4B pad]}*153
- term = [0,0,0,1,0, B/2, next_namelen] (7×u32)
- payload[0:4] = 01000000 固定标签; 权重数据从 payload[4:]
- 无独立 dtype/shape 元数据 → 形状由算子推断

### 块内三段式 (Phase 4 破译)
1. **E4M3 权重区**: 纯 E4M3 字节, 1B=1param, 零大值 (b31 全块/b9 [0:98304] 实测 std 0.03-0.08)
2. **MX 交错区**: (W:E4M3, S:E8M0) 2B 对, W × 2^(S-205); S 集中 [196,200] 与 |W| 独立 (per-matrix 量化);
   仅存在于 c=32/64 块中段、c=512 layer2 尾段 (b23.layer2 [786432:917564])
3. **fp16 尾区**: ffn2 权重 + bias (±0.006) + LN gamma (0.52-1.0, XX3b 对) + 零 pad;
   大块显著 (b9 [98304:197184] = 98,816B; b15 [360448:689232])

### 关键尺寸破译
- c=32 (20672B): qkv 3072 + proj 1024 + E4 7168 + MX 8192 + E4 1024 + misc 188
- c=64 (61760B): qkv 12288 + proj 4096 + ... + MX 16384 [40960:57344] + ...
- c=128 (197184B): E4 [0:98304] (qkv 49152+proj 16384+ffn1 32768) + fp16 [98304:] (ffn2+bias+gamma)
- c=256 (689232B): E4 [0:360448] + fp16 [360448:688640]
- c=512 SplitSwin (4 子记录): layer2 917568 = qkv 786432 E4 + mlp1 131072 MX;
  layer1 263168 = proj 262144 + norm 1024; layer0 524288 = mlp2 (1024×512) E4; layer3 263168 = 分支2 + norm
- 瓶颈 b31-38: layer0 4194320 = 2048²+16 纯 E4; layer1 = 2048²+2048; layer2 = 12×512²+128; layer3 = 2B 零
- 9 个 2B 标量记录 (b31-38.layer3 + b70.blend_scale) = fp16 零

### 陷阱记录 (Phase 4 走过的弯路)
- "4B 版本头 + 28B 尾元数据" → 实为下一条记录的 term + pad
- "(W,S) 逐元素交错全库" → 仅 MX 区; 大块 E4M3 区零大值
- "可变长编码 (大值 2B)" → fp16 尾区低字节误读伪影
- "64B 周期 [63W+1S]" → MX 区 odd 高占比的巧合
