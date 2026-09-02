# dec0 偏大根因诊断 (Phase 4 fwdtest, 基于 e0b324e)

基线 forward (`python3 phase4/semantic_fill.py`, 64×64) 显示 dec0 absmean ≈ 444-474 (bn7≈600 后, bn_proj≈440, dec0≈440; 而 dec1=1.57 → 后续收敛到 tail=0.21)。
dec0 与 enc4 拓扑同构 (都是 c=512 SwinStage, 8/7 块, 相同 ffn_hidden=892, 相同 ln_mode/rel_bias), 但 enc4 ≈ 88 (健康), dec0 ≈ 440 (偏大约 5×) — 同样的模型骨架, 同样的 fill 路径, 数据差异是唯一变量。

## 关键观察

### 1. dec0 8 个块的输出几乎相同

| block | absmean | absmax |
|------|--------:|-------:|
| 0 | 2.3392e+04 | 8.8543e+04 |
| 1 | 2.3391e+04 | 8.8544e+04 |
| 2 | 2.3390e+04 | 8.8545e+04 |
| 3 | 2.3391e+04 | 8.8546e+04 |
| 4 | 2.3389e+04 | 8.8547e+04 |
| 5 | 2.3392e+04 | 8.8544e+04 |
| 6 | 2.3391e+04 | 8.8546e+04 |
| 7 | 2.3392e+04 | 8.8547e+04 |

8 块之间差异 < 0.003% — **residual 主导, 内层 GEMM 贡献被淹**, 或每块的内层权重是同一份数据的重复。

### 2. mlp.2.weight 在块间 std 剧烈震荡

| block | mlp.2.weight std | mlp.2.weight absmean |
|------:|-----------------:|---------------------:|
| 0 | 2.14 | 0.29 |
| 1 | **3930.09** | **820.06** |
| 2 | **1141.68** | 140.94 |
| 3 | 93.10 | 8.22 |
| 4 | 369.38 | 31.04 |
| 5 | 1.66 | 0.16 |
| 6 | 941.59 | 291.22 |
| 7 | **9792.48** | **3448.54** |

对比 enc4.blocks.6 mlp.2.weight std=555 — 同一个网络拓扑, 同一种 fill, 数值却从 1.66 跳到 9792。**这不是权重, 这是被错位灌入的非权重数据**。

### 3. norm1.weight/bias 也大幅震荡

| block | norm1.weight std |
|------:|-----------------:|
| 0 | 0.1684 |
| 1 | 0.0445 |
| 2 | 0.0505 |
| 3 | 0.0462 |
| 4 | 0.0462 |
| 5 | 0.1394 |
| 6 | 0.0872 |
| 7 | 0.0693 |

合理 norm gamma 应在 1.0 附近, std 应在 0.05-0.2 之间 (取决于初始化/装填策略)。**震荡本身不一定异常, 但与 mlp.2.weight 的 4600× 震荡耦合在一起, 说明装填偏移在不同块上落点不同**。

## 根因: e0b324e 的 mlp_stream 拼装错位

`phase4/semantic_fill.py:196-205` (c=512 SplitSwin 分支):

```python
put(f'{prefix}.attn.qkv.weight', m2)            # qkv ← layer2 前段 786432 elem (E4M3 qkv)
put(f'{prefix}.attn.proj.weight', m1)           # proj ← layer1 (262144)
# 数据顺序对齐模型参数序: qkv←layer2前段, proj←layer1, mlp←layer0+layer2MX段+layer3
mlp_stream = np.concatenate([m0, m2[786432:], m3[:262144]])  # ← 错误的拼装
put(f'{prefix}.mlp.0.weight', mlp_stream)
put(f'{prefix}.mlp.2.weight', mlp_stream[456704:])
```

`m2[786432:]` 是 layer2 解码流的尾部 — 包含:
- **MX 段** (lo=786432 到 hi=917504): `mx_decode_pairs(arr[lo:hi].reshape(-1,2))` = 65536 个 MX 解码值
- **E4M3 残余** (hi=917504 到 arr 尾): ~60 个 E4M3

`m3[:262144]` 是 layer3 main 前段 (262144 elem)。

实测各源数据 std (b23 / b40 / b41):

| 数据源 | b23 (enc4.0) | b40 (dec0.0) | b41 (dec0.1) |
|--------|-------------:|-------------:|-------------:|
| m0 (layer0) | 0.076 | 0.076 | 0.076 |
| m2.e4a (qkv) | 0.044 | 0.044 | 0.044 |
| **m2.mx** | **3138** | **5.4** | **8950** |
| **m3 (layer3)** | **7.42** | **2.81** | **7.83** |

→ `mlp_stream = [0.076 std | 3138-8950 std MX | 7.4 std m3]` — **MX 段被塞进 mlp.0.weight 后半部** (从偏移 524284 开始), **m3 段被塞进 mlp.2.weight 主体** (从偏移 456704 开始, mlp.2 需要 456704 elem, 几乎全由 m3 段填充)。

结论: **e0b324e 的 mlp_stream 拼装把 MX 解码值 (本应是 scale/branch 数据) 灌入了 mlp.0/2.weight, 把 layer3 main (本应是分支权重或 norm) 灌入了 mlp.2.weight**。MX std 在不同块间从 5.4 跳到 8950, 直接驱动了 mlp.2.weight std 从 1.66 跳到 9792。

## 为什么 enc4 看起来"健康"

enc4.blocks (b23-b29) 也是 SplitSwin 同一 fill 路径, 同样错位。但 enc4.stage 输入来自 enc3 (absmean ≈ 48), 初始能量比 dec0 输入 (bn_proj ≈ 2.3e4) 低约 500×。所以 mlp.2.weight 即使错位了 555 std, 8 块累乘也只把 enc4 从 48 推到 88 (+1.8×)。而 dec0 的输入是 bn_proj 输出 2.3e4, 同一错位权重直接把 dec0 推过 2.3e4 × 大常数, 之后被 tanh 截断成 ~2.3e4。dec0 看起来是"基本不变化" (8 块输出几乎相同) — 因为错位的 GEMM 输出远大于 residual, 主导了输出, 但 residual 量级也大致稳定, 所以看似"flat"。

## 修复方向

1. **MX 段不应进 weight 矩阵**: layer2 的 MX 区在正确解读下应是某 scale/gate, 而非 mlp.0.weight 的尾部
2. **layer3 main 不是 mlp.2.weight**: b23 m3 std=7.4, b40 m3 std=2.8, b41 m3 std=7.8 — 这不是 fp16 weight (那应该 ~0.01-0.1), 也不是 norm (那应该 ~0.05)。layer3 在某些块里装的是分支/侧路数据
3. 需要回溯到 `_SplitBlock`/SwinBlock 的预算布局, 弄清每个 block 实际有哪些子记录、各自语义, 而不是简单按 layer0/1/2/3 顺序拼接

## 任务 C 状态

* ~~`mlp.2.bias` 漏装~~ — **该 bug 在 e0b324e 已修复** (填进 misc_all[512:1024]), 装填率从 143,877,332 → 143,885,012
* ~~装填路径不生效~~ — **9/10 params 已写入** (包括全部 attn/mlp/norm 权重), 不存在"没装上"
* **真正问题**: 装填路径生效, 但 **e0b324e 的 mlp_stream 拼装假设**与真实数据布局不匹配, MX/layer3 段被错位灌入 mlp.2.weight, 产生 1.66-9792 的剧烈 std 震荡, 进而把 dec0 推到 2.3e4 量级

## 建议下一步

1. **用 e0b324e 在 enc4 上验证**: enc4 实际也是 88 量级 (虽然比 dec0 低很多, 但相对输入 48 仍有 ~2× 增益) — 同样错位, 但被低能量输入掩盖。
2. **真正的 fix 应在 layer 语义识别**: 验证 b22 (enc4 entry, xform) 和 b40-b47 的 layer0/1/2/3 与权重序号的真实对应。
3. 若 layer3 main 在某些块里是 **bias**, 那么 mlp.2.weight 装到 layer3 main 是无解, 应改为装 bias。
4. 若 layer2 MX 在某些块里是 **gate/scale**, 那是 mlp_stream 之外的额外参数, 不是 mlp 权重。