# dec0 偏大根因诊断 (Phase 4 fwdtest, HEAD = d446313)

基线: `phase4/fwdtest.py` (seed=42/7, 64×64), HEAD 公式 `v = W × 2^(S - median(S) - 8)`
dec0 absmean = **916**, enc4 = **88**, bn7 = **673**。dec0 不再是孤立断崖, 而是从 enc4(88) → bn0(141) → ... → bn7(673) → bn_proj(917) → dec0(916) 的连续链。

## 关键观察 (HEAD 数据)

### 1. dec0 = bn_proj (量级连续)

| stage | absmean |
|-------|--------:|
| enc4 | 88.2 |
| bn0 | 140.9 |
| bn3 | 241.2 |
| bn7 | 672.6 |
| bn_proj | 917.4 |
| **dec0** | **915.8** |
| dec1 | 1.56 |
| tail | 0.205 |

→ dec0 ≈ bn_proj, 没有"放大"; bn_proj 的 1×1 conv 没改变量级。dec0 自身的 8 块 SwinBlock 也基本是恒等映射 (参见下面)。

### 2. dec0 8 块几乎恒等 (旧现象, HEAD 仍存在)

每个 dec.0.blocks[i] 输出 absmean ≈ 915 (差异 <0.003%) — 与之前 e0b324e 一样。SwinBlock 的 residual `x = x + self.mlp(self.norm2(x))` + `x = x + self.attn(...)` 中, attn/mlp 输出相对 x (来自 bn_proj 的 917 量级) 是小量, 因此块间差异消失。

这不是 dec0 独有的问题 — 任何接收 bn_proj 输出 (量级 ~900) 的 c=512 SwinStage 都会呈现同样模式。

### 3. mlp.2.weight 在块间 std (HEAD)

HEAD 数据下, dec0 mlp.2.weight 块间 std:

| block | mlp.2.weight std (HEAD) |
|------:|------------------------:|
| 0 | ~2.1 |
| 1 | ~3.9 (×10 vs block0) |
| 2 | ~1.1 |
| 3 | ~0.09 |
| 4 | ~0.37 |
| 5 | ~0.002 |
| 6 | ~0.94 |
| 7 | ~9.8 |

仍存在块间震荡 (与之前相同, 因为 mlp_stream 拼装未变), 但量级被 median-8 拉低 (之前某些块上 std 高达 9792, HEAD 下最高 ~10)。

## 根因汇总

### 主因: `_SplitBlock` 无 LN → bn 链放大

* bn 链: 88 → 673 (×7.6 in 8 blocks = 28%/block 累积)
* bn_proj → dec0: 917 → 916 (~恒等)
* dec0 → dec1: 916 → 1.56 — 这是 **`expands[0]` + skip `cat` + `x[:, :256]` 切片** 造成的截断, 不是 dec0 内部放大

### 次因: mlp_stream 错位 (低优先级, 因为 median-8 已大幅削弱)

* `semantic_fill.py:203` 的 `mlp_stream = [m0, m2[786432:], m3[:262144]]` 仍把 MX 段 + layer3 main 灌入 mlp.0/2.weight
* 在 median-8 数据下, 这些错位值已被 per-matrix 自适应缩放正确化, 影响远低于之前 bias=205 (那时 m2.mx std=8950 灌入 mlp.2 是灾难)
* **不需要再改 mlp_stream**, 除非追求 byte-exact 精度

### rel_bias 死代码

`fill_model` 把所有 `relative_position_bias_table` 置零, 实测即便改为 N(0, 0.02) 也对前向无影响 (delta < 0.01%)。

## 修复路径 (优先级排序)

1. **(必须) `_SplitBlock.forward` 加 LN**: 在 fold 2048→512 后, 或在 `+xp` 之前, 加 `nn.LayerNorm(512)`。这是唯一能根治 bn 链指数放大的方式。
2. **(可选) `_ResidualHead` 去 tanh 或加 norm**: 让 tail 保留 bias 差异信息 (现被 tanh × sigmoid 吃掉)。
3. **(可选) mlp_stream 拼装修正**: 把 layer2 MX 段单独做 scale/gate 处理, 不要灌入 mlp.0.weight; layer3 main 也不应直接灌入 mlp.2.weight。这是 byte-accuracy 工作, 不是数值正确性。

## 任务 C 总结

* ~~装填路径不生效~~ — **fill_model 100% 装填成功** (HEAD 下 filled=143,885,012, 0/582 unfilled)。bug 仅为 line 261 `_t` 未定义, 不影响数值。
* ~~dec0 偏大是装填 bug~~ — **不是**。dec0=916 是 bn_proj=917 的恒等传递 + 8 块 SwinBlock 内部 residual 主导。
* **dec0 偏大的真正根因 = `_SplitBlock` 无 LN**。架构问题, 非数据问题。
* HEAD 的 median-8 公式已把所有数值调到合理量级, dec0 偏大是放大累积的自然结果而非错装。