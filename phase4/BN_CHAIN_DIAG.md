# BN 链激活渐增与 dec0 装填诊断 (Phase 4 fwdtest)

生成时间: fwdtest 三任务联合跑 (`phase4/diag.py`)
基线 forward: `python3 phase4/semantic_fill.py`, 输入 `64×64`, `bn0..bn7 = 16 → 633` (≈39×)

---

## 任务 A: bn0 vs bn7 子层 absmean 对比

`_SplitBlock.forward` (calib_model.py L97-114) 子层顺序:
`ffwd(512→2048)` → `wqkv(2048→2048)` → `proj(2048→2048)` → fold2048→512 → `side(512→6144)` → fold6144→512 → sum

| 子层 | bn0 absmean | bn0 absmax | bn7 absmean | bn7 absmax | bn7/bn0 增长率 |
|------|------------:|-----------:|------------:|-----------:|---------------:|
| ffwd  | 9.72        | 4 554      | 345.91      | 86 814     | **35.6×** |
| wqkv  | 102.29      | 477        | 2 044.32    | 9 758      | **20.0×** |
| proj  | 28.89       | 146        | 1 019.54    | 4 629      | **35.3×** |
| side  | 8.34        | 4 554      | 321.17      | 86 814     | **38.5×** |
| **(块总输出)** | **16.25** | 387 | **632.21** | 15 274 | 38.9× |

输出 shape 都是 `(1,4,4,2048)` 或 `(1,4,4,6144)`。

### 定位结论

* **4 个子层全部参与了逐块放大**, 但主导者是 `wqkv` (绝对量级最大) 和 `side`/`proj` (增长率最大)。
* `wqkv` 在 bn0=102, bn7=2044 (×20), 是 **绝对振幅最高的子层**, 占据绝对值最大。
* `side` 与 `ffwd` 的增长率最高 (~35-39×), 它们是 **链式爆炸的主要驱动**。
* 四子层增长比 (×20~×39) 与外层 `bn` absmean 增长率 (×39) 同量级, 说明放大是 **均匀发生在每个子层**, 没有单点突刺 — 排除"某个子层权重严重偏置"的可能, 指向**每个 block 的内部 GEMM 本身就在几何级数放大**。
* `absmax` 同步增长: ffwd 4554→86814 (~×19), proj 146→4629 (~×32), wqkv 477→9758 (~×20), side 4554→86814 (~×19), 与 absmean 同步 → 是 **真实放大** 而非单个大值稀释均值。

### 推断根因

bn 块 (`_SplitBlock`) 没有残差 LN 也没有任何激活归一化 (forward 只有 `+` 和 `mean`), 输入来自 enc4 输出 (absmean=5.38) → ffwd 无界展开 → wqkv/proj 又各乘一次 2048×2048 → side 6144 → 三路相加再平均折叠。8 块串联, 每块相对振幅增长 ~35%, 累积到 bn7 就 ~40×。

---

## 任务 B: MX bias 敏感性

详见 [`BIAS_SCAN.md`](./BIAS_SCAN.md)。

简述: bias ∈ {203,204,205,206,206.5,207} 各档 `enc0`/`enc1`/`bn7`/`dec0`/`tail` 的 absmean **差异 < 1%**, 输出 `0.2060` 完全不变。MX bias 改动对前向传播在该 64×64 配置下基本不可观测。

---

## 任务 C: dec0 装填完整性

逐参数对比装填前 (随机初始化 std) vs 装填后 std:

| 参数 | pre std | post std | post absmean | 是否变化 |
|------|--------:|---------:|-------------:|:--------:|
| `attn.qkv.weight`    | 0.0200 | 0.0440 | 0.0338 | ✅ |
| `attn.proj.weight`   | 0.0200 | 0.0166 | 0.0109 | ✅ |
| `attn.relative_position_bias_table` | 0.0198 | 0.0000 | 0.0000 | ✅ (兜底置零) |
| `norm1.weight`       | 0.0000 | 0.1684 | 0.8540 | ✅ |
| `norm1.bias`         | 0.0000 | 0.1684 | 0.8540 | ✅ |
| `norm2.weight`       | 0.0000 | 0.1684 | 0.8540 | ✅ |
| `norm2.bias`         | 0.0000 | 0.1684 | 0.8540 | ✅ |
| `mlp.0.weight`       | 0.0200 | 0.0439 | 0.0337 | ✅ |
| `mlp.2.weight`       | 0.0200 | 0.0783 | 0.0549 | ✅ |
| **`mlp.2.bias`**     | **0.0000** | **0.0000** | **0.0000** | ❌ **UNCHANGED** |

### 结论

1. **装填路径确实生效**: 9/10 参数都已写入。但 `mlp.2.bias` 唯一未变 → `q2` 变量未消费 (见下)。
2. `attn.qkv.weight` std 从 0.0200 → **0.0440** (2.2×), `mlp.0.weight` 0.0200 → **0.0439**, `mlp.2.weight` 0.0200 → **0.0783** (3.9×)。即装填后权重能量明显高于 Kaiming 初始化。
3. norm 全部被写入到 std=0.1684 (非随机, 来自解码流), absmean=0.8540 (不是 1) — 可能是 fp16 misc 段被误读成 norm 权重。
4. **dec0=474 vs dec1=1.57 的"断崖"并非"没装上"导致**, 而是:
   - dec0 输入是 bn_proj 输出 (absmean 474) — bn 链已经放大到该量级
   - dec0 SwinBlock 8 个 block 内权重 (mlp2 std=0.078) 比 dec1-dec4 的小维权重能量更高, 加之 c=512 的 SwinBlock 的 `mlp_hidden=892` (c=512) 而非 4×, 但 GEMM 输出仍按 c=512 维度展开
   - dec1 通过 `expands` 上采样 + skip 拼接 + 1×1 conv (`x = x[:, :lo]`) 把 c 从 1024 切到 256, **维度截断本身是一道天然降幅度闸**, 配合 conv2d (`weight=None` bias 传入时是 None 时启用) 后值被洗一次, 才出现 1.57 的稳态
   - 因此"断崖"是 **bn→dec0 链 + dec0 内部 GEMM** 共同维持的高能量 + dec1 进入 1×1 conv 截断后的低能量, 物理上是合理的。

### 代码 Bug 修复

**位置**: `phase4/semantic_fill.py` fill_swin / SplitSwin 装填循环。

**Bug**: `mlp.2.bias` 在 SplitSwin 路径下未消费流。检查 `fill_model` 中:

```python
# c=512 SplitSwin 装填循环 (semantic_fill.py ~L180)
for pn_suffix, src in (('attn.qkv.weight', m2), ('attn.proj.weight', m1),
                       ('mlp.0.weight', m2), ('mlp.2.weight', m0)):
    pn = f'{prefix}.{pn_suffix}'
    if pn in pmap and pn in unfilled:
        q2 = put(pn, src) if src is not None else 0
# ↑ q2 仅在循环内赋值, 每次覆盖, 等于未消费
```

之后:

```python
misc_all = np.concatenate([x for x in (s1, s3) if len(x)]) if (len(s1) or len(s3)) else np.zeros(0)
for suffix in ('norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias'):
    pn = f'{prefix}.{suffix}'
    if pn in pmap and pn in unfilled and len(misc_all):
        put(pn, misc_all)
# ↑ 只覆盖 4 个 norm; mlp0/2.bias 未列
```

且 `mlp.0.bias`, `mlp.2.bias` 不在列表 → 兜底分支用 `_init_weights` 的零 bias 填充 (与原 init 一致), 解释了 std=0 不变。

**修复建议**: 在 `misc_all` 之后追加 `('mlp.0.bias', ...), ('mlp.2.bias', ...)`, 用 `m0`/`m1`/`m3` 残留流做来源。

### 修复后重跑前向验证

修复 `semantic_fill.py` 后重跑 `python3 phase4/semantic_fill.py`:
- 装填统计应仍是 `filled params: 143,877,332; unfilled tensors: 0/582`
- `dec0.blocks.0.mlp.2.bias` 应不再为全零 (std > 0, 来自真实解码流)
- `dec0` absmean 可能微变 (修正 bias 后 mlp 输出偏移), 但 `dec1` 1.57 / `tail` 0.206 大概率不变 (因后续 LN/conv 会吸收)

> 任务 A 提醒: 即便修复了 mlp.2.bias, dec0=474 的高能量主要来 **bn 链放大**, 不是装填 bug。要降 dec0 需在 `_SplitBlock` 内加 LN/残差, 或在 `bn_proj` 后插 norm。