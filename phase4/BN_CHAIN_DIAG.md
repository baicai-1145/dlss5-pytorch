# BN 链激活渐增诊断 (Phase 4 fwdtest)

生成时间: fwdtest 三任务联合跑 (`phase4/diag.py`)
基线 forward: `python3 phase4/semantic_fill.py`, 输入 `64×64`, `bn0..bn7 ≈ 16 → 600+` (≈35×)
最新基线 (e0b324e): `bn0..bn7 ≈ 6 → 590` (≈98× — 装填路径变化使 bn0 起点更低, 末端大致相同)

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
* `wqkv` 在 bn0=102, bn7=2044 (×20), 是 **绝对振幅最高的子层**。
* `side` 与 `ffwd` 的增长率最高 (~35-39×), 是 **链式爆炸的主要驱动**。
* 四子层增长比 (×20~×39) 与外层 `bn` absmean 增长率 (×39) 同量级 → 放大是 **均匀发生在每个子层**, 没有单点突刺, 排除"某个子层权重严重偏置"。指向 **每个 block 内部 GEMM 本身就在几何级数放大**。
* `absmax` 同步增长 (×19-×32) → 是 **真实放大** 而非单个大值稀释均值。

### 推断根因

bn 块 (`_SplitBlock`) 没有残差 LN 也没有任何激活归一化 (forward 只有 `+` 和 `mean`), 输入来自 enc4 输出 → ffwd 无界展开 → wqkv/proj 又各乘一次 2048×2048 → side 6144 → 三路相加再平均折叠。8 块串联, 每块相对振幅增长 ~35%, 累积到 bn7 就 ~40×。

**结构根因**: `_SplitBlock` 缺少任何残差 LN。若在 `_SplitBlock.forward` 末尾或 wqkv/proj 之间插入 `LayerNorm(dim)` 或简单 `x / x.std()`, bn 链放大可被钳制。但这是架构修改, 非本轮任务范围。

---

## 任务 B: MX bias 敏感性

详见 [`BIAS_SCAN.md`](./BIAS_SCAN.md)。

简述: bias ∈ {203,204,205,206,206.5,207} 各档 `enc0`/`enc1`/`bn7`/`dec0`/`tail` 的 absmean **差异 < 1%**, 输出 `0.2060` 完全不变。MX bias 改动对前向传播在该 64×64 配置下基本不可观测。

---

## 任务 C: dec0 偏大根因 — 已转 [`DEC0_DIAG.md`](./DEC0_DIAG.md)

简短总结:

* e0b324e 修复了 `mlp.2.bias` 漏装 (filled 143,877,332 → 143,885,012, +7,680 elem)。装填率100%。
* dec0=444 不是"没装上", 而是 **e0b324e 的 mlp_stream 拼装错位**: MX 段 + layer3 main 被灌入了 mlp.0/2.weight, 产生 1.66-9792 的剧烈 std 震荡, 进而把 dec0 输出推到 2.3e4 量级。
* 完整诊断 (mlp.2.weight 块间 std 表、norm1 震荡表、enc4 vs dec0 对比、根因分析、修复方向) 见 [`DEC0_DIAG.md`](./DEC0_DIAG.md)。

---

## 任务 A 推论

bn 链放大是结构问题 (无 LN) + 数据问题 (错位 mlp_stream 同等地作用在 bn 块上 — 但 bn 块的 mlp_stream 数据流是 `_SplitBlock` 专属, 与 SwinBlock mlp_stream 无关, 所以 bn 块本身用的是 layer0/1/2 各自的原始 main 流, 不走 mlp_stream 拼装路径)。

bn 链放大的**唯一**结构性原因是 `_SplitBlock` 无 LN + 无残差门控 (只有 `gate = zeros(1)` 占位)。即便装填完美, bn 链的指数级放大仍会出现。要根治, 需要:
1. 在 `_SplitBlock.forward` 中给 wqkv/proj 输出加 LN/残差
2. 给 `gate` 参数非零值, 启动 `h = h * sigmoid(gate) + ...` 形式的门控 (目前 gate 是 0, sigmoid(0)=0.5, 等同无门控)
3. 或者在 `_ResidualHead` 之前加一个全局 `bn_proj_norm = nn.LayerNorm(512)`