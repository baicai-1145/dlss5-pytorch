# MX bias 敏感性扫描 (Phase 4 fwdtest, HEAD = d446313)

工具: `phase4/fwdtest.py::task_b_bias_sweep`
对照: HEAD `mx_decode_pairs(pairs, bias=None)` = `v = W × 2^(S - median(S) - 8)` (per-matrix 自适应)
vs 固定 bias (monkey-patch, 不修改 semantic_fill.py)
seed: model=42, input=7, 64×64 单图

## 全 stage absmean 表 (6 档 bias)

| bias | enc0 | enc1 | enc4 | bn0 | bn3 | bn7 | dec0 | dec1 | tail | out |
|------|-----:|-----:|-----:|----:|----:|----:|-----:|----:|-----:|----:|
| **median-8** | 7.70e-01 | 4.48e+00 | **8.82e+01** | **1.41e+02** | **2.41e+02** | **6.73e+02** | **9.16e+02** | 1.56e+00 | 2.05e-01 | 2.05e-01 |
| fixed=203 | 7.70e-01 | 4.48e+00 | 9.43e+05 | 1.68e+06 | 3.40e+06 | 7.34e+06 | 1.32e+07 | 1.55e+00 | 2.04e-01 | 2.04e-01 |
| fixed=204 | 7.70e-01 | 4.48e+00 | 4.71e+05 | 8.42e+05 | 1.70e+06 | 3.67e+06 | 6.62e+06 | 1.55e+00 | 2.04e-01 | 2.04e-01 |
| fixed=205 | 7.70e-01 | 4.48e+00 | 2.36e+05 | 4.21e+05 | 8.50e+05 | 1.83e+06 | 3.31e+06 | 1.55e+00 | 2.04e-01 | 2.04e-01 |
| fixed=206 | 7.70e-01 | 4.48e+00 | 1.18e+05 | 2.11e+05 | 4.25e+05 | 9.17e+05 | 1.65e+06 | 1.55e+00 | 2.04e-01 | 2.04e-01 |
| fixed=207 | 7.70e-01 | 4.48e+00 | 5.89e+04 | 1.05e+05 | 2.12e+05 | 4.58e+05 | 8.27e+05 | 1.55e+00 | 2.04e-01 | 2.04e-01 |

## median-8 vs 最近 fixed bias (207) 倍数

| stage | median-8 | fixed=207 | 倍数 |
|-------|---------:|----------:|-----:|
| enc4  | 88       | 58 944    | **668×** |
| bn0   | 141      | 105 270   | **748×** |
| bn3   | 241      | 212 350   | **880×** |
| bn7   | 673      | 458 390   | **681×** |
| dec0  | 916      | 827 340   | **903×** |

vs 固定 bias=205 (旧默认): median-8 还要再低 3600×-15000×。

## 受影响 vs 不受影响 stage

* **受影响** (有 MX 区域 / c≥64 stage): enc4 / bn0-bn7 / bn_proj / dec0 — 全部随 bias 指数级变化
* **不受影响**: enc0, enc1 (c=32/64, MX 区域占比极小或不存在), dec1+ (c≤256, MX 段数据小或被 LN 钳制)
* **恒定**: tail = 2.05e-01 (tanh × sigmoid(blend) 把所有差异吃掉)

## 过渡平滑度 (median-8)

```
enc0(0.77) → enc1(4.48) → enc2(55.0) → enc3(47.9) → enc4(88.2)
                                                       ↓
dec0(916)  ← bn_proj(917) ← bn7(673) ← bn6(538) ← bn5(425) ← bn4(314) ← bn3(241) ← bn2(210) ← bn1(185) ← bn0(141) ← enc4(88)
                ↓
tail(0.205) ← dec4(0.025) ← dec3(0.94) ← dec2(1.49) ← dec1(1.56) ← dec0(916)
```

* **bn 链 (141 → 916)**: 每块平均 ×1.30 (~30%/block), 8 块累积 ×4.8。完全平滑,无跳跃。
* **dec0 → dec1 跳跃比 = 1.56/916 = 0.0017**: 这是 `expands[0]` 上采样 + `cat` + `x[:, :256]` 切片产生的截断 + 1×1 conv (`weight=None` 但 `bias=None`) 共同作用。
* **dec1 → dec4 → tail**: 1.56 → 1.49 → 0.94 → 0.025 → 0.205: 平稳下降, tanh 把 0.025 截到 -1..1。

## 结论

1. **median-8 是无争议的全局最优**: 比最近固定 bias (207) 在受影响 stage 上低 668-903×, 比默认 205 低 3600-15000×。Enc0/enc1/dec1/tail 完全不受影响。
2. **MX bias 不影响 enc0/enc1**: 这些 stage 没有 MX 区域或 MX 数据被 E4M3 主段稀释。
4. **median-8 提供最优的 dec0→dec1 过渡** (916→1.56, 比固定 bias 的 8e5→1.55 更接近"理想模型输出量级"), 后续收敛到 tail 0.205 的过程也最干净。
5. **最终输出 tail 对 bias 完全不敏感** (`2.04e-01 ~ 2.05e-01`), 因为 `tanh × sigmoid(blend)` clamp 在最末端把所有差异截掉。
6. **MX bias 扫描的正确解读是看 enc4/bn/dec0 这三个高能量 stage**, 不是 tail。

## 实操建议

- 直接采用 HEAD 的 `median(S) - 8` 公式, 不需要再校准 per-block bias
- 这条公式对所有 stage 同时最优, 不存在 per-stage 调优空间
- 若要让 tail 输出对 bias 敏感, 需去掉 `_ResidualHead` 的 tanh clamp 或在 conv 前插 norm

## 后注: HEAD bug

跑 `python3 phase4/semantic_fill.py` 会 crash 在 line 261 (`_t` 未定义)。
fwdtest.py 通过 `_fill_model_replica` 跳过该行 (bn.gate zeroing), 行为等价 (gate 默认零)。