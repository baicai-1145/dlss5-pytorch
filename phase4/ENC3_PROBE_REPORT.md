# Phase 6 Task 6 — enc3 过热根因 (b15-21 c256 stack + merge2) 体检报告

## TL;DR

**根因找到**: 14× enc3 跳变**不在 b15-21 的 swin 权重里**,而在 **merges.2 (b14, c=128→256 merge) 的 LayerNorm affine**。

* `merges.2.norm.weight` 装填的是 **b14 misc 的前 512 个 fp16 值** — 但那些值是
  **MX scale 因子** (mean=**-8.78**, std=2.21, range [-12.33, 0]),不是 LN gamma (~1.0)。
  LN gamma 全负 → 输出幅度 ~9× 放大并反号 → merge2 std 21.74 (vs enc2 的 1.55)。
* enc3 swin 块本身是**受害者不是元凶**: 单独替换 qkv/proj/mlp.0/mlp.2/rel_bias/
  norm 全部 Kaiming 化,enc3_std 纹丝不动 21.738 (因为输入 merge2 出来就已经 21.7)。
* 健康修复 (**COMBINED-C**): `merges.2.norm.weight=1, bias=0` + `enc3 norm1/2.weight=1`
  + `enc3 mlp.2.weight Kaiming` → **enc3 std 21.74 → 3.32** (Δ=-18.4), **corr 不变** (+0.7976)。
* 联合发现 #2: **enc3 的 mlp.2.weight 也是病态的** (67% 零填充,只有 2 个 hot rows)。
  只修 merge2 不修 mlp.2 时,一旦 norm=1 放行信号,病态 mlp.2 会把 enc3 推回 22.9
  (COMBINED-B)。必须一起修。

## 1. 静态体检 (张量×偏离表)

审计范围: enc.3.blocks.{0..6} 全部 8 张量 + merges.{0..3} 的 norm/reduction。
先验: LN gamma mean=1.0±0.3; LN bias mean=0±0.5; linear bias 0±0.2; rel_bias 0±0.05。

偏离 >3σ 的张量 (16 个,完整列表在 `phase4/.tmp/enc3_audit.json`):

| name | kind | obs_mean | obs_std | prior_mean | dev_σ |
|---|---|---:|---:|---:|---:|
| **merges.2.norm.weight** | ln_gamma | **-8.777** | 2.203 | +1.000 | **-32.6** |
| **merges.2.norm.bias** | ln_bias | **-8.777** | 2.203 | +0.000 | **-17.6** |
| enc.3.blocks.{0..6}.norm1.weight (7个) | ln_gamma | 0.000 | 0.000 | +1.000 | -3.33 |
| enc.3.blocks.{0..6}.norm2.weight (7个) | ln_gamma | 0.000 | 0.000 | +1.000 | -3.33 |

两类异常:
1. **merges.2 的 LN affine 被装填成 MX scale 值** (mean -8.78,严重越界) — 根因。
2. **enc3 所有 norm1/norm2.weight = 0** (fallback 兜底填 0 而不是 1) — 次级问题,
   单独存在时把 enc3 块变成近恒等映射 (信号不放大也不衰减)。

## 2. 数据流证据

| stage | baseline std |
|---|---:|
| enc2 (128ch, b9-13) | 1.554 |
| **merge2 (128→256, b14)** | **21.761** ← 14× 跳变发生在这里 |
| enc3 (256ch, b15-21) | 21.761 (透传,不再增长) |
| merge3 (256→512, b22) | 210.7 (enc4 段问题,task 5 已定位) |

替换实验证明 enc3 块权重不是元凶:

| 单张量替换 | enc3_std | corr |
|---|---:|---:|
| mlp.0.weight → Kaiming | 21.761 | +0.8464 |
| mlp.0.bias → noise | 21.761 | +0.8464 |
| qkv.weight → Kaiming | 21.761 | +0.8464 |
| proj.weight → Kaiming | 21.761 | +0.8464 |
| rel_bias → Kaiming | 21.761 | +0.8464 |
| 全部 enc3 权重 → Kaiming + norm=1 | 21.810 | — |

## 3. 替换实验结果表 (3 crops, seed=42, judge=simple delta)

| 方案 | enc2 | merge2 | enc3 | bn_avg | tail | corr |
|---|---:|---:|---:|---:|---:|---:|
| BASELINE (canonical) | 1.556 | 21.738 | 21.738 | 210.69 | 0.1640 | +0.7975 |
| merges.2.norm.weight=1, bias=0 | 1.556 | 1.904 | 1.904 | 209.49 | 0.1640 | +0.7976 |
| merges.2.norm.weight=1 (bias 保留) | 1.556 | 16.913 | 16.913 | 210.69 | 0.1640 | +0.7976 |
| merges.2.norm=1, bias=1 | 1.556 | 2.389 | 2.389 | 208.73 | 0.1641 | +0.7975 |
| **COMBINED-B**: merge2 fix + enc3 norm=1 | 1.556 | **1.904** | **22.939** ⚠ | 209.93 | 0.1641 | +0.7975 |
| **COMBINED-C**: merge2 + enc3 norm=1 + enc3 mlp.2 Kaiming | 1.556 | **1.904** | **3.316** ✓ | 210.84 | 0.1640 | +0.7976 |
| COMBINED-D: all merges + enc3 norm + mlp.2 Kaiming | 1.769 | 1.850 | 3.541 | 210.27 | 0.1640 | +0.7976 |
| b14 misc[-512:] → merges.2.norm (退化) | 1.556 | 0.017 | 0.017 ✗ | 210.34 | 0.1640 | +0.7976 |

⚠ COMBINED-B 揭示次级 bug: merge2 修好后信号放行,enc3 内部病态 mlp.2.weight
(67% 零、2 个 hot row std=0.4) 把 std 推回 22.9。**单独修 merge2 不够**。

✗ tail512 方案把 norm 填成 0.003 量级 → 信号被杀死,enc3=0.017 是**退化解**,已从推荐池排除。

## 4. 根因机理

```
b14.record (B=229,936B)
  FP16_TAIL[229936] = 155,936  ← semantic_fill.py 的硬编码边界
  → misc = fp16_decode(arr[155936:])   (36,998 vals)
  → put('merges.2.norm.weight', misc)  取前 512 vals
  → put('merges.2.norm.bias',   misc)  同前 512 vals
```

b14 misc 的**前 512 fp16 值实际是 MX scale 因子表** (负值, mean -8.78),
不是 LN gamma。真正的 gamma 可能是别的区段 (或 b14 根本没有 gamma 记录)。

LN 前向: `y = (x - μ)/σ * γ + β`,γ≈-9,β≈-9 → 输出 ≈ -9·(归一化信号) - 9
→ 幅度 ~9× 放大 + 巨大负偏置 → merge2 出口 std 21.7, mean -8.78 与 γ 的统计完全一致。

## 5. 推荐修复 (固化)

写入 `phase4/.tmp/enc3_recommended_fix.json`:

```json
{
  "fix_recipe": "COMBINED-C: merge2 + enc3 norm=1 + enc3 mlp.2 Kaiming",
  "steps": [
    "merges.2.norm.weight = 1.0   (LN gamma, canonical LN init)",
    "merges.2.norm.bias   = 0.0   (LN beta, canonical LN init)",
    "enc.3.blocks.{0..6}.norm1.weight = 1.0",
    "enc.3.blocks.{0..6}.norm2.weight = 1.0",
    "enc.3.blocks.{0..6}.mlp.2.weight ~ N(0, 1/sqrt(384))  (Kaiming, fan_in=384)"
  ],
  "effect": {
    "enc3_std":   "21.738 → 3.316",
    "merge2_std": "21.738 → 1.904",
    "corr":       "+0.7975 → +0.7976 (不变)",
    "tail_std":   "0.1640 (不变)"
  }
}
```

注: tail/bn 指标不动是预期的 — bn 段的 2.39× 过热是 enc4/b31-38 的事
(task 5 已确认 bn_avg 对 b23-29 装填敏感: 210→57),tail std 被
_SplitBlock RMSNorm + tanh head 钳死,不受上游修复影响。

## 6. 结论

1. **enc3 过热的根因张量 = `merges.2.norm.weight` (和 `.bias`)** — b14 misc 解码
   边界错误,把 MX scale 表当成 LN gamma 装填。这是单张量问题,但完整修复需要
   连带修 enc3 的 norm(兜底填 0)和 mlp.2(流耗尽 67% 零填充)两个同源 bug。
2. enc3 (b15-21) 的 swin 权重装填本身没有方向性错误 — 单独 Kaiming 化任何
   张量都不影响 enc3 std (因为病态在输入端)。
3. 下一步建议: 修 semantic_fill.py 的 b14 FP16_TAIL 边界 (155936 → 真实 gamma 区),
   同时把 c=256 块的 stream 消耗顺序改为保证 norm/ffn2 完整覆盖 (与 task 5 的
   C03 方案合并执行)。

## 7. 复现

```bash
python3 phase4/enc3_probe.py   # ~38s, 峰值 RSS ~2.4GB
# 输出:
#   phase4/.tmp/enc3_audit.json             (张量×偏离全表)
#   phase4/.tmp/enc3_recommended_fix.json   (固化修复配方)
```
