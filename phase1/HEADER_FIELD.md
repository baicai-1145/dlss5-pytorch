# blob 头部 count_field=19 字节考古

> Phase 1 收尾 — Task B  

## 0. 头部原始字节

| 偏移 | 字节数 | hex | 值 | 含义 |
|---|---|---|---|---|
| [0:8] | 8B | `32 a7 cd 08 00 00 00 00` | 0x08cda732 | magic (DLSS5 权重 blob) |
| [8:16] | 8B | `13 00 00 00 00 00 00 00` | **19** | count_field (待解读) |
| [16:35] | 19B | `block0.layer0.layer` | "block0.layer0.layer" | 第一条记录 name (**非 NUL 结尾**) |
| [35:43] | 8B | `e8 54 00 00 00 00 00 00` | 21,736 | u64 A1 (= B+40) |
| [43:51] | 8B | `e8 54 00 00 00 00 00 00` | 21,736 | u64 A2 (重复 A1) |
| [51:59] | 8B | `c0 54 00 00 00 00 00 00` | 21,696 | u64 B (payload 字节数) |

## 1. 关键观察

1. 实际记录数 = **153** (通过链式 walk 验证 152/153 + tail 截断 8B)
2. magic = 0x08cda732 ✓ (与 REPORT.md 一致)
3. **count_field = 19 ≠ 153** (不是记录数)
4. **byte [35] = 0xe8 (非 0x00)** — name **不带 NUL 终止符**, parser 必须知道 name 长度才能定位 u64 A1
5. **count_field = 19 恰好等于 "block0.layer0.layer" 的字符数** — 即第一条记录 name 的长度

## 2. 候选语义假设测试

| 假设 | 候选值 | 实测 | 匹配 |
|---|---|---|---|
| A. # 顶层 stage 容器 | 13 (stem/enc0-4/bn/dec0-4/tail) | 13 | ✗ |
| B. # unique block roles | 19 | **23** | ✗ |
| C. # distinct name patterns | 19 | **6** | ✗ |
| D. # records | 153 | 153 | ✗ |
| E. # blocks (0..70) | 71 | 71 | ✗ |
| F. # unique verbatim names | 153 | 153 | ✗ |
| G. **第一条记录的 name长度** | 19 | **19** | **✓** |
| H. 13 stages + 6 boundary | 19 | 13 stages + 10 boundary = 23 | ✗ |
| I. # cubin operator types | ? | 估计 14-15 | ✗ |
| J. NGX version 相关 (310.8.0) | ? | — | ✗ |

## 3. 决定性证据: count_field = 第一条记录 name 长度

**关键测试**: 检查 byte 35 是否为 NUL 终止符

```
byte [16:35] = "block0.layer0.layer"   (19 bytes)
byte [35]    = 0xe8              (start of u64 A1, NOT NUL)
byte [35:43] = 0x54e8            (u64 A1 LE = 21,736 = B+40)
```

由于 name **不**带 NUL 终止符, parser 必须预先知道 name 长度才能找到 A1 字段.
count_field 的角色 = **第一条 name 长度 (= 19)**.

后续记录的 name 长度在**前一条记录的 terminator**中(最后 u32 = next_namelen):
```
terminator 28B = [u32 0, u32 0, u32 0, u32 1, u32 0, u32 B/2, u32 next_namelen]
b0 terminator: (0, 0, 0, 1, 0, 10848, 19)  ← next_namelen = 19 (b1.name = "block1.layer0.layer" = 19 chars)
```

## 4. 备选解读 (次要)

虽然 count_field = 19 主要是结构性字段, 但有 19 个 unique name pattern 的次要解读:

| # | name pattern | 记录数 |
|---|---|---|
| — | `{block}.layer0.blend_scale` | 1 |
| — | `{block}.layer0.layer` | 71 |
| — | `{block}.layer1.layer` | 24 |
| — | `{block}.layer2.layer` | 24 |
| — | `{block}.layer3.layer` | 24 |
| — | `{block}.layer4.layer` | 9 |

但这只是 6 种 pattern (不是 19), 所以不是模式计数.

## 5. name 长度分布 (额外证据)

| 长度 | 记录数 | 例子 |
|---|---|---|
| 19 | 10 | `block0.layer0.layer` |
| 20 | 142 | `block10.layer0.layer` |
| 26 | 1 | `block70.layer0.blend_scale` |

不同 block 编号位数 (1位 vs 2位) 导致 name 长度变化 (19/20/26),
但头部只声明**第一条**的 name 长度, 后续由 terminator 链式给出.

## 6. 顺序记录 vs 反序记录 (额外观察)

walk_records 返回的 153 条记录**不按 block 编号排序**:
```
block0 → block1 → block10 → block11 → ... → block69 → block7 → block70 → block8 → block9 (截断)
```
说明 blob 写入顺序与 block 编号无关 — 是按某种拓扑或内存布局顺序写入的.
但这不影响 count_field 的语义.

## 7. 结论

**count_field = 19 = 第一条记录的 name 字节数** ("block0.layer0.layer" = 19 字符)

**这是一个结构性字段, 不是计数字段.** 它告诉 parser 第一条记录的 name 占多少字节,
以便定位第一条的 u64 A1 字段. 后续记录的 name 长度通过前一条的 terminator 中的
`next_namelen` (最后 u32) 链式传递.


**术语澄清**:
- count_field 这个名字 (来自 REPORT.md) 容易引起"记录数"的误解
- 实际语义 = "first_name_length" (第一条 name 长度)
- 在 README/PyTorch 装填器里, count_field **不应作为循环计数使用** — 应直接调用 walk_records 走链

## 8. 反驳常见误解

| 误解 | 实情 |
|---|---|
| count_field=19 = 19 个 stage | block_roles.json 有 23 个 unique roles |
| count_field=19 = 19 条 record | 实际 153 条 |
| count_field=19 = version/格式号 | 无版本信息 |
| count_field=19 = 巧合等于 name 长度 | **不是巧合**, name 不带 NUL 终止符必须有这个字段 |
