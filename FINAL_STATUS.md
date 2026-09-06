# FINAL_STATUS — DLSS5-NR 静态逆向终态报告（R34，33 轮收官）

> 日期：2026-09 · 仓库：dlss5-pytorch · 权重：`weights_blob.bin`（147,695,410 B，md5 `1b184b7058c99d0421536e2d79abf7c2`）
> 方法论：SASS/记录头/CB 捕获逆向优先，行为探针标注血统，拟合层仅作诊断罗盘。

---

## 1. 33 轮全景时间线（每轮一句）

| 轮 | 一句话 |
|---|---|
| R1-R2 | flat 探针 U 曲线测定（5 灰阶，±0.0005 确定性 oracle）；blob 153 记录解析框架 |
| R3 | LN γ/β 加载重构，582 参数准确对齐 |
| R4 | Attention 变体消融（Cubic vs Bounded-Softmax） |
| R5 | Record Decode v2：c128 FFN=256、c256 fp16 带、c512 4-layer 结构确立 |
| R6 | 全局 E4 尺度 ×0.25 标定（还原 ×1.0 饱和排除）；mlp.2 脏行归零 |
| R7 | A·L/B 路径比例分析；b48 up-swin 扩增 |
| R8 | b70 尾部卷积输入主序 `permute(1,0,2,3)`；空间对齐 |
| R9 | SASS 绿通道极性对齐（G 权重取负） |
| R10 | dec.4 通道根因 + G DC 偏置中和 + 物理黑电平门控 |
| R11 | G/B 行 U 型双向对齐（gbias 1.20 / bbias 0.40） |
| R12 | MV 轴分裂极性（U=−0.14 反相 / V=+1.12）+ 尾部符号对偶 |
| R13 | σ-gate 系统排查（排除）+ 官方编辑指纹：98.2% 能量在 3×3 低通带 |
| R14 | Box-Muller 高斯合成铁证（seed hash ×0x9E3779B1）；RMS 排除 |
| R15 | 高斯 lane 增益网格负封盘；tone-curve 2D 曲面突破（corr +0.52，诊断罗盘） |
| R16 | 多维记分板入库（corr/锐度/edit-ratio + 防过拟合协议） |
| R17 | MV stride 审计；up 分支 expand-GEMM 否定（字节实证） |
| R18 | b48/56/62/66 字节预算重审；expand 移除接受 |
| R19 | dec 上采样核 SASS 反推；245KB 神秘带双路径闭环 |
| R20 | 锐度通路指纹 + 血统登记清单 |
| R21 | c[0x180] = 运行时参数块（静态收口）；未装载区全图 10.4MB |
| R22 | 未装载区行为探针五族全负（辅助尺度表，非 tap 权重）；history 首正信号 |
| R23 | history blend w=0.25 标定；16 帧递归记分板无回退 |
| R24 | prev_output-lane 假设否决（门控逻辑：仅 0x60 受指针门控）；8-lane 图确立 |
| R25 | b0 复合记录发现（stem + 19072B 非链路数据） |
| R26 | off-by-one 假设否决（A/B 崩盘 + oracle 双证），enc.0 映射稳固 |
| R27 | simple_blend epilogue 全指令级解码；outview 4 平面写出；cubin_13 = Oklab pow 曲线 |
| R28 | 完整 blend 公式 oracle 验证（err 0.00061）；探针运动冻结局限记录 |
| R29 | full tail 模式入库（结构参照）；定案清单 10 条；README 状态节 |
| R30 | **0x6e = 当前输入色（H_A）三重证据定案**；驻留抑制走 c[0x60] 内部递归 |
| R31 | full tail 转正评估否决（corr 倒退 + 锐度无增益）；simple 保持默认 |
| R32 | 窗口内锐化链搜索：39 内核普查无独立锐化模块（负结果，静态侧关闭） |
| R33 | E4M3 激活量化模拟（corr 中性 Δ−0.0011）；frida 规格落地 `.frida/` |

（R1/R2/R16-R18 详情见 REPORT.md 完整表格；上表为单行摘要。）

## 2. 记分板终态（clean 1920×1050，16 帧）

| 指标 | 官方 | 我们（终态配置） |
|---|---|---|
| 纯网络 corr（held-out 8/10/13/15） | — | **+0.3927**（zero-expand，E4M3 量化中性） |
| flat oracle U 曲线 | — | **三位小数匹配**（5 灰阶 × RGB 逐点） |
| 锐度 @1px | +23.7% vs 输入 | **−12.0%** vs 官方 |
| edit-ratio（\|Δrep\|/\|Δoff\|） | — | 11.76（simple）/ 7.85（full 结构版） |
| MS-SSIM / LPIPS（含诊断 LUT） | — | 0.86525 / 0.2538（全维度无回退） |
| luma/chroma amp（Δ 分解） | — | 0.930 / 0.964（噪声级） |

终态配置：`DLSS5_TAIL_SIGN=game`、MV U=−0.14/V=+1.12、zero-expand、history blend w=0.25（control 槽）、tail mode=simple。

## 3. 血统清单汇总

- **REVERSED（真逆向，22 项）**：架构 5 级 swin + split-swin 核心；MpCubicSilu 精确公式；LN(γβ)；尾部 conv 主序 + G 极性 + blend_scale=13×2⁻¹⁰；MV 轴约定 + tail 符号对偶；E4 ×0.25；Box-Muller 合成；编辑能量指纹；cubin_13 Oklab pow 曲线；simple_blend epilogue 公式（σ 门控 + MV-bicubic + 六项归一 + 减法残差，oracle err 0.00061）；outview 4 平面布局；pre_block 8-lane 图；up 分支无 expand-GEMM；b0 复合记录 + enc.0 映射；未装载区=反量化尺度表；0x6e=当前输入色（三重证据）；窗口内无独立锐化模块（39 内核普查）；E4M3 激活量化点位（HMMA 输入侧 10 簇）。
- **CALIBRATED（标定，2 项）**：history blend 权重 w=0.25（lane 结构 REVERSED，权重运行时不可得）；E4 全局尺度 ×0.25（字节证据 + 行为双重锁定）。
- **FITTED（拟合，仅 1 项，已降级诊断罗盘）**：tone-curve 2D LUT（R15，corr +0.52 那次的双门槛通过是诊断性指标，不作为模型组件交付；记分板主线不带 LUT）。

## 4. 静态可复现性边界

**已尽（静态可复现）**：
- 权重装载：153 记录字节级对齐（bn 链残差 30779→673），load 即复现官方 E4 权重
- 架构：5 级 swin + split 核心 + 桥 + dec + b48 up + post blend 全部数对
- tail 公式：simple_blend epilogue 逐指令级（σ·MV-bicubic − net_raw）
- 输入约定：MV 轴/极性、padding、行主序、black-gate
- 后处理：cubin_13 Oklab resolve 可用 numpy 逐字复现（cap3_check.py 已验）

**不可（静态边界外）**：
- 生成纹理 ~2% 高频：需要"训练过的权重"，静态不可伪造（R13 指纹 + R22 探针双重关闭）
- attention 数值动态：窗口内 softmax/FFN 的 bit 级仿真需逐 HMMA 布局仿真（收益极低，R33 已证 E4M3 数值中性）
- x_net 真实门控权重 / c[0x180] 工作区：运行时填充，静态 blob 无初值（R21）

## 5. 剩余路线图（均超出静态边界，按性价比排序）

1. **接受现状**：+0.3927 corr + 全维度无回退 + 色调全对 —— 已是静态极限
2. **NSight CB dump**（若重返 Windows）：x_net 门控权重预估收益 ≈ +0.01 corr（R33 数值中性佐证天花板非门控）；c[0x180] 已证为插值表（R19 up 无 GEMM）—— 边际收益低，frida 已排除（NGX18 走 D3D12 驱动路径非用户态 CUDA）
3. **权重微调/蒸馏**：用官方输出做蒸馏目标训练残差（可突破生成纹理 2%），但那是训练项目非逆向项目
4. **锐度通路**：−12% 缺口 = in-window 动态（数值仿真收益低）+ 训练项，无静态解

## 6. 一键复现指南

```bash
# 依赖
pip install torch numpy pytest  # CUDA GPU (RTX 3090 级) 必需

# 权重校验
md5sum weights_blob.bin   # = 1b184b7058c99d0421536e2d79abf7c2

# 测试（应 9/9 通过）
PYTHONPATH=. python3 -m pytest tests/ -q

# 推理（clean 场景 16 帧记分板）
PYTHONPATH=. DLSS5_TAIL_SIGN=game DLSS5_MV_U_SCALE=-0.14 DLSS5_MV_V_SCALE=1.12 \
  python3 .tmp/r16_gen_replica.py          # 生成 replica npy
PYTHONPATH=. python3 eval_suite.py --data cap3_live/clean --replica-npy .tmp/replica_clean.npy

# E4M3 激活量化对照（可选）
PYTHONPATH=. DLSS5_ACT_FP8=1 DLSS5_TAIL_SIGN=game DLSS5_MV_U_SCALE=-0.14 DLSS5_MV_V_SCALE=1.12 \
  python3 .tmp/r16_gen_replica.py

# 可视化（四象限 + triptych）
# 见 eval_suite.emit_visuals / R23 r23_hist_full.py 尾部调用
```

关键开关：`DLSS5_TAIL_MODE`（simple*|full）、`DLSS5_ACT_FP8`（0*|1）、`DLSS5_TAIL_SIGN`（oracle*|game）、`DLSS5_MV_U_SCALE`（−0.14）、`DLSS5_MV_V_SCALE`（+1.12）、`DLSS5_NO_BLACK_GATE`（0*|1）。
