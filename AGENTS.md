# AGENTS.md — dlss5-pytorch 全局项目约束

逆向 NVIDIA DLSS-NR (nvngx_dlssnr.dll) 为位精确 PyTorch 副本。历史: Mac 主控 + 3090 执行 + GitHub 同步。详细轮次记录见 REPORT.md / docs/。

## 数据铁律
- cap3_live/ (探测+游戏捕获) 永不删除、永不提交 git。
- 权重 md5 必须是 1b184b70 开头; fp16 装载: fromfile→reshape(H,15360)→view(float16)→[...,:3]。
- 推理必须 bf16 autocast (fp16 权重直接跑会 NaN)。

## 验收体系 (v2, 2025-09-05 起强制, 见 eval_suite.py)
- 任何'修复'必须过完整记分板: 画质方向性(锐度/颗粒/饱和/伪影, 三列 input/replica/official 对照, 方向必须同官方) + delta分解(全局tone|平滑场|高频带|色度, 各自corr+幅度比) + MS-SSIM。
- 单一指标 (PSNR/delta-corr/edit-ratio 任何一个) 不得作为唯一验收依据——历史上已被骗三次。
- 拟合层 (tone-LUT/标定参数) 必须在 REPORT.md 标注血统: [REVERSED 真逆向 | CALIBRATED 标定 | FITTED 拟合], FITTED 层需跨场景(cold_game)迁移验证才算数。
- 每轮交付 cap3_vis/ 四象限 zoom(人物/平坦/边缘/颗粒)+triptych 供人眼验收; 人眼结论优先于指标。

## 已定案的逆向结论 (不要重复推翻, 除非有新硬证据)
- 架构: enc 1H/32→2H/64→4H/128→8H/256 swin + 16H split-swin 核心(512→1024) + 桥(1024→512) + dec 上采样跳连 + b48 up-swin 块 + post blend。
- 激活 MpCubicSilu: clamp(x,±4); p=-0.0559082*|t|+0.447266; a=t*p+0.894531; y=x*a。
- LayerNorm(有γβ) 已确认; RMSNorm 假设已消融排除。
- 尾部 conv: 输入主序 [in][out][kh][kw]; G 通道极性负; blend_scale=13×2^-10 真实记录。
- MV 约定: U=-0.14(反相), V=+1.12; tail 符号对偶 (oracle 默认, DLSS5_TAIL_SIGN=game 切换)。
- E4 全局尺度 ×0.25 (还原×1.0 会 pre-tanh ±900 全饱和, 已排除)。
- pre_block 含 Box-Muller 高斯合成 (seed×0x9E3779B1 hash+tile 坐标, sin/cos 2π, 3 lane FP16+1.0); 静态注入 hp-corr 封顶 0.05, 已负结果关闭。
- 官方编辑指纹: 98.2% 能量在 3×3 低通带 = 平滑色调校正场; 高频仅 ~2% (生成式纹理, 需训练过的权重, 静态不可伪造)。
- cubin_13 = Oklab resolve (cbrt 牛顿迭代), 非噪声门控; σ-gate 假设已排除。
- cubin_13 = 逐像素 Oklab 域 pow 曲线后处理 (RCP×27/LG2×34/EX2×42, 0×HMMA, 无空间卷积/锐化核), 运行在网络之后消费 outview 4 平面 —— 非锐度算子 (R27 定案)。
- simple_blend epilogue (R27 SASS 全解, R28 oracle 验证 err 0.00061): out = σ(x_net)·[Σ wᵢ(mv)·texᵢ(0x6e)]/norm − net_raw; MV-bicubic 权重在线计算 (FMUL 链), 六项归一化 (MUFU.RCP), sigmoid 门控 (EX2+RCP); 减法残差在 simple_blend 层级 SASS 实证 (R10 'Out=In−net_out')。
- outview 4 平面分离写出 (STG.E ×4, 3 color + 1 aux, 基址 c[0x168]) —— 平面布局非交错 RGBA (R27)。
- pre_block 8-lane 输入图 (R24): 4 tex (RGB+alpha≈depth) + 2 motion + 1 history + 1.0-const; 0x58 = MV 偏移双三次采样当前输入 (无门控=始终绑定), 0x60 = prev_output (指针 NULL 门控 = history lane); R14 'gauss 3 lane' 读法作废待修正。
- up 分支无 expand-GEMM: 字节实证 (b48 fuse zone 干净带实验更差, R17) + R19 相位 remix ±0.006/sharpE 不变 —— expand 内容非锐度杠杆, zero-expand 最优。
- b0 = [stem 2592][剩余 19072B 非链路数据] 复合记录, 但 enc.0 = [b1,b2,b3] 映射正确 (R26 off-by-one 证伪: A/B 崩盘 +0.0191 vs +0.3731 + oracle 三位小数双证); b0 剩余区语义开放不阻塞。
- 未装载区 (10.4MB) = 行为情性辅助数据 (引擎内部反量化尺度表), 非 tap 权重 (R22 五族解码全负); tap 权重只存在于运行时工作区。
- history 递归: control 槽 blend w=0.25 标定 (R23), lane 结构 REVERSED (R17-B), 混合权重 CALIBRATED; 收益场景运动量依赖 (steady Δ+0.086 vs clean ≈0)。
- 锐度载体 = 门控 MV-bicubic 残差 (simple_blend 内, R27/R28); 0x6e 绑定终审 **H_A = 当前输入色** (R30 三重证据: P3 跳变帧无幽灵 + P2 旧边位无携带 + 无偏副本计分平手; 早前 H_B 优势为官方 prev 真值泄漏伪优势)。驻留帧深抑制走网络内部状态递归 (c[0x60]) 不经 0x6e。
- 窗口内无独立锐化模块 (R32 全内核算符普查): 高频响应 = attention 对细节 token 放大 + MpCubicSilu FFN, 系数即已装载权重; HMUL2+32 增量仅 post_block blend 区 (已解码)。静态侧最终关闭; B 轨 frida 仅剩 x_net 门控权重。锐度通路静态可解部分全部关闭; 剩余 (x_net 真实门控权重/生成纹理) 需运行时抓取或训练。

## 逆向铁律
- 准确信息必须通过逆向获取 (SASS/记录头/CB/捕获dump), **禁止用猜测或数据拟合替代逆向**。拟合层只允许作诊断罗盘(残差→反推未解码通路), 一律不得作为模型组件交付或长期依赖。
- 每个未解之谜先问"DLL 里哪里能读到答案", 读不到才允许行为探针; 行为标定必须标注血统且限期被逆向替换。
- 发现自己在调参刷指标时立即停下: 那是拟合信号, 回去读 SASS/记录头。

## 工作流
- 每轮: 逆向假设→静态证据→实施→记分板验收→REPORT.md 记录(负结果也记, 标注血统 REVERSED/CALIBRATED/FITTED)→commit; Mac 侧 git am + pytest 4/4 + push GitHub。
- 提交信息格式: 'area: 描述 (关键指标变化)'。
- 长任务后台跑; bash 输出用 rg 不用裸 grep。
