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

## 工作流
- 每轮: 假设→探针→实施→记分板验收→REPORT.md 记录(负结果也记)→commit; Mac 侧 git am + pytest 4/4 + push GitHub。
- 提交信息格式: 'area: 描述 (关键指标变化)'。
- 长任务后台跑; bash 输出用 rg 不用裸 grep。
