# DLSS5 网络架构（社区首份字节级实证版）

> 来源：nvngx_dlssnr.dll (SHA256 ceb6432f...662650, NGX 310.8.0, 2026-08-11 签名, 内部代号 cg2r)
> 方法：权重 blob 153 记录字节级解析 + data.bin 15 个 cubin kernel 名 + 数学拟合交叉验证。
> 所有数字均有字节级证据，非猜测。

## 总览

**DLSS5 = Swin Transformer U-Net（tinlayout 变体）+ 宽头瓶颈 split-Swin + 双 ViT 旁路 + 残差输出**
- 总参数：~147.7M（FP8 编码 140.85 MiB blob）
- 精度：MXFP8 (E4M3)，per-block scale（布局待定位）
- 推理：sm_120 (RTX 50) 专属 cubin；sm_89 辅助 kernel 仅做捕获/后处理
- 训练目标：one-step pixel-space denoising（NVIDIA 官方论文措辞）

## 编码器（5 stage，block 数来自权重实测，非模板实例数）

| stage | 通道 c | heads | blocks | 每块字节 | 公式 |
|---|---|---|---|---|---|
| s0 | 32 | 1 | 3 | 20,672 | **20c²+6c**（FFN r=8!）|
| s1 | 64 | 2 | 3 | 61,760 | **15c²+5c** |
| s2 | 128 | 4 | 5 | 197,184 | 12c²+576（杂项 64×9）|
| s3 | 256 | 8 | 7 | 689,232 | 未破（整除性异常 2⁴）|
| s4 | 512 | 16 | 7 | 1,968,192 | 4-layer 打包 (524288/263168/917568/263168) |

- 段间 merge 块（b4/b8/b14/b22）：swin 块 + **2c²−16** downsample 投影（cubin `_ds` 后缀铁证）
- 杂项序列 64×(3,5,9)：随 stage 翻倍，疑似 window/relpos 相关
- **FFN expansion ratio 随深度递减**（8→5.5→4→…）：`BSTinlayoutConvFfn` conv-FFN 可变宽度
- head_dim 恒 32（heads = c/32，与模板名 `1h_32/2h_64/4h_128/8h_256` 一致）

## 瓶颈（b31-38，8 块，每块 12,587,154 B —— 最重）

**split-Swin：16 头 × head_dim 128 的宽注意力**（与普通 stage 的 32d 不同！）：
```
layer0 = 2048×2048 + 16       ← QKV 大 GEMM（2048 = 16h×128d）
layer1 = 2048×2048 + 2048     ← PROJ（Linear 2048→2048）
layer2 = 12×512² + 128        ← 侧支 c=512: qkv(3c²)+proj(c²)+ffwd(2·c·4c)
layer3 = 2 B 零标量            ← 门控/开关
layer4 = 512×2048 + 2048      ← ffwd expand 入口（Linear 512→2048）
```
cubin_04 kernel 全集：`cc_split_swin_16h_{qkv,proj,ffwd,ffwd_proj,ffwd_inpview,proj_pool,final_head}_512 × {chained,wait,tilesync,fp8}` 变体。

## 解码器（完全镜像）

- 8 块 @1,968,192（512ch）→ b48 up（swin+2c²+480）→ 7 块 @689,232（256ch）
- b56 up（+224）→ 5 块 @197,184（128ch）→ b62 up（+96）→ 3 块 @61,760（64ch）
- b66 up（+64）→ 3 块 @20,672（32ch）→ b70 tail（21,808 + blend_scale 2B）
- up 附加 γ 序列 64/96/224/480 → 收敛 2c（`_upsample` 后缀铁证）
- 镜像对称 57/71；14 个不对称块全是 stage 边界块 ✅

## ViT 双旁路（cubin_05）

- **1D ViT**：`cc_vit_1d_{qkv,attention,ffn_expand,ffn_contract,projection,repack_1d_to_2d,repack_2d_to_1d}` —— 编码 control/UI mask 等低维控制信号，1D↔2D repack 进主干
- **2D ViT**：`cc_vit_{qkv,attention,ffn_expand,ffn_contract,projection}` —— 空间先验处理

## 输出

- `cc_tinlayout_avg_pool_proj_block` 全局池化投影 + `blend_scale`（blob 尾 2B bf16 标量，初始 0）
- 残差加回原图（blend_scale 门控）

## 输入（DLL 字符串 + README 推断，Phase 5 验证）

Color/Depth/MVec/UI+UIAlpha+UICorrection/ControlMask/BidirectionalDistortionField/ScalingRatio

## 辅助 kernel（非网络核心）

`cuda_{font,capture,capture_output_exposure_scale,capture_mv_dilate,capture_buffer_as_texture,clear_view,copy}` + `cg2r_post_process_kernel`（全部 sm_89）

## 已知未解

1. c=256 块公式（689,232 = 2⁴×3×83×173 整除性异常）
2. MXFP8 per-block scale 字节位置（33 字节周期未发现，可能在 cubin 常量或记录头 24B 内）
3. blob header count=19 的语义
4. FFN r 随 stage 精确取值（conv-FFN 结构需 cubin 反汇编定案）
