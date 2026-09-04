# Capture v3 — 调用侧修改指南（DlssNrFeature_Dx12.cpp）

基于 OptiScaler_DLSSNR 主仓 `OptiScaler/dlssnr/DlssNrFeature_Dx12.cpp`。
共 4 处修改，全部在 `RunDlssNr`（原 `~line 1041-1280`）内。头文件用本目录的
`DlssNr_Capture.h` 整体替换原文件。

## 0. 替换头文件

把仓库里 `OptiScaler/dlssnr/DlssNr_Capture.h` 替换为本目录版本（v3）。
record 签名变了，必须同步改调用点，否则编译错误——这是故意的。

## 1. Scalar 记录块扩展（原 line ~1124-1143）

在现有 `g_capture.setScalar(...)` 块后追加 Resolve 全参数（位精确复现必需）：

```cpp
    // Capture v3: everything the resolve composition used, so the raw model output
    // can be rolled forward to "after" offline and the composition itself verified.
    if (g_capture.isActive() && !g_capture.readyToWrite())
    {
        const Config& ccfg = *Config::Instance();
        g_capture.setScalar("intensity", ccfg.DlssNrIntensity.value_or_default());
        g_capture.setScalar("style", (double) ccfg.DlssNrStyle.value_or_default());
        g_capture.setScalar("localStructure", ccfg.DlssNrLocalStructure.value_or_default());
        g_capture.setScalar("localTone", ccfg.DlssNrLocalTone.value_or_default());
        g_capture.setScalar("skinStructure", ccfg.DlssNrSkinStructure.value_or_default());
        g_capture.setScalar("autoMask", ccfg.DlssNrAutoMask.value_or_default() ? 1.0 : 0.0);
        g_capture.setScalar("whitePoint", ccfg.DlssNrWhitePointScale.value_or_default());
        g_capture.setScalar("transferStrength", ccfg.DlssNrTransferStrength.value_or_default());
        g_capture.setScalar("colourStrength", ccfg.DlssNrColourStrength.value_or_default());
        g_capture.setScalar("isHdrBuffer", isHdrBuffer ? 1.0 : 0.0);  // passthrough flag
        g_capture.setScalar("maxRatio", ccfg.DlssNrMaxRatio.value_or_default());
    }
```

注意：这段要放在 `whitePoint` / `isHdrBuffer` 都已定义的位置（原 line ~1063 之后）。

## 2. record() 调用替换（原 line ~1265）

找到：

```cpp
        if (g_capture.isActive())
        {
            g_capture.record(cmdList, device, g_nr.colorCopy,
                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, target,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
```

替换为 v3 签名（在 resolve Dispatch 之后、output 的 SRV→UAV barrier 之前插入；
此时 output 已是 SRV 态，正好可 copy）：

```cpp
        if (g_capture.isActive())
        {
            // v3: record the model's raw output while it is still in SRV (the resolve just
            // read it; the barrier below would return it to UAV). hdrCopy holds the
            // untouched frame. reset flag tells the offline replica when the recurrent
            // history restarted.
            g_capture.record(cmdList, device, g_nr.colorCopy,
                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, g_nr.output,
                             g_nr.hdrCopy, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                             g_frames, wasResetThisFrame ? 1 : 0);
```

其中 `wasResetThisFrame` 需要在 evaluate 调用前记录（原 line ~1174 之前）：

```cpp
    const bool wasResetThisFrame = g_nr.reset;   // consumed by the evaluate below
```

（evaluate 内部会把 `g_nr.reset` 置 false，所以要在调用前取快照。）

## 3. resolve 后的 barrier 顺序

原代码在 record 之后有：

```cpp
        if (g_capture.readyToWrite() && g_captureWriteAtFrame == 0)
            g_captureWriteAtFrame = g_frames + 8;
```

保持不变。v3 的 record 内部对 `g_nr.output` 做 SRV→COPY_SOURCE→SRV 往返 barrier，
与随后 `output` 的 SRV→UAV barrier 不冲突（顺序执行）。

## 4. 每帧 dump 大小估算（16 帧，1920×1050）

| plane | format | 单帧 | 16 帧 |
|---|---|---|---|
| before / model_input (colorCopy) | R10G10B10A2 | 7.9 MB | 126 MB |
| model_raw (output) | R10G10B10A2 | 7.9 MB | 126 MB |
| hdr_copy | R10G10B10A2 | 7.9 MB | 126 MB |
| after (target) | R10G10B10A2 | 7.9 MB | 126 MB |
| depth | R11G11B10 | 7.9 MB | 126 MB |
| motion | R16G16 | 7.9 MB | 126 MB |

合计 ~756 MB / 16 帧。readback 堆是默认堆的两倍大小但瞬时；5090 机器 32GB
系统内存无压力。

## 5. 触发方式（不变）

菜单 RequestCapture 或 180 帧自动触发（kAutoCaptureAfterFrames）。
建议抓两组：
- **冷启动组**：游戏加载后立即触发（frame 0 = reset 1，后续 reset 0）→ 研究递归预热
- **稳态组**：游戏运行 1 分钟后触发 → 全部 reset 0，研究稳态行为
