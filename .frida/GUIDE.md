# x_net / c[0x180] frida 抓取手册（Windows 侧傻瓜版）

> 目标：两样东西
> ① `simple_blend` 内核 launch 参数区（~0x1E8 字节）——从中找门控卷积权重指针
> ② 上采样核（upsample）的 c[0x180] 权重工作区显存内容（对比 blob 未装载 10.4MB）
>
> 全程不需要重编译任何东西。抓完把 `C:\xnet_dump\` 整个目录打包传回。

## 步骤 0：确认 frida 已装

PowerShell 里跑：

```powershell
python -m pip show frida
```

- 有输出（Name: frida）→ 继续
- 没有 → 装：

```powershell
python -m pip install frida frida-tools
```

再跑一遍 `python -m pip show frida` 确认。记下版本号（后面要用）。

## 步骤 1：把脚本放到 C 盘根

把本目录的 `xnet_dump.js` 复制到：

```
C:\xnet_dump.js
```

（直接文件管理器复制粘贴即可）

## 步骤 2：找游戏进程 PID

启动游戏（或 OptiScaler 注入的目标进程），等它跑到能出画面。

然后 PowerShell：

```powershell
tasklist | findstr /i "Control"
```

（如果进程名不是 Control，换成实际的 exe 名再 findstr）

记下 PID（最后一列数字）。假设是 `12345`。

## 步骤 3：注入

```powershell
cd C:\
frida -p 12345 -l xnet_dump.js -o xnet_dump.log
```

看到 `xnet_dump ready — dumps go to C:\xnet_dump` 就成功了。**别关这个窗口。**

## 步骤 4：触发内核 launch

回到游戏，让它跑 30-60 秒（任何有画面运动的场景都行）。simple_blend 和
upsample 内核每帧都会 launch，脚本会自动拦截并 dump。

## 步骤 5：收工

回到 frida 的 PowerShell 窗口按 `Ctrl+C`（或输入 `exit` 回车）退出。

检查产物：

```powershell
dir C:\xnet_dump\
dir C:\xnet_dump.log
```

应该看到：
- `xnet_dump.log`（launch 日志，最重要——先看这个）
- `C:\xnet_dump\params0_*.bin`（每个唯一指针 4KB dump）
- 可能还有 `ws_*.bin`（工作区 dump，如果 hook 到了 upsample）

## 步骤 6：打包传回

```powershell
tar -czf xnet_dump_all.tgz C:\xnet_dump C:\xnet_dump.log
```

把 `xnet_dump_all.tgz` 传回 3090 侧（Mac 中转即可）。

## 常见问题

| 现象 | 处理 |
|---|---|
| `Failed to attach: unable to find process` | PID 错了，重跑步骤 2 |
| `Failed to attach: need root` | 用管理员身份开 PowerShell 再跑 |
| `nvcuda.dll not found` | 游戏还没初始化 CUDA，等 30 秒再注入，或用步骤 3 的 spawn 方式 |
| frida 版本和 python 不匹配 | `python -m pip install --upgrade frida frida-tools` |
| dump 文件都是空的 | 日志里应有 `cuMemcpyDtoH failed rc=...`，把 log 传回来即可分析 |

## 传回后的分析（3090 侧做，Windows 不用管）

- 从 `xnet_dump.log` 找 simple_blend 的 launch（参数槽 0-6 = 纹理句柄）
- 用 4KB dump 反推门控权重指针 → 定位门控卷积权重的显存地址
- 对比 blob 未装载 10.4MB 的分布 → 确认工作区是否为反量化后的权重
