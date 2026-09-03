# win50 重启后恢复 SSH 手册

> 适用: UCloud Windows Server 2022 (117.50.174.150, RTX 5090)
> 背景: 该镜像账户数据库损坏 (`LookupAccountName 1332`)，sshd 真服务和 SYSTEM 计划任务均无法运行，
> 唯一可靠方案 = Administrator 身份跑 sshd。已注册计划任务 `sshd-autorun`，正常情况重启后自动拉起。

## 情况 A: 重启后 `ssh win50` 直接能连（大概率）

计划任务 ONSTART 自动生效，无需任何操作。

```bash
ssh win50 "echo ok"
```

## 情况 B: 连不上（任务没拉起）

1. **UCloud 控制台 → 主机 → VNC 登录**（用 Windows 的 Administrator 密码）

2. VNC 的 PowerShell 里跑：

```powershell
Start-ScheduledTask -TaskName "sshd-autorun"
# 若上面报任务不存在（被清理），重建：
# schtasks /Create /TN "sshd-autorun" /TR "C:\Windows\System32\OpenSSH\sshd.exe" /SC ONSTART /RU Administrator /RL HIGHEST /F
# 然后 Start-ScheduledTask sshd-autorun

Get-NetTCPConnection -LocalPort 22 -State Listen   # 看到 Listen 即成功
```

3. 回 Mac 验证：

```bash
ssh win50 "echo ok"
```

## 情况 C: 计划任务也失败（极少见，之前未发生过）

兜底 = 手工前台跑（保持 VNC 窗口不关）：

```powershell
& "C:\Windows\System32\OpenSSH\sshd.exe" -d -d
# 显示 "Server listening on 0.0.0.0 port 22" 后，Mac 侧即可连接
# 注意: -d 模式每个连接会话有限，长期用还是修 B
```

## 不要尝试的（已验证无效/有害）

| 操作 | 结果 |
|---|---|
| `Start-Service sshd` / `sc.exe start sshd` | 服务模式启动即崩（账户查询 1332） |
| `sc.exe config sshd obj= LocalSystem` | 无效，同样崩 |
| SYSTEM 身份计划任务 (`/RU SYSTEM`) | 启动即崩（同源问题） |
| `taskkill /IM sshd.exe /F`（远程 SSH 里跑） | 会杀掉自己当前连接的父进程，SSH 立断 |

## 免密原理备忘

- 公钥: Mac 的 `~/.ssh/id_ed25519.pub`（指纹 `zkOBTBIvx6wuaBMK1ZA7scYC7FHHCgtYrCszfaVcqKg`）
- 服务器端位置: `C:\ProgramData\ssh\administrators_authorized_keys`（**不是**用户目录的 `~/.ssh/authorized_keys`——Windows OpenSSH 对管理员组账户只认系统级路径）
- 权限: `icacls ... /inheritance:r /grant "SYSTEM:F" /grant "Administrators:F"` 已设置，勿动

## 连接信息

```
别名:   ssh win50        (~/.ssh/config)
IP:     117.50.174.150   (UCloud, 变动后改 ~/.ssh/config 的 HostName)
用户:   Administrator
GPU:    RTX 5090 32GB, 驱动 610.47
```
