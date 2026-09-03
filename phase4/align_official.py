"""官方 DLL 输出对齐: cap2_live 真值 → PyTorch 复刻 → PSNR/corr 归因

数据: .tmp/cap2_live/ (Capture v2 抓的 Control 实机帧)
  model_input_NN.raw  R10G10B10A2 1920x1050  — 模型实际输入 (corr=1.000 with before)
  after_NN.raw        R10G10B10A2             — 官方 NR 输出
  depth_NN.raw        R11G11B10F              — 深度引导 (R11 float)
  motion_NN.raw       R16G16F                 — 运动矢量 (mvScale 1920x1050 归一化)

对齐逻辑:
  官方: after - before = 官方网络的编辑量 (含 resolve 合成)
  复刻: residual = 复刻网络前向输出
  核心指标: corr(residual, official_delta) — 空间结构是否复现
"""
import sys, os, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase3'))
sys.path.insert(0, HERE)
from semantic_fill import load_all, fill_model
from dlss5.calib_model import DLSS5NetCalib
BLOB = os.path.join(HERE, '..', 'weights_blob.bin')
CAP = os.path.join(HERE, '..', '.tmp', 'cap2_live')

W, H = 1920, 1050
SIZE = 96


def load_rgb(name):
    u = np.fromfile(os.path.join(CAP, name), dtype=np.uint32).reshape(H, W)
    return np.stack([((u >> s) & 0x3FF) / 1023. for s in (0, 10, 20)], -1)  # H,W,3


def load_depth(name):
    """R11G11B10F: R = 6bit mantissa + 5bit exponent (无符号). 解码为线性距离."""
    u = np.fromfile(os.path.join(CAP, name), dtype=np.uint32).reshape(H, W)
    r11 = (u & 0x7FF).astype(np.uint16)
    e = (r11 >> 6) & 0x1F
    m = r11 & 0x3F
    val = np.where(e == 0, (m / 64.0) * 2.0 ** -14,
                   (1.0 + m / 64.0) * 2.0 ** (e.astype(np.float32) - 15.0))
    return val.astype(np.float32)  # H,W


def load_motion(name):
    m = np.fromfile(os.path.join(CAP, name), dtype='<f2').reshape(H, W, 2)
    return m.astype(np.float32) * np.array([1920.0, 1050.0], np.float32)  # → 像素单位


def psnr(a, b, peak=1.0):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(peak * peak / max(mse, 1e-12))


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    crops = [  # (y0, x0, 说明) — 选 NR 效果强的区域 (之前分析: 亮区 NR 弱, 中低亮度 NR 强)
        (400, 700, '中偏左'),
        (300, 1100, '中偏右'),
    ]
    print(f'加载帧 {idx}...')
    bef = load_rgb(f'model_input_{idx:02d}.raw')
    aft = load_rgb(f'after_{idx:02d}.raw')
    dep = load_depth(f'depth_{idx:02d}.raw')
    mv = load_motion(f'motion_{idx:02d}.raw')
    official = (aft - bef).transpose(2, 0, 1)  # 3,H,W
    print(f'官方编辑量: mean={official.mean():+.4f} std={official.std():.4f}')

    print('加载复刻模型 (147.7M 参数)...')
    t0 = time.time()
    torch.manual_seed(42)
    m = DLSS5NetCalib().eval()
    by = load_all()
    fill_model(m, by, blob_full=open(BLOB, 'rb').read())
    print(f'模型就绪 ({time.time()-t0:.0f}s)')

    for (y0, x0, tag) in crops:
        c = lambda a: a[y0:y0 + SIZE, x0:x0 + SIZE]
        rgb = c(bef).transpose(2, 0, 1)[None]                       # 1,3,S,S
        d = c(dep)[None, None]                                       # 1,1,S,S
        v = c(mv).transpose(2, 0, 1)[None]                           # 1,2,S,S
        off = c(aft).transpose(2, 0, 1) - c(bef).transpose(2, 0, 1)  # 3,S,S

        # 深度归一化 (中位数稳健):
        dmed = np.median(d)
        dn = np.clip(d / max(dmed, 1e-6), 0, 4) / 4.0

        t = time.time()
        with torch.no_grad():
            res = m(torch.from_numpy(rgb.copy()).float(), torch.from_numpy(dn.astype(np.float32)),
                    torch.from_numpy((v * 0.02).astype(np.float32)), torch.from_numpy(rgb.copy()).float())[0].numpy()
        dt = time.time() - t

        # 指标: 结构相关 / 幅度 / alpha 扫描 PSNR
        cc = np.corrcoef(res.ravel(), off.ravel())[0, 1]
        base = psnr(off, np.zeros_like(off))
        best_a, best_p = 0.0, -1e9
        for a in np.arange(-2.0, 2.01, 0.25):
            p = psnr(off, a * res)
            if p > best_p:
                best_p, best_a = p, a
        print(f'\n[{tag}] crop@({y0},{x0}) 前向 {dt:.0f}s')
        print(f'  corr(复刻残差, 官方编辑) = {cc:+.3f}')
        print(f'  复刻 std={res.std():.4f} vs 官方 std={off.std():.4f} → 比值 {res.std()/max(off.std(),1e-9):.2f}')
        print(f'  最优 α={best_a:+.2f}: PSNR(官方编辑, α·复刻) = {best_p:.2f} dB  (α=0 基线 {base:.2f} dB)')
        # 逐通道相关 (定位色通道差异):
        for i, ch in enumerate('RGB'):
            print(f'    {ch}: corr={np.corrcoef(res[i].ravel(), off[i].ravel())[0,1]:+.3f}  '
                  f'复刻std={res[i].std():.4f} 官方std={off[i].std():.4f}')


if __name__ == '__main__':
    main()
