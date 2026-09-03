"""Inference demo: 任意图 → DLSS5 前向 → 残差可视化

用法: python3 phase4/infer_demo.py [图片路径]
不带参数时用内置合成场景 (天空/太阳/山/棋盘格地面).
"""
import sys, os
import numpy as np
from PIL import Image
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase3'))
sys.path.insert(0, HERE)
from semantic_fill import load_all, fill_model
from dlss5.calib_model import DLSS5NetCalib
BLOB = os.path.join(HERE, '..', 'weights_blob.bin')

SIZE = 96
NOISE_STD = 0.15


def synth_scene():
    """游戏画面风格合成场景: 天空渐变 + 太阳 + 山 + 棋盘格地面 + 电线杆."""
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    img = np.zeros((3, SIZE, SIZE), np.float32)
    img[0] = 0.35 + 0.35 * (1 - y / SIZE)            # 天空 R
    img[1] = 0.45 + 0.10 * (1 - y / SIZE)            # 天空 G
    img[2] = 0.75 - 0.35 * (y / SIZE)                # 天空 B
    d2 = (x - 0.72 * SIZE) ** 2 + (y - 0.20 * SIZE) ** 2
    img[:, d2 < 81] = np.array([1.0, 0.92, 0.55])[:, None]      # 太阳
    mtn = (y > 0.60 * SIZE) & (np.abs(x - 0.30 * SIZE) < (y - 0.60 * SIZE) * 1.2)
    mtn |= (y > 0.52 * SIZE) & (np.abs(x - 0.80 * SIZE) < (y - 0.52 * SIZE) * 1.5)
    img[:, mtn] = np.array([0.16, 0.22, 0.30])[:, None]         # 山
    gnd = y > 0.82 * SIZE
    chk = ((x // 8 + y // 8) % 2 == 0)
    img[:, gnd & chk] = 0.58
    img[:, gnd & ~chk] = 0.34                                    # 棋盘格地面
    img[:, (np.abs(x - 0.48 * SIZE) < 2) & (y > 0.42 * SIZE)] = 0.08  # 电线杆
    return np.clip(img, 0, 1)


def psnr(a, b):
    return 10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        im = Image.open(path).convert('RGB').resize((SIZE, SIZE), Image.LANCZOS)
        clean = np.asarray(im).astype(np.float32).transpose(2, 0, 1) / 255.0
        print(f'输入图片: {path} → {SIZE}×{SIZE}')
    else:
        clean = synth_scene()
        print(f'内置合成场景 {SIZE}×{SIZE}')
    rng = np.random.default_rng(7)
    noisy = np.clip(clean + rng.normal(0, NOISE_STD, clean.shape), 0, 1).astype(np.float32)

    print('加载模型 (解析 blob + 装填 147.7M 参数)...')
    torch.manual_seed(42)
    m = DLSS5NetCalib().eval()
    by = load_all()
    fill_model(m, by, blob_full=open(BLOB, 'rb').read())
    print('前向推理...')
    t = torch.from_numpy(noisy).unsqueeze(0)
    with torch.no_grad():
        res = m(t, torch.zeros(1, 1, SIZE, SIZE), torch.zeros(1, 2, SIZE, SIZE), t)[0].numpy()

    print(f'\n残差: mean={res.mean():+.4f} std={res.std():.4f} range=[{res.min():.3f},{res.max():.3f}]')
    base = psnr(noisy, clean)
    print('\n--- alpha 混合扫描: out = noisy + α·res ---')
    print(f'  α= 0.00 (什么都不做)   PSNR = {base:6.2f} dB')
    best = (base, 0.0)
    for a in (0.1, 0.25, 0.5, 0.75, 1.0, -0.1, -0.25, -0.5, -1.0):
        p = psnr(np.clip(noisy + a * res, 0, 1), clean)
        if p > best[0]:
            best = (p, a)
        print(f'  α={a:+.2f}               PSNR = {p:6.2f} dB')
    print(f'  最优 α={best[1]:+.2f} → {best[0]:.2f} dB  (基线 {base:.2f} dB)')

    denoised = np.clip(noisy + best[1] * res, 0, 1)
    def to_img(a):
        return Image.fromarray((np.clip(a, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
    res_v = np.stack([(r - r.min()) / (r.max() - r.min() + 1e-9) for r in res])  # 逐通道归一化残差
    grid = Image.new('RGB', (SIZE * 2 + 4, SIZE * 2 + 4), (20, 20, 20))
    grid.paste(to_img(clean), (0, 0));        grid.paste(to_img(noisy), (SIZE + 4, 0))
    grid.paste(to_img(res_v), (0, SIZE + 4))
    grid.paste(to_img(denoised), (SIZE + 4, SIZE + 4))
    out = os.path.join(HERE, '..', '.tmp', f'infer_{"user" if path else "synth"}.png')
    grid.save(out)
    print(f'\n可视化: {out}')
    print('布局: 左上=干净参考 | 右上=噪声输入 | 左下=残差(归一化) | 右下=最优混合输出')


if __name__ == '__main__':
    main()
