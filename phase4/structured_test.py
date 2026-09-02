"""任务 A: 结构化输入测试 — 网络是否真的在工作?

测试判据 (DLSS-NR 语义: 输入噪声帧, 输出干净帧):
1. 输出与干净参考的相关性 (应 > 0.3 若网络工作)
2. 输出与噪声输入的差异 (去噪 = 移除噪声成分)
3. PSNR(out vs clean) > PSNR(noisy vs clean) → 网络在去噪
"""
import sys, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase3'))
sys.path.insert(0, os.path.join(HERE, '.'))

from semantic_fill import load_all, fill_model
from dlss5.calib_model import DLSS5NetCalib

BLOB = os.path.join(HERE, '..', 'weights_blob.bin')


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def main():
    clean = np.load(os.path.join(HERE, '..', '.tmp', 'test_structured.npy'))
    noisy = np.load(os.path.join(HERE, '..', '.tmp', 'test_noisy.npy'))

    torch.manual_seed(42)
    m = DLSS5NetCalib().eval()
    by = load_all()
    fill_model(m, by, blob_full=open(BLOB, 'rb').read())

    def run(img):
        t = torch.from_numpy(img).unsqueeze(0)
        # 模型输入: (color, depth, motion, color_t) — DLSS-NR: 当前帧+深度+运动+前一帧
        with torch.no_grad():
            out = m(t, torch.rand(1, 1, *img.shape[1:]),
                    torch.zeros(1, 2, *img.shape[1:]), t)
        return out[0].numpy()

    out_noisy = run(noisy)
    out_clean = run(clean)

    print('=== 任务 A: 结构化输入测试 ===')
    print(f'输入 clean:    range [{clean.min():.3f}, {clean.max():.3f}] mean {clean.mean():.3f}')
    print(f'输入 noisy:    range [{noisy.min():.3f}, {noisy.max():.3f}] mean {noisy.mean():.3f}')
    print(f'输出(noisy入): range [{out_noisy.min():.3f}, {out_noisy.max():.3f}] mean {out_noisy.mean():.3f}')
    print(f'输出(clean入): range [{out_clean.min():.3f}, {out_clean.max():.3f}] mean {out_clean.mean():.3f}')

    # 相关性: 展平 pearson
    def corr(a, b):
        a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))

    print()
    print('--- 相关性 (结构保留判据) ---')
    print(f'corr(out_noisy, clean) = {corr(out_noisy, clean):+.4f}   (>0.3 = 结构保留)')
    print(f'corr(out_clean, clean) = {corr(out_clean, clean):+.4f}')
    print(f'corr(noisy,    clean)  = {corr(noisy, clean):+.4f}   (基线: 噪声输入本身)')

    print()
    print('--- PSNR (去噪判据) ---')
    print(f'PSNR(noisy, clean)     = {psnr(noisy, clean):.2f} dB  (基线)')
    print(f'PSNR(out_noisy, clean) = {psnr(out_noisy, clean):.2f} dB  (应 > 基线)')
    print(f'PSNR(out_clean, clean) = {psnr(out_clean, clean):.2f} dB')

    print()
    print('--- 空间结构对比 (行剖面) ---')
    row = H = 48
    print('clean   行48:', ' '.join(f'{v:.2f}' for v in clean[0, row, 40:48]))
    print('noisy   行48:', ' '.join(f'{v:.2f}' for v in noisy[0, row, 40:48]))
    print('out_noisy行48:', ' '.join(f'{v:.2f}' for v in out_noisy[0, row, 40:48]))

    # 残差语义 (Phase 5.7 确认): out = input + tail_out
    print()
    print('--- 残差语义 out = input + tail_out ---')
    r_noisy = noisy + out_noisy; r_clean = clean + out_clean
    print(f'corr(in+tail, clean) = {corr(r_noisy, clean):+.4f}')
    print(f'PSNR(in+tail, clean) = {psnr(r_noisy, clean):.2f} dB')
    np.save(os.path.join(HERE, '..', '.tmp', 'out_noisy.npy'), out_noisy)
    np.save(os.path.join(HERE, '..', '.tmp', 'out_clean.npy'), out_clean)
    print('\n输出已存 .tmp/')


if __name__ == '__main__':
    main()
