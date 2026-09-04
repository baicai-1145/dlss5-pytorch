#!/usr/bin/env python3
"""cap3_check.py — capture v3 数据的完整性与语义校验.

对抓回的 cap3 目录做四件事:
1. 结构清点: 6 类 plane × N 帧, manifest 参数齐全
2. 语义验证: model_raw 是"模型原始输出"而非合成结果
   - 如果 resolve 的 composition 在起作用, after ≉ model_raw, 且
     |after − model_raw| 应显著大于 0 (v2 时代我们把 after 当模型输出, 这是错误)
3. resolve 复现: 用 manifest 里的 transferStrength/colourStrength/whitePoint
   在 numpy 里跑 dlssnr.hlsl 的 composition, 验证 after = Resolve(hdr_copy, model_raw)
   (这一步验证通过 = 我们彻底掌握了官方后处理, 之后的对齐目标改为 model_raw)
4. 递归检查: reset=1 后的帧序列里, 模型输出是否依赖前一帧 (帧间差分对比)

用法: PYTHONPATH=. python3 cap3_check.py .tmp/cap3_live
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dlss5  # noqa: E402

K_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

# ---- dlssnr.hlsl 常量 (AP1 / OkLab 矩阵逐字抄自 shader) ----
BT709_TO_AP1 = np.array([[0.613097, 0.339523, 0.047379],
                         [0.070194, 0.916354, 0.013452],
                         [0.020616, 0.109570, 0.869815]], np.float32)
AP1_TO_BT709 = np.array([[1.705051, -0.621792, -0.083259],
                         [-0.130256, 1.140805, -0.010548],
                         [-0.024003, -0.128969, 1.152972]], np.float32)
RGB_TO_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                       [0.2119034982, 0.6806995451, 0.1073969566],
                       [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
LMS_TO_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                       [1.9779984951, -2.4285922050, 0.4505937099],
                       [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
LAB_TO_LMS = np.array([[1.0, 0.3963377774, 0.2158037573],
                       [1.0, -0.1055613458, -0.0638541728],
                       [1.0, -0.0894841775, -1.2914855480]], np.float32)
LMS_TO_RGB = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                       [-1.2684380046, 2.6097574011, -0.3413193965],
                       [-0.0041960863, -0.7034186147, 1.7076147010]], np.float32)


def load_rgb10a2(path):
    u = np.fromfile(path, dtype=np.uint32)
    # rowPitch 由 manifest 给出; 假定 1920 宽 (校验时用 manifest 检查)
    r = ((u >> 0) & 0x3FF) / 1023.0
    g = ((u >> 10) & 0x3FF) / 1023.0
    b = ((u >> 20) & 0x3FF) / 1023.0
    a = (u >> 30) & 0x3
    rgb = np.stack([r, g, b], -1).astype(np.float32)
    return rgb, a


def load_manifest(directory):
    params = {}
    frames_meta = []
    with open(os.path.join(directory, "manifest.txt")) as f:
        for line in f:
            m = re.match(r"param (\S+) (\S+)", line)
            if m:
                params[m.group(1)] = float(m.group(2))
            m = re.match(r"frame (\d+) index (\d+) reset (\d)", line)
            if m:
                frames_meta.append(int(m.group(3)))
    return params, frames_meta


def to_oklab(c):
    lms = c @ RGB_TO_LMS.T
    lms = np.sign(lms) * np.abs(lms) ** (1.0 / 3.0)
    return lms @ LMS_TO_LAB.T


def from_oklab(lab):
    lms = lab @ LAB_TO_LMS.T
    return (lms * lms * lms) @ LMS_TO_RGB.T


def clamp_ap1(c):
    return (np.maximum(c @ BT709_TO_AP1.T, 0.0)) @ AP1_TO_BT709.T


def hue_oklab(incorrect, correct):
    inc = to_oklab(incorrect)
    cor = to_oklab(correct)
    inc_c = np.linalg.norm(inc[..., 1:3], axis=-1, keepdims=True)
    cor_c = np.linalg.norm(cor[..., 1:3], axis=-1, keepdims=True)
    inc[..., 1:3] = cor[..., 1:3] * (inc_c / np.maximum(cor_c, 1e-8))
    return clamp_ap1(from_oklab(inc))


def resolve(hdr, proxy, model, white_point, transfer, colour, max_ratio, passthrough):
    """dlssnr.hlsl CSMain resolve branch, vectorised."""
    def s2l(v):
        return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)

    p = proxy if passthrough else s2l(np.clip(proxy, 0, 1))
    m = model if passthrough else s2l(np.clip(model, 0, 1))
    norm = 1.0 if passthrough else max(white_point, 1e-4)
    original = hdr / norm

    ol = original @ K_LUMA
    pl = p @ K_LUMA
    ml = m @ K_LUMA

    ratio = np.where(ol < pl, ol / np.maximum(pl, 1e-6),
                     (ml + np.maximum(0.0, ol - pl)) / np.maximum(ml, 1e-6)).astype(np.float32)
    upgraded_hue = hue_oklab(m * ratio[..., None], m)
    upgraded = (1 - transfer) * original + transfer * upgraded_hue

    ul = upgraded @ K_LUMA
    floor = 1.0 / 512.0
    luma_ratio = np.clip((ul + floor) / (ol + floor), 0.0, max_ratio)
    result = (1 - colour) * (original * luma_ratio[..., None]) + colour * upgraded
    return np.maximum(result * norm, 0.0)


def psnr(x, y):
    return 10 * np.log10(1.0 / max(float(np.mean((x - y) ** 2)), 1e-12))


def main(directory):
    params, resets = load_manifest(directory)
    print(f"manifest params: {params}")
    print(f"frame resets: {resets}")

    files = sorted(os.listdir(directory))
    n = len([f for f in files if f.startswith("model_raw_")])
    print(f"\nplanes present: before={len([f for f in files if f.startswith('before_')])} "
          f"model_raw={n} hdr_copy={len([f for f in files if f.startswith('hdr_copy_')])} "
          f"after={len([f for f in files if f.startswith('after_')])} "
          f"depth={len([f for f in files if f.startswith('depth_')])}")

    # ---- 语义验证: model_raw vs after ----
    print("\n=== frame 0: model_raw vs after (composition 的痕迹) ===")
    mr, a_mr = load_rgb10a2(os.path.join(directory, "model_raw_00.raw"))
    af, a_af = load_rgb10a2(os.path.join(directory, "after_00.raw"))
    print(f"  mean |after - model_raw| = {np.abs(af - mr).mean():.4f} "
          f"(若 ~0 则 composition 未生效, 若 >0.01 则 v2 的 after 从来不是模型输出)")
    print(f"  alpha dist model_raw: {np.bincount(a_mr.ravel(), minlength=4) / a_mr.size}")
    print(f"  alpha dist after:     {np.bincount(a_af.ravel(), minlength=4) / a_af.size}")

    # ---- resolve 复现 ----
    if "transferStrength" in params:
        print("\n=== resolve 复现 (numpy 重放 dlssnr.hlsl) ===")
        hd, _ = load_rgb10a2(os.path.join(directory, "hdr_copy_00.raw"))
        be, _ = load_rgb10a2(os.path.join(directory, "before_00.raw"))
        # before = colorCopy = proxy (model was shown this)
        rec = resolve(hd, be, mr,
                      params.get("whitePoint", 1.0),
                      params.get("transferStrength", 1.0),
                      params.get("colourStrength", 1.0),
                      params.get("maxRatio", 4.0),
                      passthrough=params.get("isHdrBuffer", 0.0) < 0.5)
        rec8 = (np.clip(rec, 0, 1) * 1023.0).astype(np.uint32)
        print(f"  PSNR(resolve_replay, after_00) = {psnr(rec, af):.2f} dB  (10bit 量化后)")
        print(f"  (>35 dB = composition 完全掌握; 20-35 = 大体正确有细节差; <20 = 有结构错误)")

    # ---- 递归检查 ----
    print("\n=== 递归检查: reset 帧后的输出漂移 ===")
    if n >= 3:
        mrs = [load_rgb10a2(os.path.join(directory, f"model_raw_{i:02d}.raw"))[0] for i in range(min(4, n))]
        for i in range(1, len(mrs)):
            print(f"  |model_raw[{i}] - model_raw[{i-1}]| mean = {np.abs(mrs[i] - mrs[i-1]).mean():.5f}"
                  f"  (reset={resets[i] if i < len(resets) else '?'}; 静态场景下递归模型应逐帧收敛变小)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".tmp/cap3_live")
