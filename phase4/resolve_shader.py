"""OptiScaler DLSS-NR resolve composition, NumPy port of dlssnr.hlsl.

Source: .tmp/osrc/OptiScaler/shaders/dlssnr/precompile/dlssnr.hlsl
        (mode = DlssNrMode_Resolve, both `gPassthrough=0` HDR-capable path
        and `gPassthrough=1` non-HDR copy path)

Parameters:
    whitePoint      (DlssNrWhitePointScale,  default 1.0)
    transferStrength (DlssNrTransferStrength, default 1.0)
    colourStrength   (DlssNrColourStrength,   default 1.0)
    maxRatio         (DlssNrMaxRatio,         default 2.0)
    passthrough      (auto: game buffer format cannot hold linear HDR -> 1;
                            otherwise 0; R10G10B10A2_UNORM swapchains take
                            the passthrough=1 path because the format is not
                            in `FormatCanHoldLinearHdr`.)

Math chain (passthrough=0, normalised space, then * whitePoint on the way out):
    proxy      = SrgbToLinear(model_input)            # what model was shown
    model      = SrgbToLinear(model_output)           # what model returned (same scale as proxy)
    original   = hdrCopy / whitePoint                 # untouched frame, normalised
    ratio = originalLuma/proxyLuma  if originalLuma < proxyLuma
            (modelLuma + max(0, originalLuma-proxyLuma))/modelLuma  otherwise
    upgraded   = lerp(original, HueOkLab(model*ratio, model), transferStrength)
    lumaRatio  = clamp((upgradedLuma + 1/512)/(originalLuma + 1/512), 0, maxRatio)
    result     = lerp(original * lumaRatio, upgraded, colourStrength) * whitePoint

Math chain (passthrough=1, no sRGB encode/decode, proxy/original/model all
live in the swapchain's own [0,1] UNORM domain):
    proxy      = proxySample.rgb                        # as-is
    model      = modelSample.rgb                        # as-is
    original   = originalSample.rgb                     # hdrCopy / 1.0 = hdrCopy
    ratio      = ...                                     # same formula as above
    upgraded   = lerp(original, HueOkLab(model*ratio, model), transferStrength)
    lumaRatio  = clamp(..., 0, maxRatio)
    result     = lerp(original * lumaRatio, upgraded, colourStrength) * 1.0
    result     = max(result, 0)                          # clamp negative

Under passthrough=1, original == proxy (encode is a pure CopyResource of
target into colorCopy/hdrCopy), so the resolve collapses to
    after  ==  AP1_clamp(model)
when transfer=colour=1, which makes the inverse a single linear un-projection:
    model  ==  BT709_to_AP1 (after)
"""
from __future__ import annotations

import numpy as np

# BT.709 luma weights, as in dlssnr.hlsl.
kLuma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


# ---------------------------------------------------------------------------
# Colour science primitives (Bjorn Ottosson + AP1/sRGB/PQ standard constants).
# ---------------------------------------------------------------------------

def _cbrt_signed(v: np.ndarray) -> np.ndarray:
    return np.sign(v) * np.abs(v) ** (1.0 / 3.0)


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """Inverse sRGB EOTF; matches SrgbToLinear() in dlssnr.hlsl."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v < 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v < 0.0031308, v * 12.92, 1.055 * np.maximum(v, 1e-8) ** (1.0 / 2.4) - 0.055)


# OkLab matrices (Bjorn Ottosson's published constants).
_RGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_LMS_TO_LAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_LAB_TO_LMS = np.array([
    [1.0,           0.3963377774,  0.2158037573],
    [1.0,          -0.1055613458, -0.0638541728],
    [1.0,          -0.0894841775, -1.2914855480],
])
_LMS_TO_RGB = np.array([
    [ 4.0767416621, -3.3077115913,  0.2309699292],
    [-1.2684380046,  2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147,  1.7076147010],
])

# AP1 gamut clamp matrices (ACEScg primaries in BT.709 space).
_BT709_TO_AP1 = np.array([
    [0.613097, 0.339523, 0.047379],
    [0.070194, 0.916354, 0.013452],
    [0.020616, 0.109570, 0.869815],
])
_AP1_TO_BT709 = np.array([
    [ 1.705051, -0.621792, -0.083259],
    [-0.130256,  1.140805, -0.010548],
    [-0.024003, -0.128969,  1.152972],
])


def _matmul_rows(M: np.ndarray, x: np.ndarray) -> np.ndarray:
    """M @ x for x with last axis = 3. Treats x as HxWx3 (or anything *x3)."""
    return x @ M.T


def to_oklab(c: np.ndarray) -> np.ndarray:
    return _matmul_rows(_LMS_TO_LAB, _cbrt_signed(_matmul_rows(_RGB_TO_LMS, c)))


def from_oklab(lab: np.ndarray) -> np.ndarray:
    lms = _matmul_rows(_LAB_TO_LMS, lab)
    return _matmul_rows(_LMS_TO_RGB, lms * lms * lms)


def clamp_ap1(c: np.ndarray) -> np.ndarray:
    return _matmul_rows(_AP1_TO_BT709, np.maximum(_matmul_rows(_BT709_TO_AP1, c), 0.0))


def hue_oklab(incorrect: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """incorrect: a colour whose chroma magnitude we want to keep;
       correct:   a colour whose chroma direction (hue) we want to follow."""
    incorrect_lab = to_oklab(incorrect)
    correct_lab = to_oklab(correct)
    incorrect_chroma = np.linalg.norm(incorrect_lab[..., 1:], axis=-1, keepdims=True)
    correct_chroma = np.linalg.norm(correct_lab[..., 1:], axis=-1, keepdims=True)
    # avoid /0 — the HLSL uses 1.0 in that branch
    scale = np.where(correct_chroma > 0, incorrect_chroma / np.maximum(correct_chroma, 1e-30), 1.0)
    new_lab = incorrect_lab.copy()
    new_lab[..., 1:] = correct_lab[..., 1:] * scale
    return clamp_ap1(from_oklab(new_lab))


# ---------------------------------------------------------------------------
# Encode path (mode 0) — used by OptiScaler to derive model_input from target.
# ---------------------------------------------------------------------------

def encode(frame: np.ndarray, white_point: float, passthrough: bool = False) -> np.ndarray:
    """Reproduce DlssNrMode_Encode (mode=0).

    frame:  HDR linear frame (HxWx3), as written into the swapchain.
    returns: sRGB-encoded display-referred picture shown to the model, same shape.

    With `passthrough=True`, encode is a pure copy (the swapchain format
    cannot hold linear HDR, so the colour transform is skipped and the
    frame is handed to the model verbatim).
    """
    if passthrough:
        return np.maximum(frame, 0.0)
    frame = np.maximum(frame, 0.0)
    display = frame / max(white_point, 1e-4)
    # Soft knee above 0.75 luma: rolled = 0.75 + 0.25 * (1 - exp(-(L-0.75)/0.25)).
    luma = display @ kLuma
    rolled = np.where(
        luma > 0.75,
        0.75 + 0.25 * (1.0 - np.exp(-(luma - 0.75) / 0.25)),
        luma,
    )
    out = display * (rolled / np.where(luma > 1e-12, luma, 1.0))[..., None]
    return linear_to_srgb(np.maximum(out, 0.0))


# ---------------------------------------------------------------------------
# Resolve path — compose model output with the original HDR frame.
# ---------------------------------------------------------------------------

def resolve(
    proxy: np.ndarray,        # HxWx3
    model: np.ndarray,        # HxWx3
    original: np.ndarray,     # HxWx3
    *,
    white_point: float = 1.0,
    transfer_strength: float = 1.0,
    colour_strength: float = 1.0,
    max_ratio: float = 2.0,
    passthrough: bool = False,
) -> np.ndarray:
    """Forward composition: takes proxy, model output, and original frame,
    returns the resolve target.

    When `passthrough` is True (the path used when the game's DLSS buffer is
    not float-format-capable -- i.e. everything except R16G16B16A16_FLOAT /
    R32G32B32A32_FLOAT / R11G11B10_FLOAT -- so e.g. R10G10B10A2_UNORM swapchains
    take this path), proxy and model are NOT sRGB-decoded. They live in the
    swapchain's own [0,1] UNORM domain. `original` is loaded as-is too.

    When `passthrough` is False (HDR-capable buffer, real linear HDR flow),
    proxy and model are SrgbToLinear-decoded, original is divided by
    `white_point` to bring it into the [0,1] display-normalised space.
    """
    if passthrough:
        proxy_lin = proxy
        model_lin = model
        norm_scale = 1.0
    else:
        proxy_lin = srgb_to_linear(proxy)
        model_lin = srgb_to_linear(model)
        norm_scale = max(white_point, 1e-4)

    proxy_luma = proxy_lin @ kLuma
    model_luma = model_lin @ kLuma
    original_luma = original @ kLuma

    # --- ratio (luminance mapping) -------------------------------------------
    # Below the proxy's luminance: rescale model to original's luminance.
    # Above it: keep headroom (originalLuma - proxyLuma) on top of model.
    low_branch = original_luma < proxy_luma
    safe_proxy = np.maximum(proxy_luma, 1e-6)
    safe_model = np.maximum(model_luma, 1e-5)
    ratio = np.where(
        low_branch,
        original_luma / safe_proxy,
        (model_luma + np.maximum(0.0, original_luma - proxy_luma)) / safe_model,
    )

    # model_luma ≤ 1e-5 fallback: upgraded = original
    upgraded = np.where(
        (model_luma[..., None] <= 1e-5),
        original,
        # else: lerp(original, HueOkLab(model*ratio, model), transfer)
        original + (hue_oklab(model_lin * ratio[..., None], model_lin) - original)
                  * transfer_strength,
    )

    upgraded_luma = upgraded @ kLuma

    # --- final blend with floor-stabilised ratio -----------------------------
    floor = 1.0 / 512.0
    luma_ratio = np.clip(
        (upgraded_luma + floor) / (original_luma + floor),
        0.0,
        max_ratio,
    )
    # lerp(original * lumaRatio, upgraded, colourStrength)
    result = original * luma_ratio[..., None] + (upgraded - original * luma_ratio[..., None]) * colour_strength

    # Back out of normalised space.
    result = result * norm_scale
    return np.maximum(result, 0.0)


# ---------------------------------------------------------------------------
# Convenience: load R10G10B10A2 UNORM as float [0,1].
# ---------------------------------------------------------------------------

def load_rgb10a2(path: str, width: int = 1920, height: int = 1050) -> np.ndarray:
    u = np.fromfile(path, dtype=np.uint32).reshape(height, width)
    return np.stack([((u >> s) & 0x3FF) / 1023.0 for s in (0, 10, 20)], axis=-1)


# ---------------------------------------------------------------------------
# Inverse resolve — recover the model's output from the resolve target.
#
#   forward: after = resolve(proxy, model, original)
#   inverse: given (proxy, after, original, params), what model was sent?
#
#   When passthrough=True and original==proxy (encode is a pure copy in that
#   path), the resolve collapses to
#       after  ==  AP1_clamp(model)              (transfer=colour=1)
#   so the implied model is recovered as
#       model = AP1_clamp^-1 (after)
#   AP1_clamp is a non-expansive projection, so its inverse is multi-valued;
#   we use the BT.709->AP1 matrix (the inverse of the BT.709 projection
#   AP1_to_BT709). Pixels pushed outside AP1 by the model have lost their
#   exact pre-clamp colour, so the reported residual is a lower bound on
#   the true model's edit amplitude, not its exact value.
#
#   When passthrough=False (HDR-capable buffer, real linear HDR flow) the
#   collapse is
#       after / white_point  ==  AP1_clamp( SrgbToLinear(model) )
#   so
#       model = LinearToSrgb( AP1_clamp^-1 (after / white_point) )
# ---------------------------------------------------------------------------

def invert_model(
    after: np.ndarray,
    white_point: float = 1.0,
    *,
    passthrough: bool = False,
) -> np.ndarray:
    """Recover the model's output that produced `after`, assuming
    original == proxy and transfer=colour=1."""
    if passthrough:
        # after = AP1_clamp(model). Inverse: BT.709->AP1 matrix.
        model = np.maximum(after @ _BT709_TO_AP1.T, 0.0)
        return model
    # else: after/wp = AP1_clamp(SrgbToLinear(model))
    model_lin = np.maximum((after / max(white_point, 1e-4)) @ _BT709_TO_AP1.T, 0.0)
    return linear_to_srgb(model_lin)


def invert_residual(
    proxy: np.ndarray,        # sRGB-encoded display (model_input), HDR-linear if passthrough
    after: np.ndarray,        # resolve target, same domain as proxy
    white_point: float = 1.0,
    *,
    passthrough: bool = False,
) -> np.ndarray:
    """Return the model's signed edit `model_output - proxy`, under the
    proxy==original / transfer=colour=1 assumption (see invert_model docstring)."""
    return invert_model(after, white_point, passthrough=passthrough) - proxy


# ---------------------------------------------------------------------------
# CLI: probe parameter sensitivity and report simple statistics.
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Forward and inverse resolve of OptiScaler DLSS-NR composition.")
    ap.add_argument("--frame", type=int, default=0, help="frame index (0..7)")
    ap.add_argument("--white-point", type=float, default=None,
                    help="white point scale (default: 1.0, the OptiScaler default)")
    ap.add_argument("--transfer", type=float, default=1.0)
    ap.add_argument("--colour", type=float, default=1.0)
    ap.add_argument("--max-ratio", type=float, default=2.0)
    ap.add_argument("--passthrough", action="store_true",
                    help="use passthrough=1 path (R10G10B10A2_UNORM swapchains; the cap2_live capture)")
    ap.add_argument("--cap-dir", default=".tmp/cap2_live")
    ap.add_argument("--dump-residual", type=str, default=None,
                    help="path to save residual.npz with proxy, model_est, residual, after, white_point, passthrough")
    args = ap.parse_args()

    proxy = load_rgb10a2(f"{args.cap_dir}/model_input_{args.frame:02d}.raw")
    after = load_rgb10a2(f"{args.cap_dir}/after_{args.frame:02d}.raw")

    white_point = args.white_point if args.white_point is not None else 1.0

    # Forward sanity check: with model=proxy and original=proxy, what does resolve return?
    after_pred = resolve(proxy, proxy, proxy,
                         white_point=white_point,
                         transfer_strength=args.transfer,
                         colour_strength=args.colour,
                         max_ratio=args.max_ratio,
                         passthrough=args.passthrough)
    print(f"forward resolve(model=proxy, original=proxy):")
    print(f"  after_pred mean={after_pred.mean():.3f}, after actual mean={after.mean():.3f}")
    err = after - after_pred
    print(f"  diff mean={err.mean():+.4f} std={err.std():.4f}")

    # Inverse: under proxy==original assumption.
    model_est = invert_model(after, white_point, passthrough=args.passthrough)
    residual = model_est - proxy

    print(f"\nframe {args.frame}, white_point={white_point:.4f}, passthrough={args.passthrough}")
    print(f"residual = model_output - proxy (post AP1-clamp^-1):")
    for i, c in enumerate("RGB"):
        r = residual[..., i]
        print(f"  {c}: mean={r.mean():+.4f} std={r.std():.4f} "
              f"P1={np.quantile(r, 0.01):+.3f} P50={np.median(r):+.3f} P99={np.quantile(r, 0.99):+.3f}")
    print(f"  overall: mean={residual.mean():+.4f} std={residual.std():.4f} "
          f"|.|_mean={np.abs(residual).mean():.4f}")

    # Round-trip check
    after_hat = resolve(proxy, model_est, proxy,
                        white_point=white_point,
                        transfer_strength=args.transfer,
                        colour_strength=args.colour,
                        max_ratio=args.max_ratio,
                        passthrough=args.passthrough)
    err = after - after_hat
    print(f"\nreconstruction: mean|err|={np.abs(err).mean():.4f} max|err|={np.abs(err).max():.4f}")

    if args.dump_residual:
        np.savez_compressed(
            args.dump_residual,
            proxy=proxy.astype(np.float32),
            model_est=model_est.astype(np.float32),
            residual=residual.astype(np.float32),
            after=after.astype(np.float32),
            white_point=np.float32(white_point),
            passthrough=np.uint8(1 if args.passthrough else 0),
        )
        print(f"saved residual npz -> {args.dump_residual}")


if __name__ == "__main__":
    main()