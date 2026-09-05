"""Tests for eval_suite — synthetic invariance/direction checks (no data/GPU)."""
import sys, os
sys.path.insert(0, '.')
import numpy as np
import eval_suite as ES


def _img(seed=0, n=2, h=64, w=64):
    rng = np.random.default_rng(seed)
    base = np.clip(0.3 + 0.4 * rng.random((n, h, w, 1)) , 0, 1)
    img = np.repeat(base, 3, axis=-1).astype(np.float32)
    return img


def test_sharpness_directional():
    img = _img()
    # sharpen: amplify 1px differences
    sharp = img.copy()
    sharp[:, 1:, :, :] += 0.5 * (img[:, 1:, :, :] - img[:, :-1, :, :])
    s0 = ES.sharpness(img)["s1"]
    s1 = ES.sharpness(np.clip(sharp, 0, 1))["s1"]
    assert s1 > s0 * 1.05, (s0, s1)
    # blur must decrease
    import torch, torch.nn.functional as F
    t = torch.from_numpy(img.transpose(0, 3, 1, 2))
    blur = F.avg_pool2d(t, 3, 1, 1).numpy().transpose(0, 2, 3, 1)
    assert ES.sharpness(blur)["s1"] < s0


def test_grain_directional():
    img = _img()
    rng = np.random.default_rng(7)
    noisy = np.clip(img + 0.05 * rng.standard_normal(img.shape), 0, 1).astype(np.float32)
    g0 = ES.grain(img)
    g1 = ES.grain(noisy)
    assert g1["flat"] > g0["flat"]
    assert g1["edge"] > g0["edge"]


def test_color_shift_detection():
    img = _img()
    sat = np.clip(img * np.array([1.2, 1.0, 1.2]), 0, 1).astype(np.float32)
    c0 = ES.color_stats(img)
    c1 = ES.color_stats(sat)
    assert c1["sat"] > c0["sat"]
    # contrast change via luma stretch
    con = np.clip((img - 0.5) * 1.5 + 0.5, 0, 1).astype(np.float32)
    assert ES.color_stats(con)["con"] > c0["con"]


def test_decompose_identity_and_shift():
    img = _img()
    rng = np.random.default_rng(11)
    d = (0.01 * rng.standard_normal(img.shape)).astype(np.float32)
    rep = ES.decomp_report(d, d)
    # identical deltas: perfect corr, ratio 1
    assert abs(rep["smooth"]["corr"] - 1.0) < 1e-4
    assert abs(rep["hf"]["ratio"] - 1.0) < 1e-4
    # zero-delta guard returns nan corr, zero amps
    z = ES.decomp_report(np.zeros_like(img), np.zeros_like(img))
    assert np.isnan(z["smooth"]["corr"]) and z["hf"]["amp_rep"] == 0
    # pure global shift lives in tone, not hf
    const = np.zeros_like(img); const[:, :, :, 0] = 0.01; const[:, :, :, 1] = 0.02
    r2 = ES.decomp_report(const, const)
    assert r2["hf"]["amp_off"] < 0.002  # border leakage only, shift is ~all tone
    assert r2["luma"]["amp_off"] > 0
    assert r2["chroma"]["amp_off"] > 0  # G-only shift is chromatic


def test_scoreboard_runs_and_visuals(tmp_path):
    img = _img(3, n=4, h=96, w=96)
    off = np.clip(img + 0.02, 0, 1)
    fs = ES.FrameSet(img, off, name="syn")
    rep = np.clip(img + 0.01, 0, 1)
    sb = ES.Scoreboard(fs, replica=rep,
                       fit_frames=[0, 1], report_frames=[2, 3])
    d = sb.to_dict()
    assert "directional" in d and "decomp" in d and "perceptual" in d
    txt = sb.report()
    assert "SCOREBOARD" in txt
    ps = ES.emit_visuals(fs, rep, str(tmp_path), frame_idx=2)
    assert all(os.path.isfile(p) for p in ps)
