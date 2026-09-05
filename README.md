# DLSS5 PyTorch — Neural Rendering Replica

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Numpy <2.0](https://img.shields.io/badge/numpy-%3C2.0-013243.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, production-grade PyTorch replica of NVIDIA's leaked **DLSS 5 Neural Rendering** (`nvngx_dlssnr.dll`, version 310.8.0).

While the official DLL is compiled strictly for `sm_120` (NVIDIA Blackwell / RTX 50-series only, returning `0xBAD00002` on Ampere/Ada), this PyTorch implementation reconstructs the complete model architecture, decodes its proprietary multi-format weights (`E4M3`, `MXFP8`, `FP16`), and allows high-fidelity neural rendering inference on **any CUDA GPU (RTX 3090/4090/5090) or CPU**.

---

## Architecture Overview

DLSS5 Neural Rendering is an asymmetrical encoder-decoder network comprising **147,683,778 parameters (147.7M)**:

```
Inputs:
  - Color (3ch HDR, linear) ──┐
  - Depth (1ch, normalized)  ─┼─► Stem Conv (9ch -> 32ch)
  - Motion (2ch, pixels)     ─┤
  - Reference Frame (3ch)    ─┘
                                   │
                                   ▼
                    5-Stage Swin Transformer Encoder
                    Stage 0: 3 blocks (c=32)
                    Stage 1: 3 blocks (c=64)
                    Stage 2: 5 blocks (c=128)
                    Stage 3: 7 blocks (c=256)
                    Stage 4: 7 blocks (c=512)
                                   │
                                   ▼
                    8-Block 1D ViT Bottleneck (c=512)
                    [wqkv, proj, side, ffwd]
                                   │
                                   ▼
                    5-Stage Swin Transformer Decoder
                    Stage 0: 8 blocks (c=512) + UpFuse
                    Stage 1: 7 blocks (c=256) + UpFuse
                    Stage 2: 5 blocks (c=128) + UpFuse
                    Stage 3: 3 blocks (c=64)  + UpFuse
                    Stage 4: 3 blocks (c=32)
                                   │
                                   ▼
                    Tail Head (AvgPool + Global FC + 1x1 Conv)
                                   │
                                   ▼
                    Output Residual (ΔRGB linear HDR)
```

### Proprietary Weight Formats
The 147.7MB binary weights blob (`weights_blob.bin`) packages weights across 71 distinct records using mixed numerical formats:
- **`E4M3` (FP8)**: Sign + 4-bit exponent + 3-bit mantissa for high-dynamic-range attention GEMMs.
- **`MXFP8` (Microscaling)**: 2-byte pairs `(weight, scale)` with block-adaptive power-of-2 scaling.
- **`FP16` (IEEE 754)**: Little-endian float16 for LayerNorm scales/biases and UpFuse skip projections.

---

## Current Status (Round 29)

**Scoreboard (clean 1920×1050, 16 frames, official vs replica)**

| Metric | Official | Ours (pure net) |
|---|---|---|
| Pure-network corr (held-out) | — | **+0.393** (zero-expand) |
| Sharpness @1px | +23.7% over input | **−3.7%** |
| MS-SSIM / LPIPS (with tone LUT) | — | **0.86525 / 0.2538** (no regression, all dims) |

- Tone/color axes fully closed (corr +0.992, amp 0.93-0.96 with diagnostic LUT; pure-net without LUT is the honest baseline).
- Sharpness carrier identified in SASS: gated MV-bicubic residual inside `simple_blend` (`out = σ(x_net)·Σwᵢ(mv)·texᵢ(0x6e)/norm − net_raw`, R27/R28 oracle-validated at err 0.00061).
- **Remaining work**: ① tex `0x6e` runtime binding (H_A input color vs H_B prev net output) — requires Windows frida capture, the ONLY open question; ② real `x_net` gate weights (runtime params, not in blob); ③ generated-texture 2% (needs trained weights, out of scope without training).
- Structural `DLSS5_TAIL_MODE=full` implements the decoded epilogue (default `simple`); corr-neutral uncalibrated, for structure study only.

---

## Key Reverse-Engineering Breakthroughs

1. **B14 MX-Scale Leak Fix**: In earlier decoders, an FP16 tail boundary bug leaked a 512-element MX-scale table ($\mu=-8.78$) into `merges.2.norm.weight`, inverting and amplifying feature activations 9×. Restored canonical LayerNorm bounds.
2. **Restored UpFuse Skip Connections**: Identified real $2c^2$ skip-fusion GEMMs residing in blob residual B-fields (`b48/56/62/66`), replacing lossy skip truncation.
3. **Cleaned Expand LayerNorms**: Removed spurious high-scale outliers from decoder expand blocks, boosting input sensitivity by 266×.
4. **Seamless Full-Resolution Forward**: Replaced patch-average `cropmean` broadcasts with full-resolution forward inference (1920×1056 with spatial divisibility padding), eliminating 96×96 tiling artifacts.

---

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+
- **NumPy < 2.0** (`numpy>=1.24.0,<2.0.0` strictly required)
- Pillow

```bash
git clone https://github.com/baicai1145/dlss5-pytorch.git
cd dlss5-pytorch

# Install dependencies
pip install -r requirements.txt
```

Ensure `weights_blob.bin` (extracted from official `nvngx_dlssnr.dll`) is placed in the project root.

---

## CLI Usage

### 1. Neural Rendering Inference (`infer.py`)

Run inference on raw DXGI game capture buffers or standard image files:

```bash
# Fast preview on CPU (96x96 center crop with sRGB display tone-mapping)
python3 infer.py \
    --color .tmp/cap2_live/before_00.raw \
    --depth .tmp/cap2_live/depth_00.raw \
    --motion .tmp/cap2_live/motion_00.raw \
    --crop 96 96 \
    --output output_preview.png \
    --srgb

# Full-resolution 1920x1056 inference on CUDA (RTX 3090 / 4090 / 5090)
python3 infer.py \
    --color .tmp/cap2_live/before_00.raw \
    --depth .tmp/cap2_live/depth_00.raw \
    --motion .tmp/cap2_live/motion_00.raw \
    --device cuda \
    --output output_full.png \
    --srgb
```

### 2. Ground-Truth Alignment Benchmark (`evaluate.py`)

Benchmark the PyTorch replica against live ground-truth frames captured from an official RTX 5090 running `nvngx_dlssnr.dll`:

```bash
# Evaluate across all 8 captured frames
python3 evaluate.py --all-frames --crop 96 96 --device cpu
```

Output:
```
=================================================================
  NVIDIA DLSS5 PyTorch Replica — Ground Truth Alignment Benchmark
=================================================================
  Target Device : cpu
  Capture Dir   : .tmp/cap2_live
  Weights Blob  : weights_blob.bin
  Eval Mode     : Center Crop 96x96
-----------------------------------------------------------------
  Model loaded in 6.57s
-----------------------------------------------------------------
  Frame 00 | Pass-Through: 12.73 dB | PyTorch: 13.83 dB | Gain: +1.10 dB
  Frame 01 | Pass-Through: 12.72 dB | PyTorch: 13.86 dB | Gain: +1.15 dB
  Frame 02 | Pass-Through: 12.72 dB | PyTorch: 13.85 dB | Gain: +1.13 dB
  Frame 03 | Pass-Through: 12.73 dB | PyTorch: 13.82 dB | Gain: +1.09 dB
  Frame 04 | Pass-Through: 12.73 dB | PyTorch: 13.86 dB | Gain: +1.13 dB
  Frame 05 | Pass-Through: 12.74 dB | PyTorch: 13.85 dB | Gain: +1.11 dB
  Frame 06 | Pass-Through: 12.75 dB | PyTorch: 13.78 dB | Gain: +1.04 dB
  Frame 07 | Pass-Through: 12.74 dB | PyTorch: 13.84 dB | Gain: +1.11 dB
-----------------------------------------------------------------
  Benchmark Summary:
    Evaluated Frames : 8
    Pass-Through PSNR: 12.73 dB
    PyTorch Replica  : 13.84 dB
    Improvement      : +1.11 dB over baseline
    Correlation (r)  : +0.902
    Visual Comparison: .tmp/eval_comparison.png
=================================================================
```

A 4-panel visual comparison (`[ Input | Official GT | PyTorch Replica | Absolute Error ]`) is saved to `.tmp/eval_comparison.png`.

---

## Python API

```python
import dlss5

# 1. Load 147.7M model from weights blob
model = dlss5.load_model("weights_blob.bin", device="cuda")

# 2. Read native game buffers (DXGI UNORM/FLOAT)
color = dlss5.load_dxgi_color("before_00.raw")    # (1050, 1920, 3) float32 in [0, 1]
depth = dlss5.load_dxgi_depth("depth_00.raw")    # (1050, 1920) float32
motion = dlss5.load_dxgi_motion("motion_00.raw") # (1050, 1920, 2) float32 in pixel units

# 3. Full-resolution inference (auto-pads to 1056x1920 divisible by 32)
output_hdr = dlss5.infer_frame(model, color, depth, motion, device="cuda")

# 4. Display tone-mapping (Linear HDR -> sRGB)
output_srgb = dlss5.linear_to_srgb(output_hdr)
dlss5.save_image(output_srgb, "result.png")
```

---

## Repository Structure

```
dlss5-pytorch/
├── dlss5/                     # Core Python Package
│   ├── __init__.py            # Clean exports: load_model, infer_frame, etc.
│   ├── model.py               # Complete 147.7M DLSS5Net architecture
│   ├── loader.py              # Record parser and weight mounter
│   ├── mx_decode.py           # E4M3, MXFP8, and FP16 decoding routines
│   ├── postprocess.py         # Additive residual blending & sRGB curve
│   ├── data_utils.py          # DXGI capture buffer readers
│   ├── swin_block.py          # Swin Transformer window attention blocks
│   ├── vit1d_block.py         # 1D ViT bottleneck blocks
│   ├── patch_ops.py           # Window partitioning & cyclic shifts
│   └── blob_budget.py         # Architecture channel budgets
├── infer.py                   # High-level CLI for frame inference
├── evaluate.py                # Official ground-truth alignment benchmark
├── weights_blob.bin           # 147.7MB extracted weights
├── tests/                     # Unit tests
│   └── test_dlss5.py
├── requirements.txt           # Pinned dependencies (numpy < 2.0)
├── pyproject.toml             # Packaging specification
└── REPORT.md                  # Comprehensive reverse-engineering report
```

---

## Running on Cloud GPU (RTX 3090 / 24GB VRAM)

To run seamless full-resolution (1920×1056) inference without tiling on a rented RTX 3090:

```bash
# 1. Sync repository to remote host
rsync -avz --exclude '.git' --exclude '.tmp' ./ user@remote-box:~/dlss5-pytorch/

# 2. Run native CUDA inference
python3 infer.py \
    --color before_00.raw \
    --depth depth_00.raw \
    --motion motion_00.raw \
    --device cuda \
    --output full_frame_dlss5.png \
    --srgb
```

---

## License

MIT License. Educational and reverse-engineering research purposes only. DLSS and NVIDIA are registered trademarks of NVIDIA Corporation.
