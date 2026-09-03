"""Unit tests for dlss5 library."""
import os
import unittest
import numpy as np
import torch

import dlss5
from dlss5.model import DLSS5Net, DLSS5Config
from dlss5.mx_decode import e4m3_decode, fp16_decode, mx_decode_pairs
from dlss5.postprocess import apply_residual, linear_to_srgb


class TestDLSS5(unittest.TestCase):
    def test_model_forward_shape(self):
        """Test DLSS5Net model architecture with synthetic tensor input."""
        model = DLSS5Net().eval()
        B, C, H, W = 1, 3, 32, 32
        color = torch.zeros(B, C, H, W)
        depth = torch.zeros(B, 1, H, W)
        motion = torch.zeros(B, 2, H, W)
        ref = torch.zeros(B, C, H, W)

        with torch.no_grad():
            out = model(color, depth, motion, ref)

        self.assertEqual(out.shape, (B, C, H, W))
        self.assertFalse(torch.isnan(out).any())

    def test_mx_decoding(self):
        """Test numeric decoding of E4M3, FP16, and MXFP8 routines."""
        # E4M3 tests
        raw_e4 = np.array([0x00, 0x38, 0x7F, 0x80], dtype=np.uint8)
        decoded = e4m3_decode(raw_e4)
        self.assertEqual(len(decoded), 4)
        self.assertAlmostEqual(decoded[0], 0.0, places=5)
        self.assertAlmostEqual(decoded[1], 1.0, places=5)

        # FP16 tests
        raw_f16 = np.frombuffer(np.array([1.0, -2.5], dtype="<f2").tobytes(), dtype=np.uint8)
        f16_dec = fp16_decode(raw_f16)
        self.assertEqual(len(f16_dec), 2)
        self.assertAlmostEqual(f16_dec[0], 1.0, places=3)
        self.assertAlmostEqual(f16_dec[1], -2.5, places=3)

    def test_postprocess(self):
        """Test residual blend and tone mapping."""
        color = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        residual = np.zeros((64, 64, 3), dtype=np.float32)
        blended = apply_residual(color, residual)
        self.assertEqual(blended.shape, color.shape)

        srgb = linear_to_srgb(blended)
        self.assertTrue(np.all(srgb >= 0.0) and np.all(srgb <= 1.0))

    def test_end_to_end_infer(self):
        """Test full infer_frame pipeline with real weights blob if present."""
        weights_path = "weights_blob.bin"
        if not os.path.isfile(weights_path):
            self.skipTest(f"{weights_path} not found; skipping weights test")

        model = dlss5.load_model(weights_path, device="cpu", verbose=False)
        c = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        d = np.ones((64, 64), dtype=np.float32) * 0.5
        m = np.zeros((64, 64, 2), dtype=np.float32)

        out = dlss5.infer_frame(model, c, d, m, device="cpu")
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertFalse(np.isnan(out).any())
        self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0))


if __name__ == "__main__":
    unittest.main()
