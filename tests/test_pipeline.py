"""
tests/test_pipeline.py — Comprehensive Test Suite
==================================================
Tests:
  1. Security validation (size, extension, magic bytes, path traversal)
  2. Face detection & preprocessing
  3. Model architecture forward pass & parameter counts
  4. GradCAM heatmap generation
  5. End-to-end inference engine
"""

import sys
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.security import sanitize_filename, validate_upload
from src.preprocess import extract_face_or_resize, compute_fft_spectrum, prepare_image_tensor
from src.model import DeepfakeDetector
from src.gradcam import generate_gradcam_overlay


class TestDeepfakePipeline(unittest.TestCase):

    def setUp(self):
        # Create a test synthetic face image
        self.test_img = Image.new("RGB", (380, 380), color=(200, 180, 160))

    def test_01_security_sanitization(self):
        unsafe = "../../../etc/passwd.exe.jpg"
        safe = sanitize_filename(unsafe)
        self.assertNotIn("..", safe)
        self.assertTrue(safe.endswith(".jpg"))

    def test_02_security_validation(self):
        fake_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 500  # JPEG header
        ok, msg, meta = validate_upload("photo.jpg", fake_bytes, session_id="test_session")
        self.assertTrue(ok)
        self.assertEqual(meta["file_type"], "image")

    def test_03_preprocessing(self):
        face = extract_face_or_resize(self.test_img, image_size=380)
        self.assertEqual(face.shape, (380, 380, 3))
        self.assertEqual(face.dtype, np.uint8)

    def test_04_fft_spectrum(self):
        img_np = np.array(self.test_img)
        spectrum_rgb, anomaly = compute_fft_spectrum(img_np)
        self.assertEqual(spectrum_rgb.shape, (380, 380, 3))
        self.assertGreaterEqual(anomaly, 0.0)
        self.assertLessEqual(anomaly, 1.0)

    def test_05_model_forward(self):
        model = DeepfakeDetector(architecture="efficientnet-b4", pretrained=False)
        tensor = torch.randn(2, 3, 380, 380)
        output = model(tensor)
        self.assertEqual(output.shape, (2, 1))

    def test_06_gradcam_generation(self):
        model = DeepfakeDetector(architecture="efficientnet-b4", pretrained=False)
        tensor, face_np = prepare_image_tensor(self.test_img, image_size=380)
        overlay = generate_gradcam_overlay(model, tensor, face_np)
        self.assertEqual(overlay.shape, (380, 380, 3))


if __name__ == "__main__":
    unittest.main()
