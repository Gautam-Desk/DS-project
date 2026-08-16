"""
src/generate_sample_data.py — Sample Dataset Generator
======================================================
Creates a sample/benchmark dataset in data/splits/ for local training & testing:
  - data/splits/train/real/ & fake/
  - data/splits/val/real/ & fake/
  - data/splits/test/real/ & fake/

Generates synthetic faces with realistic color gradations, eyes, facial features,
and simulates deepfake artifacts (blending boundaries, high-frequency noise, warping).
"""

import os
import math
import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def generate_synthetic_face(is_fake: bool, size: int = 380, seed: int = 42) -> np.ndarray:
    """
    Generate a synthetic face-like portrait image with distinct Real vs Fake characteristics.
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Background gradient
    img = np.zeros((size, size, 3), dtype=np.uint8)
    bg_color1 = np.array([random.randint(20, 60), random.randint(20, 60), random.randint(30, 80)])
    bg_color2 = np.array([random.randint(70, 120), random.randint(70, 130), random.randint(100, 160)])
    for y in range(size):
        alpha = y / size
        img[y, :] = (1 - alpha) * bg_color1 + alpha * bg_color2

    # 2. Face Oval (skin tone)
    skin_base = np.array([
        random.randint(150, 210),  # B
        random.randint(170, 225),  # G
        random.randint(210, 250),  # R
    ], dtype=np.float32)

    center = (size // 2, int(size * 0.52))
    axes = (int(size * 0.28), int(size * 0.36))
    cv2.ellipse(img, center, axes, 0, 0, 360, skin_base.astype(np.uint8).tolist(), -1)

    # 3. Eyes
    eye_y = int(size * 0.44)
    left_eye = (int(size * 0.40), eye_y)
    right_eye = (int(size * 0.60), eye_y)
    eye_axes = (int(size * 0.05), int(size * 0.03))

    cv2.ellipse(img, left_eye, eye_axes, 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, right_eye, eye_axes, 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, left_eye, int(size * 0.018), (50, 40, 20), -1)
    cv2.circle(img, right_eye, int(size * 0.018), (50, 40, 20), -1)

    # 4. Nose & Mouth
    nose_pts = np.array([[size // 2, int(size * 0.48)], [int(size * 0.48), int(size * 0.58)], [int(size * 0.52), int(size * 0.58)]])
    cv2.polylines(img, [nose_pts], isClosed=False, color=(140, 150, 180), thickness=2)
    cv2.ellipse(img, (size // 2, int(size * 0.68)), (int(size * 0.08), int(size * 0.025)), 0, 0, 360, (100, 110, 190), -1)

    # 5. Add Realistic Texture or Deepfake Artifacts
    if is_fake:
        # FAKE ARTIFACT 1: FaceSwap blending seam around boundary
        seam_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(seam_mask, center, (int(axes[0] * 0.95), int(axes[1] * 0.95)), 0, 0, 360, 255, 3)
        seam_noise = np.random.normal(0, 35, (size, size, 3)).astype(np.int16)
        img_int = img.astype(np.int16)
        for c in range(3):
            img_int[:, :, c] += (seam_mask > 0) * seam_noise[:, :, c]

        # FAKE ARTIFACT 2: GAN high-frequency periodic grid noise in facial region
        y_grid, x_grid = np.ogrid[:size, :size]
        grid_pattern = (np.sin(x_grid * 0.6) * np.cos(y_grid * 0.6) * 18).astype(np.int16)
        face_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)
        for c in range(3):
            img_int[:, :, c] += (face_mask > 0) * grid_pattern

        # FAKE ARTIFACT 3: Blurring disparity (sharp background, softened eyes)
        img = np.clip(img_int, 0, 255).astype(np.uint8)
        blurred_face = cv2.GaussianBlur(img, (7, 7), 2.0)
        inner_mask = np.zeros((size, size), dtype=np.float32)
        cv2.ellipse(inner_mask, (size // 2, int(size * 0.50)), (int(axes[0] * 0.7), int(axes[1] * 0.7)), 0, 0, 360, 1.0, -1)
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * (1 - inner_mask * 0.4) + blurred_face[:, :, c] * (inner_mask * 0.4)).astype(np.uint8)
    else:
        # REAL: Natural film grain & smooth illumination gradients
        noise = np.random.normal(0, 4, (size, size, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Convert BGR -> RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_sample_dataset(base_dir: str = "data/splits", count_per_class: int = 30):
    """
    Generate dataset splits:
      - train: count_per_class
      - val  : count_per_class // 3
      - test : count_per_class // 3
    """
    base_path = Path(base_dir)
    splits = {
        "train": count_per_class,
        "val"  : max(6, count_per_class // 4),
        "test" : max(6, count_per_class // 4),
    }

    print(f"Creating sample dataset in '{base_dir}'...")
    total_generated = 0

    for split_name, n_samples in splits.items():
        for label_name, is_fake in [("real", False), ("fake", True)]:
            folder = base_path / split_name / label_name
            folder.mkdir(parents=True, exist_ok=True)

            for i in range(n_samples):
                seed = hash(f"{split_name}_{label_name}_{i}") % 100000
                img_rgb = generate_synthetic_face(is_fake=is_fake, size=380, seed=seed)
                file_path = folder / f"sample_{i+1:03d}.jpg"
                Image.fromarray(img_rgb).save(file_path, "JPEG", quality=92)
                total_generated += 1

    print(f"[Dataset] Generated {total_generated} sample images across train/val/test splits!")


if __name__ == "__main__":
    build_sample_dataset()
