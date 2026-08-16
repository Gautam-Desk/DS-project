"""
src/preprocess.py — Face Detection & Preprocessing Pipeline
=============================================================
Pipeline:
    Raw Image / Video Frame
      → Detect faces with MTCNN
      → Crop + align face with margin
      → Resize to target size
      → Normalize (ImageNet stats)
      → Apply augmentations (training only)
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN


# -----------------------------------------------------------------------
# ImageNet normalization constants
# -----------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------
# Face Detector (singleton — load once)
# -----------------------------------------------------------------------
_mtcnn_instance: Optional[MTCNN] = None


def get_face_detector(device: str = "cpu") -> MTCNN:
    """
    Return a shared MTCNN face detector instance (singleton).

    Args:
        device: 'cpu' or 'cuda'

    Returns:
        MTCNN instance configured for face cropping.
    """
    global _mtcnn_instance
    if _mtcnn_instance is None:
        _mtcnn_instance = MTCNN(
            image_size=224,
            margin=20,
            min_face_size=64,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            keep_all=False,          # Only return the most prominent face
            device=device,
            post_process=False,      # We'll normalize ourselves
        )
    return _mtcnn_instance


# -----------------------------------------------------------------------
# Augmentation Transforms
# -----------------------------------------------------------------------
def get_train_transforms(image_size: int = 380) -> A.Compose:
    """
    Heavy augmentation pipeline for training.
    Simulates real-world variation and compression artifacts.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Rotate(limit=15, p=0.3),
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=0.2,
        ),
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),  # simulate JPEG artifacts
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 380) -> A.Compose:
    """Minimal transforms for validation/test — no augmentation."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size: int = 380) -> A.Compose:
    """Transforms for single-image inference."""
    return get_val_transforms(image_size)


# -----------------------------------------------------------------------
# Face Extraction
# -----------------------------------------------------------------------
def extract_face(
    image: Image.Image,
    detector: Optional[MTCNN] = None,
    image_size: int = 380,
    margin_ratio: float = 0.3,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """
    Detect and crop face from a PIL image.

    Args:
        image       : PIL Image (RGB).
        detector    : MTCNN instance. If None, uses singleton.
        image_size  : Target output size.
        margin_ratio: Extra margin around detected face bounding box.
        device      : 'cpu' or 'cuda'.

    Returns:
        Cropped face as numpy array (H, W, 3) in uint8, or None if no face found.
    """
    if detector is None:
        detector = get_face_detector(device)

    img_np = np.array(image.convert("RGB"))

    # Detect face bounding boxes
    boxes, probs = detector.detect(image)

    if boxes is None or len(boxes) == 0:
        return None

    # Pick the face with highest detection probability
    best_idx = np.argmax(probs)
    box      = boxes[best_idx]
    x1, y1, x2, y2 = [int(v) for v in box]

    h, w = img_np.shape[:2]

    # Apply margin
    margin_x = int((x2 - x1) * margin_ratio)
    margin_y = int((y2 - y1) * margin_ratio)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    face_crop = img_np[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    # Resize to target
    face_crop = cv2.resize(face_crop, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
    return face_crop


def extract_face_or_resize(
    image: Image.Image,
    image_size: int = 380,
    device: str = "cpu",
) -> np.ndarray:
    """
    Try to detect face. If no face found, resize full image.
    This ensures we always return something for inference.

    Returns:
        numpy array (H, W, 3) uint8
    """
    face = extract_face(image, image_size=image_size, device=device)
    if face is not None:
        return face
    # Fall back to full image resize
    img_np = np.array(image.convert("RGB"))
    return cv2.resize(img_np, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)


# -----------------------------------------------------------------------
# Video Frame Extraction
# -----------------------------------------------------------------------
def extract_frames_from_video(
    video_path: str,
    max_frames: int = 100,
    sample_rate: int = 10,
) -> List[np.ndarray]:
    """
    Extract sampled frames from a video file.

    Args:
        video_path  : Path to video file.
        max_frames  : Maximum number of frames to return.
        sample_rate : Sample every Nth frame.

    Returns:
        List of numpy arrays (H, W, 3) in RGB.
    """
    cap    = cv2.VideoCapture(str(video_path))
    frames = []

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            # Convert BGR → RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        frame_idx += 1

    cap.release()
    return frames


# -----------------------------------------------------------------------
# Tensor Preparation (for inference)
# -----------------------------------------------------------------------
def prepare_image_tensor(
    image: Image.Image,
    image_size: int = 380,
    device: str = "cpu",
    extract_face: bool = True,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Full preprocessing for inference:
    PIL Image → face crop (if possible) → normalize → tensor.

    Args:
        image       : PIL Image.
        image_size  : Target size.
        device      : Torch device.
        extract_face: If True, try face detection first.

    Returns:
        (tensor: shape (1, 3, H, W), face_np: (H, W, 3) for GradCAM visualization)
    """
    if extract_face:
        face_np = extract_face_or_resize(image, image_size=image_size, device=device)
    else:
        face_np = np.array(image.convert("RGB"))
        face_np = cv2.resize(face_np, (image_size, image_size))

    transforms = get_inference_transforms(image_size)
    augmented  = transforms(image=face_np)
    tensor     = augmented["image"].unsqueeze(0).to(device)

    return tensor, face_np
