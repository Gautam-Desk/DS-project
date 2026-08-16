"""
src/preprocess.py — Face Detection, Preprocessing & Forensic Analysis
=======================================================================
Pipeline:
    Raw Image / Video Frame
      → Detect face (Haar Cascade / MTCNN fallback)
      → Crop + align face with configurable margin
      → Resize to target size (380x380)
      → Normalize (ImageNet statistics)
      → Optional FFT high-frequency forensic analysis
"""

from typing import Optional, Tuple, List, Dict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------
# Fast Face Detector (OpenCV Haar Cascade + MTCNN fallback)
# -----------------------------------------------------------------------
_haar_cascade = None


def get_haar_cascade():
    """Load OpenCV frontal face Haar cascade (built into opencv-python)."""
    global _haar_cascade
    if _haar_cascade is None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            _haar_cascade = None
    return _haar_cascade


def detect_face_bbox(image_rgb: np.ndarray, margin_ratio: float = 0.25) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face bounding box in RGB numpy array.
    Returns (x1, y1, x2, y2) with margin applied, or None if no face found.
    """
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    cascade = get_haar_cascade()
    if cascade is not None and not cascade.empty():
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if len(faces) > 0:
            # Pick largest face by area
            largest = max(faces, key=lambda b: b[2] * b[3])
            x, y, fw, fh = largest

            # Add margin
            mx = int(fw * margin_ratio)
            my = int(fh * margin_ratio)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + fw + mx)
            y2 = min(h, y + fh + my)
            return (x1, y1, x2, y2)

    return None


def extract_face(
    image: Image.Image,
    image_size: int = 380,
    margin_ratio: float = 0.25,
) -> Optional[np.ndarray]:
    """
    Detect and crop face from PIL image.
    Returns RGB uint8 numpy array (image_size, image_size, 3) or None.
    """
    img_np = np.array(image.convert("RGB"))
    bbox   = detect_face_bbox(img_np, margin_ratio=margin_ratio)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        face_crop = img_np[y1:y2, x1:x2]
        if face_crop.size > 0:
            return cv2.resize(face_crop, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)

    return None


def extract_face_or_resize(
    image: Image.Image,
    image_size: int = 380,
) -> np.ndarray:
    """
    Extract face if present, otherwise resize full image.
    Always returns valid (image_size, image_size, 3) RGB array.
    """
    face = extract_face(image, image_size=image_size)
    if face is not None:
        return face
    img_np = np.array(image.convert("RGB"))
    return cv2.resize(img_np, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)


# -----------------------------------------------------------------------
# FFT / Frequency Domain Forensic Analysis
# -----------------------------------------------------------------------
def compute_fft_spectrum(image_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute 2D Discrete Fourier Transform magnitude spectrum.
    Deepfakes & GANs frequently exhibit distinctive radial/azimuthal frequency spikes.

    Returns:
        (spectrum_vis: np.ndarray (H, W, 3) heatmap, anomaly_score: float [0, 1])
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f    = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-6)

    # Normalize magnitude for visualization
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max > mag_min:
        mag_norm = ((magnitude - mag_min) / (mag_max - mag_min) * 255).astype(np.uint8)
    else:
        mag_norm = np.zeros_like(magnitude, dtype=np.uint8)

    spectrum_color = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)
    spectrum_rgb   = cv2.cvtColor(spectrum_color, cv2.COLOR_BGR2RGB)

    # Anomaly metric: high frequency variance ratio
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask_high = ((x - cx)**2 + (y - cy)**2) > (r**2)

    high_freq_power = float(np.mean(np.abs(fshift)[mask_high]))
    total_power     = float(np.mean(np.abs(fshift)) + 1e-6)
    anomaly_score   = min(1.0, (high_freq_power / total_power) * 1.5)

    return spectrum_rgb, round(anomaly_score, 3)


# -----------------------------------------------------------------------
# Video Frame Extraction
# -----------------------------------------------------------------------
def extract_frames_from_video(
    video_path: str,
    max_frames: int = 100,
    sample_rate: int = 10,
) -> List[np.ndarray]:
    """
    Extract sampled frames from video file.
    Returns list of RGB numpy arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        frame_idx += 1

    cap.release()
    return frames


# -----------------------------------------------------------------------
# Transforms & Tensor Prep
# -----------------------------------------------------------------------
def prepare_image_tensor(
    image: Image.Image,
    image_size: int = 380,
    device: str = "cpu",
    extract_face_crop: bool = True,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Prepares input tensor for inference:
    PIL -> Face Crop/Resize -> Normalize -> Tensor (1, 3, H, W).
    """
    if extract_face_crop:
        face_np = extract_face_or_resize(image, image_size=image_size)
    else:
        img_np  = np.array(image.convert("RGB"))
        face_np = cv2.resize(img_np, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)

    # Normalize with ImageNet mean & std
    norm_img = face_np.astype(np.float32) / 255.0
    norm_img = (norm_img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)

    # (H, W, C) -> (1, C, H, W)
    tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).float().unsqueeze(0)
    tensor = tensor.to(torch.device(device))

    return tensor, face_np


def get_train_transforms(image_size: int = 380):
    """
    Standard data augmentation for training.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    except ImportError:
        import torchvision.transforms as T
        return T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_val_transforms(image_size: int = 380):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    except ImportError:
        import torchvision.transforms as T
        return T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
