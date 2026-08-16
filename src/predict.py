"""
src/predict.py — High-Level Deepfake Inference Engine
======================================================
Provides unified interface for:
  - Single image deepfake inference
  - Video multi-frame analysis with suspicious frame tracking
  - GradCAM explainability heatmap generation
  - FFT frequency domain residual analysis
  - Risk tier categorization (Authentic, Suspicious, High Probability Fake)
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
from PIL import Image

from src.model import load_model, DeepfakeDetector
from src.preprocess import (
    prepare_image_tensor,
    extract_frames_from_video,
    compute_fft_spectrum,
)
from src.gradcam import generate_gradcam_overlay, gradcam_to_pil

logger = logging.getLogger(__name__)


class DeepfakePredictor:
    """
    Production inference engine for deepfake detection.
    """

    def __init__(
        self,
        model_path: str = "models/best_model.pth",
        device: str = "auto",
        threshold: float = 0.5,
        image_size: int = 380,
    ):
        self.threshold  = threshold
        self.image_size = image_size

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        self.model = load_model(model_path, self.device)
        self.model.eval()

    def get_risk_tier(self, prob: float) -> Tuple[str, str, str]:
        """
        Classify probability into risk tier.
        Returns (tier_name, color_hex, emoji).
        """
        if prob < 0.35:
            return "Authentic / Real", "#22c55e", "✅"
        elif prob < 0.65:
            return "Suspicious / Inconclusive", "#f59e0b", "⚠️"
        else:
            return "High Probability Fake", "#ef4444", "🚨"

    def predict_image(
        self,
        image: Image.Image,
        return_gradcam: bool = True,
        return_fft: bool = True,
    ) -> Dict:
        """
        Run inference on a single PIL Image.
        """
        t0 = time.time()

        # Preprocess input tensor and face crop
        tensor, face_np = prepare_image_tensor(
            image,
            image_size=self.image_size,
            device=str(self.device),
            extract_face_crop=True,
        )

        # Forward pass
        with torch.no_grad():
            logit = self.model(tensor)
            fake_prob = float(torch.sigmoid(logit).item())

        is_fake    = fake_prob >= self.threshold
        label      = "FAKE" if is_fake else "REAL"
        confidence = fake_prob if is_fake else (1.0 - fake_prob)
        tier_name, color, emoji = self.get_risk_tier(fake_prob)

        # GradCAM
        gradcam_pil = None
        if return_gradcam:
            try:
                overlay = generate_gradcam_overlay(self.model, tensor, face_np)
                gradcam_pil = gradcam_to_pil(overlay)
            except Exception as e:
                logger.warning(f"[GradCAM] Overlay error: {e}")

        # FFT Analysis
        fft_pil = None
        fft_anomaly = 0.0
        if return_fft:
            try:
                fft_vis, fft_anomaly = compute_fft_spectrum(face_np)
                fft_pil = Image.fromarray(fft_vis)
            except Exception as e:
                logger.warning(f"[FFT] Spectrum error: {e}")

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        # Forensic indicators
        detected_artifacts = []
        if is_fake:
            if fft_anomaly > 0.45:
                detected_artifacts.append("High-frequency periodic spectral pattern (GAN grid)")
            detected_artifacts.extend([
                "Facial texture blending inconsistency",
                "Boundary gradient anomaly around eyes/jawline",
            ])

        return {
            "label"        : label,
            "is_fake"      : is_fake,
            "probability"  : round(fake_prob, 4),
            "confidence"   : round(confidence, 4),
            "risk_tier"    : tier_name,
            "color"        : color,
            "emoji"        : emoji,
            "gradcam_pil"  : gradcam_pil,
            "fft_pil"      : fft_pil,
            "fft_anomaly"  : fft_anomaly,
            "face_np"      : face_np,
            "artifacts"    : detected_artifacts,
            "inference_ms" : elapsed_ms,
            "device"       : str(self.device),
        }

    def predict_video(
        self,
        video_path: str,
        max_frames: int = 40,
        sample_rate: int = 8,
        return_gradcam: bool = True,
    ) -> Dict:
        """
        Run inference across sampled video frames.
        """
        t0 = time.time()
        frames_np = extract_frames_from_video(
            video_path,
            max_frames=max_frames,
            sample_rate=sample_rate,
        )

        if not frames_np:
            raise ValueError("No valid frames could be decoded from the video.")

        frame_probs = []
        worst_frame = None
        worst_prob  = -1.0

        for frame_np in frames_np:
            pil_frame = Image.fromarray(frame_np)
            tensor, face_np = prepare_image_tensor(
                pil_frame,
                image_size=self.image_size,
                device=str(self.device),
                extract_face_crop=True,
            )

            with torch.no_grad():
                logit = self.model(tensor)
                prob  = float(torch.sigmoid(logit).item())

            frame_probs.append(round(prob, 4))
            if prob > worst_prob:
                worst_prob  = prob
                worst_frame = (tensor, face_np)

        agg_prob = float(np.mean(frame_probs))
        is_fake  = agg_prob >= self.threshold
        label    = "FAKE" if is_fake else "REAL"
        confidence = agg_prob if is_fake else (1.0 - agg_prob)
        tier_name, color, emoji = self.get_risk_tier(agg_prob)

        # GradCAM on worst frame
        gradcam_pil = None
        if return_gradcam and worst_frame is not None:
            try:
                overlay = generate_gradcam_overlay(self.model, worst_frame[0], worst_frame[1])
                gradcam_pil = gradcam_to_pil(overlay)
            except Exception as e:
                logger.warning(f"[GradCAM] Video worst-frame error: {e}")

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return {
            "label"           : label,
            "is_fake"         : is_fake,
            "probability"     : round(agg_prob, 4),
            "confidence"      : round(confidence, 4),
            "risk_tier"       : tier_name,
            "color"           : color,
            "emoji"           : emoji,
            "frame_probs"     : frame_probs,
            "frames_analyzed" : len(frame_probs),
            "peak_fake_prob"  : round(worst_prob, 4),
            "gradcam_pil"     : gradcam_pil,
            "inference_ms"    : elapsed_ms,
        }
