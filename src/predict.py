"""
src/predict.py — Inference Pipeline
=====================================
Supports:
  - Single image prediction
  - Batch image prediction
  - Video prediction (frame-by-frame + aggregate)
  - Returns probability, label, and GradCAM overlay

Usage:
    from src.predict import DeepfakePredictor
    predictor = DeepfakePredictor("models/best_model.pth")
    result = predictor.predict_image(pil_image)
"""

import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from PIL import Image

from src.model import load_model
from src.preprocess import prepare_image_tensor, extract_frames_from_video
from src.gradcam import generate_gradcam_overlay, gradcam_to_pil

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Predictor Class
# -----------------------------------------------------------------------
class DeepfakePredictor:
    """
    High-level inference engine for deepfake detection.

    Args:
        model_path  : Path to trained .pth checkpoint.
        device      : 'cuda', 'cpu', or 'auto' (default).
        threshold   : Decision threshold (0.5 default).
        image_size  : Input size expected by the model.
    """

    LABELS = {0: "REAL", 1: "FAKE"}
    EMOJIS = {0: "✅", 1: "🚨"}
    COLORS = {0: "#2ECC71", 1: "#E74C3C"}

    def __init__(
        self,
        model_path : str   = "models/best_model.pth",
        device     : str   = "auto",
        threshold  : float = 0.5,
        image_size : int   = 380,
    ):
        self.threshold  = threshold
        self.image_size = image_size

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"[Predictor] Using device: {self.device}")

        # Load model
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Train the model first with: python -m src.train"
            )
        self.model = load_model(model_path, self.device)
        logger.info(f"[Predictor] Model loaded from: {model_path}")

    # ------------------------------------------------------------------
    # Image Prediction
    # ------------------------------------------------------------------
    def predict_image(
        self,
        image: Image.Image,
        return_gradcam: bool = True,
    ) -> Dict:
        """
        Predict whether a single PIL Image is real or fake.

        Args:
            image         : PIL Image (RGB or any mode).
            return_gradcam: If True, also generate GradCAM overlay.

        Returns:
            dict with keys:
                label       : "REAL" or "FAKE"
                probability : float [0, 1] — probability of being FAKE
                confidence  : float [0, 1] — confidence in the prediction
                is_fake     : bool
                color       : hex color for UI
                emoji       : status emoji
                gradcam_pil : PIL Image of GradCAM overlay (if requested)
                inference_ms: inference time in milliseconds
        """
        t0 = time.time()

        # Preprocess
        tensor, face_np = prepare_image_tensor(
            image,
            image_size   = self.image_size,
            device       = str(self.device),
            extract_face = True,
        )

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            logit      = self.model(tensor)
            fake_prob  = torch.sigmoid(logit).item()

        # Determine label
        is_fake     = fake_prob >= self.threshold
        label_idx   = int(is_fake)
        label       = self.LABELS[label_idx]
        confidence  = fake_prob if is_fake else (1.0 - fake_prob)

        # GradCAM overlay
        gradcam_pil = None
        if return_gradcam:
            try:
                overlay     = generate_gradcam_overlay(self.model, tensor, face_np)
                gradcam_pil = gradcam_to_pil(overlay)
            except Exception as e:
                logger.warning(f"[GradCAM] Failed to generate: {e}")

        elapsed_ms = (time.time() - t0) * 1000

        return {
            "label"       : label,
            "probability" : round(fake_prob, 4),
            "confidence"  : round(confidence, 4),
            "is_fake"     : bool(is_fake),
            "color"       : self.COLORS[label_idx],
            "emoji"       : self.EMOJIS[label_idx],
            "gradcam_pil" : gradcam_pil,
            "face_np"     : face_np,
            "inference_ms": round(elapsed_ms, 1),
        }

    # ------------------------------------------------------------------
    # Video Prediction
    # ------------------------------------------------------------------
    def predict_video(
        self,
        video_path    : str,
        max_frames    : int = 100,
        sample_rate   : int = 10,
        return_gradcam: bool = True,
    ) -> Dict:
        """
        Predict deepfake probability for a video file.

        Strategy: Sample frames → detect face in each → aggregate scores.

        Args:
            video_path   : Path to video file.
            max_frames   : Max number of frames to sample.
            sample_rate  : Sample every Nth frame.
            return_gradcam: Generate GradCAM for worst (most fake) frame.

        Returns:
            dict with:
                label           : "REAL" or "FAKE"
                probability     : Aggregated fake probability
                confidence      : Confidence in final decision
                frame_probs     : List of per-frame fake probabilities
                frames_analyzed : Number of frames analyzed
                gradcam_pil     : GradCAM for most suspicious frame
                inference_ms    : Total inference time
        """
        t0 = time.time()

        logger.info(f"[Video] Extracting frames from: {video_path}")
        frames_np = extract_frames_from_video(
            video_path,
            max_frames  = max_frames,
            sample_rate = sample_rate,
        )

        if not frames_np:
            raise ValueError("No frames could be extracted from the video.")

        logger.info(f"[Video] Analyzing {len(frames_np)} frames...")

        frame_probs  = []
        best_frame   = None
        best_prob    = 0.0

        self.model.eval()

        for frame_np in frames_np:
            pil_frame = Image.fromarray(frame_np)
            tensor, face_np = prepare_image_tensor(
                pil_frame,
                image_size   = self.image_size,
                device       = str(self.device),
                extract_face = True,
            )

            with torch.no_grad():
                logit     = self.model(tensor)
                fake_prob = torch.sigmoid(logit).item()

            frame_probs.append(round(fake_prob, 4))

            # Track most suspicious frame
            if fake_prob > best_prob:
                best_prob  = fake_prob
                best_frame = (tensor, face_np)

        # Aggregate: use mean probability (robust to one-off errors)
        agg_prob  = float(np.mean(frame_probs))
        is_fake   = agg_prob >= self.threshold
        label_idx = int(is_fake)
        confidence = agg_prob if is_fake else (1.0 - agg_prob)

        # GradCAM on most suspicious frame
        gradcam_pil = None
        if return_gradcam and best_frame is not None:
            try:
                overlay     = generate_gradcam_overlay(self.model, best_frame[0], best_frame[1])
                gradcam_pil = gradcam_to_pil(overlay)
            except Exception as e:
                logger.warning(f"[GradCAM] Failed: {e}")

        elapsed_ms = (time.time() - t0) * 1000

        return {
            "label"           : self.LABELS[label_idx],
            "probability"     : round(agg_prob, 4),
            "confidence"      : round(confidence, 4),
            "is_fake"         : bool(is_fake),
            "color"           : self.COLORS[label_idx],
            "emoji"           : self.EMOJIS[label_idx],
            "frame_probs"     : frame_probs,
            "frames_analyzed" : len(frame_probs),
            "gradcam_pil"     : gradcam_pil,
            "inference_ms"    : round(elapsed_ms, 1),
        }


# -----------------------------------------------------------------------
# Quick CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)

    img_path  = sys.argv[1]
    image     = Image.open(img_path).convert("RGB")
    predictor = DeepfakePredictor()
    result    = predictor.predict_image(image)

    print(f"\n{'='*40}")
    print(f"  Result     : {result['emoji']} {result['label']}")
    print(f"  Fake Prob  : {result['probability']*100:.2f}%")
    print(f"  Confidence : {result['confidence']*100:.2f}%")
    print(f"  Time       : {result['inference_ms']} ms")
    print(f"{'='*40}\n")
