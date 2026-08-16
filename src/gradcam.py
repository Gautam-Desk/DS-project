"""
src/gradcam.py — GradCAM Explainability
=========================================
Generates Gradient-weighted Class Activation Maps to visualize
which facial regions the model uses for its deepfake prediction.

High activation = regions that most influenced the FAKE decision.
Typical hotspots: eye boundaries, mouth edges, jawline, skin texture.

Usage:
    from src.gradcam import generate_gradcam_overlay
    overlay = generate_gradcam_overlay(model, image_tensor, face_rgb)
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from typing import Optional, Tuple


# -----------------------------------------------------------------------
# Manual GradCAM (no extra dependency needed)
# -----------------------------------------------------------------------
class GradCAM:
    """
    GradCAM implementation for EfficientNet-B4.

    Hooks into the last convolutional block to compute:
        CAM = ReLU( sum_k( alpha_k * A_k ) )
    where alpha_k = global average of gradients for class k.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model        : The DeepfakeDetector model.
            target_layer : The layer to hook (last conv block).
        """
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Clean up hooks after use."""
        for hook in self._hooks:
            hook.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor : Preprocessed image tensor (1, 3, H, W).
            target_class : Class index. None = use predicted class.

        Returns:
            Heatmap as numpy array (H, W) in range [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)  # (1, 1)
        score  = output[0, 0]

        # Backward pass to get gradients w.r.t. target layer
        score.backward()

        # Compute weights: global average pool of gradients
        gradients   = self.gradients   # (1, C, H, W)
        activations = self.activations # (1, C, H, W)

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def __del__(self):
        self.remove_hooks()


# -----------------------------------------------------------------------
# Overlay Generator
# -----------------------------------------------------------------------
def colorize_heatmap(cam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Convert grayscale CAM to colored heatmap and resize.

    Args:
        cam  : Heatmap numpy array (H, W) in [0, 1].
        size : Target (width, height) for resizing.

    Returns:
        BGR heatmap numpy array (H, W, 3).
    """
    cam_resized   = cv2.resize(cam, size, interpolation=cv2.INTER_LINEAR)
    cam_uint8     = np.uint8(255 * cam_resized)
    heatmap_bgr   = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return heatmap_bgr


def generate_gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    face_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Generate a GradCAM overlay on the face image.

    Args:
        model        : Trained DeepfakeDetector model.
        input_tensor : Preprocessed input tensor (1, 3, H, W).
        face_rgb     : Original face image as RGB numpy array (H, W, 3).
        alpha        : Opacity of heatmap overlay (0=invisible, 1=only heatmap).

    Returns:
        Overlay image as RGB numpy array (H, W, 3).
    """
    # Target: last block of EfficientNet backbone
    target_layer = model.backbone._blocks[-1]

    cam_engine = GradCAM(model, target_layer)

    try:
        cam = cam_engine.generate(input_tensor)
    finally:
        cam_engine.remove_hooks()

    h, w = face_rgb.shape[:2]

    # Colorize and resize heatmap to match face image
    heatmap_bgr = colorize_heatmap(cam, (w, h))
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(face_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def gradcam_to_pil(overlay: np.ndarray) -> Image.Image:
    """Convert numpy overlay array to PIL Image."""
    return Image.fromarray(overlay.astype(np.uint8))


# -----------------------------------------------------------------------
# Side-by-Side Comparison
# -----------------------------------------------------------------------
def make_comparison_figure(
    original_rgb : np.ndarray,
    overlay_rgb  : np.ndarray,
    label        : str,
    confidence   : float,
) -> np.ndarray:
    """
    Create a side-by-side original | GradCAM overlay image with labels.

    Returns:
        Combined image as numpy array (H, W*2, 3).
    """
    h, w = original_rgb.shape[:2]
    canvas = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
    canvas[:, :w]        = original_rgb
    canvas[:, w + 20:]   = overlay_rgb
    # Divider line
    canvas[:, w:w + 20]  = 255

    return canvas
