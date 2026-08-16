"""
src/gradcam.py — GradCAM Explainability Engine
================================================
Generates Gradient-weighted Class Activation Maps to visualize
which facial regions the model used for its deepfake prediction.

High activation = regions that most influenced the FAKE decision.
Typical hotspots: eye boundaries, mouth blending, jawline artifacts, texture irregularities.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    """
    GradCAM implementation compatible with torchvision EfficientNet & ResNet backbones.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        if target_layer is None:
            if hasattr(model, "target_conv"):
                self.target_layer = model.target_conv
            elif hasattr(model, "backbone") and hasattr(model.backbone, "features"):
                self.target_layer = model.backbone.features[-1]
            elif hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
                self.target_layer = model.backbone.layer4[-1]
            else:
                # Find last Conv2d layer
                convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                self.target_layer = convs[-1] if convs else None
        else:
            self.target_layer = target_layer

        self.gradients   = None
        self.activations = None
        self._hooks      = []
        if self.target_layer is not None:
            self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate GradCAM heatmap for the binary output.
        """
        if self.target_layer is None:
            # Fallback if no conv layer found
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        score  = output[0, 0]

        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        # Global average pool over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def __del__(self):
        self.remove_hooks()


def colorize_heatmap(cam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Convert grayscale CAM to BGR color heatmap."""
    cam_resized = cv2.resize(cam, size, interpolation=cv2.INTER_LINEAR)
    cam_uint8   = np.uint8(255 * cam_resized)
    return cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)


def generate_gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    face_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Generate blended GradCAM heatmap over original face image.
    """
    cam_engine = GradCAM(model)
    try:
        cam = cam_engine.generate(input_tensor)
    finally:
        cam_engine.remove_hooks()

    h, w = face_rgb.shape[:2]
    heatmap_bgr = colorize_heatmap(cam, (w, h))
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(face_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def gradcam_to_pil(overlay: np.ndarray) -> Image.Image:
    return Image.fromarray(overlay.astype(np.uint8))
