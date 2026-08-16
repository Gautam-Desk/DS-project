"""
src/model.py — Deepfake Detection Model Architecture
=====================================================
Primary model: EfficientNet-B4 / B0 fine-tuned binary classifier.

Architecture:
    Input (380x380x3)
      → EfficientNet-B4 backbone (Torchvision ImageNet pretrained)
      → Global Average Pooling
      → FC(512) → GELU → Dropout(0.4)
      → FC(256) → GELU → Dropout(0.3)
      → FC(1)   → Sigmoid Logits
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torchvision.models as models


class DeepfakeDetector(nn.Module):
    """
    EfficientNet-based binary classifier for deepfake detection.

    Args:
        architecture (str): 'efficientnet-b4', 'efficientnet-b0', or 'resnet50'.
        dropout_1 (float): Dropout rate for first FC layer.
        dropout_2 (float): Dropout rate for second FC layer.
        hidden_dim_1 (int): Size of first hidden FC layer.
        hidden_dim_2 (int): Size of second hidden FC layer.
        pretrained (bool): Load ImageNet pretrained weights.
    """

    def __init__(
        self,
        architecture: str = "efficientnet-b4",
        dropout_1: float = 0.4,
        dropout_2: float = 0.3,
        hidden_dim_1: int = 512,
        hidden_dim_2: int = 256,
        pretrained: bool = True,
    ):
        super(DeepfakeDetector, self).__init__()
        self.architecture = architecture.lower()

        # --- Backbone selection ---
        if self.architecture == "efficientnet-b4":
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b4(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.target_conv = self.backbone.features[-1]

        elif self.architecture == "efficientnet-b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.target_conv = self.backbone.features[-1]

        elif self.architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.target_conv = self.backbone.layer4[-1]

        else:
            # Default to EfficientNet-B0 for universal compatibility
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.target_conv = self.backbone.features[-1]

        # --- Custom Classifier Head ---
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim_1),
            nn.GELU(),
            nn.Dropout(p=dropout_1),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.GELU(),
            nn.Dropout(p=dropout_2),
            nn.Linear(hidden_dim_2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize FC layer weights with Kaiming normal."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def freeze_backbone(self):
        """Freeze backbone parameters (warm-up phase training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen — training classifier head only.")

    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] Backbone unfrozen — fine-tuning all layers.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            Logits tensor of shape (B, 1) — apply sigmoid for probability.
        """
        features = self.backbone(x)           # (B, in_features)
        logits   = self.classifier(features)   # (B, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probability (0=Real, 1=Fake)."""
        return torch.sigmoid(self.forward(x))

    def get_num_params(self) -> Dict[str, int]:
        """Returns dict of total and trainable parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def load_model(model_path: str, device: torch.device) -> DeepfakeDetector:
    """
    Load a saved model checkpoint.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Determine architecture
    arch = "efficientnet-b4"
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        arch = checkpoint["config"].get("model", {}).get("architecture", "efficientnet-b4")

    model = DeepfakeDetector(architecture=arch, pretrained=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Model] Loaded checkpoint (epoch: {checkpoint.get('epoch', '?')}, Val AUC: {checkpoint.get('val_auc', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
