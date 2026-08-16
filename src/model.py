"""
src/model.py — Deepfake Detection Model Architecture
=====================================================
Primary model: EfficientNet-B4 fine-tuned binary classifier.

Architecture:
    Input (380x380x3)
      → EfficientNet-B4 backbone (ImageNet pretrained)
      → Global Average Pooling (done inside EfficientNet)
      → FC(512) → GELU → Dropout(0.4)
      → FC(256) → GELU → Dropout(0.3)
      → FC(1)   → Sigmoid
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class DeepfakeDetector(nn.Module):
    """
    EfficientNet-B4 based binary classifier for deepfake detection.

    Args:
        dropout_1 (float): Dropout rate for first FC layer.
        dropout_2 (float): Dropout rate for second FC layer.
        hidden_dim_1 (int): Size of first hidden FC layer.
        hidden_dim_2 (int): Size of second hidden FC layer.
        pretrained (bool): Load ImageNet pretrained weights.
    """

    def __init__(
        self,
        dropout_1: float = 0.4,
        dropout_2: float = 0.3,
        hidden_dim_1: int = 512,
        hidden_dim_2: int = 256,
        pretrained: bool = True,
    ):
        super(DeepfakeDetector, self).__init__()

        # --- Backbone ---
        if pretrained:
            self.backbone = EfficientNet.from_pretrained("efficientnet-b4")
        else:
            self.backbone = EfficientNet.from_name("efficientnet-b4")

        # Get the number of features from the backbone's final FC layer
        in_features = self.backbone._fc.in_features

        # Replace the built-in classifier with Identity (we build our own)
        self.backbone._fc = nn.Identity()

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

        # --- Weight initialization for classifier ---
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
        features = self.backbone(x)          # (B, in_features)
        logits   = self.classifier(features)  # (B, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probability (0=Real, 1=Fake)."""
        return torch.sigmoid(self.forward(x))

    def get_num_params(self) -> dict:
        """Returns dict of total and trainable parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def load_model(model_path: str, device: torch.device) -> DeepfakeDetector:
    """
    Load a saved model checkpoint.

    Args:
        model_path: Path to .pth checkpoint file.
        device: torch device to load model onto.

    Returns:
        Loaded DeepfakeDetector model in eval mode.
    """
    model = DeepfakeDetector(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)

    # Support both raw state_dict and full checkpoint dicts
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Model] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        print(f"[Model] Best Val AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick sanity check
    model  = DeepfakeDetector(pretrained=False)
    params = model.get_num_params()
    print(f"Total params    : {params['total']:,}")
    print(f"Trainable params: {params['trainable']:,}")

    dummy  = torch.randn(2, 3, 380, 380)
    output = model(dummy)
    print(f"Output shape    : {output.shape}")  # Expected: (2, 1)
    print(f"Probabilities   : {torch.sigmoid(output).detach()}")
