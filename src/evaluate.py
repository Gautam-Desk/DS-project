"""
src/evaluate.py — Model Evaluation & Forensic Metrics
======================================================
Computes:
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC & Precision-Recall Curve
  - Confusion Matrix
  - Full Classification Report
  - Generates publication-ready figures to reports/figures/
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_recall_curve,
)

from src.model import load_model
from src.data_loader import build_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on DataLoader and return (y_true, y_pred, y_proba)."""
    model.eval()
    all_labels = []
    all_probs  = []

    for images, labels in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
        images = images.to(device)
        logits = model(images)
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().flatten().tolist())

    y_true  = np.array(all_labels)
    y_proba = np.array(all_probs)
    y_pred  = (y_proba >= threshold).astype(int)
    return y_true, y_pred, y_proba


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy" : float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"   : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score" : float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc"  : float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 1.0,
        "avg_prec" : float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 1.0,
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "reports/figures/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real (Pred)", "Fake (Pred)"],
        yticklabels=["Real (True)", "Fake (True)"],
        ax=ax, annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title("Confusion Matrix — Deepfake Detection", fontsize=13, pad=10, weight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: str = "reports/figures/roc_curve.png"):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#6366f1", lw=2.5, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5)
    ax.set_title("ROC Curve", fontsize=13, weight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate(
    model_path: str = "models/best_model.pth",
    config_path: str = "config.yaml",
    split: str = "test",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    loaders = build_dataloaders(
        splits_dir=cfg["data"]["splits_dir"],
        image_size=cfg["preprocessing"]["image_size"],
        batch_size=cfg["evaluation"].get("batch_size", 32),
        num_workers=0,
    )

    loader = loaders.get(split, loaders["test"])
    threshold = cfg["evaluation"].get("threshold", 0.5)

    y_true, y_pred, y_proba = get_predictions(model, loader, device, threshold)
    metrics = compute_metrics(y_true, y_pred, y_proba)

    print("\n" + "="*45)
    print(f"  📊 {split.upper()} Evaluation Results")
    print("="*45)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("="*45 + "\n")

    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_proba)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.pth")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    evaluate(args.model, args.config, args.split)
