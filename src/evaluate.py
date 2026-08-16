"""
src/evaluate.py — Model Evaluation & Metrics
=============================================
Computes:
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC
  - Confusion Matrix
  - Classification Report
  - Saves plots to reports/figures/

Usage:
    python -m src.evaluate --model models/best_model.pth
"""

import os
import argparse
import logging
import yaml
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# Style
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = {"real": "#2ECC71", "fake": "#E74C3C"}


# -----------------------------------------------------------------------
# Inference Pass
# -----------------------------------------------------------------------
@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on a DataLoader and collect predictions.

    Returns:
        (y_true, y_pred, y_proba) — numpy arrays.
    """
    model.eval()
    all_labels = []
    all_probs  = []

    for images, labels in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().flatten().tolist())

    y_true  = np.array(all_labels)
    y_proba = np.array(all_probs)
    y_pred  = (y_proba >= threshold).astype(int)

    return y_true, y_pred, y_proba


# -----------------------------------------------------------------------
# Metrics Computation
# -----------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute all binary classification metrics."""
    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, zero_division=0),
        "f1_score" : f1_score(y_true, y_pred, zero_division=0),
        "auc_roc"  : roc_auc_score(y_true, y_proba),
        "avg_prec" : average_precision_score(y_true, y_proba),
    }


def print_metrics(metrics: Dict[str, float], split: str = "Test"):
    """Pretty print metrics table."""
    print(f"\n{'='*50}")
    print(f"  📊 {split} Set Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1-Score      : {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC       : {metrics['auc_roc']:.4f}")
    print(f"  Avg Precision : {metrics['avg_prec']:.4f}")
    print(f"{'='*50}\n")


# -----------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "reports/figures/confusion_matrix.png",
):
    """Plot and save a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdYlGn_r",
        xticklabels=["Real (Pred)", "Fake (Pred)"],
        yticklabels=["Real (True)", "Fake (True)"],
        linewidths=1,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_title("Confusion Matrix — Deepfake Detection", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str = "reports/figures/roc_curve.png",
):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc         = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#E74C3C", lw=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#E74C3C")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Deepfake Detection", fontsize=14, weight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved: {save_path}")


def plot_precision_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str = "reports/figures/precision_recall.png",
):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_prec = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#3498DB", lw=2.5, label=f"AP = {avg_prec:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color="#3498DB")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, weight="bold", pad=12)
    ax.legend(fontsize=11)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Precision-recall curve saved: {save_path}")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = "reports/figures/training_history.png",
):
    """Plot training and validation loss/AUC curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], color="#E74C3C", lw=2, marker="o", ms=4, label="Train")
    ax1.plot(epochs, history["val_loss"],   color="#3498DB", lw=2, marker="o", ms=4, label="Val")
    ax1.set_title("Loss per Epoch", fontsize=13, weight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.legend()

    # AUC
    ax2.plot(epochs, history["train_auc"], color="#E74C3C", lw=2, marker="o", ms=4, label="Train")
    ax2.plot(epochs, history["val_auc"],   color="#3498DB", lw=2, marker="o", ms=4, label="Val")
    ax2.set_title("AUC-ROC per Epoch", fontsize=13, weight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC")
    ax2.set_ylim([0.5, 1.0])
    ax2.legend()

    plt.suptitle("Training History", fontsize=15, weight="bold", y=1.01)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training history saved: {save_path}")


# -----------------------------------------------------------------------
# Full Evaluation Pipeline
# -----------------------------------------------------------------------
def evaluate(
    model_path: str = "models/best_model.pth",
    config_path: str = "config.yaml",
    split: str = "test",
):
    """
    Load model, run evaluation on a split, generate all reports & plots.

    Args:
        model_path  : Path to saved .pth checkpoint.
        config_path : Path to config.yaml.
        split       : 'val' or 'test'.
    """
    cfg    = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_path, device)

    # Load data
    loaders   = build_dataloaders(
        splits_dir  = cfg["data"]["splits_dir"],
        image_size  = cfg["preprocessing"]["image_size"],
        batch_size  = cfg["evaluation"]["batch_size"],
        num_workers = cfg["training"]["num_workers"],
    )
    loader    = loaders[split]
    threshold = cfg["evaluation"]["threshold"]

    # Run predictions
    y_true, y_pred, y_proba = get_predictions(model, loader, device, threshold)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)
    print_metrics(metrics, split=split.capitalize())

    # Print classification report
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    # Generate plots
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_proba)
    plot_precision_recall(y_true, y_proba)

    logger.info("✅ Evaluation complete. Plots saved to reports/figures/")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="models/best_model.pth")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split",  default="test", choices=["val", "test"])
    args = parser.parse_args()
    evaluate(args.model, args.config, args.split)
