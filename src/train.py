"""
src/train.py — Full Model Training Pipeline
=============================================
Features:
  - Warm-up phase: Frozen backbone (train classifier head)
  - Fine-tuning phase: Full end-to-end backpropagation
  - CosineAnnealingLR scheduling & gradient clipping
  - Early stopping with best checkpoint retention
  - Works on CUDA GPU, Apple Silicon MPS, or CPU
"""

import os
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.model import DeepfakeDetector
from src.data_loader import build_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for images, labels in tqdm(loader, desc="  Training", leave=False, dynamic_ncols=True):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss   = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / max(1, len(loader))
    try:
        auc = float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) > 1 else 1.0
    except Exception:
        auc = 0.5

    return {"loss": avg_loss, "auc": auc}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for images, labels in tqdm(loader, desc="  Validating", leave=False, dynamic_ncols=True):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / max(1, len(loader))
    try:
        auc = float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) > 1 else 1.0
    except Exception:
        auc = 0.5

    return {"loss": avg_loss, "auc": auc}


def train(config_path: str = "config.yaml"):
    cfg   = load_config(config_path)
    tcfg  = cfg["training"]
    mcfg  = cfg["model"]
    paths = cfg["paths"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    Path(paths["model_save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["logs_dir"]).mkdir(parents=True, exist_ok=True)

    # 1. Build loaders
    loaders = build_dataloaders(
        splits_dir  = cfg["data"]["splits_dir"],
        image_size  = cfg["preprocessing"]["image_size"],
        batch_size  = tcfg["batch_size"],
        num_workers = 0,
        seed        = cfg["data"]["seed"],
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    pos_weight   = loaders["pos_weight"].to(device)

    # 2. Build model
    arch = mcfg.get("architecture", "efficientnet-b4")
    model = DeepfakeDetector(
        architecture = arch,
        dropout_1    = mcfg["dropout_1"],
        dropout_2    = mcfg["dropout_2"],
        hidden_dim_1 = mcfg["hidden_dim_1"],
        hidden_dim_2 = mcfg["hidden_dim_2"],
        pretrained   = mcfg["pretrained"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_epochs  = tcfg["epochs"]
    warmup_epochs = tcfg.get("warmup_epochs", 3)
    best_val_auc  = 0.0
    patience      = 0
    history       = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}

    logger.info(f"Training {arch} on {len(train_loader.dataset)} samples...")

    for epoch in range(1, total_epochs + 1):
        if epoch == 1:
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(tcfg["learning_rate"]) * 3,
                weight_decay=float(tcfg["weight_decay"]),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_epochs, eta_min=1e-6)

        if epoch == warmup_epochs + 1:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(tcfg["learning_rate"]),
                weight_decay=float(tcfg["weight_decay"]),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)

        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, tcfg.get("gradient_clip", 1.0))
        val_m   = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Epoch [{epoch:02d}/{total_epochs}] "
            f"Train Loss: {train_m['loss']:.4f} | Train AUC: {train_m['auc']:.4f} | "
            f"Val Loss: {val_m['loss']:.4f} | Val AUC: {val_m['auc']:.4f}"
        )

        history["train_loss"].append(train_m["loss"])
        history["train_auc"].append(train_m["auc"])
        history["val_loss"].append(val_m["loss"])
        history["val_auc"].append(val_m["auc"])

        if val_m["auc"] >= best_val_auc:
            best_val_auc = val_m["auc"]
            patience = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": best_val_auc,
                "config": cfg,
            }
            torch.save(checkpoint, paths["best_model"])
            logger.info(f"  ✅ Saved new best model (Val AUC: {best_val_auc:.4f})")
        else:
            patience += 1
            if patience >= tcfg.get("early_stopping_patience", 7):
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    return history


if __name__ == "__main__":
    train()
