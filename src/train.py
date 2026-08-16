"""
src/train.py — Full Training Loop
===================================
Features:
  - Warm-up phase: frozen backbone, trains classifier head only
  - Fine-tuning phase: all layers unfrozen
  - AMP (Automatic Mixed Precision) for faster GPU training
  - Early stopping with patience
  - CosineAnnealingLR scheduler
  - Gradient clipping
  - Best model checkpoint saving
  - Weights & Biases logging (optional)
  - Detailed per-epoch console output

Usage:
    python -m src.train
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
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.model import DeepfakeDetector
from src.data_loader import build_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Load Config
# -----------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------
# Training Step
# -----------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one full training epoch."""
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    pbar = tqdm(loader, desc="  Training", leave=False, dynamic_ncols=True)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return {"loss": avg_loss, "auc": auc}


# -----------------------------------------------------------------------
# Validation Step
# -----------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation pass."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for images, labels in tqdm(loader, desc="  Validating", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return {"loss": avg_loss, "auc": auc}


# -----------------------------------------------------------------------
# Main Trainer
# -----------------------------------------------------------------------
def train(config_path: str = "config.yaml"):
    """
    Full training pipeline.

    Phases:
        Phase 1 (warm-up): Backbone frozen, train only classifier.
        Phase 2 (fine-tune): All layers unlocked, full end-to-end training.
    """
    cfg    = load_config(config_path)
    tcfg   = cfg["training"]
    mcfg   = cfg["model"]
    paths  = cfg["paths"]

    # --- Device ---
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Using device: {device}")

    # --- Output directories ---
    Path(paths["model_save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["logs_dir"]).mkdir(parents=True, exist_ok=True)

    # --- Data ---
    logger.info("Loading datasets...")
    loaders = build_dataloaders(
        splits_dir  = cfg["data"]["splits_dir"],
        image_size  = cfg["preprocessing"]["image_size"],
        batch_size  = tcfg["batch_size"],
        num_workers = tcfg["num_workers"],
        seed        = cfg["data"]["seed"],
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    pos_weight   = loaders["pos_weight"].to(device)

    # --- Model ---
    logger.info(f"Building model: {mcfg['architecture']}")
    model = DeepfakeDetector(
        dropout_1   = mcfg["dropout_1"],
        dropout_2   = mcfg["dropout_2"],
        hidden_dim_1= mcfg["hidden_dim_1"],
        hidden_dim_2= mcfg["hidden_dim_2"],
        pretrained  = mcfg["pretrained"],
    ).to(device)

    params = model.get_num_params()
    logger.info(f"Total params: {params['total']:,} | Trainable: {params['trainable']:,}")

    # --- Loss ---
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- AMP Scaler ---
    scaler = GradScaler(enabled=(tcfg["mixed_precision"] and device.type == "cuda"))

    # --- Early stopping state ---
    best_val_auc     = 0.0
    patience_counter = 0
    history          = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}

    total_epochs  = tcfg["epochs"]
    warmup_epochs = tcfg["warmup_epochs"]

    logger.info(f"Starting training for {total_epochs} epochs "
                f"({warmup_epochs} warm-up + {total_epochs - warmup_epochs} fine-tune)")

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        # --- Phase switch ---
        if epoch == 1:
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=tcfg["learning_rate"] * 5,   # higher LR for warm-up head
                weight_decay=tcfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=warmup_epochs,
                eta_min=1e-6,
            )

        if epoch == warmup_epochs + 1:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=tcfg["learning_rate"],
                weight_decay=tcfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=1e-6,
            )

        # --- Train & Validate ---
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, tcfg["gradient_clip"])
        val_metrics   = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch [{epoch:02d}/{total_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} | Train AUC: {train_metrics['auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val AUC: {val_metrics['auc']:.4f} | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        )

        # --- Record history ---
        history["train_loss"].append(train_metrics["loss"])
        history["train_auc"].append(train_metrics["auc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])

        # --- Save best model ---
        if val_metrics["auc"] > best_val_auc:
            best_val_auc     = val_metrics["auc"]
            patience_counter = 0
            checkpoint = {
                "epoch"           : epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state" : optimizer.state_dict(),
                "val_auc"         : best_val_auc,
                "config"          : cfg,
            }
            torch.save(checkpoint, paths["best_model"])
            logger.info(f"  ✅ New best model saved! Val AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{tcfg['early_stopping_patience']}")

        # --- Early stopping ---
        if patience_counter >= tcfg["early_stopping_patience"]:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    logger.info(f"\n🎉 Training complete! Best Val AUC: {best_val_auc:.4f}")
    logger.info(f"   Best model saved to: {paths['best_model']}")
    return history


if __name__ == "__main__":
    train()
