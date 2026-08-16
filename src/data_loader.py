"""
src/data_loader.py — PyTorch Dataset & DataLoader
===================================================
Expects processed dataset split directories:
    data/splits/
        train/
            real/   *.jpg, *.png
            fake/   *.jpg, *.png
        val/
            real/
            fake/
        test/
            real/
            fake/

Labels: 0 = Real, 1 = Fake
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

from src.preprocess import get_train_transforms, get_val_transforms


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
class DeepfakeDataset(Dataset):
    """
    Binary deepfake detection dataset.

    Reads images from:
        root/real/*.{jpg,png,jpeg,webp}
        root/fake/*.{jpg,png,jpeg,webp}

    Args:
        root_dir   : Root directory containing 'real' and 'fake' subfolders.
        transform  : Albumentations transform pipeline.
        image_size : Target image size (used if transform is None).
        max_samples: Cap the number of samples (useful for quick experiments).
        seed       : Random seed for reproducible sampling.
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    LABEL_MAP      = {"real": 0, "fake": 1}

    def __init__(
        self,
        root_dir: str,
        transform=None,
        image_size: int = 380,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.root_dir   = Path(root_dir)
        self.transform  = transform if transform is not None else get_val_transforms(image_size)
        self.image_size = image_size
        self.samples: list = []  # List of (image_path, label) tuples

        self._load_samples(max_samples, seed)

    def _load_samples(self, max_samples: Optional[int], seed: int):
        """Scan directory and build sample list."""
        for class_name, label in self.LABEL_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"[Dataset] Warning: directory not found: {class_dir}")
                continue

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTS:
                    self.samples.append((str(img_path), label))

        if not self.samples:
            raise RuntimeError(f"No samples found in {self.root_dir}. "
                               "Ensure 'real/' and 'fake/' subdirectories exist.")

        # Optional: cap samples
        if max_samples is not None and max_samples < len(self.samples):
            random.seed(seed)
            self.samples = random.sample(self.samples, max_samples)

        # Compute class distribution
        labels    = [s[1] for s in self.samples]
        n_real    = labels.count(0)
        n_fake    = labels.count(1)
        print(f"[Dataset] Loaded {len(self.samples)} samples from {self.root_dir.name}: "
              f"{n_real} real, {n_fake} fake")

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for balanced loss (handles imbalanced datasets).

        Returns:
            pos_weight tensor for BCEWithLogitsLoss.
        """
        labels = [s[1] for s in self.samples]
        n_real = labels.count(0)
        n_fake = labels.count(1)
        if n_fake == 0 or n_real == 0:
            return torch.tensor([1.0])
        weight = torch.tensor([n_real / n_fake])
        print(f"[Dataset] Class weight (pos_weight): {weight.item():.3f}")
        return weight

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image)

        # Apply transforms
        augmented = self.transform(image=img_np)
        tensor    = augmented["image"]

        return tensor, label

    def get_sample_path(self, idx: int) -> str:
        """Return file path for a given index."""
        return self.samples[idx][0]


# -----------------------------------------------------------------------
# DataLoader Factory
# -----------------------------------------------------------------------
def build_dataloaders(
    splits_dir: str = "data/splits",
    image_size: int = 380,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Args:
        splits_dir  : Root directory containing 'train', 'val', 'test' folders.
        image_size  : Input image size.
        batch_size  : Batch size.
        num_workers : Number of DataLoader workers.
        seed        : Random seed.

    Returns:
        Dictionary: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    splits_path = Path(splits_dir)

    train_dataset = DeepfakeDataset(
        root_dir=str(splits_path / "train"),
        transform=get_train_transforms(image_size),
        image_size=image_size,
        seed=seed,
    )
    val_dataset = DeepfakeDataset(
        root_dir=str(splits_path / "val"),
        transform=get_val_transforms(image_size),
        image_size=image_size,
        seed=seed,
    )
    test_dataset = DeepfakeDataset(
        root_dir=str(splits_path / "test"),
        transform=get_val_transforms(image_size),
        image_size=image_size,
        seed=seed,
    )

    # Fix seed for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train"       : train_loader,
        "val"         : val_loader,
        "test"        : test_loader,
        "pos_weight"  : train_dataset.get_class_weights(),
    }
