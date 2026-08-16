"""
src/data_loader.py — PyTorch Dataset & DataLoader
===================================================
Handles directory dataset loading:
    data/splits/
        train/ (real/, fake/)
        val/   (real/, fake/)
        test/  (real/, fake/)

Labels: 0 = Real, 1 = Fake
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocess import get_train_transforms, get_val_transforms, IMAGENET_MEAN, IMAGENET_STD


class DeepfakeDataset(Dataset):
    """
    Binary deepfake detection dataset for image files.
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
        self.samples: List[Tuple[str, int]] = []

        self._load_samples(max_samples, seed)

    def _load_samples(self, max_samples: Optional[int], seed: int):
        for class_name, label in self.LABEL_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTS:
                    self.samples.append((str(img_path), label))

        if not self.samples:
            # Fallback: scan any subdirectories or root
            for p in self.root_dir.glob("**/*"):
                if p.suffix.lower() in self.SUPPORTED_EXTS:
                    lbl = 1 if "fake" in str(p).lower() else 0
                    self.samples.append((str(p), lbl))

        if max_samples is not None and max_samples < len(self.samples):
            random.seed(seed)
            self.samples = random.sample(self.samples, max_samples)

    def get_class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples]
        n_real = labels.count(0)
        n_fake = labels.count(1)
        if n_fake == 0 or n_real == 0:
            return torch.tensor([1.0])
        return torch.tensor([n_real / n_fake])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image)

        if hasattr(self.transform, "__call__"):
            try:
                # Albumentations dict format
                augmented = self.transform(image=img_np)
                tensor = augmented["image"]
            except TypeError:
                # Torchvision format
                tensor = self.transform(img_np)
        else:
            # Manual fallback
            resized = Image.fromarray(img_np).resize((self.image_size, self.image_size))
            arr = np.array(resized, dtype=np.float32) / 255.0
            arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()

        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).float()

        return tensor, label


def build_dataloaders(
    splits_dir: str = "data/splits",
    image_size: int = 380,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders.
    Default num_workers=0 for seamless Windows compatibility.
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

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=(len(train_dataset) > batch_size),
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return {
        "train"      : train_loader,
        "val"        : val_loader,
        "test"       : test_loader,
        "pos_weight" : train_dataset.get_class_weights(),
    }
