"""Datasets, augmentation and class-imbalance handling."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms as T

from .utils import IMAGENET_MEAN, IMAGENET_STD

CLASSES = ["benign", "malignant"]  # index 1 = malignant (positive class)


def train_transforms(size: int):
    # Histology has no canonical orientation, so flips/rotations are safe;
    # colour jitter simulates stain variation between labs.
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(90)], p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transforms(size: int):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SyntheticCells(Dataset):
    """Fake 'tissue' patches: malignant = larger, darker, irregular nuclei.

    Only for smoke-testing the pipeline offline. NOT clinically meaningful.
    """

    def __init__(self, n=400, size=96, malignant_frac=0.2, transform=None, seed=0):
        rng = np.random.RandomState(seed)
        self.labels = (rng.rand(n) < malignant_frac).astype(int)
        self.seeds = rng.randint(0, 2**31 - 1, n)
        self.size, self.transform = size, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        rng = np.random.RandomState(self.seeds[i])
        y = int(self.labels[i])
        img = Image.new("RGB", (self.size, self.size), (235, 200, 220))
        d = ImageDraw.Draw(img)
        for _ in range(rng.randint(8, 14)):
            x, yy = rng.randint(0, self.size, 2)
            r = rng.randint(7, 12) if y else rng.randint(3, 6)
            col = (70, 30, 120) if y else (140, 100, 170)
            jitter = rng.randint(-3, 4) if y else 0
            d.ellipse([x - r, yy - r + jitter, x + r, yy + r], fill=col)
        img = img.filter(ImageFilter.GaussianBlur(0.8))
        if self.transform:
            img = self.transform(img)
        return img, y


class TransformSubset(Dataset):
    """Subset that applies its own transform (train vs val augmentations)."""

    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.base[self.indices[i]]
        return self.transform(img), y


class _RawFolder(Dataset):
    """ImageFolder-like dataset returning PIL images (benign/malignant subfolders)."""

    def __init__(self, root):
        root = Path(root)
        self.items = []
        for idx, name in enumerate(CLASSES):
            for p in sorted((root / name).rglob("*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    self.items.append((p, idx))
        if not self.items:
            raise FileNotFoundError(
                f"No images found under {root}/{{benign,malignant}}. See README for layout.")
        self.labels = np.array([y for _, y in self.items])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return Image.open(p).convert("RGB"), y


class _RawPCamHF(Dataset):
    """PCam via HuggingFace datasets — avoids Google Drive quota errors."""

    def __init__(self, max_samples=None, seed=42):
        from datasets import load_dataset
        print("Downloading PCam from HuggingFace (no quota issues)...")
        ds = load_dataset("1aurent/PatchCamelyon", split="train", trust_remote_code=True)
        if max_samples and max_samples < len(ds):
            ds = ds.shuffle(seed=seed).select(range(max_samples))
        self._ds = ds
        self.labels = np.array([int(x["label"]) for x in ds], dtype=np.int64)
        print(f"Loaded {len(self._ds)} images  "
              f"({int(self.labels.sum())} malignant / {int((self.labels==0).sum())} benign)")

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        row = self._ds[i]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return img.convert("RGB"), int(row["label"])


def _stratified_split(labels, val_frac, seed):
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        k = max(1, int(len(idx) * val_frac))
        va += list(idx[:k])
        tr += list(idx[k:])
    return tr, va


def build_datasets(name, data_dir, size, val_frac=0.2, seed=42, max_samples=None):
    """Returns (train_ds, val_ds, train_labels)."""
    if name == "synthetic":
        base = SyntheticCells(n=max_samples or 600, size=size)
    elif name == "folder":
        base = _RawFolder(data_dir)
    elif name == "pcam":
        base = _RawPCamHF(max_samples=max_samples, seed=seed)
        labels = base.labels
        tr, va = _stratified_split(labels, val_frac, seed)
        return (TransformSubset(base, tr, train_transforms(size)),
                TransformSubset(base, va, eval_transforms(size)), labels[tr])
    else:
        raise ValueError(f"unknown dataset {name!r}")

    labels = np.asarray(base.labels)
    if max_samples and name == "folder" and len(base) > max_samples:
        keep = np.random.RandomState(seed).permutation(len(base))[:max_samples]
        base = Subset(base, keep)
        labels = labels[keep]
    tr, va = _stratified_split(labels, val_frac, seed)
    return (TransformSubset(base, tr, train_transforms(size)),
            TransformSubset(base, va, eval_transforms(size)), labels[tr])


def make_loaders(train_ds, val_ds, train_labels, batch_size, balanced_sampling=True, workers=0):
    """Class-imbalance handling: oversample the minority (positive) class."""
    sampler = None
    if balanced_sampling:
        counts = np.bincount(train_labels, minlength=2).astype(float)
        w = (1.0 / np.maximum(counts, 1))[train_labels]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(train_labels), replacement=True)
    train = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                       shuffle=sampler is None, num_workers=workers)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train, val


def class_weights(train_labels, device):
    counts = np.bincount(train_labels, minlength=2).astype(float)
    w = counts.sum() / (2.0 * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)
