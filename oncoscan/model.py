import torch
import torch.nn as nn
from torchvision import models


def build_model(pretrained: bool = True) -> nn.Module:
    """ResNet-18 backbone with a 2-class head (benign / malignant)."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
    return m


def save_checkpoint(path, model, threshold, size, meta=None):
    torch.save({"state_dict": model.state_dict(), "threshold": threshold,
                "size": size, "meta": meta or {}}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["threshold"], ckpt["size"], ckpt.get("meta", {})
