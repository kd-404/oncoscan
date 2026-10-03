"""Minimal Grad-CAM for the ResNet backbone (last conv block)."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def grad_cam(model, x, target_class=1):
    """x: (1,3,H,W) normalised tensor. Returns heatmap in [0,1], shape (H,W)."""
    feats, grads = [], []
    layer = model.layer4
    h1 = layer.register_forward_hook(lambda m, i, o: feats.append(o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        model.zero_grad()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = model(x)
            logits[0, target_class].backward()
    finally:
        h1.remove()
        h2.remove()
    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats[0]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp(min=1e-8)
    return cam.detach().cpu().numpy()


def overlay(pil_img: Image.Image, cam: np.ndarray, alpha=0.45) -> Image.Image:
    img = np.asarray(pil_img.convert("RGB").resize(cam.shape[::-1]), dtype=np.float32) / 255
    heat = np.stack([cam, np.clip(1 - np.abs(cam - 0.5) * 2, 0, 1), 1 - cam], -1)  # blue->red
    out = (1 - alpha) * img + alpha * heat
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
