import torch
from PIL import Image

from .data import eval_transforms
from .gradcam import grad_cam, overlay
from .model import load_checkpoint
from .utils import get_device


class Predictor:
    def __init__(self, ckpt_path, device=None):
        self.device = device or get_device()
        self.model, self.threshold, self.size, self.meta = load_checkpoint(ckpt_path, self.device)
        self.tf = eval_transforms(self.size)

    def predict(self, img: Image.Image, with_cam=True):
        img = img.convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = torch.softmax(self.model(x), 1)[0, 1].item()
        result = {"p_malignant": p, "malignant": p >= self.threshold, "cam": None}
        if with_cam:
            result["cam"] = overlay(img.resize((self.size, self.size)), grad_cam(self.model, x))
        return result
