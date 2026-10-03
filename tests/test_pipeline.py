import numpy as np
import torch

from oncoscan.data import SyntheticCells, build_datasets, class_weights, make_loaders
from oncoscan.gradcam import grad_cam
from oncoscan.metrics import pick_threshold, report
from oncoscan.model import build_model


def test_pick_threshold_hits_target_recall():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([.9, .8, .4, .3, .35, .2, .1, .05])
    t = pick_threshold(y, p, 1.0)
    assert report(y, p, t)["recall"] == 1.0


def test_balanced_sampler_oversamples_minority():
    tr, va, labels = build_datasets("synthetic", "", 64, max_samples=300)
    dl, _ = make_loaders(tr, va, labels, 300)
    _, y = next(iter(dl))
    assert 0.35 < y.float().mean() < 0.65 or labels.mean() > 0.35
    assert class_weights(labels, "cpu")[1] > class_weights(labels, "cpu")[0]


def test_model_and_gradcam_shapes():
    m = build_model(pretrained=False).eval()
    x = torch.randn(1, 3, 64, 64)
    assert m(x).shape == (1, 2)
    cam = grad_cam(m, x)
    assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1
