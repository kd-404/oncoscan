"""Train OncoScan.  python -m oncoscan.train --dataset synthetic --epochs 3"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import build_datasets, class_weights, make_loaders
from .metrics import pick_threshold, report
from .model import build_model, save_checkpoint
from .utils import get_device, seed_everything


def predict_probs(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pcam", "folder", "synthetic"], default="synthetic")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--size", type=int, default=96, help="input resolution (small = fast on a Mac)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default="runs/oncoscan.pt")
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    print(f"device: {device}")

    tr_ds, va_ds, tr_labels = build_datasets(args.dataset, args.data_dir, args.size,
                                             max_samples=args.max_samples)
    train_dl, val_dl = make_loaders(tr_ds, va_ds, tr_labels, args.batch_size)
    print(f"train={len(tr_ds)} val={len(va_ds)} positives(train)={int(tr_labels.sum())}")

    model = build_model(pretrained=not args.no_pretrained).to(device)
    # Weighted loss on top of oversampling; positive-class errors cost more.
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(tr_labels, device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score, best_state = -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        y_true, probs = predict_probs(model, val_dl, device)
        thr = pick_threshold(y_true, probs, args.target_recall)
        m = report(y_true, probs, thr)
        print(f"epoch {epoch} loss={total/len(tr_ds):.4f} val recall={m['recall']:.3f} "
              f"precision={m['precision']:.3f} auc={m.get('roc_auc', float('nan')):.3f} thr={thr:.3f}")
        score = m.get("pr_auc", m["recall"])
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y_true, probs = predict_probs(model, val_dl, device)
    thr = pick_threshold(y_true, probs, args.target_recall)
    final = report(y_true, probs, thr)
    print("final:", json.dumps(final, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.out, model.cpu(), thr, args.size, meta={"dataset": args.dataset, **final})
    Path(args.out).with_suffix(".json").write_text(json.dumps(final, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
