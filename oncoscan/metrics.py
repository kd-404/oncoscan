import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def pick_threshold(y_true, probs, target_recall=0.95):
    """Highest threshold whose recall >= target (maximises precision subject to recall).

    Screening context: a missed cancer (false negative) costs far more than a
    false alarm, so we fix recall first and accept lower precision.
    """
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    best = 0.0
    for t in np.unique(probs)[::-1]:
        recall = ((probs >= t) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)
        if recall >= target_recall:
            return float(min(t, 0.5))  # never stricter than 0.5
    return best


def report(y_true, probs, threshold):
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "specificity": tn / max(tn + fp, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
        out["pr_auc"] = float(average_precision_score(y_true, probs))
    return out
