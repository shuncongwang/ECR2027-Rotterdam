from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, confusion_matrix, roc_curve
)


def choose_threshold_youden(y_true, y_score) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    return float(thresholds[idx])


def calculate_binary_metrics(y_true, y_score, threshold=0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "AUC": roc_auc_score(y_true, y_score),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "NPV": npv,
        "Threshold": float(threshold),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def bootstrap_metrics(y_true, y_score, threshold, n_bootstrap=2000, ci=0.95, seed=42):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    names = ["AUC", "F1", "Sensitivity", "Specificity", "Accuracy", "Balanced_Accuracy", "Precision", "NPV"]
    samples = {k: [] for k in names}

    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if np.unique(yt).size < 2:
            continue
        m = calculate_binary_metrics(yt, ys, threshold)
        for k in names:
            if np.isfinite(m[k]):
                samples[k].append(m[k])

    alpha = 1.0 - ci
    out = {}
    point = calculate_binary_metrics(y_true, y_score, threshold)
    for k in names:
        vals = np.asarray(samples[k], dtype=float)
        if vals.size:
            lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        else:
            lo, hi = np.nan, np.nan
        out[k] = point[k]
        out[f"{k}_95CI_Lower"] = lo
        out[f"{k}_95CI_Upper"] = hi
        out[f"{k}_95CI"] = f"{point[k]:.3f} ({lo:.3f}-{hi:.3f})" if np.isfinite(lo) else "NA"
    out.update({k: point[k] for k in ["Threshold", "TP", "TN", "FP", "FN"]})
    return out
