"""Threshold-interactive validation metrics computed from a run's predictions.csv.

Anomalib's own internal Evaluator/F1AdaptiveThreshold picks one fixed threshold at
training time. For a GUI where the user drags a threshold slider and expects the
confusion matrix / precision / recall to update live without retraining, we instead
treat predictions.csv (per-image gt_label + pred_score) as the source of truth and
recompute everything with scikit-learn on demand.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def load_predictions(predictions_csv_path) -> pd.DataFrame:
    return pd.read_csv(predictions_csv_path)


def compute_auroc(df: pd.DataFrame) -> float:
    if df["gt_label"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(df["gt_label"], df["pred_score"]))


def compute_roc_curve(df: pd.DataFrame) -> dict:
    if df["gt_label"].nunique() < 2:
        return {"fpr": [], "tpr": [], "thresholds": []}
    fpr, tpr, thresholds = roc_curve(df["gt_label"], df["pred_score"])
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}


def compute_at_threshold(df: pd.DataFrame, threshold: float) -> dict:
    y_true = df["gt_label"].to_numpy()
    y_pred = (df["pred_score"].to_numpy() >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "confusion_matrix": cm.tolist(),  # rows=true[good,anomaly], cols=pred[good,anomaly]
        "precision_good": float(precision[0]),
        "recall_good": float(recall[0]),
        "f1_good": float(f1[0]),
        "precision_anomaly": float(precision[1]),
        "recall_anomaly": float(recall[1]),
        "f1_anomaly": float(f1[1]),
        "threshold": threshold,
    }


def compute_run_range(run_dir, low_pct: float = 1.0, high_pct: float = 99.0) -> tuple[float, float]:
    """Percentile-based (vmin, vmax) across every heatmap in a run, used to normalize the
    anomaly heatmap overlay consistently across all images in that run (see heatmap.py's
    module docstring for why per-image normalization is wrong). Computed once at training
    time and stored in metrics.json; also usable as a fallback for older runs that predate
    that, by rescanning the run's heatmaps/*.npy files directly.
    """
    heatmaps_dir = Path(run_dir) / "heatmaps"
    all_values = [np.load(p).ravel() for p in heatmaps_dir.glob("*.npy")]
    if not all_values:
        return 0.0, 1.0
    stacked = np.concatenate(all_values)
    vmin = float(np.percentile(stacked, low_pct))
    vmax = float(np.percentile(stacked, high_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def summarize(predictions_csv_path, threshold: float = 0.5) -> dict:
    df = load_predictions(predictions_csv_path)
    return {
        "auroc": compute_auroc(df),
        "roc_curve": compute_roc_curve(df),
        "at_threshold": compute_at_threshold(df, threshold),
        "n_good": int((df["gt_label"] == 0).sum()),
        "n_anomaly": int((df["gt_label"] == 1).sum()),
    }
