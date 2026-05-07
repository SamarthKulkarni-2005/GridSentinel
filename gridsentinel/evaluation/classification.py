"""Classification evaluation metrics for theft detection."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        upper = (y_proba <= bins[i + 1]) if i == n_bins - 1 else (y_proba < bins[i + 1])
        mask = (y_proba >= bins[i]) & upper
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_proba[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Returns dict with: precision, recall, f1, fpr, mcc, roc_auc, pr_auc, ece.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    mcc = float(matthews_corrcoef(y_true, y_pred))

    # False positive rate from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / max(fp + tn, 1))

    # ROC-AUC and PR-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except ValueError:
        pr_auc = 0.0

    ece = expected_calibration_error(y_true, y_proba)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": fpr,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ece": ece,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    logger.info(
        "Classification metrics: precision=%.3f recall=%.3f f1=%.3f fpr=%.4f mcc=%.3f",
        metrics["precision"], metrics["recall"], metrics["f1"],
        metrics["fpr"], metrics["mcc"],
    )
    return metrics
