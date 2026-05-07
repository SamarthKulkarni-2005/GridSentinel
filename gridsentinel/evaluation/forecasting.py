"""Forecasting evaluation metrics — MAPE, RMSE, PICP, PINAW."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error with zero-division guard."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def picp(y_true: np.ndarray, q5: np.ndarray, q95: np.ndarray) -> float:
    """Prediction Interval Coverage Probability — fraction inside [Q5, Q95]."""
    y_true = np.asarray(y_true, dtype=float)
    q5 = np.asarray(q5, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    return float(np.mean((y_true >= q5) & (y_true <= q95)))


def pinaw(q5: np.ndarray, q95: np.ndarray, y_true: np.ndarray) -> float:
    """Prediction Interval Normalised Average Width."""
    q5 = np.asarray(q5, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    width = float(np.mean(q95 - q5))
    y_range = float(np.max(y_true) - np.min(y_true))
    return float(width / (y_range + 1e-9))


def evaluate_forecast(
    y_true: np.ndarray,
    q5: np.ndarray,
    q95: np.ndarray,
) -> Dict[str, float]:
    """Compute all forecasting metrics.

    y_pred = (q5 + q95) / 2 used for MAPE/RMSE.
    Returns dict with: mape, rmse, picp, pinaw.
    """
    y_true = np.asarray(y_true, dtype=float)
    q5 = np.asarray(q5, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    y_pred = (q5 + q95) / 2.0

    metrics = {
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "picp": picp(y_true, q5, q95),
        "pinaw": pinaw(q5, q95, y_true),
    }

    logger.info(
        "Forecast metrics: MAPE=%.4f RMSE=%.4f PICP=%.4f PINAW=%.4f",
        metrics["mape"], metrics["rmse"], metrics["picp"], metrics["pinaw"],
    )
    return metrics
