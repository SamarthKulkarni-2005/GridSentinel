"""CASS signal functions — 7 signals for theft suspicion scoring."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats

logger = logging.getLogger(__name__)


def signal_dtw_divergence(
    meter_series: np.ndarray,
    cluster_centroid: np.ndarray,
    norm_denom: float = 3.0,
) -> float:
    """Compute normalised DTW distance between meter series and cluster centroid."""
    try:
        from tslearn.metrics import dtw as tslearn_dtw
        dist = float(tslearn_dtw(meter_series.reshape(-1, 1), cluster_centroid.reshape(-1, 1)))
    except Exception as exc:
        logger.warning("tslearn DTW failed (%s); falling back to Euclidean", exc)
        min_len = min(len(meter_series), len(cluster_centroid))
        dist = float(np.linalg.norm(meter_series[:min_len] - cluster_centroid[:min_len]))

    return float(np.clip(dist / max(norm_denom, 1e-9), 0.0, 1.0))


def signal_voltage_stability(
    voltage_array: np.ndarray,
    norm_std: float = 0.15,
) -> float:
    """Compute voltage instability signal from all three phases concatenated."""
    sigma = float(np.std(voltage_array))
    # High sigma → high instability → high signal value
    return float(np.clip(sigma / max(norm_std, 1e-9), 0.0, 1.0))


def signal_billing_ratio(billed_kwh: float, meter_kwh: float) -> float:
    """Compute billing ratio anomaly signal."""
    if meter_kwh == 0:
        return 1.0
    ratio = billed_kwh / meter_kwh
    return float(np.clip(abs(1.0 - ratio), 0.0, 1.0))


def signal_entropy(consumption_series: np.ndarray) -> float:
    """Compute consumption entropy signal — low entropy means suspicious pattern."""
    if len(consumption_series) == 0:
        return 0.0

    # Discretise into 20 equal-width bins
    counts, _ = np.histogram(consumption_series, bins=20)
    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    h = float(scipy.stats.entropy(probs))
    h_norm = h / np.log(20)  # normalise to [0, 1]
    # Low entropy (predictable) = 0; high entropy (erratic) = 1
    return float(np.clip(1.0 - h_norm, 0.0, 1.0))


def signal_night_load_anomaly(
    consumption_series: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> float:
    """Compute night-time load anomaly signal.

    Night = 22:00–05:59; Day = 06:00–21:59.
    Returns high score when night consumption is disproportionately high.
    """
    if len(consumption_series) == 0 or len(timestamps) == 0:
        return 0.0

    hours = timestamps.hour
    night_mask = (hours >= 22) | (hours < 6)
    day_mask = ~night_mask

    night_vals = consumption_series[night_mask]
    day_vals = consumption_series[day_mask]

    day_avg = float(np.mean(day_vals)) if len(day_vals) > 0 else 0.0
    night_avg = float(np.mean(night_vals)) if len(night_vals) > 0 else 0.0

    if day_avg == 0:
        return 0.0

    ratio = night_avg / day_avg
    return float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))


def signal_dt_balance_error(sum_meter_kwh: float, dt_feeder_kwh: float) -> float:
    """Compute DT-level energy balance error signal."""
    if dt_feeder_kwh == 0:
        return 0.0
    error = abs(sum_meter_kwh - dt_feeder_kwh) / dt_feeder_kwh
    return float(np.clip(error / 0.5, 0.0, 1.0))


def signal_repeat_anomaly(
    consumption_series: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    window_count: int = 5,
) -> float:
    """Compute repeat anomaly signal using rolling Z-score detection.

    Uses a 96-interval rolling window; flags windows where Z-score > 3.0.
    Returns the fraction of anomalous windows, clipped to [0, 1].
    """
    if len(consumption_series) < 4:
        return 0.0

    window_size = min(96, len(consumption_series))
    anomaly_flags = []

    for start in range(0, len(consumption_series) - window_size + 1, window_size):
        window = consumption_series[start : start + window_size]
        if len(window) < 2:
            continue
        mean_w = np.mean(window)
        std_w = np.std(window)
        if std_w < 1e-9:
            anomaly_flags.append(False)
            continue
        z_scores = np.abs((window - mean_w) / std_w)
        anomaly_flags.append(bool(np.any(z_scores > 3.0)))

    if not anomaly_flags:
        return 0.0

    anomaly_array = np.array(anomaly_flags, dtype=bool)
    count = int(anomaly_array.sum())
    return float(np.clip(count / window_count, 0.0, 1.0))
