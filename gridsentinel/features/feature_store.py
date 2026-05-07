"""Feature store — assembles final feature matrices for ML models."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gridsentinel.features.cass_signals import (
    signal_billing_ratio,
    signal_dt_balance_error,
    signal_dtw_divergence,
    signal_entropy,
    signal_night_load_anomaly,
    signal_repeat_anomaly,
    signal_voltage_stability,
)
from gridsentinel.features.gsi_signals import HOUR_PROFILE
from gridsentinel.features.holidays import is_holiday

logger = logging.getLogger(__name__)

# Features passed to XGBoost
FEATURE_NAMES = [
    "dtw_divergence", "voltage_stability", "billing_ratio",
    "entropy", "night_load_anomaly", "dt_balance_error", "repeat_anomaly",
    "power_factor_mean", "kwh_mean_7d", "kwh_std_7d",
    "kwh_trend_slope",
    "hour_of_day", "day_of_week", "month",
]

# Features used by LSTM forecaster
FORECAST_FEATURES = [
    "kwh", "temperature_c", "power_factor",
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_holiday",
    "ev_density", "solar_irradiance",
]


def _try_dtw_clustering(
    series_matrix: np.ndarray,
    n_clusters: int = 10,
    seed: int = 42,
    timeout_seconds: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit DTW K-Means; fall back to sklearn KMeans on timeout or error."""
    import signal as _signal

    def _handler(signum, frame):
        raise TimeoutError("DTW clustering timed out")

    # Try tslearn DTW clustering
    try:
        from tslearn.clustering import TimeSeriesKMeans

        n_clusters_actual = min(n_clusters, len(series_matrix))

        # Use signal-based timeout on Unix; skip on Windows
        use_timeout = hasattr(_signal, "SIGALRM")
        if use_timeout:
            old_handler = _signal.signal(_signal.SIGALRM, _handler)
            _signal.alarm(int(timeout_seconds))

        try:
            km = TimeSeriesKMeans(
                n_clusters=n_clusters_actual,
                metric="dtw",
                random_state=seed,
                max_iter=10,
                n_init=1,
                verbose=False,
            )
            labels = km.fit_predict(series_matrix.reshape(len(series_matrix), -1, 1))
            centroids = km.cluster_centers_.squeeze(-1)  # (k, T)
        finally:
            if use_timeout:
                _signal.alarm(0)
                _signal.signal(_signal.SIGALRM, old_handler)

        logger.info("DTW clustering succeeded: %d clusters", n_clusters_actual)
        return labels, centroids

    except Exception as exc:
        logger.warning("DTW clustering failed (%s); using KMeans fallback", exc)
        return _kmeans_fallback(series_matrix, n_clusters, seed)


def _kmeans_fallback(
    series_matrix: np.ndarray,
    n_clusters: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """KMeans fallback using mean/std features when DTW is unavailable."""
    from sklearn.cluster import KMeans

    n_clusters_actual = min(n_clusters, len(series_matrix))
    # Feature vector: mean, std per series
    features = np.column_stack([
        series_matrix.mean(axis=1),
        series_matrix.std(axis=1),
    ])
    km = KMeans(n_clusters=n_clusters_actual, random_state=seed, n_init=10)
    labels = km.fit_predict(features)

    # Compute centroids as mean of assigned series
    centroids = np.zeros((n_clusters_actual, series_matrix.shape[1]))
    for k in range(n_clusters_actual):
        mask = labels == k
        if mask.any():
            centroids[k] = series_matrix[mask].mean(axis=0)
        else:
            centroids[k] = series_matrix.mean(axis=0)

    logger.info("KMeans fallback clustering: %d clusters", n_clusters_actual)
    return labels, centroids


def build_meter_features(
    meter_df: pd.DataFrame,
    dt_df: pd.DataFrame,
    lookback_window: int = 96,
    n_clusters: int = 10,
    dtw_norm_denom: float = 3.0,
    voltage_std_norm: float = 0.15,
    repeat_window_count: int = 5,
    cluster_labels: Optional[np.ndarray] = None,
    cluster_centroids: Optional[np.ndarray] = None,
    meter_ids_train: Optional[list] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build per-meter feature matrix for XGBoost theft detection.

    Returns (feature_df, cluster_labels, cluster_centroids).
    If cluster_labels/centroids are None, fits clustering on this data.
    """
    logger.info("Building meter features for %d rows", len(meter_df))
    meter_df = meter_df.sort_values(["meter_id", "timestamp"])
    meter_ids = meter_df["meter_id"].unique()

    # ── Prepare daily time series per meter for DTW clustering ───────────────
    pivot = (
        meter_df.groupby(["meter_id", "timestamp"])["kwh"]
        .mean()
        .unstack("timestamp")
        .fillna(0.0)
    )
    series_matrix = pivot.values  # (n_meters, T)

    # Truncate or pad to lookback_window
    T = min(lookback_window, series_matrix.shape[1])
    series_matrix = series_matrix[:, :T]

    # ── Fit or assign clusters ────────────────────────────────────────────────
    if cluster_labels is None or cluster_centroids is None:
        cluster_labels, cluster_centroids = _try_dtw_clustering(series_matrix, n_clusters)

    meter_to_cluster = {mid: lbl for mid, lbl in zip(pivot.index, cluster_labels)}

    # ── DT aggregates for balance error ──────────────────────────────────────
    dt_feeder = (
        dt_df.groupby(["dt_id", "timestamp"])["feeder_kwh"]
        .mean()
        .reset_index()
    )
    dt_sum_meter = (
        meter_df.groupby(["dt_id", "timestamp"])["kwh"]
        .sum()
        .reset_index()
        .rename(columns={"kwh": "sum_meter_kwh"})
    )
    dt_balance = dt_sum_meter.merge(dt_feeder, on=["dt_id", "timestamp"], how="left")
    dt_balance["balance_error"] = dt_balance.apply(
        lambda r: signal_dt_balance_error(r["sum_meter_kwh"], r.get("feeder_kwh", 0.0)), axis=1
    )
    dt_balance_mean = (
        dt_balance.groupby("dt_id")["balance_error"].mean().to_dict()
    )

    # ── Per-meter feature computation ─────────────────────────────────────────
    records = []
    for meter_id in meter_ids:
        m_data = meter_df[meter_df["meter_id"] == meter_id].sort_values("timestamp")
        if len(m_data) == 0:
            continue

        ts = m_data["timestamp"]
        kwh_arr = m_data["kwh"].values.astype(float)
        voltage_all = np.concatenate([
            m_data["voltage_r"].values,
            m_data["voltage_y"].values,
            m_data["voltage_b"].values,
        ]).astype(float)

        # DTW divergence
        cluster_idx = meter_to_cluster.get(meter_id, 0)
        centroid = cluster_centroids[cluster_idx]
        meter_series = series_matrix[list(pivot.index).index(meter_id)] if meter_id in pivot.index else kwh_arr[:T]
        dtw_div = signal_dtw_divergence(meter_series, centroid, norm_denom=dtw_norm_denom)

        # Voltage stability
        v_stab = signal_voltage_stability(voltage_all, norm_std=voltage_std_norm)

        # Billing ratio
        billed = float(m_data["billed_kwh"].iloc[-1]) if "billed_kwh" in m_data.columns else 0.0
        total_kwh = float(kwh_arr.sum())
        bill_ratio = signal_billing_ratio(billed, total_kwh)

        # Entropy
        ent = signal_entropy(kwh_arr)

        # Night load anomaly
        if hasattr(ts, "dt"):
            dt_idx = pd.DatetimeIndex(ts.values)
        else:
            dt_idx = ts
        night_anom = signal_night_load_anomaly(kwh_arr, dt_idx)

        # DT balance error
        dt_id = m_data["dt_id"].iloc[0]
        dt_bal_err = dt_balance_mean.get(dt_id, 0.0)

        # Repeat anomaly
        rep_anom = signal_repeat_anomaly(kwh_arr, window_count=repeat_window_count)

        # Statistical features
        pf_mean = float(m_data["power_factor"].mean())

        # 7-day rolling stats (use last 7d * 96 intervals)
        last_7d = kwh_arr[-672:] if len(kwh_arr) >= 672 else kwh_arr
        kwh_mean_7d = float(np.mean(last_7d))
        kwh_std_7d = float(np.std(last_7d))

        # Linear trend slope
        if len(kwh_arr) >= 2:
            x = np.arange(len(kwh_arr), dtype=float)
            slope = float(np.polyfit(x, kwh_arr, 1)[0])
        else:
            slope = 0.0

        # Temporal features from last timestamp
        last_ts = pd.Timestamp(ts.iloc[-1])
        hour = last_ts.hour
        dow = last_ts.dayofweek
        month = last_ts.month

        # is_theft label
        is_theft = int(m_data["is_theft"].iloc[-1]) if "is_theft" in m_data.columns else 0

        records.append({
            "meter_id": meter_id,
            "dt_id": dt_id,
            "dtw_divergence": dtw_div,
            "voltage_stability": v_stab,
            "billing_ratio": bill_ratio,
            "entropy": ent,
            "night_load_anomaly": night_anom,
            "dt_balance_error": dt_bal_err,
            "repeat_anomaly": rep_anom,
            "power_factor_mean": pf_mean,
            "kwh_mean_7d": kwh_mean_7d,
            "kwh_std_7d": kwh_std_7d,
            "kwh_trend_slope": slope,
            "hour_of_day": hour,
            "day_of_week": dow,
            "month": month,
            "is_theft": is_theft,
        })

    feature_df = pd.DataFrame(records)
    logger.info("Built meter feature matrix: %s", feature_df.shape)
    return feature_df, cluster_labels, cluster_centroids


def build_forecast_features(
    meter_df: pd.DataFrame,
    dt_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-DT feature matrix for LSTM demand forecasting."""
    logger.info("Building forecast features")

    dt_agg = (
        meter_df.groupby(["dt_id", "timestamp"])
        .agg(kwh=("kwh", "sum"), power_factor=("power_factor", "mean"))
        .reset_index()
    )

    dt_merged = dt_agg.merge(
        dt_df[["dt_id", "timestamp", "temperature_c", "ev_density", "solar_irradiance"]],
        on=["dt_id", "timestamp"],
        how="left",
    )
    dt_merged["temperature_c"] = dt_merged["temperature_c"].fillna(28.0)
    dt_merged["ev_density"] = dt_merged["ev_density"].fillna(0.1)
    dt_merged["solar_irradiance"] = dt_merged["solar_irradiance"].fillna(0.0)

    ts = pd.DatetimeIndex(dt_merged["timestamp"])
    hour = ts.hour.astype(float)
    day = ts.dayofweek.astype(float)
    month = ts.month.astype(float)

    dt_merged["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    dt_merged["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dt_merged["day_sin"] = np.sin(2 * np.pi * day / 7.0)
    dt_merged["day_cos"] = np.cos(2 * np.pi * day / 7.0)
    dt_merged["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    dt_merged["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    dt_merged["is_weekend"] = (ts.dayofweek >= 5).astype(float)
    dt_merged["is_holiday"] = [
        float(is_holiday(pd.Timestamp(t).date())) for t in dt_merged["timestamp"]
    ]

    return dt_merged.sort_values(["dt_id", "timestamp"]).reset_index(drop=True)


def compute_thermal_hours(
    dt_df: pd.DataFrame,
    window_hours: int = 6,
    threshold_pct: float = 0.80,
    power_factor: float = 0.9,
) -> pd.DataFrame:
    """Compute hours_above_80pct per DT per timestamp using rolling window.

    Returns dt_df with added column 'hours_above_80pct'.
    """
    dt_df = dt_df.copy().sort_values(["dt_id", "timestamp"])
    results = []

    for dt_id, group in dt_df.groupby("dt_id"):
        group = group.copy()
        capacity_kw = group["capacity_kva"].iloc[0] * power_factor
        load_pct = group["feeder_kwh"] / max(capacity_kw, 1e-9)
        above = (load_pct > threshold_pct).astype(float)

        # Each row represents 15 min = 0.25 hours; window = 6h = 24 intervals
        intervals_per_hour = 4
        window_intervals = window_hours * intervals_per_hour

        rolling_hours = above.rolling(window=window_intervals, min_periods=1).sum() * 0.25
        group["hours_above_80pct"] = rolling_hours.values
        results.append(group)

    return pd.concat(results).sort_values(["dt_id", "timestamp"]).reset_index(drop=True)
