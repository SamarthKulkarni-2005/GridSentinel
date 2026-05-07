"""Deterministic synthetic data generators for GridSentinel tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_meter_df(
    n_meters: int = 100,
    n_days: int = 30,
    seed: int = 42,
    theft_rate: float = 0.05,
) -> pd.DataFrame:
    """Generate synthetic meter_df with is_theft column.

    Theft meters have inflated night load (×1.3 to ×2.5 during 22:00–05:59).
    """
    rng = np.random.default_rng(seed)
    n_per_day = 96  # 15-min intervals
    n_intervals = n_days * n_per_day
    n_dts = max(1, n_meters // 20)  # 20 meters per DT

    # Timestamps (UTC)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = pd.date_range(start, periods=n_intervals, freq="15min")

    records = []
    theft_meters = set(rng.choice(n_meters, size=max(1, int(n_meters * theft_rate)), replace=False).tolist())

    for m_idx in range(n_meters):
        meter_id = f"MTR_{m_idx:04d}"
        dt_id = f"DT_{(m_idx // 20):02d}"
        is_theft = 1 if m_idx in theft_meters else 0

        # Base load profile (diurnal pattern)
        hours = np.array([t.hour for t in timestamps])
        base_load = 0.5 + 0.5 * np.sin(np.pi * (hours - 6) / 12.0)
        base_load = np.clip(base_load, 0.05, 1.0)

        # Add meter-specific variation
        meter_scale = rng.uniform(0.5, 2.0)
        kwh = base_load * meter_scale + rng.normal(0, 0.05, n_intervals)
        kwh = np.clip(kwh, 0.0, None)

        # Simulate theft: inflate night load
        if is_theft:
            night_mask = (hours >= 22) | (hours < 6)
            theft_factor = rng.uniform(1.3, 2.5, n_intervals)
            kwh[night_mask] *= theft_factor[night_mask]

        # Voltage: nominal ±10V variation
        voltage_r = rng.normal(230.0, 5.0, n_intervals)
        voltage_y = voltage_r + rng.uniform(-2.0, 2.0, n_intervals)
        voltage_b = voltage_r + rng.uniform(-2.0, 2.0, n_intervals)
        current = kwh / (voltage_r * 0.9 / 1000.0 + 1e-9)  # approximate
        power_factor = rng.uniform(0.85, 0.99, n_intervals)

        # Billed kwh: monthly total
        monthly_kwh = kwh.sum() / n_days * 30  # approximate monthly

        for i, ts in enumerate(timestamps):
            records.append({
                "meter_id": meter_id,
                "dt_id": dt_id,
                "timestamp": ts,
                "kwh": float(kwh[i]),
                "voltage_r": float(voltage_r[i]),
                "voltage_y": float(voltage_y[i]),
                "voltage_b": float(voltage_b[i]),
                "current": float(np.clip(current[i], 0.0, 1000.0)),
                "power_factor": float(power_factor[i]),
                "billed_kwh": float(monthly_kwh),
                "is_theft": is_theft,
            })

    return pd.DataFrame(records)


def make_dt_df(
    n_dts: int = 5,
    n_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic dt_df with actual_kwh column for forecast evaluation."""
    rng = np.random.default_rng(seed)
    n_per_day = 96
    n_intervals = n_days * n_per_day

    start = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = pd.date_range(start, periods=n_intervals, freq="15min")

    records = []
    for d_idx in range(n_dts):
        dt_id = f"DT_{d_idx:02d}"
        capacity_kva = 500.0
        age_years = 5.0 + d_idx * 3.75  # 5, 8.75, 12.5, 16.25, 20 years

        hours = np.array([t.hour for t in timestamps])
        feeder_base = 150.0 + 100.0 * np.sin(np.pi * (hours - 6) / 12.0)
        feeder_kwh = np.clip(feeder_base + rng.normal(0, 10, n_intervals), 0.0, None)

        temp_c = 28.0 + 4.0 * np.sin(2.0 * np.pi * hours / 24.0)
        ev_density = 0.1
        solar = np.zeros(n_intervals)
        daylight = (hours >= 6) & (hours <= 18)
        solar[daylight] = np.maximum(0.0, 800.0 * np.sin(np.pi * (hours[daylight] - 6) / 12.0))

        for i, ts in enumerate(timestamps):
            records.append({
                "dt_id": dt_id,
                "timestamp": ts,
                "feeder_kwh": float(feeder_kwh[i]),
                "capacity_kva": capacity_kva,
                "age_years": age_years,
                "temperature_c": float(temp_c[i]),
                "ev_density": ev_density,
                "solar_irradiance": float(solar[i]),
                "actual_kwh": float(feeder_kwh[i]),
            })

    return pd.DataFrame(records)


def make_small_meter_df(
    n_meters: int = 10,
    n_days: int = 7,
    seed: int = 0,
    theft_rate: float = 0.2,
) -> pd.DataFrame:
    """Return small meter_df for fast unit tests."""
    return make_meter_df(n_meters=n_meters, n_days=n_days, seed=seed, theft_rate=theft_rate)


def make_small_dt_df(
    n_dts: int = 2,
    n_days: int = 7,
    seed: int = 0,
) -> pd.DataFrame:
    """Return small dt_df for fast unit tests."""
    return make_dt_df(n_dts=n_dts, n_days=n_days, seed=seed)
