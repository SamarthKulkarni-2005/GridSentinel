"""Data loader — reads CSV or Parquet, performs column mapping and DT derivation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Column name mapping from CSV to internal schema ──────────────────────────
_METER_RENAME = {
    "Meter_ID": "meter_id",
    "Transformer_ID": "dt_id",
    "Timestamp": "timestamp",
    "Active_Power (kWh)": "kwh",
    "Voltage": "voltage_r",
    "Current": "current",
    "Power_Factor": "power_factor",
    "Is_Theft (0/1)": "is_theft",
    "Imputation_Flag": "imputation_flag",
    "Imputation_Confidence": "imputation_confidence",
    # passthrough columns (kept for reference)
    "Peer_Group_ID": "peer_group_id",
    "Theft_Type": "theft_type",
    "GSI_Event_Type": "gsi_event_type",
    "Topology_Confidence": "topology_confidence",
}


def load_from_csv(csv_path: str | Path, rng_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load meter and DT data from the synthetic CSV file.

    Returns (meter_df, dt_df) conforming to the GridSentinel schema.
    Three-phase voltages are synthesised from the single Voltage column.
    DT-level features are derived from meter aggregates.
    """
    csv_path = Path(csv_path)
    logger.info("Loading CSV from %s", csv_path)

    raw = pd.read_csv(csv_path, dtype=str)
    logger.info("Raw CSV shape: %s", raw.shape)

    # ── Rename columns ────────────────────────────────────────────────────────
    raw = raw.rename(columns={k: v for k, v in _METER_RENAME.items() if k in raw.columns})

    # ── Cast types ────────────────────────────────────────────────────────────
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    for col in ["kwh", "voltage_r", "current", "power_factor",
                "imputation_confidence", "topology_confidence"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw["is_theft"] = pd.to_numeric(raw.get("is_theft", 0), errors="coerce").fillna(0).astype(int)
    raw["imputation_flag"] = pd.to_numeric(raw.get("imputation_flag", 0), errors="coerce").fillna(0).astype(int)

    # ── Synthesise three-phase voltages ───────────────────────────────────────
    rng = np.random.default_rng(rng_seed)
    noise_y = rng.uniform(-2.0, 2.0, size=len(raw))
    noise_b = rng.uniform(-2.0, 2.0, size=len(raw))
    raw["voltage_y"] = raw["voltage_r"] + noise_y
    raw["voltage_b"] = raw["voltage_r"] + noise_b

    # ── Derive billed_kwh (monthly total per meter) ───────────────────────────
    raw["year_month"] = raw["timestamp"].dt.to_period("M")
    monthly = (
        raw.groupby(["meter_id", "year_month"])["kwh"]
        .sum()
        .reset_index()
        .rename(columns={"kwh": "billed_kwh"})
    )
    raw = raw.merge(monthly, on=["meter_id", "year_month"], how="left")
    raw = raw.drop(columns=["year_month"])

    # ── Build meter_df ────────────────────────────────────────────────────────
    meter_cols = [
        "meter_id", "dt_id", "timestamp", "kwh",
        "voltage_r", "voltage_y", "voltage_b",
        "current", "power_factor", "billed_kwh", "is_theft",
        "imputation_flag", "imputation_confidence",
    ]
    # Add optional passthrough columns if present
    for opt in ["peer_group_id", "theft_type", "gsi_event_type", "topology_confidence"]:
        if opt in raw.columns:
            meter_cols.append(opt)

    meter_df = raw[[c for c in meter_cols if c in raw.columns]].copy()

    # ── Build dt_df from meter aggregates ─────────────────────────────────────
    dt_df = _build_dt_df(raw)

    logger.info("meter_df shape: %s", meter_df.shape)
    logger.info("dt_df shape: %s", dt_df.shape)
    return meter_df, dt_df


def _build_dt_df(meter_raw: pd.DataFrame) -> pd.DataFrame:
    """Derive transformer-level DataFrame from meter readings."""
    # feeder_kwh = sum of meter kwh per DT per timestamp
    agg = (
        meter_raw.groupby(["dt_id", "timestamp"])["kwh"]
        .sum()
        .reset_index()
        .rename(columns={"kwh": "feeder_kwh"})
    )

    # Derive hour from timestamp for diurnal signals
    agg["hour"] = agg["timestamp"].dt.hour

    # capacity_kva: fixed per DT (500 kVA default)
    dt_ids = agg["dt_id"].unique()
    dt_capacity = {dt_id: 500.0 for dt_id in dt_ids}
    agg["capacity_kva"] = agg["dt_id"].map(dt_capacity)

    # age_years: fixed per DT (5–20 years based on DT number)
    dt_age = {}
    for i, dt_id in enumerate(sorted(dt_ids)):
        dt_age[dt_id] = 5.0 + (i % 4) * 5.0  # cycles through 5, 10, 15, 20
    agg["age_years"] = agg["dt_id"].map(dt_age)

    # temperature_c = 28 + 4*sin(2π*hour/24)
    agg["temperature_c"] = 28.0 + 4.0 * np.sin(2.0 * np.pi * agg["hour"] / 24.0)

    # ev_density = 0.1 (default)
    agg["ev_density"] = 0.1

    # solar_irradiance = max(0, 800*sin(π*(hour-6)/12)) during daylight (6-18h)
    solar = np.zeros(len(agg))
    daylight_mask = (agg["hour"] >= 6) & (agg["hour"] <= 18)
    solar[daylight_mask] = np.maximum(
        0.0,
        800.0 * np.sin(np.pi * (agg.loc[daylight_mask, "hour"] - 6) / 12.0),
    )
    agg["solar_irradiance"] = solar

    # actual_kwh = feeder_kwh (for forecast evaluation)
    agg["actual_kwh"] = agg["feeder_kwh"]

    agg = agg.drop(columns=["hour"])
    return agg


def load_from_parquet(
    meter_path: str | Path,
    dt_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load meter and DT data from Parquet files."""
    meter_path = Path(meter_path)
    dt_path = Path(dt_path)
    logger.info("Loading meter parquet from %s", meter_path)
    meter_df = pd.read_parquet(meter_path)
    logger.info("Loading DT parquet from %s", dt_path)
    dt_df = pd.read_parquet(dt_path)
    return meter_df, dt_df
