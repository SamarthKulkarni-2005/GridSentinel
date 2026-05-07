"""Schema validation for meter_df and dt_df DataFrames."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Required columns and expected dtypes ─────────────────────────────────────
METER_REQUIRED = {
    "meter_id": "object",
    "dt_id": "object",
    "timestamp": "datetime64[ns, UTC]",
    "kwh": "float64",
    "voltage_r": "float64",
    "voltage_y": "float64",
    "voltage_b": "float64",
    "current": "float64",
    "power_factor": "float64",
    "billed_kwh": "float64",
}

METER_OPTIONAL = {
    "is_theft": "int64",
    "imputation_flag": "int64",
    "imputation_confidence": "float64",
}

DT_REQUIRED = {
    "dt_id": "object",
    "timestamp": "datetime64[ns, UTC]",
    "feeder_kwh": "float64",
    "capacity_kva": "float64",
    "age_years": "float64",
    "temperature_c": "float64",
    "actual_kwh": "float64",
}

DT_OPTIONAL = {
    "last_maintained": "datetime64[ns, UTC]",
    "ev_density": "float64",
    "solar_irradiance": "float64",
}


def _coerce_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp column is timezone-aware UTC datetime."""
    if "timestamp" not in df.columns:
        return df
    ts = df["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        df["timestamp"] = pd.to_datetime(ts, utc=True)
    elif ts.dt.tz is None:
        df["timestamp"] = ts.dt.tz_localize("UTC")
    else:
        df["timestamp"] = ts.dt.tz_convert("UTC")
    return df


def validate_meter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce meter_df schema; raise on critical failures."""
    logger.info("Validating meter_df (%d rows)", len(df))
    df = df.copy()
    df = _coerce_timestamps(df)

    # Check required columns
    missing = [c for c in METER_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"meter_df missing required columns: {missing}")

    # Coerce numeric dtypes
    for col, dtype in METER_REQUIRED.items():
        if col in ("meter_id", "dt_id"):
            df[col] = df[col].astype(str)
        elif col == "timestamp":
            pass  # already handled
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Check nulls in required columns
    for col in METER_REQUIRED:
        null_count = df[col].isna().sum()
        if null_count > 0:
            logger.error("meter_df: %d nulls in required column '%s'", null_count, col)
            raise ValueError(f"meter_df: {null_count} nulls in required column '{col}'")

    # Handle optional columns
    for col, dtype in METER_OPTIONAL.items():
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.warning("meter_df: %d nulls in optional column '%s'", null_count, col)
            if "int" in dtype:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip power_factor to [0, 1]
    df["power_factor"] = df["power_factor"].clip(0.0, 1.0)

    logger.info("meter_df validation passed")
    return df


def validate_dt_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce dt_df schema; raise on critical failures."""
    logger.info("Validating dt_df (%d rows)", len(df))
    df = df.copy()
    df = _coerce_timestamps(df)

    missing = [c for c in DT_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"dt_df missing required columns: {missing}")

    for col, dtype in DT_REQUIRED.items():
        if col == "dt_id":
            df[col] = df[col].astype(str)
        elif col == "timestamp":
            pass
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    for col in DT_REQUIRED:
        null_count = df[col].isna().sum()
        if null_count > 0:
            logger.error("dt_df: %d nulls in required column '%s'", null_count, col)
            raise ValueError(f"dt_df: {null_count} nulls in required column '{col}'")

    for col, dtype in DT_OPTIONAL.items():
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.warning("dt_df: %d nulls in optional column '%s'", null_count, col)
            if col == "last_maintained":
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            # Fill missing optional columns with defaults
            if col == "ev_density":
                df[col] = 0.0
            elif col == "solar_irradiance":
                df[col] = 0.0

    logger.info("dt_df validation passed")
    return df
