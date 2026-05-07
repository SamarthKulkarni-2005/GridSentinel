"""Train the Bi-LSTM demand forecaster on synthetic_grid_data.csv and save to models/."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make project root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from gridsentinel.features.feature_store import FORECAST_FEATURES, build_forecast_features
from gridsentinel.ingestion.loader import load_from_csv
from gridsentinel.models.demand_forecaster import DemandForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_forecaster")

CSV_CANDIDATES = [
    ROOT.parent / "synthetic_grid_data.csv",
    Path("D:/BESCOM/synthetic_grid_data.csv"),
    ROOT / "synthetic_grid_data.csv",
]
MODEL_PATH  = ROOT / "models" / "demand_forecaster.pt"
SCALER_PATH = ROOT / "models" / "demand_scaler.pkl"


def main() -> None:
    # ── Find CSV ──────────────────────────────────────────────────────────────
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        logger.error("synthetic_grid_data.csv not found. Tried: %s", CSV_CANDIDATES)
        sys.exit(1)
    logger.info("Loading CSV from %s", csv_path)

    meter_df, dt_df = load_from_csv(csv_path)
    logger.info("meter_df: %s  dt_df: %s", meter_df.shape, dt_df.shape)

    # ── Build forecast feature matrix (per DT per timestamp) ─────────────────
    logger.info("Building forecast features …")
    features_df = build_forecast_features(meter_df, dt_df)
    logger.info("Forecast feature matrix: %s", features_df.shape)

    feature_cols = [c for c in FORECAST_FEATURES if c in features_df.columns]
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    # ── Time-based 80/20 split (respect temporal order) ──────────────────────
    all_ts   = sorted(features_df["timestamp"].unique())
    cutoff   = all_ts[int(len(all_ts) * 0.8)]
    train_df = features_df[features_df["timestamp"] < cutoff].copy()
    val_df   = features_df[features_df["timestamp"] >= cutoff].copy()
    logger.info("Train rows: %d  Val rows: %d  (cutoff: %s)", len(train_df), len(val_df), cutoff)

    # ── Train ─────────────────────────────────────────────────────────────────
    forecaster = DemandForecaster(
        hidden_size=64,        # lighter model — faster on CPU
        num_layers=2,
        dropout=0.2,
        quantiles=[0.05, 0.5, 0.95],
        epochs=40,
        batch_size=128,
        lr=0.001,
        patience=8,
        lookback_window=24,    # 24-hour lookback (hourly data)
        random_state=42,
    )

    logger.info("Training Bi-LSTM (this takes ~2 min on CPU) …")
    forecaster.fit(
        train_df, val_df,
        feature_cols=feature_cols,
        target_col="kwh",
        group_col="dt_id",     # build sequences per DT to avoid boundary mixing
    )

    # ── Quick validation MAPE ─────────────────────────────────────────────────
    preds = forecaster.predict(val_df, feature_cols=feature_cols)  # (N, 3)
    q50   = preds[:, 1]  # median = point forecast
    y_val = val_df["kwh"].values
    mask  = y_val > 0.1
    mape  = float(np.mean(np.abs((y_val[mask] - q50[mask]) / y_val[mask]))) * 100
    logger.info("Validation MAPE (q50): %.2f%%", mape)

    # ── Save ──────────────────────────────────────────────────────────────────
    forecaster.save(MODEL_PATH, SCALER_PATH)
    logger.info("Model saved → %s", MODEL_PATH)
    logger.info("Scaler saved → %s", SCALER_PATH)
    logger.info("Done. Restart the API server to load the new model.")


if __name__ == "__main__":
    main()
