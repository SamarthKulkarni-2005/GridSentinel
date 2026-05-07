"""Bi-LSTM demand forecaster with quantile outputs and StandardScaler normalisation."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

FORECAST_FEATURES = [
    "kwh", "temperature_c", "power_factor",
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_holiday",
    "ev_density", "solar_irradiance",
]


# ── PyTorch model definition ─────────────────────────────────────────────────

class _BiLSTMModel(nn.Module):
    """Bi-directional LSTM for quantile demand forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_outputs: int = 3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, seq, features) -> (batch, n_outputs)."""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    err = target - pred
    return torch.mean(torch.max(quantile * err, (quantile - 1.0) * err))


# ── Forecaster class ─────────────────────────────────────────────────────────

class DemandForecaster:
    """Bi-LSTM quantile demand forecaster (default: Q5, Q50, Q95)."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        quantiles: List[float] = None,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 10,
        random_state: int = 42,
        lookback_window: int = 24,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.quantiles = quantiles if quantiles is not None else [0.05, 0.5, 0.95]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.lookback_window = lookback_window

        self.model: Optional[_BiLSTMModel] = None
        self.scaler: StandardScaler = StandardScaler()
        self._input_size: int = len(FORECAST_FEATURES)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        lb = min(self.lookback_window, len(X) - 1)
        if lb < 1:
            return np.empty((0, self.lookback_window, X.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float32)
        Xs, ys = [], []
        for i in range(lb, len(X)):
            Xs.append(X[i - lb: i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    def _build_grouped_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        group_col: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build sequences per group to prevent cross-group contamination."""
        all_X, all_y = [], []
        for _, grp in df.groupby(group_col):
            grp = grp.sort_values("timestamp") if "timestamp" in grp.columns else grp
            X = self.scaler.transform(grp[feature_cols].values.astype(np.float32))
            y = grp[target_col].values.astype(np.float32)
            Xs, ys = self._make_sequences(X, y)
            if len(Xs) > 0:
                all_X.append(Xs)
                all_y.append(ys)
        if not all_X:
            n_feat = len(feature_cols)
            return np.empty((0, self.lookback_window, n_feat), np.float32), np.empty(0, np.float32)
        return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        feature_cols: List[str] = None,
        target_col: str = "kwh",
        group_col: str = None,
    ) -> "DemandForecaster":
        """Train Bi-LSTM with Adam optimiser and early stopping.

        group_col: when set, sequences are built per group to avoid
                   cross-group boundary contamination (e.g. per DT).
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if feature_cols is None:
            feature_cols = [c for c in FORECAST_FEATURES if c in train_data.columns]

        self._input_size = len(feature_cols)

        # Always fit scaler on all training features at once
        self.scaler.fit(train_data[feature_cols].values.astype(np.float32))

        if group_col and group_col in train_data.columns:
            X_tr, y_tr = self._build_grouped_sequences(train_data, feature_cols, target_col, group_col)
            X_vl, y_vl = self._build_grouped_sequences(val_data, feature_cols, target_col, group_col)
        else:
            X_tr_raw = self.scaler.transform(train_data[feature_cols].values.astype(np.float32))
            y_tr_raw = train_data[target_col].values.astype(np.float32)
            X_vl_raw = self.scaler.transform(val_data[feature_cols].values.astype(np.float32))
            y_vl_raw = val_data[target_col].values.astype(np.float32)
            X_tr, y_tr = self._make_sequences(X_tr_raw, y_tr_raw)
            X_vl, y_vl = self._make_sequences(X_vl_raw, y_vl_raw)

        if len(X_tr) == 0:
            logger.warning("No training sequences — initialising untrained model")
            self.model = _BiLSTMModel(
                input_size=self._input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                n_outputs=len(self.quantiles),
            ).to(self.device)
            return self

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr, device=self.device), torch.tensor(y_tr, device=self.device)),
            batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_vl, device=self.device), torch.tensor(y_vl, device=self.device)),
            batch_size=self.batch_size, shuffle=False,
        )

        self.model = _BiLSTMModel(
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_outputs=len(self.quantiles),
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=max(1, self.patience // 2), factor=0.5
        )

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = self.model(xb)  # (batch, n_quantiles)
                loss = sum(
                    _quantile_loss(preds[:, i], yb, q)
                    for i, q in enumerate(self.quantiles)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = self.model(xb)
                    val_loss += sum(
                        _quantile_loss(preds[:, i], yb, q).item()
                        for i, q in enumerate(self.quantiles)
                    )

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 5 == 0:
                logger.info("Epoch %d: train=%.4f val=%.4f", epoch + 1, train_loss, val_loss)

            if no_improve >= self.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        logger.info("DemandForecaster training complete (%d quantiles)", len(self.quantiles))
        return self

    def predict(
        self,
        data: pd.DataFrame,
        feature_cols: List[str] = None,
    ) -> np.ndarray:
        """Return quantile forecasts as array of shape (N, n_quantiles).

        Columns correspond to self.quantiles in order (e.g. q5, q50, q95).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")

        if feature_cols is None:
            feature_cols = [c for c in FORECAST_FEATURES if c in data.columns]

        X_raw = data[feature_cols].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_raw)

        lb = min(self.lookback_window, len(X_scaled) - 1)
        if lb < 1:
            return np.zeros((len(data), len(self.quantiles)), dtype=np.float32)

        Xs = [X_scaled[i - lb: i] for i in range(lb, len(X_scaled))]
        X_tensor = torch.tensor(np.array(Xs, dtype=np.float32), device=self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()  # (N-lb, n_quantiles)

        # Pad first lb rows by repeating the first valid prediction
        pad = np.tile(preds[0:1], (lb, 1))
        preds_full = np.concatenate([pad, preds], axis=0)  # (N, n_quantiles)

        # Enforce monotonicity across quantiles column by column
        preds_full = np.sort(preds_full, axis=1)

        return preds_full

    def save(self, model_path: str | Path, scaler_path: str | Path) -> None:
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            torch.save({
                "state_dict":    self.model.state_dict(),
                "hidden_size":   self.hidden_size,
                "num_layers":    self.num_layers,
                "dropout":       self.dropout,
                "lookback_window": self.lookback_window,
                "quantiles":     self.quantiles,
                "input_size":    self._input_size,
            }, str(model_path))

        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        logger.info("DemandForecaster saved → %s", model_path)

    def load(self, model_path: str | Path, scaler_path: str | Path, input_size: int = None) -> "DemandForecaster":
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)

        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.hidden_size    = checkpoint.get("hidden_size", self.hidden_size)
        self.num_layers     = checkpoint.get("num_layers", self.num_layers)
        self.dropout        = checkpoint.get("dropout", self.dropout)
        self.lookback_window = checkpoint.get("lookback_window", self.lookback_window)
        self.quantiles      = checkpoint.get("quantiles", self.quantiles)
        self._input_size    = checkpoint.get("input_size", input_size or len(FORECAST_FEATURES))

        self.model = _BiLSTMModel(
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_outputs=len(self.quantiles),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        logger.info("DemandForecaster loaded ← %s (%d quantiles, lookback=%d)",
                    model_path, len(self.quantiles), self.lookback_window)
        return self
