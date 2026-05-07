"""XGBoost-based theft detector with SHAP explainability support."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "dtw_divergence", "voltage_stability", "billing_ratio",
    "entropy", "night_load_anomaly", "dt_balance_error", "repeat_anomaly",
    "power_factor_mean", "kwh_mean_7d", "kwh_std_7d",
    "kwh_trend_slope",
    "hour_of_day", "day_of_week", "month",
]


class TheftDetector:
    """XGBoost classifier for electricity theft detection."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 10.0,
        eval_metric: str = "aucpr",
        early_stopping_rounds: int = 30,
        random_state: int = 42,
    ) -> None:
        """Initialise TheftDetector with XGBoost hyperparameters."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "TheftDetector":
        """Train XGBoost with early stopping; compute scale_pos_weight from data."""
        # Compute actual class ratio
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        if n_pos == 0:
            logger.warning("No positive samples in training set; using default scale_pos_weight")
            spw = self.scale_pos_weight
        else:
            spw = n_neg / n_pos
            logger.info("scale_pos_weight computed from data: %.2f", spw)

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=spw,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            use_label_encoder=False,
            verbosity=0,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iteration = self.model.best_iteration
        logger.info("TheftDetector training complete. Best iteration: %d", best_iteration)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(theft) in [0, 1] for each sample."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # P(class=1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary theft labels using the given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        """Save XGBoost model to JSON file."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info("TheftDetector saved to %s", path)

    def load(self, path: str | Path) -> "TheftDetector":
        """Load XGBoost model from JSON file."""
        path = Path(path)
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        logger.info("TheftDetector loaded from %s", path)
        return self
