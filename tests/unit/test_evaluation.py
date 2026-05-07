"""Unit tests for classification and forecasting evaluation functions."""
from __future__ import annotations

import numpy as np
import pytest

from gridsentinel.evaluation.classification import (
    evaluate_classification,
    expected_calibration_error,
)
from gridsentinel.evaluation.forecasting import (
    evaluate_forecast,
    mape,
    picp,
    pinaw,
    rmse,
)


# ── Classification evaluation ─────────────────────────────────────────────────

class TestExpectedCalibrationError:
    def test_perfect_calibration_returns_zero(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])
        ece = expected_calibration_error(y_true, y_proba)
        assert ece == pytest.approx(0.0, abs=0.05)

    def test_worst_calibration_positive(self):
        y_true = np.zeros(100, dtype=int)
        y_proba = np.ones(100)  # predicts 1.0 for all negatives
        ece = expected_calibration_error(y_true, y_proba)
        assert ece > 0.0

    def test_returns_float(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])
        result = expected_calibration_error(y_true, y_proba)
        assert isinstance(result, float)

    def test_range(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        y_proba = rng.random(100)
        ece = expected_calibration_error(y_true, y_proba)
        assert 0.0 <= ece <= 1.0


class TestEvaluateClassification:
    def _make_data(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        n = 200
        y_true = rng.integers(0, 2, n)
        y_proba = np.clip(y_true * 0.7 + rng.normal(0, 0.15, n), 0.0, 1.0)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_true, y_pred, y_proba

    def test_keys_present(self):
        y_true, y_pred, y_proba = self._make_data()
        result = evaluate_classification(y_true, y_pred, y_proba)
        for key in ["precision", "recall", "f1", "fpr", "mcc", "roc_auc", "pr_auc", "ece"]:
            assert key in result

    def test_all_correct_high_f1(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.9, 0.9])
        result = evaluate_classification(y_true, y_pred, y_proba)
        assert result["f1"] == pytest.approx(1.0, abs=0.01)

    def test_fpr_range(self):
        y_true, y_pred, y_proba = self._make_data()
        result = evaluate_classification(y_true, y_pred, y_proba)
        assert 0.0 <= result["fpr"] <= 1.0

    def test_returns_float_values(self):
        y_true, y_pred, y_proba = self._make_data()
        result = evaluate_classification(y_true, y_pred, y_proba)
        for k in ["precision", "recall", "f1", "fpr", "mcc", "roc_auc", "pr_auc", "ece"]:
            assert isinstance(result[k], float), f"{k} is not float"


# ── Forecasting evaluation ────────────────────────────────────────────────────

class TestMAPE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mape(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_division_by_zero_guard(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([1.0, 1.0])
        # Should not raise
        result = mape(y_true, y_pred)
        assert result >= 0.0

    def test_returns_float(self):
        result = mape(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
        assert isinstance(result, float)

    def test_positive_error(self):
        result = mape(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
        assert result > 0.0


class TestRMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0, abs=1e-6)

    def test_returns_float(self):
        result = rmse(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
        assert isinstance(result, float)


class TestPICP:
    def test_all_inside_returns_one(self):
        y = np.array([1.0, 2.0, 3.0])
        q5 = y - 0.5
        q95 = y + 0.5
        assert picp(y, q5, q95) == pytest.approx(1.0, abs=1e-6)

    def test_all_outside_returns_zero(self):
        y = np.array([5.0, 6.0])
        q5 = np.array([0.0, 0.0])
        q95 = np.array([1.0, 1.0])
        assert picp(y, q5, q95) == pytest.approx(0.0, abs=1e-6)

    def test_returns_float(self):
        y = np.array([1.0, 2.0, 3.0])
        result = picp(y, y - 1, y + 1)
        assert isinstance(result, float)


class TestPINAW:
    def test_zero_width_returns_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        result = pinaw(y, y, y)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_positive_width(self):
        y = np.array([1.0, 2.0, 3.0])
        q5 = y - 0.5
        q95 = y + 0.5
        result = pinaw(q5, q95, y)
        assert result > 0.0

    def test_returns_float(self):
        y = np.array([1.0, 2.0, 3.0])
        result = pinaw(y - 0.5, y + 0.5, y)
        assert isinstance(result, float)


class TestEvaluateForecast:
    def _make_data(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        y_true = rng.uniform(50, 200, 100)
        q5 = y_true * 0.9 - rng.uniform(0, 5, 100)
        q95 = y_true * 1.1 + rng.uniform(0, 5, 100)
        return y_true, q5, q95

    def test_keys_present(self):
        y_true, q5, q95 = self._make_data()
        result = evaluate_forecast(y_true, q5, q95)
        for key in ["mape", "rmse", "picp", "pinaw"]:
            assert key in result

    def test_picp_reasonable(self):
        y_true, q5, q95 = self._make_data()
        result = evaluate_forecast(y_true, q5, q95)
        assert 0.0 <= result["picp"] <= 1.0

    def test_returns_float_values(self):
        y_true, q5, q95 = self._make_data()
        result = evaluate_forecast(y_true, q5, q95)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float"
