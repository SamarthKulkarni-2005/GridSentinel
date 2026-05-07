"""Unit tests for CASS signal functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridsentinel.features.cass_signals import (
    signal_billing_ratio,
    signal_dt_balance_error,
    signal_dtw_divergence,
    signal_entropy,
    signal_night_load_anomaly,
    signal_repeat_anomaly,
    signal_voltage_stability,
)


# ── signal_dtw_divergence ─────────────────────────────────────────────────────

class TestDTWDivergence:
    def test_identical_series_returns_zero(self):
        arr = np.ones(96)
        result = signal_dtw_divergence(arr, arr, norm_denom=3.0)
        assert result == 0.0

    def test_large_distance_clipped_to_one(self):
        arr1 = np.zeros(96)
        arr2 = np.ones(96) * 1000.0
        result = signal_dtw_divergence(arr1, arr2, norm_denom=3.0)
        assert result == 1.0

    def test_returns_float(self):
        arr = np.random.default_rng(0).random(96)
        centroid = np.random.default_rng(1).random(96)
        result = signal_dtw_divergence(arr, centroid)
        assert isinstance(result, float)

    def test_range_zero_to_one(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            arr = rng.random(96)
            centroid = rng.random(96)
            result = signal_dtw_divergence(arr, centroid, norm_denom=1.0)
            assert 0.0 <= result <= 1.0


# ── signal_voltage_stability ──────────────────────────────────────────────────

class TestVoltageStability:
    def test_constant_voltage_returns_zero(self):
        arr = np.full(100, 230.0)
        result = signal_voltage_stability(arr, norm_std=0.15)
        assert result == 0.0

    def test_high_variance_clipped_to_one(self):
        arr = np.array([0.0, 1000.0] * 50)
        result = signal_voltage_stability(arr, norm_std=0.15)
        assert result == 1.0

    def test_returns_float(self):
        arr = np.random.default_rng(0).random(50) * 10.0
        result = signal_voltage_stability(arr, norm_std=0.15)
        assert isinstance(result, float)

    @pytest.mark.parametrize("norm_std", [0.01, 0.15, 1.0])
    def test_range_with_various_norms(self, norm_std):
        arr = np.random.default_rng(42).random(100)
        result = signal_voltage_stability(arr, norm_std=norm_std)
        assert 0.0 <= result <= 1.0


# ── signal_billing_ratio ──────────────────────────────────────────────────────

class TestBillingRatio:
    def test_equal_kwh_returns_zero(self):
        result = signal_billing_ratio(100.0, 100.0)
        assert result == 0.0

    def test_zero_meter_kwh_returns_one(self):
        result = signal_billing_ratio(100.0, 0.0)
        assert result == 1.0

    def test_large_discrepancy_clipped(self):
        result = signal_billing_ratio(10.0, 100.0)
        assert result == pytest.approx(0.9, abs=0.01)

    def test_returns_float(self):
        result = signal_billing_ratio(50.0, 100.0)
        assert isinstance(result, float)

    @pytest.mark.parametrize("billed,metered", [
        (0.0, 100.0), (100.0, 50.0), (200.0, 100.0)
    ])
    def test_range(self, billed, metered):
        result = signal_billing_ratio(billed, metered)
        assert 0.0 <= result <= 1.0


# ── signal_entropy ────────────────────────────────────────────────────────────

class TestEntropy:
    def test_constant_series_returns_one(self):
        arr = np.ones(100)
        result = signal_entropy(arr)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_uniform_random_returns_near_zero(self):
        rng = np.random.default_rng(42)
        arr = rng.uniform(0.0, 10.0, 10000)
        result = signal_entropy(arr)
        assert result < 0.2  # Should be close to 0 (max entropy)

    def test_empty_returns_zero(self):
        result = signal_entropy(np.array([]))
        assert result == 0.0

    def test_returns_float(self):
        arr = np.random.default_rng(0).random(100)
        result = signal_entropy(arr)
        assert isinstance(result, float)

    def test_range(self):
        for seed in range(5):
            arr = np.random.default_rng(seed).random(200)
            result = signal_entropy(arr)
            assert 0.0 <= result <= 1.0


# ── signal_night_load_anomaly ─────────────────────────────────────────────────

class TestNightLoadAnomaly:
    def _make_timestamps(self, n: int = 96) -> pd.DatetimeIndex:
        return pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")

    def test_zero_day_avg_returns_zero(self):
        ts = self._make_timestamps(96)
        # All night timestamps
        night_ts = pd.date_range("2024-01-01 22:00", periods=32, freq="15min", tz="UTC")
        arr = np.ones(32)
        result = signal_night_load_anomaly(arr, night_ts)
        assert result == 0.0

    def test_equal_night_day_returns_zero(self):
        ts = self._make_timestamps(96)
        arr = np.ones(96)
        result = signal_night_load_anomaly(arr, ts)
        assert result == 0.0

    def test_high_night_returns_positive(self):
        ts = self._make_timestamps(96)
        arr = np.ones(96)
        hours = ts.hour
        night_mask = (hours >= 22) | (hours < 6)
        arr[night_mask] = 5.0
        result = signal_night_load_anomaly(arr, ts)
        assert result > 0.0

    def test_returns_float(self):
        ts = self._make_timestamps(96)
        arr = np.random.default_rng(0).random(96)
        result = signal_night_load_anomaly(arr, ts)
        assert isinstance(result, float)

    def test_range(self):
        ts = self._make_timestamps(96)
        arr = np.random.default_rng(0).random(96)
        result = signal_night_load_anomaly(arr, ts)
        assert 0.0 <= result <= 1.0


# ── signal_dt_balance_error ───────────────────────────────────────────────────

class TestDTBalanceError:
    def test_zero_feeder_returns_zero(self):
        result = signal_dt_balance_error(100.0, 0.0)
        assert result == 0.0

    def test_equal_returns_zero(self):
        result = signal_dt_balance_error(100.0, 100.0)
        assert result == 0.0

    def test_large_error_clipped(self):
        result = signal_dt_balance_error(200.0, 100.0)
        assert result == 1.0  # error=1.0/0.5=2.0, clipped to 1.0

    def test_returns_float(self):
        result = signal_dt_balance_error(90.0, 100.0)
        assert isinstance(result, float)

    @pytest.mark.parametrize("meter,feeder", [
        (50.0, 100.0), (100.0, 90.0), (0.0, 100.0)
    ])
    def test_range(self, meter, feeder):
        result = signal_dt_balance_error(meter, feeder)
        assert 0.0 <= result <= 1.0


# ── signal_repeat_anomaly ─────────────────────────────────────────────────────

class TestRepeatAnomaly:
    def test_short_series_returns_zero(self):
        result = signal_repeat_anomaly(np.array([1.0, 2.0, 1.0]))
        assert result == 0.0

    def test_constant_series_no_anomaly(self):
        arr = np.ones(500)
        result = signal_repeat_anomaly(arr, window_count=5)
        assert result == 0.0

    def test_spike_detected(self):
        arr = np.ones(500)
        arr[100] = 1000.0  # single spike
        result = signal_repeat_anomaly(arr, window_count=5)
        assert result > 0.0

    def test_returns_float(self):
        arr = np.random.default_rng(0).random(200)
        result = signal_repeat_anomaly(arr, window_count=5)
        assert isinstance(result, float)

    def test_range(self):
        arr = np.random.default_rng(42).random(300)
        result = signal_repeat_anomaly(arr, window_count=5)
        assert 0.0 <= result <= 1.0
