"""Unit tests for GSI signal functions."""
from __future__ import annotations

import numpy as np
import pytest

from gridsentinel.features.gsi_signals import (
    HOUR_PROFILE,
    signal_calendar,
    signal_ev_load_risk,
    signal_load_quantile,
    signal_power_factor_penalty,
    signal_pv_duck_curve,
    signal_temperature_derating,
    signal_thermal_soak,
    signal_transformer_age,
)


class TestLoadQuantile:
    def test_zero_capacity_returns_one(self):
        result = signal_load_quantile(100.0, 0.0)
        assert result == 1.0

    def test_below_capacity_returns_less_than_one(self):
        result = signal_load_quantile(100.0, 500.0, 0.9)
        assert 0.0 < result < 1.0

    def test_exceeds_capacity_clipped_to_one(self):
        result = signal_load_quantile(1000.0, 100.0, 0.9)
        assert result == 1.0

    def test_zero_load_returns_zero(self):
        result = signal_load_quantile(0.0, 500.0, 0.9)
        assert result == 0.0

    def test_returns_float(self):
        result = signal_load_quantile(200.0, 500.0, 0.9)
        assert isinstance(result, float)


class TestTemperatureDerating:
    def test_below_threshold_returns_zero(self):
        result = signal_temperature_derating(25.0, threshold=32.0, rate=0.05)
        assert result == 0.0

    def test_at_threshold_returns_zero(self):
        result = signal_temperature_derating(32.0, threshold=32.0, rate=0.05)
        assert result == 0.0

    def test_above_threshold_positive(self):
        result = signal_temperature_derating(40.0, threshold=32.0, rate=0.05)
        assert result > 0.0

    def test_clipped_to_one(self):
        result = signal_temperature_derating(1000.0, threshold=32.0, rate=0.05)
        assert result == 1.0

    def test_returns_float(self):
        result = signal_temperature_derating(35.0)
        assert isinstance(result, float)


class TestPowerFactorPenalty:
    def test_zero_pf_returns_one(self):
        result = signal_power_factor_penalty(0.0)
        assert result == 1.0

    def test_unity_pf_returns_zero(self):
        result = signal_power_factor_penalty(1.0)
        assert result == 0.0

    def test_low_pf_higher_penalty(self):
        r1 = signal_power_factor_penalty(0.7)
        r2 = signal_power_factor_penalty(0.9)
        assert r1 > r2

    def test_returns_float(self):
        result = signal_power_factor_penalty(0.85)
        assert isinstance(result, float)

    @pytest.mark.parametrize("pf", [0.1, 0.5, 0.8, 0.95, 1.0])
    def test_range(self, pf):
        result = signal_power_factor_penalty(pf)
        assert 0.0 <= result <= 1.0


class TestThermalSoak:
    def test_zero_hours_returns_zero(self):
        result = signal_thermal_soak(0.0, tau=4.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_many_hours_approaches_one(self):
        result = signal_thermal_soak(100.0, tau=4.0)
        assert result > 0.99

    def test_returns_float(self):
        result = signal_thermal_soak(2.0, tau=4.0)
        assert isinstance(result, float)

    def test_range(self):
        for h in [0, 1, 4, 10, 50]:
            result = signal_thermal_soak(float(h))
            assert 0.0 <= result <= 1.0


class TestEVLoadRisk:
    def test_zero_density_returns_zero(self):
        result = signal_ev_load_risk(0.0, hour=19)
        assert result == 0.0

    def test_peak_hour_higher_risk(self):
        r_peak = signal_ev_load_risk(0.5, hour=19)
        r_off = signal_ev_load_risk(0.5, hour=2)
        assert r_peak > r_off

    def test_returns_float(self):
        result = signal_ev_load_risk(0.3, hour=18)
        assert isinstance(result, float)

    def test_range_all_hours(self):
        for h in range(24):
            result = signal_ev_load_risk(0.5, hour=h)
            assert 0.0 <= result <= 1.0


class TestTransformerAge:
    def test_zero_age_returns_zero(self):
        result = signal_transformer_age(0.0)
        assert result == 0.0

    def test_age_25_returns_one(self):
        result = signal_transformer_age(25.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_older_higher_signal(self):
        r10 = signal_transformer_age(10.0)
        r20 = signal_transformer_age(20.0)
        assert r20 > r10

    def test_returns_float(self):
        result = signal_transformer_age(15.0)
        assert isinstance(result, float)

    def test_range(self):
        for age in [0, 5, 10, 15, 20, 25, 30]:
            result = signal_transformer_age(float(age))
            assert 0.0 <= result <= 1.0


class TestCalendarSignal:
    def test_event_returns_high_risk(self):
        result = signal_calendar(12, 0, 1, is_event=True)
        assert result >= 0.9

    def test_weekend_higher_than_weekday(self):
        r_sat = signal_calendar(12, 5, 7, is_event=False)
        r_mon = signal_calendar(12, 0, 7, is_event=False)
        assert r_sat >= r_mon

    def test_returns_float(self):
        result = signal_calendar(18, 2, 6, False)
        assert isinstance(result, float)

    def test_range_all_hours(self):
        for h in range(24):
            result = signal_calendar(h, 1, 6, False)
            assert 0.0 <= result <= 1.0


class TestPVDuckCurve:
    def test_zero_irradiance_returns_zero(self):
        result = signal_pv_duck_curve(0.0, 100.0, 80.0)
        assert result == 0.0

    def test_no_delta_returns_zero(self):
        result = signal_pv_duck_curve(500.0, 100.0, 100.0)
        assert result == 0.0

    def test_high_irradiance_high_delta_positive(self):
        result = signal_pv_duck_curve(800.0, 200.0, 50.0)
        assert result > 0.0

    def test_returns_float(self):
        result = signal_pv_duck_curve(400.0, 120.0, 80.0)
        assert isinstance(result, float)

    def test_range(self):
        for irr in [0, 100, 500, 1000]:
            result = signal_pv_duck_curve(float(irr), 200.0, 100.0)
            assert 0.0 <= result <= 1.0
