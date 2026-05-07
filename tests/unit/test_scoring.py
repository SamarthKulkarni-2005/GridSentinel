"""Unit tests for CASS, GSI, economic, and GSS scoring functions."""
from __future__ import annotations

import numpy as np
import pytest

from gridsentinel.scoring.cass import cass_label, compute_cass, compute_g_pv
from gridsentinel.scoring.economic import compute_baseline_cost, compute_cost
from gridsentinel.scoring.gsi import compute_gsi, gsi_label
from gridsentinel.scoring.gss import (
    check_pareto_constraints,
    compute_gss_core,
    compute_gss_final,
    compute_s_delay,
    compute_s_energy,
)

# ── Weights fixtures ──────────────────────────────────────────────────────────

CASS_WEIGHTS = {
    "dtw_divergence": 0.25,
    "voltage_stability": 0.10,
    "billing_ratio": 0.20,
    "entropy": 0.10,
    "night_load_anomaly": 0.15,
    "dt_balance_error": 0.15,
    "repeat_anomaly": 0.05,
}

GSI_WEIGHTS = {
    "load_quantile": 0.30,
    "temperature_derating": 0.15,
    "power_factor_penalty": 0.15,
    "thermal_soak": 0.10,
    "ev_load_risk": 0.10,
    "transformer_age": 0.10,
    "calendar_signal": 0.05,
    "pv_duck_curve": 0.05,
}

GSS_CORE_WEIGHTS = {"w_cass": 0.35, "w_gsi": 0.20, "w_econ": 0.30, "w_robust": 0.15}
GSS_FINAL_WEIGHTS = {
    "w_cass": 0.30, "w_gsi": 0.15, "w_econ": 0.25,
    "w_robust": 0.10, "w_delay": 0.08, "w_energy": 0.07, "w_calib": 0.05,
}


# ── CASS scoring ──────────────────────────────────────────────────────────────

class TestComputeCASS:
    def _make_signals(self, value: float = 0.5) -> dict:
        return {k: value for k in CASS_WEIGHTS}

    def test_all_zero_signals_low_score(self):
        result = compute_cass(self._make_signals(0.0), CASS_WEIGHTS, 0.35, 8.0, 0.0)
        assert result < 10.0

    def test_all_one_signals_high_score(self):
        result = compute_cass(self._make_signals(1.0), CASS_WEIGHTS, 0.35, 8.0, 0.0)
        assert result > 80.0

    def test_high_pv_reduces_score(self):
        signals = self._make_signals(0.7)
        score_no_pv = compute_cass(signals, CASS_WEIGHTS, 0.35, 8.0, 0.0)
        score_with_pv = compute_cass(signals, CASS_WEIGHTS, 0.35, 8.0, 0.3)
        assert score_with_pv < score_no_pv

    def test_range(self):
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = compute_cass(self._make_signals(v), CASS_WEIGHTS, 0.35, 8.0, 0.0)
            assert 0.0 <= result <= 100.0

    def test_returns_float(self):
        result = compute_cass(self._make_signals(0.5), CASS_WEIGHTS, 0.35, 8.0, 0.0)
        assert isinstance(result, float)


class TestCassLabel:
    @pytest.mark.parametrize("score,expected", [
        (0.0, "Normal"), (34.9, "Normal"),
        (35.0, "Watch"), (59.9, "Watch"),
        (60.0, "Inspect"), (79.9, "Inspect"),
        (80.0, "Immediate"), (100.0, "Immediate"),
    ])
    def test_labels(self, score, expected):
        assert cass_label(score) == expected


class TestComputeGPV:
    def test_zero_irradiance(self):
        assert compute_g_pv(0.0) == 0.0

    def test_max_irradiance_clipped(self):
        assert compute_g_pv(10000.0) == pytest.approx(0.3, abs=0.001)

    def test_range(self):
        for irr in [0, 100, 500, 1000]:
            result = compute_g_pv(float(irr))
            assert 0.0 <= result <= 0.3


# ── GSI scoring ───────────────────────────────────────────────────────────────

class TestComputeGSI:
    def _make_signals(self, value: float = 0.5) -> dict:
        return {k: value for k in GSI_WEIGHTS}

    def test_zero_signals_returns_zero(self):
        result = compute_gsi(self._make_signals(0.0), GSI_WEIGHTS, 1.0, 1.0)
        assert result == 0.0

    def test_high_signals_high_score(self):
        result = compute_gsi(self._make_signals(1.0), GSI_WEIGHTS, 1.0, 1.0)
        assert result > 80.0

    def test_zero_confidence_returns_zero(self):
        result = compute_gsi(self._make_signals(1.0), GSI_WEIGHTS, 0.0, 1.0)
        assert result == 0.0

    def test_range(self):
        for v in [0.0, 0.25, 0.5, 1.0]:
            result = compute_gsi(self._make_signals(v), GSI_WEIGHTS, 1.0, 1.0)
            assert 0.0 <= result <= 100.0

    def test_returns_float(self):
        result = compute_gsi(self._make_signals(0.5), GSI_WEIGHTS, 0.9, 0.95)
        assert isinstance(result, float)


class TestGSILabel:
    @pytest.mark.parametrize("score,expected", [
        (0.0, "Stable"), (29.9, "Stable"),
        (30.0, "Caution"), (54.9, "Caution"),
        (55.0, "Stressed"), (74.9, "Stressed"),
        (75.0, "Critical"), (100.0, "Critical"),
    ])
    def test_labels(self, score, expected):
        assert gsi_label(score) == expected


# ── Economic model ────────────────────────────────────────────────────────────

class TestComputeCost:
    def test_no_fp_fn_zero_cost(self):
        assert compute_cost(0, 0, 8500.0, 5500.0, 3.0) == 0.0

    def test_fp_only(self):
        assert compute_cost(2, 0, 8500.0, 5500.0, 3.0) == pytest.approx(17000.0)

    def test_fn_only(self):
        assert compute_cost(0, 1, 8500.0, 5500.0, 3.0) == pytest.approx(16500.0)

    def test_combined_cost(self):
        cost = compute_cost(1, 1, 8500.0, 5500.0, 3.0)
        assert cost == pytest.approx(8500.0 + 5500.0 * 3.0)

    def test_returns_float(self):
        result = compute_cost(5, 3, 8500.0, 5500.0, 3.0)
        assert isinstance(result, float)


class TestComputeBaselineCost:
    def test_zero_cases(self):
        assert compute_baseline_cost(0, 5500.0, 3.0) == 0.0

    def test_formula(self):
        result = compute_baseline_cost(10, 5500.0, 3.0)
        assert result == pytest.approx(10 * 5500.0 * 3.0)


# ── GSS scoring ───────────────────────────────────────────────────────────────

class TestCheckParetoConstraints:
    CFG = {"max_fpr": 0.02, "min_recall": 0.85, "max_mape": 0.07}

    def test_all_pass(self):
        assert check_pareto_constraints(0.01, 0.90, 0.05, self.CFG)

    def test_fpr_fails(self):
        assert not check_pareto_constraints(0.03, 0.90, 0.05, self.CFG)

    def test_recall_fails(self):
        assert not check_pareto_constraints(0.01, 0.80, 0.05, self.CFG)

    def test_mape_fails(self):
        assert not check_pareto_constraints(0.01, 0.90, 0.08, self.CFG)


class TestComputeGSSCore:
    def test_constraints_not_met_returns_zero(self):
        result = compute_gss_core(0.9, 0.9, 0.9, 0.9, GSS_CORE_WEIGHTS, False)
        assert result == 0.0

    def test_all_perfect_returns_one(self):
        result = compute_gss_core(1.0, 1.0, 1.0, 1.0, GSS_CORE_WEIGHTS, True)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_returns_float(self):
        result = compute_gss_core(0.5, 0.5, 0.5, 0.5, GSS_CORE_WEIGHTS, True)
        assert isinstance(result, float)

    def test_range(self):
        for v in [0.0, 0.5, 1.0]:
            result = compute_gss_core(v, v, v, v, GSS_CORE_WEIGHTS, True)
            assert 0.0 <= result <= 1.0


class TestComputeGSSFinal:
    def test_constraints_not_met_returns_zero(self):
        result = compute_gss_final(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, GSS_FINAL_WEIGHTS, False)
        assert result == 0.0

    def test_all_perfect_returns_one(self):
        result = compute_gss_final(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, GSS_FINAL_WEIGHTS, True)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_range(self):
        for v in [0.0, 0.5, 1.0]:
            result = compute_gss_final(v, v, v, v, v, v, v, GSS_FINAL_WEIGHTS, True)
            assert 0.0 <= result <= 1.0


class TestSDelay:
    def test_zero_latency_returns_one(self):
        assert compute_s_delay(0.0) == pytest.approx(1.0, abs=0.001)

    def test_large_latency_near_zero(self):
        assert compute_s_delay(1000.0) < 0.01

    def test_range(self):
        for d in [0, 1, 7, 30, 100]:
            result = compute_s_delay(float(d))
            assert 0.0 <= result <= 1.0


class TestSEnergy:
    def test_zero_feeder_returns_one(self):
        assert compute_s_energy(100.0, 0.0) == 1.0

    def test_perfect_balance_returns_one(self):
        assert compute_s_energy(100.0, 100.0) == pytest.approx(1.0, abs=0.001)

    def test_large_error_clipped(self):
        result = compute_s_energy(0.0, 100.0)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_range(self):
        for m, f in [(50, 100), (100, 100), (150, 100)]:
            result = compute_s_energy(float(m), float(f))
            assert 0.0 <= result <= 1.0
