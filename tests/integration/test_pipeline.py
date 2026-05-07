"""Integration test — full pipeline on synthetic 100-meter × 5-DT dataset."""
from __future__ import annotations

import pytest

from tests.fixtures.sample_data import make_dt_df, make_meter_df
from gridsentinel.config import load_config, reset_config
from gridsentinel.pipeline import GridSentinelPipeline, PipelineResult


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate deterministic 100-meter × 5-DT dataset (7 days for speed)."""
    meter_df = make_meter_df(n_meters=100, n_days=7, seed=42, theft_rate=0.05)
    dt_df = make_dt_df(n_dts=5, n_days=7, seed=42)
    return meter_df, dt_df


@pytest.fixture(scope="module")
def pipeline_result(synthetic_data):
    """Run the full pipeline and return the result."""
    meter_df, dt_df = synthetic_data
    reset_config()
    pipeline = GridSentinelPipeline()
    result = pipeline.run(meter_df, dt_df)
    return result


class TestPipelineResult:
    def test_gss_final_in_range(self, pipeline_result):
        assert 0.0 <= pipeline_result.gss_final <= 1.0

    def test_gss_core_in_range(self, pipeline_result):
        assert 0.0 <= pipeline_result.gss_core <= 1.0

    def test_constraints_met_is_bool(self, pipeline_result):
        assert isinstance(pipeline_result.constraints_met, bool)

    def test_meter_scores_not_empty(self, pipeline_result):
        assert len(pipeline_result.meter_scores) > 0

    def test_dt_scores_not_empty(self, pipeline_result):
        assert len(pipeline_result.dt_scores) > 0

    def test_meter_scores_columns(self, pipeline_result):
        cols = pipeline_result.meter_scores.columns.tolist()
        assert "meter_id" in cols
        assert "cass_score" in cols
        assert "cass_label" in cols

    def test_dt_scores_columns(self, pipeline_result):
        cols = pipeline_result.dt_scores.columns.tolist()
        assert "dt_id" in cols
        assert "gsi_score" in cols
        assert "gsi_label" in cols

    def test_cass_scores_in_range(self, pipeline_result):
        scores = pipeline_result.meter_scores["cass_score"]
        assert (scores >= 0.0).all()
        assert (scores <= 100.0).all()

    def test_gsi_scores_in_range(self, pipeline_result):
        scores = pipeline_result.dt_scores["gsi_score"]
        assert (scores >= 0.0).all()
        assert (scores <= 100.0).all()

    def test_classification_metrics_keys(self, pipeline_result):
        for key in ["precision", "recall", "f1", "fpr", "mcc"]:
            assert key in pipeline_result.classification_metrics

    def test_forecast_metrics_keys(self, pipeline_result):
        for key in ["mape", "rmse", "picp", "pinaw"]:
            assert key in pipeline_result.forecast_metrics

    def test_economic_cost_non_negative(self, pipeline_result):
        assert pipeline_result.economic_cost_inr >= 0.0

    def test_y_proba_in_range(self, pipeline_result):
        probas = pipeline_result.meter_scores["y_proba"]
        assert (probas >= 0.0).all()
        assert (probas <= 1.0).all()
