"""GridSentinel configuration loader — single source of truth for all parameters."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)

_LOG_LEVEL = os.environ.get("GS_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


# ── Pydantic sub-models ──────────────────────────────────────────────────────

class DataConfig(BaseModel):
    raw_meter_path: str
    raw_dt_path: str
    resample_freq: str
    lookback_window: int


class CASSWeights(BaseModel):
    dtw_divergence: float
    voltage_stability: float
    billing_ratio: float
    entropy: float
    night_load_anomaly: float
    dt_balance_error: float
    repeat_anomaly: float

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "CASSWeights":
        total = sum(self.model_dump().values())
        assert abs(total - 1.0) < 1e-6, f"CASS weights sum to {total}, not 1.0"
        return self


class CASSConfig(BaseModel):
    weights: CASSWeights
    sigmoid_shift: float
    sigmoid_scale: float
    dtw_norm_denom: float
    voltage_std_norm: float
    night_ratio_scale: float
    dt_error_norm: float
    repeat_window_count: int


class GSIWeights(BaseModel):
    load_quantile: float
    temperature_derating: float
    power_factor_penalty: float
    thermal_soak: float
    ev_load_risk: float
    transformer_age: float
    calendar_signal: float
    pv_duck_curve: float

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "GSIWeights":
        total = sum(self.model_dump().values())
        assert abs(total - 1.0) < 1e-6, f"GSI weights sum to {total}, not 1.0"
        return self


class GSIConfig(BaseModel):
    weights: GSIWeights
    temp_threshold: float
    temp_derate_rate: float
    thermal_tau: float
    age_ref: float


class XGBoostConfig(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    scale_pos_weight: float
    eval_metric: str
    early_stopping_rounds: int
    random_state: int


class LSTMConfig(BaseModel):
    hidden_size: int
    num_layers: int
    dropout: float
    quantiles: List[float]
    epochs: int
    batch_size: int
    lr: float
    patience: int
    random_state: int


class ModelsConfig(BaseModel):
    xgboost: XGBoostConfig
    lstm: LSTMConfig


class EconomicsConfig(BaseModel):
    fp_cost_inr: float
    fn_cost_per_month_inr: float
    default_theft_duration_months: float


class GSSCoreWeights(BaseModel):
    w_cass: float
    w_gsi: float
    w_econ: float
    w_robust: float

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "GSSCoreWeights":
        total = self.w_cass + self.w_gsi + self.w_econ + self.w_robust
        assert abs(total - 1.0) < 1e-6, f"GSS core weights sum to {total}, not 1.0"
        return self


class GSSFinalWeights(BaseModel):
    w_cass: float
    w_gsi: float
    w_econ: float
    w_robust: float
    w_delay: float
    w_energy: float
    w_calib: float

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "GSSFinalWeights":
        total = (self.w_cass + self.w_gsi + self.w_econ + self.w_robust +
                 self.w_delay + self.w_energy + self.w_calib)
        assert abs(total - 1.0) < 1e-6, f"GSS final weights sum to {total}, not 1.0"
        return self


class ConstraintsConfig(BaseModel):
    max_fpr: float
    min_recall: float
    max_mape: float


class RobustnessConfig(BaseModel):
    noise_std: float
    noise_seed: int


class APIConfig(BaseModel):
    host: str
    port: int
    workers: int


class Config(BaseModel):
    """Top-level configuration model — loaded once and cached as singleton."""
    data: DataConfig
    cass: CASSConfig
    gsi: GSIConfig
    models: ModelsConfig
    economics: EconomicsConfig
    gss_core: GSSCoreWeights
    gss_final: GSSFinalWeights
    constraints: ConstraintsConfig
    robustness: RobustnessConfig
    api: APIConfig


# ── Singleton loader ─────────────────────────────────────────────────────────

_CONFIG_INSTANCE: Optional[Config] = None
_CONFIG_PATH: Optional[Path] = None


def load_config(path: Optional[Path] = None) -> Config:
    """Load and validate config from YAML; return cached singleton."""
    global _CONFIG_INSTANCE, _CONFIG_PATH

    if path is None:
        path = Path(__file__).parent.parent / "config" / "default.yaml"

    path = Path(path)

    if _CONFIG_INSTANCE is not None and _CONFIG_PATH == path:
        return _CONFIG_INSTANCE

    logger.info("Loading config from %s", path)
    with open(path, "r") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    cfg = Config(**raw)
    _CONFIG_INSTANCE = cfg
    _CONFIG_PATH = path
    logger.info("Config loaded and validated successfully")
    return cfg


def reset_config() -> None:
    """Reset the singleton (used in tests)."""
    global _CONFIG_INSTANCE, _CONFIG_PATH
    _CONFIG_INSTANCE = None
    _CONFIG_PATH = None
