"""Pydantic request/response schemas for the GridSentinel REST API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Shared primitives ─────────────────────────────────────────────────────────

class StressFactor(BaseModel):
    parameter: str
    value: str


class DtwPoint(BaseModel):
    hour: int
    peer_avg: float
    target: float


class DemandPoint(BaseModel):
    h: str
    actual: float
    pred: float


# ── Meter / DT record models ──────────────────────────────────────────────────

class MeterRecord(BaseModel):
    meter_id: str
    dt_id: str
    timestamp: str
    kwh: float
    voltage_r: float
    voltage_y: float
    voltage_b: float
    current: float
    power_factor: float
    billed_kwh: float
    is_theft: Optional[int] = 0
    imputation_flag: Optional[int] = 0
    imputation_confidence: Optional[float] = 1.0


class DTRecord(BaseModel):
    dt_id: str
    timestamp: str
    feeder_kwh: float
    capacity_kva: float
    age_years: float
    temperature_c: float
    actual_kwh: float
    ev_density: Optional[float] = 0.0
    solar_irradiance: Optional[float] = 0.0


# ── /api/anomaly/clusters ─────────────────────────────────────────────────────

class AnomalyClusterItem(BaseModel):
    feeder: str
    risk: float
    logic: str
    status: str


# ── /api/anomaly/{meter_id} ───────────────────────────────────────────────────

class AnomalyDetailResponse(BaseModel):
    meter_id: str
    dtw_series: List[Dict[str, Any]]
    cass_score: float


# ── /api/demand/{dt_id} ───────────────────────────────────────────────────────

class DemandDetailResponse(BaseModel):
    transformer_id: str
    gsi_score: float
    action: str
    stress_factors: List[StressFactor]
    demand_series: List[Dict[str, Any]]


# ── /api/chat ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    trigger_xai: Optional[bool] = False
    target_id: Optional[str] = None
    target_type: Optional[str] = None   # "meter" | "transformer"


# ── /score/meter ──────────────────────────────────────────────────────────────

class MeterScoreRequest(BaseModel):
    meter_id: str
    records: List[Dict[str, Any]] = Field(..., description="List of meter reading records as dicts")


class MeterScoreResponse(BaseModel):
    meter_id: str
    cass_score: float
    cass_label: str
    y_proba: float
    top_features: List[Dict[str, float]]


# ── /score/transformer ────────────────────────────────────────────────────────

class DTScoreRequest(BaseModel):
    dt_id: str
    dt_records: List[Dict[str, Any]] = Field(..., description="DT readings as dicts")
    meter_records: Optional[List[Dict[str, Any]]] = Field(default=None)


class DTScoreResponse(BaseModel):
    dt_id: str
    gsi_score: float
    gsi_label: str
    q5_forecast: float
    q95_forecast: float


# ── /evaluate ─────────────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    meter_records: List[Dict[str, Any]]
    dt_records: List[Dict[str, Any]]


class EvaluateResponse(BaseModel):
    classification_metrics: Dict[str, float]
    forecast_metrics: Dict[str, float]
    economic_cost_inr: float
    gss_core: float
    gss_final: float
    constraints_met: bool
    shap_summary_path: str


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str


class ErrorResponse(BaseModel):
    error: str
    code: int
