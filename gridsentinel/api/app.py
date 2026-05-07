"""FastAPI application for GridSentinel REST API."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from gridsentinel.api.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from gridsentinel.api.database import AuditLog, SessionLocal, User, get_db
from gridsentinel.api.schemas import (
    AnomalyClusterItem,
    AnomalyDetailResponse,
    ChatRequest,
    ChatResponse,
    DemandDetailResponse,
    DTScoreRequest,
    DTScoreResponse,
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    MeterScoreRequest,
    MeterScoreResponse,
    StressFactor,
)
from gridsentinel.config import load_config
from gridsentinel.features.cass_signals import (
    signal_billing_ratio,
    signal_dt_balance_error,
    signal_entropy,
    signal_night_load_anomaly,
    signal_repeat_anomaly,
    signal_voltage_stability,
)
from gridsentinel.features.feature_store import FEATURE_NAMES, build_meter_features, build_forecast_features
from gridsentinel.features.gsi_signals import (
    signal_calendar,
    signal_ev_load_risk,
    signal_load_quantile,
    signal_power_factor_penalty,
    signal_pv_duck_curve,
    signal_temperature_derating,
    signal_thermal_soak,
    signal_transformer_age,
)
from gridsentinel.features.holidays import is_holiday
from gridsentinel.ingestion.loader import load_from_csv
from gridsentinel.ingestion.validator import validate_dt_df, validate_meter_df
from gridsentinel.models.demand_forecaster import DemandForecaster, FORECAST_FEATURES
from gridsentinel.models.explainability import compute_shap_values, get_top_shap_features
from gridsentinel.models.theft_detector import FEATURE_NAMES as TD_FEATURE_NAMES
from gridsentinel.pipeline import GridSentinelPipeline, PipelineResult
from gridsentinel.scoring.cass import cass_label, compute_cass, compute_g_pv
from gridsentinel.scoring.gsi import compute_gsi, gsi_label

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GridSentinel AI",
    description="Smart grid theft detection and stress scoring REST API",
    version="2.0",
)

# ── CORS — allow the Vite dev server (port 5173) and any localhost origin ─────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory data store populated at startup ─────────────────────────────────
_store: Dict[str, Any] = {}

# ── Global pipeline singleton ─────────────────────────────────────────────────
_pipeline: GridSentinelPipeline | None = None

# ── CASS signal weights (used for SHAP proxy) ────────────────────────────────
_CASS_WEIGHTS = {
    "dtw_divergence":     0.25,
    "voltage_stability":  0.20,
    "billing_ratio":      0.20,
    "entropy":            0.10,
    "night_load_anomaly": 0.15,
    "dt_balance_error":   0.05,
    "repeat_anomaly":     0.05,
}

# ── LLM System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are GridSentinel AI, an expert system for BESCOM's smart electrical grid.
You analyze meter telemetry, detect electricity theft using CASS scores, and monitor Grid Stress Index (GSI).
Keep responses concise (2-3 sentences). Use technical but clear language.
When given a meter ID, comment on its CASS score and theft likelihood.
When given a transformer ID, comment on its GSI score and grid stress.
Always end with a recommended action."""


def _get_pipeline() -> GridSentinelPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = GridSentinelPipeline()
    return _pipeline


def _write_audit(
    db: Session,
    action: str,
    target_id: Optional[str] = None,
    target_type: Optional[str] = None,
    details: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    """Insert an audit log entry — silently ignores errors to avoid breaking endpoints."""
    try:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            target_id=target_id,
            target_type=target_type,
            details=details,
            timestamp=datetime.utcnow(),
        )
        db.add(entry)
        db.commit()
    except Exception as exc:
        logger.warning("Audit log write failed: %s", exc)
        db.rollback()


def _seed_admin() -> None:
    """Create the default admin user if it doesn't already exist."""
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == "admin").first()
        if existing is None:
            admin = User(
                username="admin",
                hashed_password=hash_password("bescom2026"),
                role="admin",
            )
            db.add(admin)
            db.commit()
            logger.info("Seeded default admin user.")
    except Exception as exc:
        logger.warning("Admin seed failed: %s", exc)
        db.rollback()
    finally:
        db.close()


# ── Startup: load CSV and pre-compute scores ──────────────────────────────────

@app.on_event("startup")
async def startup_load_data() -> None:
    """Load synthetic CSV, cache meter/DT data, seed admin user."""
    _seed_admin()

    candidates = [
        Path(os.environ.get("GS_DATA_PATH", "")),
        Path(__file__).parent.parent.parent / "synthetic_grid_data.csv",
        Path(__file__).parent.parent.parent.parent / "synthetic_grid_data.csv",
        Path("D:/BESCOM/synthetic_grid_data.csv"),
        Path("../synthetic_grid_data.csv"),
        Path("../../synthetic_grid_data.csv"),
    ]
    for p in candidates:
        if p and p.exists():
            try:
                meter_df, dt_df = load_from_csv(p)
                _store["meter_df"] = meter_df
                _store["dt_df"] = dt_df
                _compute_risk_scores(meter_df, dt_df)
                _load_forecaster(meter_df, dt_df)
                logger.info("Loaded CSV from %s: %d meter rows, %d DT rows", p, len(meter_df), len(dt_df))
                return
            except Exception as exc:
                logger.warning("Failed to load CSV from %s: %s", p, exc)
    logger.warning("synthetic_grid_data.csv not found — /api endpoints will return demo data")


def _compute_risk_scores(meter_df: pd.DataFrame, dt_df: pd.DataFrame | None = None) -> None:
    """Pre-compute CASS score per meter using the actual 7-signal pipeline."""
    cfg = load_config()
    w = cfg.cass.weights
    weights = {
        "dtw_divergence":     w.dtw_divergence,
        "voltage_stability":  w.voltage_stability,
        "billing_ratio":      w.billing_ratio,
        "entropy":            w.entropy,
        "night_load_anomaly": w.night_load_anomaly,
        "dt_balance_error":   w.dt_balance_error,
        "repeat_anomaly":     w.repeat_anomaly,
    }

    # DTW centroid: mean hourly profile across all meters (Euclidean proxy)
    mean_hourly = (
        meter_df.assign(_h=meter_df["timestamp"].dt.hour)
        .groupby("_h")["kwh"].mean()
        .reindex(range(24), fill_value=0.0)
        .values
    )
    centroid_norm = max(float(np.linalg.norm(mean_hourly)), cfg.cass.dtw_norm_denom)

    # DT feeder totals for balance error signal
    dt_feeder_map: dict = {}
    dt_meter_sum: dict = {}
    if dt_df is not None:
        dt_feeder_map = dt_df.groupby("dt_id")["feeder_kwh"].sum().to_dict()
        dt_meter_sum  = meter_df.groupby("dt_id")["kwh"].sum().to_dict()

    # Mean solar irradiance per DT (for PV correction)
    dt_solar_map: dict = {}
    if dt_df is not None and "solar_irradiance" in dt_df.columns:
        dt_solar_map = dt_df.groupby("dt_id")["solar_irradiance"].mean().to_dict()

    meter_to_dt = (
        meter_df[["meter_id", "dt_id"]].drop_duplicates()
        .set_index("meter_id")["dt_id"].to_dict()
    )

    scores: dict = {}
    for meter_id, grp in meter_df.groupby("meter_id"):
        kwh = grp["kwh"].values
        ts  = pd.DatetimeIndex(grp["timestamp"])

        # 1. DTW divergence (Euclidean against population mean profile)
        hourly = (
            grp.assign(_h=ts.hour).groupby("_h")["kwh"].mean()
            .reindex(range(24), fill_value=0.0).values
        )
        dtw_val = float(np.clip(np.linalg.norm(hourly - mean_hourly) / centroid_norm, 0, 1))

        # 2. Voltage stability (all three phases)
        volt_arr = grp[["voltage_r", "voltage_y", "voltage_b"]].values.flatten()
        volt_sig = signal_voltage_stability(volt_arr, norm_std=cfg.cass.voltage_std_norm)

        # 3. Billing ratio
        billed   = float(grp["billed_kwh"].iloc[-1]) if "billed_kwh" in grp.columns else float(kwh.sum())
        bill_sig = signal_billing_ratio(billed, max(float(kwh.sum()), 1e-9))

        # 4. Entropy
        ent_sig = signal_entropy(kwh)

        # 5. Night-load anomaly
        night_sig = signal_night_load_anomaly(kwh, ts)

        # 6. DT balance error
        dt_id      = meter_to_dt.get(meter_id)
        feeder_kwh = dt_feeder_map.get(dt_id, 0.0) if dt_id else 0.0
        sum_m_kwh  = dt_meter_sum.get(dt_id, 0.0)  if dt_id else 0.0
        bal_sig    = signal_dt_balance_error(sum_m_kwh, feeder_kwh) if feeder_kwh > 0 else 0.0

        # 7. Repeat anomaly
        rep_sig = signal_repeat_anomaly(kwh, timestamps=ts, window_count=cfg.cass.repeat_window_count)

        signals = {
            "dtw_divergence":     dtw_val,
            "voltage_stability":  volt_sig,
            "billing_ratio":      bill_sig,
            "entropy":            ent_sig,
            "night_load_anomaly": night_sig,
            "dt_balance_error":   bal_sig,
            "repeat_anomaly":     rep_sig,
        }

        solar = dt_solar_map.get(dt_id, 0.0) if dt_id else 0.0
        g_pv  = compute_g_pv(solar)
        scores[meter_id] = compute_cass(signals, weights, cfg.cass.sigmoid_shift, cfg.cass.sigmoid_scale, g_pv)

    _store["meter_scores"] = pd.Series(scores).sort_values(ascending=False)
    logger.info("CASS scores computed for %d meters", len(scores))


def _load_forecaster(meter_df: pd.DataFrame, dt_df: pd.DataFrame) -> None:
    """Load trained Bi-LSTM and pre-compute forecast features for all DTs."""
    from gridsentinel.features.feature_store import build_forecast_features
    model_path  = Path(__file__).parent.parent.parent / "models" / "demand_forecaster.pt"
    scaler_path = Path(__file__).parent.parent.parent / "models" / "demand_scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        logger.warning("Demand forecaster not found at %s — run scripts/train_forecaster.py first", model_path)
        return
    try:
        fc = DemandForecaster()
        fc.load(model_path, scaler_path)
        _store["forecaster"] = fc
        _store["forecast_features"] = build_forecast_features(meter_df, dt_df)
        logger.info("Demand forecaster loaded (lookback=%d, quantiles=%s)", fc.lookback_window, fc.quantiles)
    except Exception as exc:
        logger.warning("Failed to load demand forecaster: %s", exc)


def _status_from_risk(risk: float) -> str:
    if risk >= 80: return "critical"
    if risk >= 60: return "high"
    if risk >= 35: return "moderate"
    return "normal"


# ── Error handler ─────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(status_code=500, content={"error": str(exc), "code": 500})


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(status="ok", version="2.0")


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
async def register(body: dict, db: Session = Depends(get_db)):
    """Register a new user."""
    username = body.get("username", "").strip()
    password = body.get("password", "")
    role     = body.get("role", "viewer")

    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password are required")

    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")

    user = User(
        username=username,
        hashed_password=hash_password(password),
        role=role,
    )
    db.add(user)
    db.commit()
    return {"message": "ok"}


@app.post("/api/auth/login")
async def login(body: dict, db: Session = Depends(get_db)):
    """Verify credentials and return a JWT access token."""
    username = body.get("username", "").strip()
    password = body.get("password", "")

    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = create_access_token({"sub": user.username, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "username": user.username}


# OAuth2 token alias (required by OAuth2PasswordBearer tokenUrl)
@app.post("/api/auth/token")
async def token_alias(body: dict, db: Session = Depends(get_db)):
    """Alias of /api/auth/login for OAuth2 compatibility."""
    return await login(body, db)


# ── Audit log endpoint ────────────────────────────────────────────────────────

@app.get("/api/audit/log")
async def audit_log(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return last 50 audit log entries (protected)."""
    entries = (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id":          e.id,
            "user_id":     e.user_id,
            "action":      e.action,
            "target_id":   e.target_id,
            "target_type": e.target_type,
            "details":     e.details,
            "timestamp":   e.timestamp.isoformat() if e.timestamp else None,
        }
        for e in entries
    ]


# ── /api/anomaly/clusters ─────────────────────────────────────────────────────

@app.get("/api/anomaly/clusters", response_model=List[AnomalyClusterItem])
async def anomaly_clusters(top_n: int = 8) -> List[AnomalyClusterItem]:
    """Return top-N meters ranked by risk score for the anomaly table."""
    scores: pd.Series | None = _store.get("meter_scores")
    if scores is None or scores.empty:
        return [
            AnomalyClusterItem(feeder="MTR_000001", risk=88.0, logic="DTW + CASS", status="critical"),
            AnomalyClusterItem(feeder="MTR_000002", risk=71.0, logic="CASS",       status="high"),
            AnomalyClusterItem(feeder="MTR_000003", risk=52.0, logic="DTW",        status="moderate"),
        ]

    top = scores.head(top_n)
    results = []
    for meter_id, risk in top.items():
        status = _status_from_risk(float(risk))
        logic  = "DTW + CASS" if risk >= 65 else "CASS"
        results.append(AnomalyClusterItem(
            feeder=str(meter_id),
            risk=round(float(risk), 1),
            logic=logic,
            status=status,
        ))
    return results


# ── /api/anomaly/{meter_id} ───────────────────────────────────────────────────

@app.get("/api/anomaly/{meter_id}", response_model=AnomalyDetailResponse)
async def anomaly_detail(meter_id: str, db: Session = Depends(get_db)) -> AnomalyDetailResponse:
    """Return hourly DTW divergence series and CASS score for a meter."""
    # Write audit log
    _write_audit(db, action="anomaly_detail", target_id=meter_id, target_type="meter",
                 details=f"Fetched anomaly detail for {meter_id}")

    meter_df: pd.DataFrame | None = _store.get("meter_df")

    if meter_df is None or meter_df.empty:
        series = [{"hour": h, "peer_avg": 3.5, "target": 3.5 + (h % 4) * 0.4} for h in range(24)]
        return AnomalyDetailResponse(meter_id=meter_id, dtw_series=series, cass_score=72.0)

    normalised = meter_id.replace("-", "_")
    mask = meter_df["meter_id"] == normalised
    if not mask.any():
        mask = meter_df["meter_id"].str.contains(meter_id.split("_")[-1].lstrip("0") or "0", regex=False)

    meter_slice = meter_df[mask]
    if meter_slice.empty:
        series = [{"hour": h, "peer_avg": 3.5, "target": 3.5 + (h % 3) * 0.5} for h in range(24)]
        return AnomalyDetailResponse(meter_id=meter_id, dtw_series=series, cass_score=68.0)

    dt_id = meter_slice["dt_id"].iloc[0]
    peer_df = meter_df[(meter_df["dt_id"] == dt_id) & (meter_df["meter_id"] != normalised)]

    meter_slice = meter_slice.copy()
    meter_slice["hour"] = meter_slice["timestamp"].dt.hour
    hourly_target = meter_slice.groupby("hour")["kwh"].mean()

    if not peer_df.empty:
        peer_cp = peer_df.copy()
        peer_cp["hour"] = peer_cp["timestamp"].dt.hour
        hourly_peer = peer_cp.groupby("hour")["kwh"].mean()
    else:
        hourly_peer = hourly_target * 0.85

    dtw_series = []
    for h in range(24):
        dtw_series.append({
            "hour": h,
            "peer_avg": round(float(hourly_peer.get(h, hourly_peer.mean())), 3),
            "target":   round(float(hourly_target.get(h, hourly_target.mean())), 3),
        })

    scores = _store.get("meter_scores", pd.Series(dtype=float))
    cass = float(scores.get(normalised, 60.0))

    return AnomalyDetailResponse(meter_id=normalised, dtw_series=dtw_series, cass_score=round(cass, 2))


def _fallback_series(vals: pd.Series) -> list:
    """Smoothed actual as stand-in pred when model is unavailable."""
    smoothed = vals.rolling(3, center=True, min_periods=1).mean()
    return [
        {
            "h":      f"{h:02d}",
            "actual": round(float(vals.get(h, 0)), 2),
            "pred":   round(float(smoothed.get(h, 0)), 2),
            "q5":     round(float(vals.get(h, 0)) * 0.88, 2),
            "q95":    round(float(vals.get(h, 0)) * 1.12, 2),
        }
        for h in range(24)
    ]


# ── /api/demand/{dt_id} ───────────────────────────────────────────────────────

@app.get("/api/demand/{dt_id}", response_model=DemandDetailResponse)
async def demand_detail(dt_id: str, db: Session = Depends(get_db)) -> DemandDetailResponse:
    """Return hourly actual vs predicted demand and GSI score for a transformer."""
    # Write audit log
    _write_audit(db, action="demand_detail", target_id=dt_id, target_type="transformer",
                 details=f"Fetched demand detail for {dt_id}")

    dt_df: pd.DataFrame | None = _store.get("dt_df")

    if dt_df is None or dt_df.empty:
        series = [
            {
                "h": f"{h:02d}",
                "actual": 400 + h * 20,
                "pred":   420 + h * 20,
                "q5":     round((400 + h * 20) * 0.88, 2),
                "q95":    round((400 + h * 20) * 1.12, 2),
            }
            for h in range(24)
        ]
        return DemandDetailResponse(
            transformer_id=dt_id, gsi_score=55.0,
            action="Monitor Load Trend",
            stress_factors=[StressFactor(parameter="Data", value="Unavailable")],
            demand_series=series,
        )

    mask = dt_df["dt_id"] == dt_id
    if not mask.any():
        mask = dt_df["dt_id"].str.contains(dt_id.split("_")[-1], regex=False)

    dt_slice = dt_df[mask]
    if dt_slice.empty:
        series = [
            {
                "h": f"{h:02d}",
                "actual": 400 + h * 15,
                "pred":   415 + h * 15,
                "q5":     round((400 + h * 15) * 0.88, 2),
                "q95":    round((400 + h * 15) * 1.12, 2),
            }
            for h in range(24)
        ]
        return DemandDetailResponse(
            transformer_id=dt_id, gsi_score=50.0,
            action="No Data",
            stress_factors=[],
            demand_series=series,
        )

    dt_slice = dt_slice.copy()
    dt_slice["hour"] = dt_slice["timestamp"].dt.hour
    hourly_actual = dt_slice.groupby("hour")["feeder_kwh"].mean()
    vals = hourly_actual.reindex(range(24), fill_value=hourly_actual.mean())

    # ── Real Bi-LSTM predictions ───────────────────────────────────────────
    forecaster: DemandForecaster | None = _store.get("forecaster")
    ff_df: pd.DataFrame | None = _store.get("forecast_features")

    if forecaster is not None and ff_df is not None:
        dt_feats = ff_df[ff_df["dt_id"] == dt_id].sort_values("timestamp")
        feature_cols = [c for c in FORECAST_FEATURES if c in dt_feats.columns]
        if len(dt_feats) > forecaster.lookback_window and feature_cols:
            preds = forecaster.predict(dt_feats, feature_cols=feature_cols)  # (N, n_q)
            n_q = preds.shape[1]
            dt_feats = dt_feats.copy()
            dt_feats["hour"] = pd.DatetimeIndex(dt_feats["timestamp"]).hour
            dt_feats["_q0"] = preds[:, 0]
            dt_feats["_q1"] = preds[:, 1] if n_q >= 2 else preds[:, 0]
            dt_feats["_q2"] = preds[:, -1]
            hourly_q0  = dt_feats.groupby("hour")["_q0"].mean().reindex(range(24), fill_value=0)
            hourly_q1  = dt_feats.groupby("hour")["_q1"].mean().reindex(range(24), fill_value=0)
            hourly_q2  = dt_feats.groupby("hour")["_q2"].mean().reindex(range(24), fill_value=0)
            demand_series = [
                {
                    "h":      f"{h:02d}",
                    "actual": round(float(vals.get(h, 0)), 2),
                    "pred":   round(float(hourly_q1.get(h, 0)), 2),
                    "q5":     round(float(hourly_q0.get(h, 0)), 2),
                    "q95":    round(float(hourly_q2.get(h, 0)), 2),
                }
                for h in range(24)
            ]
        else:
            demand_series = _fallback_series(vals)
    else:
        demand_series = _fallback_series(vals)

    last = dt_slice.sort_values("timestamp").iloc[-1]
    cfg  = load_config()
    gsi_weights = cfg.gsi.weights.model_dump()

    feeder_load   = float(last["feeder_kwh"])
    baseline_load = float(dt_slice["feeder_kwh"].mean())
    cap_kva       = float(last["capacity_kva"])
    age           = float(last["age_years"])
    temp          = float(last["temperature_c"])
    ev_dens       = float(last.get("ev_density", 0.1))
    solar         = float(last.get("solar_irradiance", 0.0))
    ts            = pd.Timestamp(last["timestamp"])

    signals = {
        "load_quantile":        signal_load_quantile(feeder_load, cap_kva, 0.9),
        "temperature_derating": signal_temperature_derating(temp, cfg.gsi.temp_threshold, cfg.gsi.temp_derate_rate),
        "power_factor_penalty": signal_power_factor_penalty(0.9),
        "thermal_soak":         signal_thermal_soak(0.0, cfg.gsi.thermal_tau),
        "ev_load_risk":         signal_ev_load_risk(ev_dens, ts.hour),
        "transformer_age":      signal_transformer_age(age),
        "calendar_signal":      signal_calendar(ts.hour, ts.dayofweek, ts.month, is_holiday(ts.date())),
        "pv_duck_curve":        signal_pv_duck_curve(solar, feeder_load, baseline_load),
    }
    gsi = compute_gsi(signals=signals, weights=gsi_weights, u_tconf=1.0, mape_scale=1.0)
    label = gsi_label(gsi)

    action_map = {
        "Stable":   "No Action Required",
        "Caution":  "Monitor Load Trend",
        "Stressed": "Prepare Load-Shedding Plan",
        "Critical": "Immediate Grid Operator Notification",
    }

    stress_factors = [
        StressFactor(parameter="GSI Label",        value=label),
        StressFactor(parameter="Load Quantile",    value=f"{signals['load_quantile']:.2f}"),
        StressFactor(parameter="Temp Derating",    value="High" if signals["temperature_derating"] > 0.3 else "Normal"),
        StressFactor(parameter="Transformer Age",  value=f"{age:.0f} yrs"),
        StressFactor(parameter="EV Load Risk",     value=f"{signals['ev_load_risk']:.2f}"),
        StressFactor(parameter="Feeder Load",      value=f"{feeder_load:.1f} kWh"),
    ]

    return DemandDetailResponse(
        transformer_id=dt_id,
        gsi_score=round(gsi, 1),
        action=action_map.get(label, label),
        stress_factors=stress_factors,
        demand_series=demand_series,
    )


# ── /api/chat — Ollama LLM with keyword fallback ─────────────────────────────

def _keyword_response(msg: str) -> ChatResponse:
    """Keyword-based fallback chat response."""
    msg_lower = msg.lower()
    meter_match = re.search(r"mtr[_\-]?(\d+)", msg_lower)
    dt_match    = re.search(r"dt[_\-]?(\d+)", msg_lower)

    if meter_match:
        num     = meter_match.group(1).zfill(6)
        full_id = f"MTR_{num}"
        scores  = _store.get("meter_scores", pd.Series(dtype=float))
        risk    = float(scores.get(full_id, 68.0))
        label   = "Critical — immediate field inspection advised" if risk >= 80 \
                  else "Elevated — flag for next billing audit" if risk >= 50 \
                  else "Within normal operating parameters"
        return ChatResponse(
            response=f"CASS analysis for {full_id}: Risk Score {risk:.1f}/100. {label}. "
                     f"DTW divergence pattern loaded below.",
            trigger_xai=True,
            target_id=full_id,
            target_type="meter",
        )

    if dt_match:
        full_id = f"DT_{dt_match.group(1)}"
        return ChatResponse(
            response=f"Fetching GSI telemetry for transformer {full_id}. "
                     f"Demand forecast and stress analysis loaded into panel.",
            trigger_xai=True,
            target_id=full_id,
            target_type="transformer",
        )

    if any(k in msg_lower for k in ["theft", "anomal", "steal"]):
        scores = _store.get("meter_scores", pd.Series(dtype=float))
        top3   = scores.head(3).index.tolist() if not scores.empty else ["MTR_000001"]
        return ChatResponse(
            response=f"Top anomalous meters by CASS score: {', '.join(top3)}. "
                     f"Click any row in the Anomalous Cluster Feed table to inspect DTW divergence."
        )

    if any(k in msg_lower for k in ["stress", "gsi", "grid", "load"]):
        return ChatResponse(
            response="Grid Stress Index computed from 8 GSI signals: load quantile, temperature derating, "
                     "power factor penalty, thermal soak, EV load risk, transformer age, calendar signal, "
                     "and PV duck curve. Query a specific transformer (e.g. 'DT_1 stress') for details."
        )

    return ChatResponse(
        response="GridSentinel ready. Try querying a meter (e.g. 'MTR_000001 anomaly') "
                 "or transformer (e.g. 'DT_1 stress') for detailed analysis."
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Route natural language queries via Ollama LLM (with keyword fallback)."""
    msg = request.message.strip()
    msg_lower = msg.lower()

    # Determine trigger_xai and target from message patterns (same as before)
    meter_match = re.search(r"mtr[_\-]?(\d+)", msg_lower)
    dt_match    = re.search(r"dt[_\-]?(\d+)", msg_lower)
    trigger_xai = bool(meter_match or dt_match)
    target_id   = None
    target_type = None

    if meter_match:
        target_id   = f"MTR_{meter_match.group(1).zfill(6)}"
        target_type = "meter"
    elif dt_match:
        target_id   = f"DT_{dt_match.group(1)}"
        target_type = "transformer"

    # Build context string from live data
    context_lines = []
    if target_id and target_type == "meter":
        scores = _store.get("meter_scores", pd.Series(dtype=float))
        risk   = float(scores.get(target_id, 68.0))
        context_lines.append(f"Meter {target_id}: CASS risk score = {risk:.1f}/100, status = {_status_from_risk(risk)}.")
    elif target_id and target_type == "transformer":
        context_lines.append(f"Transformer {target_id} is being queried for GSI and demand forecast.")

    context_str = " ".join(context_lines)
    user_content = f"{msg}\n\nContext: {context_str}" if context_str else msg

    # Try Ollama first
    try:
        import ollama  # type: ignore

        def _call_ollama() -> str:
            result = ollama.chat(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
            )
            return result["message"]["content"]

        llm_text = await asyncio.to_thread(_call_ollama)
        return ChatResponse(
            response=llm_text,
            trigger_xai=trigger_xai,
            target_id=target_id,
            target_type=target_type,
        )

    except Exception as exc:
        logger.warning("Ollama unavailable (%s), using keyword fallback.", exc)
        return _keyword_response(msg)


# ── /api/stream/scores — Server-Sent Events ───────────────────────────────────

@app.get("/api/stream/scores")
async def stream_scores():
    """SSE stream: sends top 3 meter risk scores + DT_1 GSI every 10 seconds."""

    async def event_generator():
        while True:
            try:
                scores: pd.Series = _store.get("meter_scores", pd.Series(dtype=float))
                top3 = []
                if not scores.empty:
                    for meter_id, risk in scores.head(3).items():
                        noise = random.uniform(-2, 2)
                        top3.append({
                            "meter_id": str(meter_id),
                            "risk":     round(float(risk) + noise, 1),
                            "status":   _status_from_risk(float(risk)),
                        })
                else:
                    top3 = [
                        {"meter_id": "MTR_000001", "risk": round(88.0 + random.uniform(-2, 2), 1), "status": "critical"},
                        {"meter_id": "MTR_000002", "risk": round(71.0 + random.uniform(-2, 2), 1), "status": "high"},
                        {"meter_id": "MTR_000003", "risk": round(52.0 + random.uniform(-2, 2), 1), "status": "moderate"},
                    ]

                # Compute a simple GSI for DT_1 or use a demo value
                gsi_val = 62 + random.randint(-3, 3)

                payload = json.dumps({
                    "scores":    top3,
                    "gsi":       gsi_val,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                yield f"data: {payload}\n\n"
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("SSE stream error: %s", exc)
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── /api/map/zones ────────────────────────────────────────────────────────────

@app.get("/api/map/zones")
async def map_zones():
    """Return per-DT risk summary for the Koramangala grid map."""
    scores: pd.Series = _store.get("meter_scores", pd.Series(dtype=float))
    meter_df: pd.DataFrame | None = _store.get("meter_df")
    dt_df: pd.DataFrame | None    = _store.get("dt_df")

    if scores.empty or meter_df is None:
        # Demo fallback
        return [
            {"dt_id": "DT_1", "status": "critical", "gsi_score": 72, "meter_count": 10, "critical_meters": 3},
            {"dt_id": "DT_2", "status": "high",     "gsi_score": 58, "meter_count": 10, "critical_meters": 1},
            {"dt_id": "DT_3", "status": "moderate", "gsi_score": 44, "meter_count": 10, "critical_meters": 0},
            {"dt_id": "DT_4", "status": "normal",   "gsi_score": 28, "meter_count": 10, "critical_meters": 0},
            {"dt_id": "DT_5", "status": "high",     "gsi_score": 61, "meter_count": 10, "critical_meters": 2},
        ]

    # Map each meter to its DT
    meter_to_dt = (
        meter_df[["meter_id", "dt_id"]].drop_duplicates()
        .set_index("meter_id")["dt_id"].to_dict()
    )

    # Build per-DT aggregates from meter CASS scores
    dt_scores: dict = {}
    for meter_id, score in scores.items():
        dt_id = meter_to_dt.get(meter_id)
        if dt_id:
            dt_scores.setdefault(dt_id, []).append(float(score))

    # Per-DT GSI from dt_df last row
    dt_gsi: dict = {}
    if dt_df is not None:
        cfg = load_config()
        gsi_weights = cfg.gsi.weights.model_dump()
        for dt_id, grp in dt_df.groupby("dt_id"):
            last = grp.sort_values("timestamp").iloc[-1]
            feeder = float(last["feeder_kwh"])
            baseline = float(grp["feeder_kwh"].mean())
            cap = float(last["capacity_kva"])
            age = float(last["age_years"])
            temp = float(last["temperature_c"])
            ev = float(last.get("ev_density", 0.1))
            solar = float(last.get("solar_irradiance", 0.0))
            ts = pd.Timestamp(last["timestamp"])
            sigs = {
                "load_quantile":        signal_load_quantile(feeder, cap, 0.9),
                "temperature_derating": signal_temperature_derating(temp, cfg.gsi.temp_threshold, cfg.gsi.temp_derate_rate),
                "power_factor_penalty": signal_power_factor_penalty(0.9),
                "thermal_soak":         signal_thermal_soak(0.0, cfg.gsi.thermal_tau),
                "ev_load_risk":         signal_ev_load_risk(ev, ts.hour),
                "transformer_age":      signal_transformer_age(age),
                "calendar_signal":      signal_calendar(ts.hour, ts.dayofweek, ts.month, is_holiday(ts.date())),
                "pv_duck_curve":        signal_pv_duck_curve(solar, feeder, baseline),
            }
            dt_gsi[dt_id] = round(compute_gsi(signals=sigs, weights=gsi_weights, u_tconf=1.0, mape_scale=1.0), 1)

    result = []
    for dt_id, cass_list in sorted(dt_scores.items()):
        max_cass       = max(cass_list)
        critical_count = sum(1 for s in cass_list if s >= 80)
        high_count     = sum(1 for s in cass_list if 60 <= s < 80)
        # DT status: driven by worst meter
        if critical_count > 0:
            status = "critical"
        elif high_count > 0:
            status = "high"
        elif max_cass >= 35:
            status = "moderate"
        else:
            status = "normal"
        result.append({
            "dt_id":            dt_id,
            "status":           status,
            "gsi_score":        dt_gsi.get(dt_id, 50),
            "meter_count":      len(cass_list),
            "critical_meters":  critical_count,
            "max_cass":         round(max_cass, 1),
        })

    return result


# ── /api/economic/summary ─────────────────────────────────────────────────────

@app.get("/api/economic/summary")
async def economic_summary():
    """Return economic impact summary based on meter risk scores."""
    scores: pd.Series = _store.get("meter_scores", pd.Series(dtype=float))

    if scores.empty:
        # Demo fallback
        critical_count = 3
        watch_count    = 7
    else:
        critical_count = int((scores > 80).sum())
        watch_count    = int(((scores >= 50) & (scores <= 80)).sum())

    estimated_protection = critical_count * 5500 * 3
    investigation_cost   = watch_count * 8500
    net_benefit          = estimated_protection - investigation_cost

    return {
        "critical_count":            critical_count,
        "watch_count":               watch_count,
        "estimated_protection_inr":  estimated_protection,
        "investigation_cost_inr":    investigation_cost,
        "net_benefit_inr":           net_benefit,
    }


# ── /api/shap/{meter_id} ──────────────────────────────────────────────────────

@app.get("/api/shap/{meter_id}")
async def shap_detail(meter_id: str):
    """Return SHAP-proxy feature contributions for a meter (7 CASS signals)."""
    meter_df: pd.DataFrame | None = _store.get("meter_df")
    normalised = meter_id.replace("-", "_")

    # Defaults if no data
    default_signals = {
        "dtw_divergence":     0.35,
        "voltage_stability":  0.62,
        "billing_ratio":      0.48,
        "entropy":            0.29,
        "night_load_anomaly": 0.71,
        "dt_balance_error":   0.18,
        "repeat_anomaly":     0.44,
    }

    if meter_df is not None and not meter_df.empty:
        mask = meter_df["meter_id"] == normalised
        if not mask.any():
            mask = meter_df["meter_id"].str.contains(
                meter_id.split("_")[-1].lstrip("0") or "0", regex=False
            )
        meter_slice = meter_df[mask]
    else:
        meter_slice = pd.DataFrame()

    if meter_slice.empty:
        signals = default_signals
    else:
        meter_slice = meter_slice.copy()
        kwh_arr     = meter_slice["kwh"].values.astype(float)
        ts_index    = pd.DatetimeIndex(meter_slice["timestamp"].values)
        billed      = float(meter_slice["billed_kwh"].iloc[-1]) if "billed_kwh" in meter_slice.columns else float(kwh_arr.sum())
        total_kwh   = float(kwh_arr.sum())

        cfg = load_config()
        voltage_all = np.concatenate([
            meter_slice["voltage_r"].values,
            meter_slice["voltage_y"].values,
            meter_slice["voltage_b"].values,
        ]).astype(float)

        # DTW divergence: compare meter's hourly profile against population mean
        full_meter_df: pd.DataFrame | None = _store.get("meter_df")
        if full_meter_df is not None and not full_meter_df.empty:
            mean_hourly = (
                full_meter_df.assign(_h=full_meter_df["timestamp"].dt.hour)
                .groupby("_h")["kwh"].mean()
                .reindex(range(24), fill_value=0.0)
                .values
            )
            centroid_norm = max(float(np.linalg.norm(mean_hourly)), cfg.cass.dtw_norm_denom)
            meter_hourly = (
                meter_slice.assign(_h=ts_index.hour)
                .groupby("_h")["kwh"].mean()
                .reindex(range(24), fill_value=0.0)
                .values
            )
            dtw_val = float(np.clip(np.linalg.norm(meter_hourly - mean_hourly) / centroid_norm, 0.0, 1.0))
        else:
            dtw_val = 0.0

        # DT balance error: meter sum vs feeder total for the same transformer
        dt_df_store: pd.DataFrame | None = _store.get("dt_df")
        dt_id_val = meter_slice["dt_id"].iloc[0] if "dt_id" in meter_slice.columns else None
        if dt_df_store is not None and dt_id_val is not None and full_meter_df is not None:
            feeder_total = float(dt_df_store[dt_df_store["dt_id"] == dt_id_val]["feeder_kwh"].sum())
            meter_total  = float(full_meter_df[full_meter_df["dt_id"] == dt_id_val]["kwh"].sum())
            bal_val = signal_dt_balance_error(meter_total, feeder_total) if feeder_total > 0 else 0.0
        else:
            bal_val = 0.0

        signals = {
            "dtw_divergence":     dtw_val,
            "voltage_stability":  signal_voltage_stability(voltage_all, cfg.cass.voltage_std_norm),
            "billing_ratio":      signal_billing_ratio(billed, total_kwh),
            "entropy":            signal_entropy(kwh_arr),
            "night_load_anomaly": signal_night_load_anomaly(kwh_arr, ts_index),
            "dt_balance_error":   bal_val,
            "repeat_anomaly":     signal_repeat_anomaly(kwh_arr, window_count=cfg.cass.repeat_window_count),
        }

    features = []
    for feature_name, value in signals.items():
        weight       = _CASS_WEIGHTS.get(feature_name, 0.1)
        contribution = float(value) * weight
        features.append({
            "feature":      feature_name,
            "value":        round(float(value), 4),
            "contribution": round(contribution, 4),
        })

    features.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return features


# ── /api/report/summary — PDF generation ─────────────────────────────────────

@app.get("/api/report/summary")
async def report_summary():
    """Generate and return a PDF GridSentinel intelligence report."""
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError:
        raise HTTPException(status_code=500, detail="fpdf2 not installed. Run: pip install fpdf2")

    scores: pd.Series = _store.get("meter_scores", pd.Series(dtype=float))

    # Economic summary data
    if scores.empty:
        critical_count = 3
        watch_count    = 7
    else:
        critical_count = int((scores > 80).sum())
        watch_count    = int(((scores >= 50) & (scores <= 80)).sum())

    estimated_protection = critical_count * 5500 * 3
    investigation_cost   = watch_count * 8500
    net_benefit          = estimated_protection - investigation_cost

    # Top 5 meters
    if not scores.empty:
        top5 = [(str(mid), round(float(risk), 1), _status_from_risk(float(risk)))
                for mid, risk in scores.head(5).items()]
    else:
        top5 = [
            ("MTR_000001", 88.0, "critical"),
            ("MTR_000002", 74.5, "priority"),
            ("MTR_000003", 62.1, "high"),
            ("MTR_000004", 54.3, "high"),
            ("MTR_000005", 41.0, "moderate"),
        ]

    # GSI values per transformer — use real computed scores if available
    dt_df_store: pd.DataFrame | None = _store.get("dt_df")
    meter_df_store: pd.DataFrame | None = _store.get("meter_df")
    if dt_df_store is not None and meter_df_store is not None:
        cfg_pdf = load_config()
        gsi_weights_pdf = cfg_pdf.gsi.weights.model_dump()
        gsi_per_dt = []
        for dt_id, grp in dt_df_store.groupby("dt_id"):
            last = grp.sort_values("timestamp").iloc[-1]
            feeder = float(last["feeder_kwh"])
            baseline = float(grp["feeder_kwh"].mean())
            cap = float(last["capacity_kva"])
            age = float(last["age_years"])
            temp = float(last["temperature_c"])
            ev = float(last.get("ev_density", 0.1))
            solar = float(last.get("solar_irradiance", 0.0))
            ts_pdf = pd.Timestamp(last["timestamp"])
            sigs_pdf = {
                "load_quantile":        signal_load_quantile(feeder, cap, 0.9),
                "temperature_derating": signal_temperature_derating(temp, cfg_pdf.gsi.temp_threshold, cfg_pdf.gsi.temp_derate_rate),
                "power_factor_penalty": signal_power_factor_penalty(0.9),
                "thermal_soak":         signal_thermal_soak(0.0, cfg_pdf.gsi.thermal_tau),
                "ev_load_risk":         signal_ev_load_risk(ev, ts_pdf.hour),
                "transformer_age":      signal_transformer_age(age),
                "calendar_signal":      signal_calendar(ts_pdf.hour, ts_pdf.dayofweek, ts_pdf.month, is_holiday(ts_pdf.date())),
                "pv_duck_curve":        signal_pv_duck_curve(solar, feeder, baseline),
            }
            gsi_val = round(compute_gsi(signals=sigs_pdf, weights=gsi_weights_pdf, u_tconf=1.0, mape_scale=1.0))
            gsi_per_dt.append((str(dt_id), gsi_val))
        gsi_per_dt = sorted(gsi_per_dt, key=lambda x: x[1], reverse=True)[:5]
    else:
        gsi_per_dt = [
            ("DT_1", 62), ("DT_2", 55), ("DT_3", 38), ("DT_4", 71), ("DT_5", 48)
        ]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 50, 120)
    pdf.cell(0, 10, "GridSentinel AI -- BESCOM Intelligence Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", ln=True, align="C")
    pdf.ln(6)

    # Section 1: Top 5 Anomalous Meters
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Section 1: Top 5 Anomalous Meters", ln=True)
    pdf.ln(2)

    col_widths = [60, 35, 40]
    headers    = ["Meter ID", "Risk Score", "Status"]
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 230, 255)
    for hdr, w in zip(headers, col_widths):
        pdf.cell(w, 8, hdr, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    pdf.set_fill_color(245, 245, 255)
    for i, (mid, risk, stat) in enumerate(top5):
        fill = (i % 2 == 0)
        pdf.cell(col_widths[0], 7, mid,           border=1, fill=fill)
        pdf.cell(col_widths[1], 7, f"{risk}/100", border=1, fill=fill, align="C")
        pdf.cell(col_widths[2], 7, stat.upper(),  border=1, fill=fill, align="C")
        pdf.ln()

    pdf.ln(6)

    # Section 2: Economic Summary
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Section 2: Economic Impact Summary", ln=True)
    pdf.ln(2)

    eco_rows = [
        ("Critical Meters (risk > 80)",          str(critical_count)),
        ("Watch List Meters (risk 50-80)",        str(watch_count)),
        ("Estimated Revenue Protection (INR)",    f"Rs {estimated_protection:,}"),
        ("Investigation Cost (INR)",              f"Rs {investigation_cost:,}"),
        ("Net Benefit (INR)",                     f"Rs {net_benefit:,}"),
    ]
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 230, 255)
    pdf.cell(100, 8, "Parameter", border=1, fill=True)
    pdf.cell(70,  8, "Value",     border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for i, (param, val) in enumerate(eco_rows):
        fill = (i % 2 == 0)
        pdf.set_fill_color(245, 245, 255)
        pdf.cell(100, 7, param, border=1, fill=fill)
        pdf.cell(70,  7, val,   border=1, fill=fill)
        pdf.ln()

    pdf.ln(6)

    # Section 3: GSI per Transformer
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Section 3: Grid Stress Index per Transformer", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 230, 255)
    pdf.cell(60, 8, "Transformer ID", border=1, fill=True)
    pdf.cell(40, 8, "GSI Score",      border=1, fill=True)
    pdf.cell(60, 8, "Status",         border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for i, (dt_id, gsi) in enumerate(gsi_per_dt):
        fill = (i % 2 == 0)
        pdf.set_fill_color(245, 245, 255)
        gsi_status = "Critical" if gsi >= 75 else "Stressed" if gsi >= 55 else "Caution" if gsi >= 35 else "Stable"
        pdf.cell(60, 7, dt_id,      border=1, fill=fill)
        pdf.cell(40, 7, str(gsi),   border=1, fill=fill, align="C")
        pdf.cell(60, 7, gsi_status, border=1, fill=fill, align="C")
        pdf.ln()

    pdf_bytes = pdf.output()

    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=gridsentinel_report.pdf"},
    )


# ── Existing scored endpoints (unchanged) ─────────────────────────────────────

@app.post("/score/meter", response_model=MeterScoreResponse)
async def score_meter(request: MeterScoreRequest) -> MeterScoreResponse:
    """Score a single meter's theft suspicion (CASS)."""
    try:
        meter_id = request.meter_id
        df = pd.DataFrame(request.records)
        df = validate_meter_df(df)

        cfg = load_config()
        cass_weights = cfg.cass.weights.model_dump()

        kwh_arr     = df["kwh"].values.astype(float)
        voltage_all = np.concatenate([df["voltage_r"].values, df["voltage_y"].values, df["voltage_b"].values]).astype(float)
        ts_index    = pd.DatetimeIndex(df["timestamp"].values)
        billed      = float(df["billed_kwh"].iloc[-1])
        total_kwh   = float(kwh_arr.sum())
        pf_mean     = float(df["power_factor"].mean())

        signals = {
            "dtw_divergence":    0.0,
            "voltage_stability": signal_voltage_stability(voltage_all, cfg.cass.voltage_std_norm),
            "billing_ratio":     signal_billing_ratio(billed, total_kwh),
            "entropy":           signal_entropy(kwh_arr),
            "night_load_anomaly":signal_night_load_anomaly(kwh_arr, ts_index),
            "dt_balance_error":  0.0,
            "repeat_anomaly":    signal_repeat_anomaly(kwh_arr, window_count=cfg.cass.repeat_window_count),
        }

        score = compute_cass(signals=signals, weights=cass_weights,
                             sigmoid_shift=cfg.cass.sigmoid_shift, sigmoid_scale=cfg.cass.sigmoid_scale, g_pv=0.0)
        label    = cass_label(score)
        y_proba  = score / 100.0
        top_features: list[Dict[str, Any]] = []

        pipeline = _get_pipeline()
        if pipeline.theft_detector is not None and pipeline.theft_detector.model is not None:
            last_ts = pd.Timestamp(df["timestamp"].iloc[-1])
            trend_slope = float(np.polyfit(np.arange(len(kwh_arr), dtype=float), kwh_arr, 1)[0]) if len(kwh_arr) >= 2 else 0.0
            x_feat  = np.array([[
                signals["dtw_divergence"], signals["voltage_stability"], signals["billing_ratio"],
                signals["entropy"], signals["night_load_anomaly"], signals["dt_balance_error"],
                signals["repeat_anomaly"], pf_mean,
                float(kwh_arr[-672:].mean()) if len(kwh_arr) >= 672 else float(kwh_arr.mean()),
                float(kwh_arr[-672:].std())  if len(kwh_arr) >= 672 else float(kwh_arr.std()),
                trend_slope, float(last_ts.hour), float(last_ts.dayofweek), float(last_ts.month),
            ]])
            y_proba = float(pipeline.theft_detector.predict_proba(x_feat)[0])
            try:
                shap_vals    = compute_shap_values(pipeline.theft_detector.model, x_feat, TD_FEATURE_NAMES)
                top_features = get_top_shap_features(shap_vals, TD_FEATURE_NAMES, n=10)
            except Exception:
                top_features = [{"name": k, "shap_value": v} for k, v in signals.items()]

        return MeterScoreResponse(meter_id=meter_id, cass_score=score, cass_label=label,
                                  y_proba=y_proba, top_features=top_features)
    except Exception as exc:
        logger.error("score_meter failed: %s", exc, exc_info=True)
        raise


@app.post("/score/transformer", response_model=DTScoreResponse)
async def score_transformer(request: DTScoreRequest) -> DTScoreResponse:
    """Score a single transformer's grid stress (GSI)."""
    try:
        dt_id   = request.dt_id
        dt_df_r = pd.DataFrame(request.dt_records)
        dt_df_r = validate_dt_df(dt_df_r)

        cfg         = load_config()
        gsi_weights = cfg.gsi.weights.model_dump()
        last        = dt_df_r.sort_values("timestamp").iloc[-1]
        ts_last     = pd.Timestamp(last["timestamp"])

        feeder_load   = float(last["feeder_kwh"])
        baseline_load = float(dt_df_r["feeder_kwh"].mean())
        cap_kva       = float(last["capacity_kva"])
        age           = float(last["age_years"])
        temp          = float(last["temperature_c"])
        ev_dens       = float(last.get("ev_density", 0.1))
        solar         = float(last.get("solar_irradiance", 0.0))
        pf            = float(dt_df_r.get("power_factor", pd.Series([0.9])).mean()) if "power_factor" in dt_df_r.columns else 0.9
        mean_kwh      = float(dt_df_r["feeder_kwh"].mean())

        signals = {
            "load_quantile":        signal_load_quantile(mean_kwh * 1.1, cap_kva, pf),
            "temperature_derating": signal_temperature_derating(temp, cfg.gsi.temp_threshold, cfg.gsi.temp_derate_rate),
            "power_factor_penalty": signal_power_factor_penalty(pf),
            "thermal_soak":         signal_thermal_soak(0.0, cfg.gsi.thermal_tau),
            "ev_load_risk":         signal_ev_load_risk(ev_dens, ts_last.hour),
            "transformer_age":      signal_transformer_age(age),
            "calendar_signal":      signal_calendar(ts_last.hour, ts_last.dayofweek, ts_last.month, is_holiday(ts_last.date())),
            "pv_duck_curve":        signal_pv_duck_curve(solar, feeder_load, baseline_load),
        }

        gsi_score = compute_gsi(signals=signals, weights=gsi_weights, u_tconf=1.0, mape_scale=1.0)
        label     = gsi_label(gsi_score)
        return DTScoreResponse(dt_id=dt_id, gsi_score=gsi_score, gsi_label=label,
                               q5_forecast=mean_kwh * 0.9, q95_forecast=mean_kwh * 1.1)
    except Exception as exc:
        logger.error("score_transformer failed: %s", exc, exc_info=True)
        raise


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Run the full GridSentinel pipeline and return all metrics."""
    try:
        meter_df = pd.DataFrame(request.meter_records)
        dt_df    = pd.DataFrame(request.dt_records)
        pipeline = _get_pipeline()
        result: PipelineResult = pipeline.run(meter_df, dt_df)
        return EvaluateResponse(
            classification_metrics={k: float(v) for k, v in result.classification_metrics.items()},
            forecast_metrics={k: float(v) for k, v in result.forecast_metrics.items()},
            economic_cost_inr=float(result.economic_cost_inr),
            gss_core=float(result.gss_core),
            gss_final=float(result.gss_final),
            constraints_met=bool(result.constraints_met),
            shap_summary_path=result.shap_summary_path,
        )
    except Exception as exc:
        logger.error("evaluate failed: %s", exc, exc_info=True)
        raise


def create_app() -> FastAPI:
    """Return the FastAPI application instance."""
    return app


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port, workers=cfg.api.workers)
