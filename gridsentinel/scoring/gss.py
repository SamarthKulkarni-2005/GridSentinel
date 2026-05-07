"""Unified System Score (GSS) — core and extended formulations."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def check_pareto_constraints(
    fpr: float,
    recall: float,
    mape: float,
    cfg: Dict[str, float],
) -> bool:
    """Check whether system meets Pareto-validity constraints.

    Returns False if any constraint is violated; GSS is then 0.0.
    """
    meets = (
        fpr < cfg["max_fpr"]
        and recall > cfg["min_recall"]
        and mape < cfg["max_mape"]
    )
    if not meets:
        logger.warning(
            "Pareto constraints FAILED: fpr=%.4f (max %.4f), "
            "recall=%.4f (min %.4f), mape=%.4f (max %.4f)",
            fpr, cfg["max_fpr"],
            recall, cfg["min_recall"],
            mape, cfg["max_mape"],
        )
    return meets


def compute_gss_core(
    s_cass: float,
    s_gsi: float,
    s_econ: float,
    s_robust: float,
    weights: Dict[str, float],
    constraints_met: bool,
) -> float:
    """Compute core Unified System Score (GSS).

    s_cass = MCC × (1 - FPR)
    s_gsi  = (1 - MAPE) × PICP × (1 - PINAW)
    s_econ = 1 - C(T) / C_baseline
    s_robust = 1 - |F1_clean - F1_noisy| / F1_clean
    Returns GSS in [0, 1].
    """
    if not constraints_met:
        return 0.0

    gss = (
        weights["w_cass"] * s_cass
        + weights["w_gsi"] * s_gsi
        + weights["w_econ"] * s_econ
        + weights["w_robust"] * s_robust
    )
    return round(float(np.clip(gss, 0.0, 1.0)), 4)


def compute_gss_final(
    s_cass: float,
    s_gsi: float,
    s_econ: float,
    s_robust: float,
    s_delay: float,
    s_energy: float,
    s_calib: float,
    weights: Dict[str, float],
    constraints_met: bool,
) -> float:
    """Compute extended (final) Unified System Score (GSS).

    Additional components:
    s_delay  = exp(-0.1 × detection_latency_days)
    s_energy = 1 - |Σmeter_kwh - dt_feeder_kwh| / dt_feeder_kwh
    s_calib  = 1 - ECE
    Returns GSS in [0, 1].
    """
    if not constraints_met:
        return 0.0

    gss = (
        weights["w_cass"]   * s_cass
        + weights["w_gsi"]    * s_gsi
        + weights["w_econ"]   * s_econ
        + weights["w_robust"] * s_robust
        + weights["w_delay"]  * s_delay
        + weights["w_energy"] * s_energy
        + weights["w_calib"]  * s_calib
    )
    return round(float(np.clip(gss, 0.0, 1.0)), 4)


def compute_s_delay(detection_latency_days: float) -> float:
    """Compute detection latency component: exp(-0.1 × latency_days)."""
    return float(np.clip(np.exp(-0.1 * detection_latency_days), 0.0, 1.0))


def compute_s_energy(sum_meter_kwh: float, dt_feeder_kwh: float) -> float:
    """Compute system-level energy consistency score."""
    if dt_feeder_kwh == 0:
        return 1.0
    error = abs(sum_meter_kwh - dt_feeder_kwh) / dt_feeder_kwh
    return float(np.clip(1.0 - error, 0.0, 1.0))
