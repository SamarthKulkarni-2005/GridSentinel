"""CASS score computation and labelling."""
from __future__ import annotations

import logging
from math import exp
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def compute_cass(
    signals: Dict[str, float],
    weights: Dict[str, float],
    sigmoid_shift: float,
    sigmoid_scale: float,
    g_pv: float,
) -> float:
    """Compute CASS score from 7 signals using weighted sigmoid.

    Returns CASS in [0, 100].
    g_pv: PV correction factor (0 = no solar effect, reduces false alarms).
    """
    raw = sum(weights[k] * signals[k] for k in weights if k in signals)
    sigmoid_val = 1.0 / (1.0 + exp(-((raw - sigmoid_shift) * sigmoid_scale)))
    cass = sigmoid_val * (1.0 - g_pv) * 100.0
    return round(float(np.clip(cass, 0.0, 100.0)), 2)


def cass_label(score: float) -> str:
    """Return human-readable CASS label and recommended action."""
    if score < 35:
        return "Normal"
    elif score < 60:
        return "Watch"
    elif score < 80:
        return "Inspect"
    else:
        return "Immediate"


def compute_g_pv(solar_irradiance: float) -> float:
    """Compute PV correction factor from solar irradiance.

    g_pv = clip(solar_irradiance / 1000 * 0.3, 0, 0.3)
    Reduces CASS up to 30% when solar is high.
    """
    return float(np.clip(solar_irradiance / 1000.0 * 0.3, 0.0, 0.3))
