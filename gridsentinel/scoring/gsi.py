"""GSI score computation and labelling."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def compute_gsi(
    signals: Dict[str, float],
    weights: Dict[str, float],
    u_tconf: float,
    mape_scale: float,
) -> float:
    """Compute GSI score from 8 signals weighted and modulated by forecast confidence.

    u_tconf: uncertainty factor = clip(1 - PINAW, 0, 1)
    mape_scale: clip(1 - MAPE, 0, 1)
    Returns GSI in [0, 100].
    """
    base = sum(weights[k] * signals[k] for k in weights if k in signals)
    gsi = base * u_tconf * mape_scale * 100.0
    return round(float(np.clip(gsi, 0.0, 100.0)), 2)


def gsi_label(score: float) -> str:
    """Return human-readable GSI label and recommended action."""
    if score < 30:
        return "Stable"
    elif score < 55:
        return "Caution"
    elif score < 75:
        return "Stressed"
    else:
        return "Critical"
