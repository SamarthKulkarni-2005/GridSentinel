"""Economic cost model for theft detection system evaluation."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_cost(
    fp_count: int,
    fn_count: int,
    fp_cost: float,
    fn_monthly_cost: float,
    theft_months: float,
) -> float:
    """Compute total investigation and loss cost in INR.

    C(T) = FP × fp_cost + FN × fn_monthly_cost × theft_months
    """
    cost = fp_count * fp_cost + fn_count * fn_monthly_cost * theft_months
    logger.debug(
        "Economic cost: FP=%d (×%.0f) + FN=%d (×%.0f×%.1f) = %.2f INR",
        fp_count, fp_cost, fn_count, fn_monthly_cost, theft_months, cost,
    )
    return float(cost)


def compute_baseline_cost(
    total_cases: int,
    fn_monthly_cost: float,
    theft_months: float,
) -> float:
    """Compute baseline cost assuming all theft cases are undetected (all FN).

    C_baseline = total_cases × fn_monthly_cost × theft_months
    """
    return float(total_cases * fn_monthly_cost * theft_months)
