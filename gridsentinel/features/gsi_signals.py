"""GSI signal functions — 8 signals for grid stress index scoring."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# EV charging hour profile (0-23) — peak risk at 18-20h
HOUR_PROFILE: dict[int, float] = {
    0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1,
    6: 0.3, 7: 0.5, 8: 0.4, 9: 0.3, 10: 0.2, 11: 0.2,
    12: 0.2, 13: 0.2, 14: 0.2, 15: 0.3, 16: 0.5, 17: 0.7,
    18: 0.9, 19: 1.0, 20: 0.9, 21: 0.7, 22: 0.5, 23: 0.3,
}


def signal_load_quantile(
    q95_kwh: float,
    capacity_kva: float,
    power_factor: float = 0.9,
) -> float:
    """Compute load utilisation signal at Q95 forecast level."""
    capacity_kw = capacity_kva * power_factor
    if capacity_kw <= 0:
        return 1.0
    return float(np.clip(q95_kwh / capacity_kw, 0.0, 1.0))


def signal_temperature_derating(
    temperature_c: float,
    threshold: float = 32.0,
    rate: float = 0.05,
) -> float:
    """Compute transformer derating signal due to high ambient temperature."""
    return float(np.clip(max(0.0, (temperature_c - threshold) * rate), 0.0, 1.0))


def signal_power_factor_penalty(power_factor: float) -> float:
    """Compute power factor penalty signal — low PF increases apparent load."""
    if power_factor <= 0:
        return 1.0
    return float(np.clip((1.0 / power_factor) ** 2 - 1.0, 0.0, 1.0))


def signal_thermal_soak(hours_above_80pct: float, tau: float = 4.0) -> float:
    """Compute thermal soak signal from sustained high loading."""
    return float(np.clip(1.0 - np.exp(-hours_above_80pct / max(tau, 1e-9)), 0.0, 1.0))


def signal_ev_load_risk(ev_density: float, hour: int) -> float:
    """Compute EV charging risk signal based on density and time of day."""
    hour_factor = HOUR_PROFILE.get(int(hour) % 24, 0.1)
    return float(np.clip(ev_density * hour_factor, 0.0, 1.0))


def signal_transformer_age(age_years: float) -> float:
    """Compute transformer age degradation signal."""
    return float(np.clip(np.log1p(age_years) / np.log(26.0), 0.0, 1.0))


def signal_calendar(
    hour: int,
    day_of_week: int,
    month: int,
    is_event: bool,
) -> float:
    """Compute calendar-based peak risk signal."""
    hour_risk = HOUR_PROFILE.get(int(hour) % 24, 0.1)

    # Weekday = 0 (Mon) to 6 (Sun)
    day_risk = 0.8 if day_of_week >= 5 else 0.5  # weekend/holiday higher

    # Indian seasonal/festive peak months
    if month in (5, 6, 10, 11, 12):
        month_risk = 0.9
    elif month in (1, 2, 3, 4):
        month_risk = 0.6
    else:
        month_risk = 0.5

    event_risk = 0.95 if is_event else hour_risk

    return float(np.clip(max(hour_risk, day_risk, month_risk, event_risk), 0.0, 1.0))


def signal_pv_duck_curve(
    solar_irradiance: float,
    feeder_load: float,
    baseline_load: float,
) -> float:
    """Compute PV duck curve risk signal from rapid load ramp at dusk."""
    if solar_irradiance == 0:
        return 0.0

    delta = abs(feeder_load - baseline_load) / (baseline_load + 1e-9)
    irradiance_factor = float(np.clip(solar_irradiance / 1000.0, 0.0, 1.0))
    return float(np.clip(delta * irradiance_factor, 0.0, 1.0))
