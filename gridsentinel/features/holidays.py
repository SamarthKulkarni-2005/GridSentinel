"""Indian public holidays — static list used for GSI calendar signal."""
from __future__ import annotations

from datetime import date

# Indian public holidays for 2025 and 2026
HOLIDAYS: dict[str, str] = {
    # 2025
    "2025-01-01": "New Year's Day",
    "2025-01-14": "Makar Sankranti",
    "2025-01-26": "Republic Day",
    "2025-02-26": "Maha Shivaratri",
    "2025-03-14": "Holi",
    "2025-03-31": "Id-ul-Fitr (Eid)",
    "2025-04-06": "Ram Navami",
    "2025-04-10": "Mahavir Jayanti",
    "2025-04-14": "Dr. Ambedkar Jayanti",
    "2025-04-18": "Good Friday",
    "2025-05-12": "Buddha Purnima",
    "2025-06-07": "Id-ul-Zuha (Bakrid)",
    "2025-06-27": "Muharram",
    "2025-08-15": "Independence Day",
    "2025-08-16": "Janmashtami",
    "2025-09-05": "Milad-un-Nabi",
    "2025-10-02": "Gandhi Jayanti",
    "2025-10-02": "Mahatma Gandhi Birthday",
    "2025-10-20": "Dussehra",
    "2025-10-31": "Halloween / Sardar Patel Jayanti",
    "2025-11-05": "Diwali",
    "2025-11-15": "Guru Nanak Jayanti",
    "2025-12-25": "Christmas Day",
    # 2026
    "2026-01-01": "New Year's Day",
    "2026-01-14": "Makar Sankranti",
    "2026-01-26": "Republic Day",
    "2026-03-03": "Maha Shivaratri",
    "2026-03-20": "Holi",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Ambedkar Jayanti",
    "2026-05-01": "Labour Day",
    "2026-05-31": "Buddha Purnima",
    "2026-08-15": "Independence Day",
    "2026-10-02": "Gandhi Jayanti",
    "2026-10-19": "Diwali",
    "2026-11-03": "Guru Nanak Jayanti",
    "2026-12-25": "Christmas Day",
}


def is_holiday(dt: date) -> bool:
    """Return True if the given date is an Indian public holiday."""
    return dt.strftime("%Y-%m-%d") in HOLIDAYS


def get_holiday_name(dt: date) -> str | None:
    """Return holiday name for the given date, or None if not a holiday."""
    return HOLIDAYS.get(dt.strftime("%Y-%m-%d"))
