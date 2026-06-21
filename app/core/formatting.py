from __future__ import annotations

import math
from typing import Any

import pandas as pd

MISSING_VALUE = "—"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_fr_number(value: Any, decimals: int = 1, *, sign: bool = False, trim_zeros: bool = False) -> str:
    if _is_missing(value):
        return MISSING_VALUE
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return MISSING_VALUE
    if not math.isfinite(numeric):
        return MISSING_VALUE
    sign_prefix = ""
    if sign:
        sign_prefix = "+" if numeric > 0 else "−" if numeric < 0 else ""
    text = f"{abs(numeric) if sign else numeric:.{decimals}f}"
    if trim_zeros and "." in text:
        text = text.rstrip("0").rstrip(".")
    return sign_prefix + text.replace("-", "−").replace(".", ",")


def format_fr_unit(value: Any, unit: str = "", decimals: int = 1, *, sign: bool = False, trim_zeros: bool = False) -> str:
    number = format_fr_number(value, decimals=decimals, sign=sign, trim_zeros=trim_zeros)
    return number if number == MISSING_VALUE or not unit else f"{number} {unit}"


def format_fr_kg(value: Any, decimals: int = 1, *, sign: bool = False, trim_zeros: bool = False) -> str:
    return format_fr_unit(value, "kg", decimals, sign=sign, trim_zeros=trim_zeros)


def format_fr_kg_per_week(value: Any, decimals: int = 1, *, sign: bool = False) -> str:
    return format_fr_unit(value, "kg/semaine", decimals, sign=sign)


def format_fr_date(value: Any) -> str:
    if _is_missing(value):
        return MISSING_VALUE
    date = pd.Timestamp(value)
    if pd.isna(date):
        return MISSING_VALUE
    return date.strftime("%d/%m/%Y")
