from __future__ import annotations

import re

import pandas as pd

_ISO_DATE_RE = re.compile(r"^\s*\d{4}-\d{1,2}-\d{1,2}(?:[T\s].*)?$")


def _to_utc_naive(value, *, dayfirst: bool) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return pd.NaT
        if _ISO_DATE_RE.match(text):
            parsed = pd.to_datetime(text, errors="coerce", yearfirst=True, utc=True)
        else:
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst, utc=True)
    else:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).tz_convert(None)


def normalize_datetime_series(values, *, dayfirst: bool = True, normalize_day: bool = False) -> pd.Series:
    """Convert mixed date inputs to timezone-naive UTC timestamps deterministically.

    ISO-like strings (YYYY-MM-DD...) are parsed as ISO/year-first. Other strings
    keep the caller's day-first policy, which is used for French CSV dates.
    Existing datetime objects are parsed directly. Index and order are preserved.
    """
    series = pd.Series(values)
    parsed = pd.Series((_to_utc_naive(v, dayfirst=dayfirst) for v in series), index=series.index, dtype="datetime64[ns]")
    if normalize_day:
        parsed = parsed.dt.normalize()
    return parsed
