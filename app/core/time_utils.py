from __future__ import annotations

import pandas as pd


def normalize_datetime_series(values, *, dayfirst: bool = True, normalize_day: bool = False) -> pd.Series:
    """Convert mixed naive/aware datetimes to timezone-naive UTC timestamps.

    Aware values are converted to UTC before dropping timezone information. Naive
    values are interpreted as already timezone-naive. Optional day normalization
    is only applied for explicitly daily business calculations.
    """
    series = pd.Series(values)
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, utc=True)
    parsed = parsed.dt.tz_convert(None)
    if normalize_day:
        parsed = parsed.dt.normalize()
    return parsed
