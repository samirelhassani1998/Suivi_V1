from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from app.core.business import FINAL_TARGET_WEIGHT_KG
from app.core.time_utils import normalize_datetime_series


@dataclass(frozen=True)
class TruncatedProjection:
    dates: list[pd.Timestamp]
    values: list[float]
    stop_date: pd.Timestamp | None


def _clean_points(dates: Iterable, values: Iterable) -> pd.DataFrame:
    out = pd.DataFrame({"Date": list(dates), "value": list(values)})
    if out.empty:
        return pd.DataFrame(columns=["Date", "value"])
    out["Date"] = normalize_datetime_series(out["Date"], normalize_day=False)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Date", "value"])
    return out.sort_values("Date", kind="mergesort").reset_index(drop=True)


def truncate_projection_at_floor(dates, values, floor_weight: float = FINAL_TARGET_WEIGHT_KG) -> TruncatedProjection:
    """Return a display projection stopped at the first crossing of floor_weight."""
    data = _clean_points(dates, values)
    if data.empty:
        return TruncatedProjection([], [], None)

    kept_dates: list[pd.Timestamp] = []
    kept_values: list[float] = []
    prev_date: pd.Timestamp | None = None
    prev_value: float | None = None

    for row in data.itertuples(index=False):
        date = pd.Timestamp(row.Date)
        value = float(row.value)
        if value <= floor_weight:
            if prev_date is not None and prev_value is not None and prev_value > floor_weight and value != prev_value:
                span_seconds = (date - prev_date).total_seconds()
                ratio = (prev_value - floor_weight) / (prev_value - value)
                crossing = prev_date + pd.Timedelta(seconds=span_seconds * max(0.0, min(1.0, ratio)))
            else:
                crossing = date
            if not kept_dates or kept_dates[-1] != crossing:
                kept_dates.append(crossing)
                kept_values.append(float(floor_weight))
            else:
                kept_values[-1] = float(floor_weight)
            return TruncatedProjection(kept_dates, kept_values, crossing)
        kept_dates.append(date)
        kept_values.append(value)
        prev_date, prev_value = date, value

    return TruncatedProjection(kept_dates, kept_values, None)


def constrain_interval_dataframe(df: pd.DataFrame, *, central_col: str = "prevision", lower_col: str = "borne_basse", upper_col: str = "borne_haute", floor_weight: float = FINAL_TARGET_WEIGHT_KG) -> pd.DataFrame:
    """Constrain display forecast intervals using the central forecast stop date."""
    if df is None or df.empty or "Date" not in df.columns or central_col not in df.columns:
        return pd.DataFrame(columns=[] if df is None else df.columns)
    truncated = truncate_projection_at_floor(df["Date"], df[central_col], floor_weight)
    if not truncated.dates:
        return df.iloc[0:0].copy()
    out = pd.DataFrame({"Date": truncated.dates, central_col: truncated.values})
    source = df.copy()
    source["Date"] = normalize_datetime_series(source["Date"], normalize_day=False)
    for col in [c for c in [lower_col, upper_col] if c in source.columns]:
        s = source[["Date", col]].dropna().sort_values("Date")
        vals = []
        x = s["Date"].astype("int64").to_numpy(dtype=float)
        y = pd.to_numeric(s[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        for d, center in zip(out["Date"], out[central_col]):
            raw = float(np.interp(pd.Timestamp(d).value, x, y)) if len(x) else center
            vals.append(max(raw, floor_weight))
        out[col] = vals
    if lower_col in out.columns:
        out[lower_col] = np.minimum(out[lower_col], out[central_col])
    if upper_col in out.columns:
        out[upper_col] = np.maximum(out[upper_col], out[central_col])
    for col in df.columns:
        if col not in out.columns and col != "Date":
            out[col] = df[col].iloc[0] if len(df[col]) else np.nan
    return out[df.columns.intersection(out.columns).tolist()]
