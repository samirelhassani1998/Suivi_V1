from __future__ import annotations

import pandas as pd

from app.core.time_utils import normalize_datetime_series


def _canonical_for_change_detection(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = normalize_datetime_series(out["Date"], normalize_day=True)
    if "Poids (Kgs)" in out.columns:
        out["Poids (Kgs)"] = pd.to_numeric(out["Poids (Kgs)"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return out.reset_index(drop=True)


def has_unsaved_changes(edited: pd.DataFrame, saved: pd.DataFrame) -> bool:
    """Return whether edited journal rows differ semantically from saved rows."""
    return not _canonical_for_change_detection(edited).equals(_canonical_for_change_detection(saved))
