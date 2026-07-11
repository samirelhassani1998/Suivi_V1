from __future__ import annotations

import pandas as pd


def _canonical_for_change_detection(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    if "Poids (Kgs)" in out.columns:
        out["Poids (Kgs)"] = pd.to_numeric(out["Poids (Kgs)"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return out.reset_index(drop=True)


def has_unsaved_changes(edited: pd.DataFrame, saved: pd.DataFrame) -> bool:
    """Return whether edited journal rows differ semantically from saved rows."""
    return not _canonical_for_change_detection(edited).equals(_canonical_for_change_detection(saved))
